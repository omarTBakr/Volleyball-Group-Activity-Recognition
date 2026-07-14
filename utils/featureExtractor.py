"""
Frozen CNN feature extractor for the temporal baselines (B4+).

Wraps a ResNet backbone with its classification head removed and returns
one feature vector per image. Weights come either from torchvision's
ImageNet pretraining or from a fine-tuned project checkpoint (e.g.
Baseline 1's backbone, which is what B4 is defined on).

The module is **always frozen**: parameters have ``requires_grad=False``,
it stays in eval mode (BatchNorm uses running stats, never batch stats —
otherwise extracted features would depend on batch composition), and
``forward`` runs under ``torch.inference_mode()``.

Inputs must already be transformed (224×224 warp + ImageNet Normalize),
i.e. exactly what the project's transform pipelines produce.

Usage::

    from utils.featureExtractor import FeatureExtractor

    extractor = FeatureExtractor()                              # ImageNet weights
    extractor = FeatureExtractor(checkpoint="baseline1_run2.pt")  # B1 backbone
    feats = extractor(images)   # (B, 3, H, W) -> (B, 2048)
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

_SUPPORTED_BACKBONES = {
    "resnet50": models.ResNet50_Weights,
    "resnet101": models.ResNet101_Weights,
}


class FeatureExtractor(nn.Module):
    """
    Frozen ResNet feature extractor: images → feature vectors.

    Parameters
    ----------
    model_name : str
        Backbone name — one of ``"resnet50"``, ``"resnet101"``.
        Unknown names raise (no silent fallback).
    checkpoint : str or None
        Optional checkpoint filename under ``MODEL_SAVE_DIR`` (as saved by
        ``utils.utility.save_model``). Loads the backbone weights from it,
        e.g. Baseline 1's fine-tuned ResNet. ``None`` → ImageNet weights
        (``Weights.DEFAULT``, same as the training scripts).

    """

    def __init__(self, model_name: str = "resnet50", checkpoint: str | None = None) -> None:
        super().__init__()

        if model_name not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{model_name}'. "
                f"Choose from {sorted(_SUPPORTED_BACKBONES)}.",
            )

        weights = None if checkpoint else _SUPPORTED_BACKBONES[model_name].DEFAULT
        backbone = getattr(models, model_name)(weights=weights)

        self.feature_dim = backbone.fc.in_features

        if checkpoint:
            self._load_backbone_from_checkpoint(backbone, checkpoint)

        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Permanently frozen: no grads, BN in eval mode.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @staticmethod
    def _load_backbone_from_checkpoint(backbone: nn.Module, checkpoint: str) -> None:
        """Load backbone weights from a project checkpoint (head weights skipped)."""
        from configs.path_config import MODEL_SAVE_DIR

        path = MODEL_SAVE_DIR / checkpoint
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)

        # Checkpoints wrap the resnet as "backbone.<layer>..." and may carry a
        # trained fc head; strip the prefix and drop fc so shapes always match.
        cleaned = {
            k.removeprefix("module.").removeprefix("backbone."): v
            for k, v in state.items()
        }
        cleaned = {k: v for k, v in cleaned.items() if not k.startswith("fc.")}

        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        missing = [k for k in missing if not k.startswith("fc.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint '{checkpoint}' does not match the {type(backbone).__name__} "
                f"backbone (missing: {missing[:5]}, unexpected: {unexpected[:5]}). "
                "Check model_name against the checkpoint's architecture.",
            )

    def train(self, mode: bool = True):
        # Stay in eval mode even if a parent module calls .train() —
        # frozen BN running stats must not switch to batch statistics.
        super().train(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) — already transformed/normalized.
        returns : (B, feature_dim)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D batch (B, C, H, W), got shape {tuple(x.shape)}")
        # no_grad, not inference_mode: inference tensors cannot be saved for
        # backward, which breaks training any trainable layers (LSTM/MLP)
        # stacked on top of these features.
        with torch.no_grad():
            return self.backbone(x)


if __name__ == "__main__":
    extractor = FeatureExtractor()
    x = torch.randn(32, 3, 224, 224)
    feats = extractor(x)
    print(f"features: {feats.shape}")            # (32, 2048)
    assert feats.shape == (32, extractor.feature_dim)

    # Frozen-ness sanity checks
    assert not any(p.requires_grad for p in extractor.parameters())
    extractor.train()
    assert not extractor.backbone.training, "must stay in eval mode"

    # Same input twice → identical features (BN uses running stats)
    assert torch.equal(extractor(x), extractor(x))
    print("all checks passed")
