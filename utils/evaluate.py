"""
Post-training evaluation script for baseline models.

Loads a saved model checkpoint, runs inference on the test split,
and generates all standard visualisation plots (confusion matrix,
classification report, precision-recall AUC, mAP vs F1).

Usage
-----
    uv run python -m utils.evaluate --model baseline1_run1.pt
    uv run python -m utils.evaluate --model baseline1_run1.pt --baseline baseline3
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.labels import (
    IDX_TO_GROUP_ACTIVITY,
    NUM_GROUP_ACTIVITIES,
    NUM_PERSON_ACTIONS,
)
from configs.path_config import BASE_DIR
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from utils.load_model_config import build_transforms
from utils.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_map_f1,
    plot_precision_recall_auc,
)
from utils.utility import get_device, load_model

# Per-baseline batch_unpack callables follow the same contract as
# utils.utility.BatchUnpack: ``batch → (model_inputs_tuple, target_tensor)`` or
# ``None`` to skip a batch. Defining the type alias here so any future baseline
# can register without circular imports.
from collections.abc import Callable

BatchUnpack = Callable[[Any], tuple[tuple[torch.Tensor, ...], torch.Tensor] | None]

# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    IDX_TO_GROUP_ACTIVITY[i] for i in range(NUM_GROUP_ACTIVITIES)
]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Model & Dataset Factory
# ═════════════════════════════════════════════════════════════════════════════


def _detect_stage_b_pool(feature_dim: int, cfg: DictConfig) -> str:
    """Infer the right GroupActivityModel pool mode for a saved checkpoint.

    Old (pre-concat-pool) checkpoints have a classifier first-layer Linear of
    shape ``(hidden_dim, feature_dim)``; new ones have
    ``(hidden_dim, 2 * feature_dim)``. ``_build_model`` calls this *after* the
    state_dict has been read at the call site (see ``_load_checkpoint``), but
    that wouldn't work — we'd have to build the model before loading. So we
    open the checkpoint here, peek at the saved shape, and decide.

    Falls back to ``cfg.stage_b.pool`` (or ``"concat"``) when the checkpoint
    can't be found yet — that path is hit only for callers that pre-build a
    model with no checkpoint context, which evaluate.py doesn't do.

    Returns
    -------
    str
        ``"max"`` (legacy single-pool), or whatever ``cfg.stage_b.pool``
        specifies (``"max"`` / ``"mean"`` / ``"concat"``).
    """
    fname = _PENDING_CKPT.get("filename")
    fallback = cfg.stage_b.get("pool", "concat")
    if fname is None:
        return fallback

    from configs.path_config import MODEL_SAVE_DIR

    try:
        ckpt = torch.load(MODEL_SAVE_DIR / fname, map_location="cpu", weights_only=False)
        saved_in = ckpt["model_state_dict"]["classifier.0.weight"].shape[1]
    except (FileNotFoundError, KeyError):
        return fallback

    if saved_in == feature_dim:
        # Legacy single-pool checkpoint (max-only was the default at save time).
        print(
            f"  ⓘ Detected legacy single-pool Stage B checkpoint "
            f"(classifier_in={saved_in} = feature_dim). Using pool='max'."
        )
        return "max"
    if saved_in == 2 * feature_dim:
        print(
            f"  ⓘ Detected concat-pool Stage B checkpoint "
            f"(classifier_in={saved_in} = 2·feature_dim). Using pool='concat'."
        )
        return "concat"

    # Unknown shape — fall back to YAML and let load_state_dict surface a clear error.
    print(
        f"  ⚠ Unexpected classifier_in={saved_in} for feature_dim={feature_dim}; "
        f"falling back to cfg.stage_b.pool='{fallback}'."
    )
    return fallback


# Module-global passing the in-flight checkpoint filename to the model builder
# without changing the public _build_model signature. Set inside `evaluate()`
# just before _build_model is called; consulted by the checkpoint-shape
# detectors (_detect_stage_b_pool, _detect_lstm_shape).
_PENDING_CKPT: dict[str, str | None] = {"filename": None}


def _detect_lstm_shape(cfg: DictConfig) -> tuple[int, int]:
    """Infer (hidden_dim, num_layers) of a saved TemporalImageClassifier.

    Guards against config drift: if baseline4.yaml was edited after training
    (e.g. ``lstm.num_layers`` 1 → 2), building from the YAML alone would make
    ``load_state_dict`` fail. The saved LSTM weights carry the truth:
    ``lstm.weight_ih_l0`` has shape ``(4·hidden, input)`` and a key
    ``lstm.weight_ih_l{k}`` exists per layer k.
    """
    fallback = (cfg.lstm.hidden_dim, cfg.lstm.num_layers)
    fname = _PENDING_CKPT.get("filename")
    if fname is None:
        return fallback

    from configs.path_config import MODEL_SAVE_DIR

    try:
        ckpt = torch.load(MODEL_SAVE_DIR / fname, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        hidden = state["lstm.weight_ih_l0"].shape[0] // 4
        layers = 1 + max(
            (int(k.split("_l")[-1]) for k in state if k.startswith("lstm.weight_ih_l")),
        )
    except (FileNotFoundError, KeyError, ValueError):
        return fallback

    if (hidden, layers) != fallback:
        print(
            f"  ⓘ Checkpoint LSTM shape (hidden={hidden}, layers={layers}) "
            f"overrides cfg (hidden={fallback[0]}, layers={fallback[1]})."
        )
    return hidden, layers


def _build_model(baseline_name: str, cfg: DictConfig) -> nn.Module:
    """Instantiate the correct model architecture for *baseline_name*.

    This function is the single place to extend when adding new baselines.
    The model is returned uninitialized — ``_load_checkpoint`` will restore
    the trained weights from the saved state_dict.
    """
    if baseline_name == "baseline1":
        from models.baseline1 import Model as Baseline1Model

        return Baseline1Model(
            num_classes=NUM_GROUP_ACTIVITIES,
            backbone_name=cfg.model.name,
            dropout=cfg.model.get("dropout", 0.0),
        )

    if baseline_name == "baseline3":
        # baseline3 evaluates the Stage B group-activity model. It's a
        # GroupActivityModel that wraps a PersonActionResNet whose fc is
        # replaced by Identity at construction. The saved Stage B checkpoint
        # restores both the backbone weights and the MLP classifier head.
        #
        # We honor cfg.stage_b.pool by default, but old checkpoints saved
        # before the concat-pool change have a Linear(feature_dim, …) head
        # rather than Linear(2·feature_dim, …). We peek at the saved
        # classifier.0.weight shape and force pool="max" when it matches the
        # legacy shape — that's the only post-concat pool layout with the
        # same input dim as the legacy max-only head.
        from models.baseline3 import GroupActivityModel, PersonActionResNet

        person = PersonActionResNet(
            num_classes=NUM_PERSON_ACTIONS,
            backbone_name=cfg.model.name,
            pretrained=False,
        )
        pool = _detect_stage_b_pool(person.feature_dim, cfg)
        return GroupActivityModel(
            person_model=person,
            num_classes=NUM_GROUP_ACTIVITIES,
            hidden_dim=cfg.stage_b.hidden_dim,
            dropout=cfg.stage_b.get("dropout", 0.4),
            pool=pool,
        )

    if baseline_name == "baseline4":
        # Temporal full-image classifier: frozen extractor → LSTM → head.
        # Build with checkpoint=None (ImageNet init) — _load_checkpoint then
        # restores EVERYTHING from the saved baseline4 state_dict, including
        # the frozen extractor weights, so baseline1's checkpoint is not
        # needed at evaluation time. LSTM shape is read from the checkpoint
        # to survive later YAML edits.
        from models.baseline4 import TemporalImageClassifier

        lstm_hidden, lstm_layers = _detect_lstm_shape(cfg)
        return TemporalImageClassifier(
            num_classes=NUM_GROUP_ACTIVITIES,
            backbone_name=cfg.model.name,
            checkpoint='baseline1_run2.pt',
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=cfg.get("dropout", 0.3),
        )

    # B5–B8 (temporal crop baselines): add a branch here once their model
    # classes exist in models/ — mirror the baseline3 pattern (crop-mode
    # dataset + mask-aware batch_unpack) with n_frames from the YAML.
    raise ValueError(
        f"Evaluation not implemented for baseline: '{baseline_name}'. "
        "Supported: baseline1, baseline3, baseline4."
    )


def _dataset_kwargs_for(baseline_name: str, cfg: DictConfig) -> dict[str, Any]:
    """Return the ``VolleyballDataset`` keyword arguments for *baseline_name*."""
    if baseline_name == "baseline1":
        return {"full_image": True, "n_frames": 1}

    if baseline_name == "baseline3":
        return {"full_image": False, "crop": True, "n_frames": 1}

    if baseline_name == "baseline4":
        return {"full_image": True, "n_frames": cfg.get("n_frames", 9)}

    raise ValueError(
        f"Dataset config not defined for baseline: '{baseline_name}'. "
        "Supported: baseline1, baseline3, baseline4."
    )


def _batch_unpack_for(baseline_name: str) -> BatchUnpack:
    """Return the batch_unpack callable wired to *baseline_name*'s forward signature.

    Backward-compat default (anything not registered here): assume the legacy
    ``(data, target)`` 2-tuple batch contract with single-input
    ``model(data)`` — matches baseline1.
    """
    if baseline_name == "baseline3":
        # Reuse Stage B's exact unpacker — same contract as training time, so
        # the model sees identical inputs and there's no chance of drift.
        from models.baseline3 import stage_b_unpack

        return stage_b_unpack

    # Default: legacy 2-tuple ``(data, target)`` → ``model(data)``.
    # Covers baseline1 (single frame) and baseline4 (frame sequence) — both
    # use full-image mode whose collate emits plain (images, labels) batches.
    def _default(batch):
        if not batch or len(batch) < 2:
            return None
        return (batch[0],), batch[1]

    return _default


# ═════════════════════════════════════════════════════════════════════════════
# 2. Data Loading
# ═════════════════════════════════════════════════════════════════════════════


def _build_test_loader(
    cfg: DictConfig, dataset_kwargs: dict[str, Any]
) -> DataLoader:
    """Build and return the test-split ``DataLoader``."""
    transforms_dict = build_transforms(cfg)

    test_dataset = VolleyballDataset(
        mode="test",
        transform=transforms_dict["test"],
        **dataset_kwargs,
    )
    return DataLoader(
        test_dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Checkpoint Loading
# ═════════════════════════════════════════════════════════════════════════════


def _load_checkpoint(
    model: nn.Module, filename: str, device: torch.device
) -> nn.Module:
    """Load saved weights into *model*, move to *device*, and set eval mode."""
    print(f"Loading checkpoint '{filename}' ...")
    try:
        model, _, _, _, _ = load_model(filename, model)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        print(
            "Make sure the filename is correct (e.g. 'baseline1_run1.pt') "
            "and it exists in your saved_models/ folder."
        )
        sys.exit(1)

    model.to(device)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 4. Inference
# ═════════════════════════════════════════════════════════════════════════════


def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    batch_unpack: BatchUnpack | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return ``(y_true, y_pred, y_score)``.

    Parameters
    ----------
    model, loader, device
        Standard PyTorch inference setup.
    batch_unpack : BatchUnpack, optional
        Callable mapping a raw batch to ``(model_inputs_tuple, target)``.
        When ``None`` (default), falls back to the legacy 2-tuple contract
        ``(data, target)`` → ``model(data)`` — same behavior as the previous
        ``_unpack_batch`` for ``len(batch) == 2``. Crop-mode baselines must
        pass an unpacker that surfaces the mask alongside ``crops``.

    Returns
    -------
    (y_true, y_pred, y_score)
        Arrays of shape (N,), (N,), and (N, num_classes) respectively.
        ``y_score`` is the softmax over output logits, used by the PR-AUC
        and mAP plots.
    """
    if batch_unpack is None:
        def batch_unpack(b):
            if not b or len(b) < 2:
                return None
            return (b[0],), b[1]

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_score_all: list[np.ndarray] = []

    print(f"\nEvaluating on {device} ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if not batch:
                continue
            unpacked = batch_unpack(batch)
            if unpacked is None:
                continue
            inputs, target = unpacked
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
            if target.numel() == 0:
                continue

            inputs = tuple(t.to(device) for t in inputs)

            output = model(*inputs)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            y_true_all.extend(target.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            y_score_all.extend(probs.cpu().numpy())

    return (
        np.array(y_true_all),
        np.array(y_pred_all),
        np.array(y_score_all),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. Plot Generation
# ═════════════════════════════════════════════════════════════════════════════


def _generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    baseline_name: str,
) -> None:
    """Generate and save all evaluation plots."""
    print("\nGenerating plots ...")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, baseline=baseline_name)
    print("  ✓ Confusion matrix")

    plot_classification_report(y_true, y_pred, class_names, baseline=baseline_name)
    print("  ✓ Classification report")

    plot_precision_recall_auc(y_true, y_score, class_names, baseline=baseline_name)
    print("  ✓ Precision-Recall AUC curves")

    plot_map_f1(y_true, y_score, y_pred, class_names, baseline=baseline_name)
    print("  ✓ mAP & F1 bar chart")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Public Entry Point
# ═════════════════════════════════════════════════════════════════════════════


def evaluate(model_filename: str, baseline_name: str) -> None:
    """Evaluate a saved checkpoint and generate all metric plots.

    Parameters
    ----------
    model_filename : str
        Filename of the saved checkpoint (e.g. ``"baseline1_run1.pt"``).
    baseline_name : str
        Baseline identifier — used to select the model architecture,
        load the matching Hydra config, and name the output sub-folder
        under ``plots/``.
    """
    device = get_device()

    # ── Config ────────────────────────────────────────────────────────
    configs_dir = str(BASE_DIR / "configs")
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(config_name=baseline_name)

    # ── Pipeline ──────────────────────────────────────────────────────
    # Surface the checkpoint name to the builder so architecture details can
    # be auto-detected from the saved shapes (baseline3's pool mode,
    # baseline4's LSTM hidden/layers) instead of trusting a possibly-edited
    # YAML.
    _PENDING_CKPT["filename"] = model_filename
    try:
        model = _build_model(baseline_name, cfg)
    finally:
        _PENDING_CKPT["filename"] = None

    dataset_kwargs = _dataset_kwargs_for(baseline_name, cfg)
    batch_unpack = _batch_unpack_for(baseline_name)
    test_loader = _build_test_loader(cfg, dataset_kwargs)

    model = _load_checkpoint(model, model_filename, device)

    y_true, y_pred, y_score = _run_inference(
        model, test_loader, device, batch_unpack=batch_unpack,
    )

    _generate_plots(y_true, y_pred, y_score, CLASS_NAMES, baseline_name)

    print(
        f"\n✓ Done! All plots saved to 'plots/{baseline_name}/'."
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model and plot all metrics.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Filename of the saved model (e.g. 'baseline1_run1.pt')",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline1",
        help="Baseline identifier (defaults to 'baseline1')",
    )
    args = parser.parse_args()

    evaluate(args.model, args.baseline)
