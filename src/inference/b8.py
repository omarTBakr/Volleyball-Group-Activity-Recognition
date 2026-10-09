"""Load Baseline 8 from a checkpoint and run it on one clip.

The architecture is recovered from the checkpoint's tensor shapes (same
helpers ``utils.evaluate`` uses), so no Hydra config is needed. The Stage-B
checkpoint already contains the ResNet backbone, so neither the ImageNet
weights nor the B3 checkpoint are required at inference time.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.labels import IDX_TO_GROUP_ACTIVITY, NUM_GROUP_ACTIVITIES, NUM_PERSON_ACTIONS
from models.baseline8 import GroupTemporalClassifier, PersonTemporalLSTM, _group_dims, _person_dims
from utils.utility import remap_sequential_indices

DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "saved_models" / "baseline8_stage_b_run1.pt"


def default_ckpt() -> Path:
    """Checkpoint path: ``$VB_B8_CKPT`` if set, else the repo's saved model."""
    return Path(os.environ.get("VB_B8_CKPT", DEFAULT_CKPT))


def load_b8(ckpt_path: str | Path | None = None, device: str | torch.device = "cpu") -> GroupTemporalClassifier:
    """Build B8 from *ckpt_path* and return it in eval mode on *device*."""
    ckpt_path = Path(ckpt_path) if ckpt_path else default_ckpt()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"B8 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    lstm1_hidden, lstm1_layers = _person_dims(state, prefix="person.")
    lstm2_hidden, pool, T, head_hidden = _group_dims(state)

    person = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name="resnet50",
        checkpoint=None,
        lstm_hidden=lstm1_hidden,
        lstm_layers=lstm1_layers,
        pretrained_backbone=False,
    )
    model = GroupTemporalClassifier(
        person_model=person,
        num_classes=NUM_GROUP_ACTIVITIES,
        lstm2_hidden=lstm2_hidden,
        pool=pool,
        T=T,
        hidden_dim=head_hidden,
    )
    model.load_state_dict(remap_sequential_indices(state, model))
    return model.to(device).eval()


@torch.no_grad()
def predict_clip(
    model: GroupTemporalClassifier,
    crops: torch.Tensor,
    masks: torch.Tensor,
    team_ids: torch.Tensor,
) -> dict:
    """Run B8 on one batched clip ``(1, T, P, 3, 224, 224)``.

    Returns ``{"label", "index", "probs"}`` where ``probs`` maps each of the
    8 group-activity names to its softmax probability.
    """
    device = next(model.parameters()).device
    logits = model(crops.to(device), masks.to(device), team_ids.to(device))
    probs = F.softmax(logits.float(), dim=1)[0].cpu()
    index = int(probs.argmax())
    return {
        "label": IDX_TO_GROUP_ACTIVITY[index],
        "index": index,
        "probs": {IDX_TO_GROUP_ACTIVITY[i]: float(p) for i, p in enumerate(probs)},
    }
