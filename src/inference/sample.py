"""Pick a clip from the validation split and package it for B8."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision import transforms

from configs.labels import IDX_TO_GROUP_ACTIVITY
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from src.data.unpackers import group_team_unpack

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_DATASETS: dict[str, VolleyballDataset] = {}


@dataclass
class ClipSample:
    """One clip: model inputs plus the raw frames/boxes needed for overlays."""

    video_id: str
    clip_id: str
    index: int
    gt_index: int
    crops: torch.Tensor      # (1, T, P, 3, 224, 224)
    masks: torch.Tensor      # (1, P)
    team_ids: torch.Tensor   # (1, P)
    frames: list[np.ndarray] = field(repr=False)          # T RGB uint8 frames
    boxes: list[np.ndarray] = field(repr=False)           # T arrays (P_t, 4) xyxy, aligned with crops

    @property
    def gt_label(self) -> str:
        return IDX_TO_GROUP_ACTIVITY[self.gt_index]


def get_dataset(mode: str = "validation") -> VolleyballDataset:
    """Cached team-aware crop dataset for *mode* (9 frames per clip, B8's setup)."""
    if mode not in _DATASETS:
        _DATASETS[mode] = VolleyballDataset(
            mode=mode, n_frames=9, full_image=False, crop=True,
            transform=_TF, with_teams=True,
        )
    return _DATASETS[mode]


def _valid_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp boxes and drop degenerate ones, mirroring ``_crop_boxes`` so rows stay aligned with crops."""
    kept = []
    for x1, y1, x2, y2 in boxes.tolist():
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            kept.append((x1, y1, x2, y2))
    return np.asarray(kept, dtype=np.int32).reshape(-1, 4)


def load_clip(index: int, mode: str = "validation") -> ClipSample | None:
    """Load clip *index* of *mode*; ``None`` if it has no usable player crops."""
    ds = get_dataset(mode)
    batch = collate_fn([ds[index]])
    unpacked = group_team_unpack(batch)
    if unpacked is None:
        return None
    (crops, masks, team_ids), glabels = unpacked

    video_id, clip_id, _, frame_names, persons = ds._records[index]
    frames, boxes = [], []
    for fname in frame_names:
        img = ds._load_image(video_id, clip_id, fname)
        frames.append(np.asarray(img))
        boxes.append(_valid_boxes(persons[fname][0], *img.size))

    return ClipSample(
        video_id=video_id, clip_id=clip_id, index=index, gt_index=int(glabels[0]),
        crops=crops, masks=masks, team_ids=team_ids, frames=frames, boxes=boxes,
    )


def find_clip(video_id: str, clip_id: str, mode: str = "validation") -> ClipSample:
    """Load a specific clip; raises ``KeyError`` if it isn't in *mode*'s split."""
    ds = get_dataset(mode)
    for i, (vid, cid, _) in enumerate(ds.samples):
        if vid == str(video_id) and cid == str(clip_id):
            sample = load_clip(i, mode)
            if sample is None:
                raise KeyError(f"Clip {video_id}/{clip_id} has no usable player crops")
            return sample
    raise KeyError(f"Clip {video_id}/{clip_id} is not in the {mode} split")


def random_validation_clip(seed: int | None = None) -> ClipSample:
    """A random validation clip (reproducible when *seed* is given)."""
    ds = get_dataset("validation")
    rng = random.Random(seed)
    for _ in range(50):
        sample = load_clip(rng.randrange(len(ds)), "validation")
        if sample is not None:
            return sample
    raise RuntimeError("Could not find a validation clip with player crops")
