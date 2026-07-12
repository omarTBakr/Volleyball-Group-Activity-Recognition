"""
Shared implementation for the volleyball dataset loaders.

``BaseVolleyballDataset`` holds every piece of logic that does not depend
on the physical frame storage: split filtering, frame-window selection,
person cropping, label mapping, ``__getitem__`` routing, and the collate
function.  Concrete subclasses only provide:

    _load_frame_index() -> dict[(video_id, clip_id), list[frame_name]]
    _load_image(video_id, clip_id, frame_name) -> PIL.Image

Two subclasses exist:

    src.data.data_loader.VolleyballDataset         — frames from LMDB
    src.data.kaggle_data_loader.VolleyballDataset  — frames from disk

Both keep their original public interface (constructor args, return
shapes, ``collate_fn``), so existing training scripts work unchanged.

Supported configurations (all baselines B1–B8):

    full_image=True,  n_frames=1  →  (image, group_label)                [B1]
    full_image=True,  n_frames=9  →  (images, group_label)               [B4]
    crop=True,        n_frames=1  →  (crops, person_labels, group_label) [B3]
    crop=True,        n_frames=9  →  (crops, person_labels, group_label) [B5-B8]
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch  # ty:ignore[unresolved-import]
from PIL import Image
from torch.utils.data import Dataset  # ty:ignore[unresolved-import]
from torchvision.transforms import ToTensor  # ty:ignore[unresolved-import]

from configs.data_split import (
    TEST_VIDEOS_NUMBERS,
    TRAIN_VIDEOS_NUMBERS,
    VALIDATION_VIDEO_NUMBERS,
)
from configs.labels import GROUP_ACTIVITY_TO_IDX, PERSON_ACTION_TO_IDX
from src.pickle_dump import load_from_pickle

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_video_ids_for_mode(mode: str) -> list[int]:
    """Return the video IDs for the given split ("train"/"validation"/"test")."""
    mode_map = {
        "train": TRAIN_VIDEOS_NUMBERS,
        "validation": VALIDATION_VIDEO_NUMBERS,
        "test": TEST_VIDEOS_NUMBERS,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {list(mode_map.keys())}.")
    return mode_map[mode]


# ── Dataset ──────────────────────────────────────────────────────────────────


class BaseVolleyballDataset(Dataset):
    """
    Storage-agnostic volleyball group-activity dataset.

    Parameters
    ----------
    mode : str
        Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
    n_frames : int
        Number of frames to sample per clip. Must be a positive odd number.
    full_image : bool
        If True, return full-resolution frames. Default True.
    crop : bool
        If True, return per-person crops from bounding boxes. Default False.
    transform : callable or None
        A torchvision transform applied to each image / crop.

    """

    def __init__(
        self,
        mode: str = "train",
        n_frames: int = 1,
        full_image: bool = True,
        crop: bool = False,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        if crop and full_image:
            raise ValueError("crop and full_image cannot both be True. Choose one.")
        if n_frames % 2 == 0 or n_frames <= 0:
            raise ValueError(
                "n_frames must be a positive odd number to have a clear middle frame.",
            )

        self.mode = mode
        self.n_frames = n_frames
        self.full_image = full_image
        self.crop = crop
        self.transform = transform

        self._master_data: dict = load_from_pickle()
        self._frame_index: dict[tuple[str, str], list[str]] = self._load_frame_index()

        self.samples: list[tuple[str, str, dict]] = []
        self._build_samples()

    # ── Storage hooks (implemented by subclasses) ────────────────────────

    def _load_frame_index(self) -> dict[tuple[str, str], list[str]]:
        """Return ``(video_id, clip_id) -> sorted list of frame filenames``."""
        raise NotImplementedError

    def _load_image(self, video_id: str, clip_id: str, frame_name: str) -> Image.Image:
        """Load one frame as a PIL RGB image."""
        raise NotImplementedError

    # ── Index building ────────────────────────────────────────────────────

    def _build_samples(self) -> None:
        """Filter the master pickle by the video IDs in the current split."""
        video_ids = _get_video_ids_for_mode(self.mode)
        valid_prefixes = {str(v) for v in video_ids}

        for clip_key, clip_data in self._master_data.items():
            # clip_key is "video_id/clip_id", e.g. "0/13286"
            video_id, clip_id = clip_key.split("/", 1)
            if video_id in valid_prefixes:
                self.samples.append((video_id, clip_id, clip_data))

    # ── Label mapping ─────────────────────────────────────────────────────

    def _group_label(self, video_id: str, clip_id: str, clip_data: dict) -> int:
        """
        Map the clip's ``scene_class`` string to its integer label.

        Fails loudly on labels missing from ``GROUP_ACTIVITY_TO_IDX`` —
        a silent fallback here once masked a corrupted annotation merge
        that mislabeled 62% of the dataset.
        """
        scene_class = clip_data.get("scene_class")
        if scene_class is None:
            logger.warning(
                "Clip %s/%s has no scene_class; defaulting to label 0. "
                "Re-run `python -m src.json_parser` to enrich the master JSON.",
                video_id, clip_id,
            )
            return 0
        if scene_class not in GROUP_ACTIVITY_TO_IDX:
            raise KeyError(
                f"Unknown scene_class '{scene_class}' for clip {video_id}/{clip_id}. "
                f"Expected one of {list(GROUP_ACTIVITY_TO_IDX)}. "
                "The master JSON/pickle is likely corrupted — regenerate it with "
                "`python -m src.json_parser` then `python -m src.pickle_dump`.",
            )
        return GROUP_ACTIVITY_TO_IDX[scene_class]

    # ── Frame selection ───────────────────────────────────────────────────

    def _find_valid_middle_index(
        self, all_frames: list[str], clip_id: str, clip_data: dict,
    ) -> int:
        """Find the index of the best middle frame, ensuring annotations exist if cropping."""
        middle_name = f"{clip_id}.jpg"
        if middle_name in all_frames:
            best_idx = all_frames.index(middle_name)
        else:
            best_idx = len(all_frames) // 2

        # If we need crops, ensure the selected middle frame actually has person annotations
        if self.crop:
            search_offsets = [0]
            for i in range(1, len(all_frames)):
                search_offsets.extend([i, -i])

            for offset in search_offsets:
                idx = best_idx + offset
                if 0 <= idx < len(all_frames):
                    fname = all_frames[idx]
                    if self._get_persons_for_frame(clip_data, fname):
                        return idx
        return best_idx

    def _pad_frame_sequence(
        self, selected: list[str], all_frames: list[str], start: int, end: int,
    ) -> list[str]:
        """Pad a sequence of frames symmetrically at boundaries to ensure length == n_frames."""
        while len(selected) < self.n_frames:
            if start > 0:
                start -= 1
                selected.insert(0, all_frames[start])
            elif end < len(all_frames):
                selected.append(all_frames[end])
                end += 1
            elif len(selected) > 0:
                # Duplicate extreme boundary frame if completely out of physical frames
                selected.append(selected[-1])
            else:
                break
        return selected

    def _select_frame_names(
        self, video_id: str, clip_id: str, clip_data: dict,
    ) -> list[str]:
        """Select exactly ``n_frames`` filenames centered on the best valid middle frame."""
        all_frames = self._frame_index.get((video_id, clip_id), [])
        if not all_frames:
            return []

        mid_idx = self._find_valid_middle_index(all_frames, clip_id, clip_data)

        half = self.n_frames // 2
        start = max(0, mid_idx - half)
        end = min(len(all_frames), mid_idx + half + 1)

        selected = all_frames[start:end]
        return self._pad_frame_sequence(selected, all_frames, start, end)

    # ── Person annotations ────────────────────────────────────────────────

    def _get_persons_for_frame(self, clip_data: dict, frame_name: str) -> list[dict]:
        """
        Return person entries for a specific frame.

        Prefers tracking data (which carries consistent player IDs) and
        falls back to action detections when tracking is unavailable.
        """
        persons = clip_data.get("tracking", {}).get(frame_name, [])
        if persons:
            return persons
        return clip_data.get("actions", {}).get(frame_name, [])

    def _crop_persons(
        self, image: Image.Image, persons: list[dict],
    ) -> tuple[list[Image.Image], list[int]]:
        """
        Crop person regions from the image and collect action labels.

        Handles both tracking boxes ``[x1, y1, x2, y2]`` and detection
        boxes ``[x, y, w, h]``.
        """
        crops, labels = [], []
        img_w, img_h = image.size

        for person in persons:
            box = person["box"]

            # Tracking uses [x1, y1, x2, y2]; detections use [x, y, w, h]
            if "id" in person:
                x1, y1, x2, y2 = box
            else:
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crops.append(image.crop((x1, y1, x2, y2)))
            action = person.get("action", "standing")
            labels.append(PERSON_ACTION_TO_IDX.get(action, PERSON_ACTION_TO_IDX["standing"]))

        return crops, labels

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Return a sample depending on the configured mode.

        Returns
        -------
        For ``full_image=True, n_frames=1``:
            ``(image_tensor, group_label)``

        For ``full_image=True, n_frames>1``:
            ``(images_tensor [T,C,H,W], group_label)``

        For ``crop=True, n_frames=1``:
            ``(crops_tensor [P,C,H,W], person_labels [P], group_label)``

        For ``crop=True, n_frames>1``:
            ``(crops_tensor [T,P,C,H,W], person_labels [P], group_label)``

        """
        video_id, clip_id, clip_data = self.samples[idx]

        group_label = self._group_label(video_id, clip_id, clip_data)
        frame_names = self._select_frame_names(video_id, clip_id, clip_data)

        if self.full_image:
            return self._getitem_full_image(video_id, clip_id, frame_names, group_label)
        return self._getitem_crop(video_id, clip_id, frame_names, clip_data, group_label)

    def _getitem_full_image(
        self,
        video_id: str,
        clip_id: str,
        frame_names: list[str],
        group_label: int,
    ) -> tuple[torch.Tensor, int]:
        """Load full frame(s) and return as tensor(s)."""
        if not frame_names:
            raise RuntimeError(
                f"No frames found for video={video_id} clip={clip_id}. "
                "Check that the frame storage (LMDB / dataset directory) is complete.",
            )

        images = []
        for fname in frame_names:
            img = self._load_image(video_id, clip_id, fname)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        if self.n_frames == 1:
            return images[0], group_label
        return torch.stack(images, dim=0), group_label

    def _getitem_crop(
        self,
        video_id: str,
        clip_id: str,
        frame_names: list[str],
        clip_data: dict,
        group_label: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load cropped person images and return as tensor(s)."""
        _tf = self.transform or ToTensor()

        if self.n_frames == 1:
            middle_frame = frame_names[len(frame_names) // 2] if frame_names else None
            if middle_frame is None:
                return torch.empty(0), torch.empty(0, dtype=torch.long), group_label

            img = self._load_image(video_id, clip_id, middle_frame)
            persons = self._get_persons_for_frame(clip_data, middle_frame)
            crops, person_labels = self._crop_persons(img, persons)
            crops = [_tf(c) for c in crops]

            if not crops:
                return torch.empty(0), torch.empty(0, dtype=torch.long), group_label

            return (
                torch.stack(crops, dim=0),                       # (P, C, H, W)
                torch.tensor(person_labels, dtype=torch.long),   # (P,)
                group_label,
            )

        # Temporal crops — use tracking data per frame for accurate boxes
        all_frame_crops = []
        last_person_labels = []

        for fname in frame_names:
            img = self._load_image(video_id, clip_id, fname)
            persons = self._get_persons_for_frame(clip_data, fname)
            crops, person_labels = self._crop_persons(img, persons)
            crops = [_tf(c) for c in crops]

            if crops:
                all_frame_crops.append(torch.stack(crops, dim=0))  # (P, C, H, W)
                last_person_labels = person_labels
            else:
                all_frame_crops.append(None)

        valid_frames = [f for f in all_frame_crops if f is not None]
        if not valid_frames:
            return torch.empty(0), torch.empty(0, dtype=torch.long), group_label

        # Frames can carry different person counts (tracking dropouts /
        # detection fallback) — zero-pad each frame to the clip max so the
        # temporal stack is rectangular instead of crashing.
        max_p = max(f.shape[0] for f in valid_frames)
        if any(f.shape[0] != max_p for f in valid_frames):
            valid_frames = [
                torch.cat([f, f.new_zeros(max_p - f.shape[0], *f.shape[1:])], dim=0)
                if f.shape[0] < max_p else f
                for f in valid_frames
            ]

        return (
            torch.stack(valid_frames, dim=0),                    # (T, P, C, H, W)
            torch.tensor(last_person_labels, dtype=torch.long),  # (P,)
            group_label,
        )


# ── Collate function ─────────────────────────────────────────────────────────


def _pad_and_stack_crops(
    crops_list: Sequence[torch.Tensor],
    person_labels_list: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-pad batch crops (temporal or single) up to maximum player/time counts."""
    max_players = max(
        (c.shape[0] if c.dim() == 4 else c.shape[1] for c in crops_list if c.numel() > 0),
        default=0,
    )

    if max_players == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)

    batch_size = len(crops_list)
    sample_shape = next(c for c in crops_list if c.numel() > 0).shape

    if len(sample_shape) == 4:
        # Single-frame crops (P, C, H, W)
        _, C, H, W = sample_shape
        padded_crops = torch.zeros(batch_size, max_players, C, H, W)
        padded_labels = torch.zeros(batch_size, max_players, dtype=torch.long)
        masks = torch.zeros(batch_size, max_players, dtype=torch.bool)

        for i, (crops, plabels) in enumerate(zip(crops_list, person_labels_list)):
            if crops.numel() == 0:
                continue
            n = crops.shape[0]
            padded_crops[i, :n] = crops
            padded_labels[i, :n] = plabels
            masks[i, :n] = True

    elif len(sample_shape) == 5:
        # Temporal crops (T, P, C, H, W)
        max_T = max((c.shape[0] for c in crops_list if c.numel() > 0), default=0)
        _, _, C, H, W = sample_shape
        padded_crops = torch.zeros(batch_size, max_T, max_players, C, H, W)
        padded_labels = torch.zeros(batch_size, max_players, dtype=torch.long)
        masks = torch.zeros(batch_size, max_players, dtype=torch.bool)

        for i, (crops, plabels) in enumerate(zip(crops_list, person_labels_list)):
            if crops.numel() == 0:
                continue
            t, n = crops.shape[0], crops.shape[1]
            padded_crops[i, :t, :n] = crops
            padded_labels[i, :n] = plabels
            masks[i, :n] = True
    else:
        raise ValueError(f"Unexpected crop shape: {sample_shape}")

    return padded_crops, padded_labels, masks


def collate_fn(batch: list[tuple[Any, ...]]) -> tuple[torch.Tensor, ...]:
    """
    Custom collate function for variable numbers of player crops per clip.

    Handles batches from both full-image mode (2-tuples) and crop mode
    (3-tuples).  For crop mode, pads the player dimension to the maximum
    count in the batch and returns a mask indicating valid players.

    Returns
    -------
    For full-image mode:
        ``(images_batch, labels_batch)``

    For crop mode:
        ``(crops_batch, person_labels_batch, group_labels_batch, masks_batch)``

    """
    if not batch:
        return ()

    if len(batch[0]) == 2:
        # Full-image mode
        images, labels = zip(*batch)
        shapes = [img.shape for img in images]

        if all(s == shapes[0] for s in shapes):
            return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)

        # Variable temporal size (n_frames > 1 with missing physical frames)
        max_T = max(s[0] for s in shapes)
        _, C, H, W = shapes[0]
        padded_images = torch.zeros(len(images), max_T, C, H, W)
        for i, img in enumerate(images):
            padded_images[i, : img.shape[0]] = img

        return padded_images, torch.tensor(labels, dtype=torch.long)

    # Crop mode — variable number of players
    crops_list, person_labels_list, group_labels = zip(*batch)
    padded_crops, padded_labels, masks = _pad_and_stack_crops(crops_list, person_labels_list)

    return (
        padded_crops,
        padded_labels,
        torch.tensor(group_labels, dtype=torch.long),
        masks,
    )
