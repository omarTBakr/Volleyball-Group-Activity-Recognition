"""
Kaggle-specific volleyball dataset loader — identical interface to
data_loader.py but reads frames directly from disk instead of LMDB.

Use this on Kaggle where /kaggle/working/ lacks the space to build
the LMDB (~50 GB). Frames are served straight from the read-only
input mount at MAIN_DATASET_DIR, which is already on a fast SSD.

Drop-in replacement: just change the import in your training script:

    # instead of:
    from src.data.data_loader import VolleyballDataset, collate_fn
    # use:
    from src.data.data_loader_kaggle import VolleyballDataset, collate_fn

Everything else (constructor args, return shapes, collate_fn) is identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor

from configs.data_split import (
    TEST_VIDEOS_NUMBERS,
    TRAIN_VIDEOS_NUMBERS,
    VALIDATION_VIDEO_NUMBERS,
)
from configs.labels import GROUP_ACTIVITY_TO_IDX, PERSON_ACTION_TO_IDX
from configs.path_config import MAIN_DATASET_DIR
from src.pickle_dump import load_from_pickle

# ── Frame index ──────────────────────────────────────────────────────────────

_FRAME_INDEX_CACHE: dict[tuple[str, str], list[str]] | None = None


def _build_frame_index(
    dataset_dir: Path,
) -> dict[tuple[str, str], list[str]]:
    """
    Walk dataset_dir once and build a mapping:
        (video_id, clip_id) -> sorted list of frame filenames

    Result is cached in memory so the directory walk only happens once
    per process.
    """
    global _FRAME_INDEX_CACHE
    if _FRAME_INDEX_CACHE is not None:
        return _FRAME_INDEX_CACHE

    index: dict[tuple[str, str], list[str]] = {}

    for video_dir in sorted(dataset_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name

        for clip_dir in sorted(video_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            clip_id = clip_dir.name

            frames = sorted(f.name for f in clip_dir.iterdir() if f.suffix == ".jpg")
            if frames:
                index[(video_id, clip_id)] = frames

    _FRAME_INDEX_CACHE = index
    return _FRAME_INDEX_CACHE


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_video_ids_for_mode(mode: str) -> list[int]:
    """Retrieve video IDs based on the requested dataset split mode."""
    mode_map = {
        "train": TRAIN_VIDEOS_NUMBERS,
        "validation": VALIDATION_VIDEO_NUMBERS,
        "test": TEST_VIDEOS_NUMBERS,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {list(mode_map.keys())}.")
    return mode_map[mode]


# ── Dataset ──────────────────────────────────────────────────────────────────


class VolleyballDataset(Dataset):
    """
    Kaggle-compatible volleyball dataset loader.

    Identical interface to the original VolleyballDataset in data_loader.py
    but loads frames directly from disk instead of an LMDB database.

    Parameters
    ----------
    mode : str
        Dataset split — "train", "validation", or "test".
    n_frames : int
        Number of frames to sample per clip. Must be a positive odd number.
    full_image : bool
        If True, return full-resolution frames. Default True.
    crop : bool
        If True, return per-person crops. Default False.
    transform : callable or None
        Torchvision transform applied to every image / crop.
    dataset_dir : Path or None
        Override for the frames root directory. Defaults to MAIN_DATASET_DIR.

    """

    def __init__(
        self,
        mode: str = "train",
        n_frames: int = 1,
        full_image: bool = True,
        crop: bool = False,
        transform: Callable | None = None,
        dataset_dir: Path | str | None = None,
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
        
        # Verify Dataset Directory
        self.dataset_dir = Path(dataset_dir) if dataset_dir else MAIN_DATASET_DIR
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found or not a directory: {self.dataset_dir}"
            )

        # Load annotation data
        self._master_data: dict = load_from_pickle()

        # Build / retrieve cached frame index (one disk walk per process)
        self._frame_index = _build_frame_index(self.dataset_dir)

        # Filter samples for this split
        self.samples: list[tuple[str, str, dict]] = []
        self._build_samples()

    # ── Initialization & Setup ───────────────────────────────────────────

    def _build_samples(self) -> None:
        """Filter the global annotation data to match the requested dataset split."""
        video_ids = _get_video_ids_for_mode(self.mode)
        valid_prefixes = {str(v) for v in video_ids}

        for clip_key, clip_data in self._master_data.items():
            video_id, clip_id = clip_key.split("/", 1)
            if video_id in valid_prefixes:
                self.samples.append((video_id, clip_id, clip_data))

    # ── Frame Selection ──────────────────────────────────────────────────

    def _find_valid_middle_index(self, all_frames: list[str], clip_id: str, clip_data: dict) -> int:
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
                    persons = self._get_persons_for_frame(clip_data, fname)
                    if persons:
                        return idx
        return best_idx

    def _pad_frame_sequence(self, selected: list[str], all_frames: list[str], start: int, end: int) -> list[str]:
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

    def _select_frame_names(self, video_id: str, clip_id: str, clip_data: dict) -> list[str]:
        """Select exactly `n_frames` filenames centered symmetrically on the best valid middle frame."""
        all_frames = self._frame_index.get((video_id, clip_id), [])
        if not all_frames:
            return []

        mid_idx = self._find_valid_middle_index(all_frames, clip_id, clip_data)

        half = self.n_frames // 2
        start = max(0, mid_idx - half)
        end = min(len(all_frames), mid_idx + half + 1)
        
        selected = all_frames[start:end]
        selected = self._pad_frame_sequence(selected, all_frames, start, end)

        return selected

    # ── Image and Annotation Processing ──────────────────────────────────

    def _load_image(self, video_id: str, clip_id: str, frame_name: str) -> Image.Image:
        """Load an individual frame directly from disk."""
        path = self.dataset_dir / video_id / clip_id / frame_name
        
        if not path.is_file():
            raise FileNotFoundError(f"Frame image not found: {path}")
            
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open or decode image {path}: {e}") from e

    def _get_persons_for_frame(self, clip_data: dict, frame_name: str) -> list[dict]:
        """
        Return person entries for a specific frame.
        Prefers tracking data (which carries player IDs) and falls back to action detections.
        """
        tracking = clip_data.get("tracking", {})
        persons = tracking.get(frame_name, [])
        if persons:
            return persons
            
        actions = clip_data.get("actions", {})
        return actions.get(frame_name, [])

    def _crop_persons(
        self,
        image: Image.Image,
        persons: list[dict],
    ) -> tuple[list[Image.Image], list[int]]:
        """
        Extract cropped person bounding boxes from the base image.
        Supports tracking format `[x1, y1, x2, y2]` and detection format `[x, y, w, h]`.
        """
        crops, labels = [], []
        img_w, img_h = image.size

        for person in persons:
            box = person["box"]

            if "id" in person:
                x1, y1, x2, y2 = box
            else:
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h

            # Enforce bounding box fits within physical image dimensions
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

    # ── Dataset Interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        video_id, clip_id, clip_data = self.samples[idx]

        scene_class = clip_data.get("scene_class")
        group_label = GROUP_ACTIVITY_TO_IDX.get(scene_class, 0) if scene_class else 0

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
        if not frame_names:
            raise RuntimeError(
                f"No frames found for video={video_id} clip={clip_id}. "
                "Check that MAIN_DATASET_DIR points to a valid dataset."
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
                torch.stack(crops, dim=0),  # (P, C, H, W)
                torch.tensor(person_labels, dtype=torch.long),  # (P,)
                group_label,
            )

        # Temporal crops mode (n_frames > 1)
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

        return (
            torch.stack(valid_frames, dim=0),  # (T, P, C, H, W)
            torch.tensor(last_person_labels, dtype=torch.long),  # (P,)
            group_label,
        )


# ── Collate function ─────────────────────────────────────────────────────────

def _pad_and_stack_crops(crops_list: Sequence[torch.Tensor], person_labels_list: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper to zero-pad batch crops (temporal or single) up to maximum player and/or time counts."""
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

    elif len(sample_shape) >= 5:
        # Temporal crops (T, P, C, H, W)
        max_T = max((c.shape[0] for c in crops_list if c.numel() > 0), default=0)
        _, _, C, H, W = sample_shape
        padded_crops = torch.zeros(batch_size, max_T, max_players, C, H, W)
        padded_labels = torch.zeros(batch_size, max_players, dtype=torch.long)
        masks = torch.zeros(batch_size, max_players, dtype=torch.bool)

        for i, (crops, plabels) in enumerate(zip(crops_list, person_labels_list)):
            if crops.numel() == 0:
                continue
            t = crops.shape[0]
            n = crops.shape[1]
            padded_crops[i, :t, :n] = crops
            padded_labels[i, :n] = plabels
            masks[i, :n] = True
    else:
        raise ValueError(f"Unexpected crop dimension format: {sample_shape}")
        
    return padded_crops, padded_labels, masks


def collate_fn(batch: list[tuple[Any, ...]]) -> tuple[torch.Tensor, ...]:
    """
    Custom PyTorch DataLoader collate function.
    
    Dynamically routes behavior:
    - Full-image mode   -> (images_batch, labels_batch)
    - Crop mode         -> (crops_batch, person_labels_batch, group_labels_batch, masks_batch)
    """
    if not batch:
        return ()

    if len(batch[0]) == 2:
        images, labels = zip(*batch)
        shapes = [img.shape for img in images]
        
        # Consistent Temporal size across batch
        if all(s == shapes[0] for s in shapes):
            return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)
            
        # Variable Temporal size (e.g. n_frames > 1 and some frames omitted)
        max_T = max(s[0] for s in shapes)
        _, C, H, W = shapes[0]
        padded_images = torch.zeros(len(images), max_T, C, H, W)
        
        for i, img in enumerate(images):
            t = img.shape[0]
            padded_images[i, :t] = img
            
        return padded_images, torch.tensor(labels, dtype=torch.long)

    # Crop Mode Routing
    crops_list, person_labels_list, group_labels = zip(*batch)
    padded_crops, padded_labels, masks = _pad_and_stack_crops(crops_list, person_labels_list)
    
    return (
        padded_crops,
        padded_labels,
        torch.tensor(group_labels, dtype=torch.long),
        masks,
    )


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print("── Full image (n_frames=5) ──")
    ds = VolleyballDataset("validation", full_image=True, n_frames=5, transform=tf)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    images, labels = next(iter(dl))
    print(f"  images : {images.shape}")  # Expected: (4, 5, 3, 256, 256)
    print(f"  labels : {labels.shape}")  # Expected: (4,)

    print("── Crop (n_frames=9) ──")
    ds2 = VolleyballDataset(
        "validation", full_image=False, crop=True, n_frames=9, transform=tf
    )
    dl2 = DataLoader(ds2, batch_size=4, shuffle=True, collate_fn=collate_fn)
    crops, plabels, glabels, masks = next(iter(dl2))
    print(f"  crops   : {crops.shape}")
    print(f"  plabels : {plabels.shape}")
    print(f"  glabels : {glabels.shape}")
    print(f"  masks   : {masks.shape}")

    print(f"\nDataset sizes — full: {len(ds)}  crop: {len(ds2)}")

