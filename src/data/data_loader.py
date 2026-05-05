"""
Generic volleyball dataset loader supporting all baselines (B1–B8).

Loads annotation data from the master pickle and frame pixel data from
the frames LMDB (produced by ``load_frames_into_lmdb``).  All image
I/O is memory-mapped — no per-frame disk reads during training.

Configurable via constructor parameters to return different data shapes:

    full_image=True,  n_frames=1  →  (image, group_label)               [B1]
    full_image=True,  n_frames=9  →  (images, group_label)              [B4]
    crop=True,        n_frames=1  →  (crops, person_labels, group_label) [B3]
    crop=True,        n_frames=9  →  (crops, person_labels, group_label) [B5-B8]
"""

from __future__ import annotations

import torch  # ty:ignore[unresolved-import]
from PIL import Image
from torch.utils.data import Dataset,DataLoader  # ty:ignore[unresolved-import]
from torchvision.transforms import ToTensor  # ty:ignore[unresolved-import]
from torchvision import transforms  # ty:ignore[unresolved-import]

from configs.data_split import (
    TEST_VIDEOS_NUMBERS,
    TRAIN_VIDEOS_NUMBERS,
    VALIDATION_VIDEO_NUMBERS,
)
from configs.labels import GROUP_ACTIVITY_TO_IDX, PERSON_ACTION_TO_IDX
from configs.path_config import FRAMES_LMDB_DIR
from src.load_frames_into_lmdb import (
    decode_frame,
    get_all_frame_keys,
    open_frames_lmdb,
)
from src.pickle_dump import load_from_pickle

# ── Helpers ─────────────────────────────────────────────────────────────────

_FRAME_INDEX_CACHE: dict[tuple[str, str], list[str]] | None = None


def _get_frame_index() -> dict[tuple[str, str], list[str]]:
    """Singleton helper to load the frame index mapping."""
    global _FRAME_INDEX_CACHE
    if _FRAME_INDEX_CACHE is None:
        keys = get_all_frame_keys()
        
        index: dict[tuple[str, str], list[str]] = {}
        for key in keys:
            vid, cid, fname = key.split("/", 2)
            index.setdefault((vid, cid), []).append(fname)
            
        for v in index.values():
            v.sort()
            
        _FRAME_INDEX_CACHE = index
    return _FRAME_INDEX_CACHE


def _get_video_ids_for_mode(mode: str) -> list[int]:
    """
    Return the list of video IDs for the given split.

    Parameters
    ----------
    mode : str
        One of ``"train"``, ``"validation"``, or ``"test"``.

    """
    mode_map = {
        "train": TRAIN_VIDEOS_NUMBERS,
        "validation": VALIDATION_VIDEO_NUMBERS,
        "test": TEST_VIDEOS_NUMBERS,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {list(mode_map.keys())}.")
    return mode_map[mode]


# ── Dataset ─────────────────────────────────────────────────────────────────


class VolleyballDataset(Dataset):
    """
    Generic PyTorch Dataset for the volleyball group-activity dataset.

    Data is loaded from the pre-built LMDB so there is no redundant
    file I/O on every instantiation.

    Supports all baseline configurations (B1–B8) through constructor flags:

    - ``full_image=True``  : return full-resolution frames.
    - ``crop=True``        : return per-person crops from bounding boxes.
    - ``n_frames=1``       : middle frame only (B1, B3).
    - ``n_frames=9``       : temporal window of 9 frames (B4–B8).

    Parameters
    ----------
    mode : str
        Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
    n_frames : int
        Number of frames to sample per clip. Must be a positive odd number.
        Default is 1 (middle frame only).
    full_image : bool
        If True, return the full frame image(s). Default True.
    crop : bool
        If True, return cropped person images. Default False.
    transform : callable or None
        A torchvision transform to apply to each image/crop.

    """

    def __init__(
        self,
        mode: str = "train",
        n_frames: int = 1,
        full_image: bool = True,
        crop: bool = False,
        transform=None,
    ) -> None:
        super().__init__()

        if crop and full_image:
            raise ValueError(
                "crop and full_image cannot both be True. Please choose one.",
            )
        if n_frames % 2 == 0 or n_frames <= 0:
            raise ValueError(
                "n_frames must be a positive odd number to have a clear middle frame.",
            )
        if not FRAMES_LMDB_DIR.exists():
            raise FileNotFoundError(
                f"Frames LMDB not found at {FRAMES_LMDB_DIR}. "
                "Run `python -m src.load_frames_into_lmdb` first.",
            )

        self.mode = mode
        self.n_frames = n_frames
        self.full_image = full_image
        self.crop = crop
        self.transform = transform

        # Load the master annotation data from pickle
        self._master_data: dict = load_from_pickle()

        # Load the index of available frames (fast, memory-efficient)
        self._frame_index: dict[tuple[str, str], list[str]] = _get_frame_index()

        # Build the sample index filtered by split
        self.samples: list[tuple[str, str, dict]] = []
        self._build_samples()

    # ── Index building ──────────────────────────────────────────────────

    def _build_samples(self) -> None:
        """
        Filter the master pickle by the video IDs in the current split.

        Each sample is stored as ``(video_id, clip_id, clip_data)`` where
        *clip_data* is the full dict from the pickle (actions, persons,
        tracking, scene_class).
        """
        video_ids = _get_video_ids_for_mode(self.mode)
        valid_prefixes = {str(v) for v in video_ids}

        for clip_key, clip_data in self._master_data.items():
            # clip_key is "video_id/clip_id", e.g. "0/13286"
            video_id, clip_id = clip_key.split("/", 1)
            if video_id in valid_prefixes:
                self.samples.append((video_id, clip_id, clip_data))

    # ── Frame helpers ───────────────────────────────────────────────────

    def _select_frame_names(self, video_id: str, clip_id: str) -> list[str]:
        """
        Select ``n_frames`` frame filenames centred on the middle frame.

        Uses the frames pickle index (no disk I/O).
        """
        all_frames = self._frame_index.get((video_id, clip_id), [])

        if not all_frames:
            return []

        middle_name = f"{clip_id}.jpg"

        if middle_name in all_frames:
            mid_idx = all_frames.index(middle_name)
        else:
            mid_idx = len(all_frames) // 2

        half = self.n_frames // 2
        start = max(0, mid_idx - half)
        end = min(len(all_frames), mid_idx + half + 1)

        selected = all_frames[start:end]

        # Pad if we're at the boundary
        while len(selected) < self.n_frames and len(selected) < len(all_frames):
            if start > 0:
                start -= 1
                selected.insert(0, all_frames[start])
            elif end < len(all_frames):
                selected.append(all_frames[end])
                end += 1
            else:
                break

        return selected

    def _load_image(self, video_id: str, clip_id: str, frame_name: str) -> Image.Image:
        """Decode a frame directly from the memory-mapped LMDB."""
        key = f"{video_id}/{clip_id}/{frame_name}".encode("utf-8")
        
        env = open_frames_lmdb()
        with env.begin() as txn:
            raw = txn.get(key)
            
        if raw is None:
            raise KeyError(f"Frame not found in LMDB: {key.decode('utf-8')}")
            
        return decode_frame(raw)

    def _get_persons_for_frame(
        self, clip_data: dict, frame_name: str,
    ) -> list[dict]:
        """
        Get person entries for a specific frame.

        Prefers **tracking** data (which carries consistent player IDs)
        and falls back to **action detections** when tracking is
        unavailable for that frame.

        Parameters
        ----------
        clip_data : dict
            The full clip dict from the master pickle.
        frame_name : str
            e.g. ``"13286.jpg"``.

        Returns
        -------
        list[dict]
            Each dict has at least ``"box"`` and either ``"action"`` or
            ``"label"`` depending on the source.

        """
        # 1. Try tracking (preferred — has player IDs)
        tracking = clip_data.get("tracking", {})
        persons = tracking.get(frame_name, [])
        if persons:
            return persons

        # 2. Fallback to action detections
        actions = clip_data.get("actions", {})
        return actions.get(frame_name, [])

    def _crop_persons(
        self, image: Image.Image, persons: list[dict],
    ) -> tuple[list[Image.Image], list[int]]:
        """
        Crop person regions from the image and collect labels.

        Handles both tracking boxes ``[x1, y1, x2, y2]`` and detection
        boxes ``[x, y, w, h]``.

        Returns
        -------
        tuple
            ``(crops, labels)`` where *crops* is a list of PIL Images and
            *labels* is a list of integer action indices.

        """
        crops = []
        labels = []
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

            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

            action = person.get("action", "standing")
            labels.append(
                PERSON_ACTION_TO_IDX.get(action, PERSON_ACTION_TO_IDX["standing"]),
            )

        return crops, labels

    # ── Dataset interface ───────────────────────────────────────────────

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

        scene_class = clip_data.get("scene_class")
        group_label = GROUP_ACTIVITY_TO_IDX.get(scene_class, 0) if scene_class else 0

        frame_names = self._select_frame_names(video_id, clip_id)

        if self.full_image:
            return self._getitem_full_image(
                video_id, clip_id, frame_names, group_label,
            )
        return self._getitem_crop(
            video_id, clip_id, frame_names, clip_data, group_label,
        )

    def _getitem_full_image(
        self,
        video_id: str,
        clip_id: str,
        frame_names: list[str],
        group_label: int,
    ):
        """Load full frame(s) and return as tensor(s)."""
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
    ):
        """Load cropped person images and return as tensor(s)."""
        if self.n_frames == 1:
            # Single frame crops — use the middle frame
            middle_frame = frame_names[len(frame_names) // 2] if frame_names else None
            if middle_frame is None:
                return torch.empty(0), torch.empty(0, dtype=torch.long), group_label

            img = self._load_image(video_id, clip_id, middle_frame)
            persons = self._get_persons_for_frame(clip_data, middle_frame)
            crops, person_labels = self._crop_persons(img, persons)

            _tf = self.transform or ToTensor()
            crops = [_tf(c) for c in crops]

            if not crops:
                return torch.empty(0), torch.empty(0, dtype=torch.long), group_label

            crops_tensor = torch.stack(crops, dim=0)                       # (P, C, H, W)
            labels_tensor = torch.tensor(person_labels, dtype=torch.long)  # (P,)
            return crops_tensor, labels_tensor, group_label

        # Temporal crops — use tracking data per frame for accurate boxes
        all_frame_crops = []
        last_person_labels = []

        for fname in frame_names:
            img = self._load_image(video_id, clip_id, fname)
            persons = self._get_persons_for_frame(clip_data, fname)
            crops, person_labels = self._crop_persons(img, persons)

            _tf = self.transform or ToTensor()
            crops = [_tf(c) for c in crops]

            if crops:
                all_frame_crops.append(torch.stack(crops, dim=0))  # (P, C, H, W)
                last_person_labels = person_labels
            else:
                all_frame_crops.append(None)

        valid_frames = [f for f in all_frame_crops if f is not None]
        if not valid_frames:
            return torch.empty(0), torch.empty(0, dtype=torch.long), group_label

        crops_tensor = torch.stack(valid_frames, dim=0)
        labels_tensor = torch.tensor(last_person_labels, dtype=torch.long)
        return crops_tensor, labels_tensor, group_label


# ── Collate function ────────────────────────────────────────────────────────


def collate_fn(batch):
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
    if len(batch[0]) == 2:
        # Full-image mode
        images, labels = zip(*batch)
        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)

    # Crop mode — variable number of players
    crops_list, person_labels_list, group_labels = zip(*batch)

    max_players = max(
        (c.shape[0] if len(c.shape) == 4 else c.shape[1] for c in crops_list if c.numel() > 0), default=0,
    )

    if max_players == 0:
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            torch.tensor(group_labels, dtype=torch.long),
            torch.empty(0, dtype=torch.bool),
        )

    batch_size = len(batch)
    sample_shape = next(c for c in crops_list if c.numel() > 0).shape

    if len(sample_shape) == 4:
        # (P, C, H, W) — single frame crops
        C, H, W = sample_shape[1], sample_shape[2], sample_shape[3]
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
        # (T, P, C, H, W) — temporal crops
        T, _, C, H, W = sample_shape
        padded_crops = torch.zeros(batch_size, T, max_players, C, H, W)
        padded_labels = torch.zeros(batch_size, max_players, dtype=torch.long)
        masks = torch.zeros(batch_size, max_players, dtype=torch.bool)

        for i, (crops, plabels) in enumerate(zip(crops_list, person_labels_list)):
            if crops.numel() == 0:
                continue
            n = crops.shape[1]
            padded_crops[i, :, :n] = crops
            padded_labels[i, :n] = plabels
            masks[i, :n] = True
    else:
        raise ValueError(f"Unexpected crop shape: {sample_shape}")

    return (
        padded_crops,
        padded_labels,
        torch.tensor(group_labels, dtype=torch.long),
        masks,
    )

if __name__ == "__main__":
    dataset_full_image = VolleyballDataset(
        "validation",
        full_image=True,
        n_frames=5,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
    )
    dataloader_full_image = DataLoader(dataset_full_image, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for images, labels in dataloader_full_image:
        print(images.shape)
        print(labels.shape)
        break

    dataset_crop = VolleyballDataset(
        "validation",
        full_image=False,
        n_frames=5,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
    )
    dataloader_crop = DataLoader(dataset_crop, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for images, labels, group_labels, masks in dataloader_crop:
        print(images.shape)
        print(labels.shape)
        print(group_labels.shape)
        print(masks.shape)
        break

    print(f"Full image dataset size: {len(dataset_full_image)}")
    print(f"Crop dataset size: {len(dataset_crop)}")