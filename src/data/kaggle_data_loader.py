"""
Kaggle-specific volleyball dataset loader — identical interface to
data_loader.py but reads frames AND annotations directly from disk.

Use this on Kaggle where /kaggle/working/ lacks the space to build
the LMDB (~50 GB). Frames are served straight from the read-only
input mount at MAIN_DATASET_DIR, which is already on a fast SSD.

Annotations are parsed straight from the raw text files (tracking,
action detections, per-video annotations.txt) into the same master-dict
structure the pickle held — no volleyball_master.json / pickle needed.
Skipping the pickle avoids materializing its unused "persons" detections
(~1.2 GB) and works out of the box on a fresh Kaggle mount. If the
annotation directories are missing, the loader transparently falls back
to the pickle, so existing setups keep working.

Drop-in replacement: just change the import in your training script:

    # instead of:
    from src.data.data_loader import VolleyballDataset, collate_fn
    # use:
    from src.data.kaggle_data_loader import VolleyballDataset, collate_fn

Everything else (constructor args, return shapes, collate_fn) is identical.
All dataset logic lives in :mod:`src.data.base_dataset`; this module only
supplies the direct-from-disk storage backend.
"""  # noqa: D205

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from PIL import Image

from configs.path_config import (
    MAIN_DATASET_DIR,
    VOLLEYBALL_ANNOTATIONS_DIR,
    VOLLEYBALL_DETECTION_DIR,
    VOLLEYBALL_TRACKING_DIR,
)
from src.data.base_dataset import BaseVolleyballDataset, collate_fn
from src.json_parser import (
    parse_detection_file,
    parse_scene_annotations,
    parse_tracking_file,
)
from src.pickle_dump import load_from_pickle

logger = logging.getLogger(__name__)

__all__ = ["VolleyballDataset", "collate_fn", "free_annotation_cache"]

# ── Disk-direct annotation index ─────────────────────────────────────────────
#
# Built once per process and shared by the train/val/test datasets, exactly
# like the pickle cache it replaces. Mirrors json_parser.create_master_json +
# enrich_with_scene_labels, minus the never-used "persons" detections.

_ANNOTATION_CACHE: dict | None = None


def _iter_clip_dirs(root: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(video_id, clip_id)`` for every clip folder under root."""
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        for clip_dir in sorted(video_dir.iterdir()):
            if clip_dir.is_dir():
                yield video_dir.name, clip_dir.name


def _build_annotations_from_disk() -> dict:
    """
    Parse the raw annotation text files into a master dict keyed by
    ``"video_id/clip_id"``, matching the pickle's structure per clip:
    ``{"actions": {...}, "tracking": {...}, "scene_class": str | None}``.
    """
    global _ANNOTATION_CACHE
    if _ANNOTATION_CACHE is not None:
        return _ANNOTATION_CACHE

    # Clip enumeration follows json_parser.create_master_json (detection
    # dir); tracking dir is an equivalent fallback with the same layout.
    if VOLLEYBALL_DETECTION_DIR.is_dir():
        enum_root = VOLLEYBALL_DETECTION_DIR
    elif VOLLEYBALL_TRACKING_DIR.is_dir():
        enum_root = VOLLEYBALL_TRACKING_DIR
    else:
        raise FileNotFoundError(
            "Neither annotation source exists: "
            f"{VOLLEYBALL_DETECTION_DIR} nor {VOLLEYBALL_TRACKING_DIR}",
        )

    # Scene labels stay scoped per video — frame names collide across videos.
    scene_labels: dict[str, dict[str, str]] = {}
    if VOLLEYBALL_ANNOTATIONS_DIR.is_dir():
        for video_folder in sorted(VOLLEYBALL_ANNOTATIONS_DIR.iterdir()):
            if video_folder.is_dir():
                scene_labels[video_folder.name] = parse_scene_annotations(
                    video_folder / "annotations.txt",
                )

    master: dict[str, dict] = {}
    for video_id, clip_id in _iter_clip_dirs(enum_root):
        actions = (
            parse_detection_file(
                VOLLEYBALL_DETECTION_DIR / video_id / clip_id / "action_detections.txt",
            )
            if VOLLEYBALL_DETECTION_DIR.is_dir() else {}
        )
        master[f"{video_id}/{clip_id}"] = {
            "actions": actions,
            "tracking": parse_tracking_file(
                VOLLEYBALL_TRACKING_DIR / video_id / clip_id / f"{clip_id}.txt",
            ),
            "scene_class": scene_labels.get(video_id, {}).get(f"{clip_id}.jpg"),
        }

    logger.info("Built annotation index from disk: %d clips.", len(master))
    _ANNOTATION_CACHE = master
    return master


def free_annotation_cache() -> None:
    """
    Drop the disk-built annotation cache (analog of
    ``pickle_dump.free_master_data_cache``). Call after every dataset is
    constructed, before DataLoader workers fork.
    """
    global _ANNOTATION_CACHE
    _ANNOTATION_CACHE = None

# ── Frame index ──────────────────────────────────────────────────────────────

_FRAME_INDEX_CACHE: dict[tuple[str, str], list[str]] | None = None


def _build_frame_index(dataset_dir: Path) -> dict[tuple[str, str], list[str]]:
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
        for clip_dir in sorted(video_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            frames = sorted(f.name for f in clip_dir.iterdir() if f.suffix == ".jpg")
            if frames:
                index[(video_dir.name, clip_dir.name)] = frames

    _FRAME_INDEX_CACHE = index
    return _FRAME_INDEX_CACHE


# ── Dataset ──────────────────────────────────────────────────────────────────


class VolleyballDataset(BaseVolleyballDataset):
    """
    Disk-backed (Kaggle-compatible) volleyball dataset.

    See :class:`src.data.base_dataset.BaseVolleyballDataset` for the full
    parameter and return-shape documentation.

    Parameters
    ----------
    dataset_dir : Path or str or None
        Override for the frames root directory. Defaults to MAIN_DATASET_DIR.
        All other parameters are inherited from the base class.

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
        self.dataset_dir = Path(dataset_dir) if dataset_dir else MAIN_DATASET_DIR
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found or not a directory: {self.dataset_dir}",
            )
        super().__init__(
            mode=mode, n_frames=n_frames, full_image=full_image,
            crop=crop, transform=transform,
        )

    def _load_master_data(self) -> dict:
        """Build annotations straight from disk; fall back to the pickle."""
        try:
            return _build_annotations_from_disk()
        except FileNotFoundError as e:
            logger.warning(
                "Raw annotation dirs unavailable (%s) — falling back to the "
                "master pickle.", e,
            )
            return load_from_pickle()

    def _load_frame_index(self) -> dict[tuple[str, str], list[str]]:
        return _build_frame_index(self.dataset_dir)

    def _load_image(self, video_id: str, clip_id: str, frame_name: str) -> Image.Image:
        """Load an individual frame directly from disk."""
        path = self.dataset_dir / video_id / clip_id / frame_name

        if not path.is_file():
            raise FileNotFoundError(f"Frame image not found: {path}")

        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open or decode image {path}: {e}") from e


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader  # ty:ignore[unresolved-import]
    from torchvision import transforms  # ty:ignore[unresolved-import]

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
        "validation", full_image=False, crop=True, n_frames=9, transform=tf,
    )
    dl2 = DataLoader(ds2, batch_size=4, shuffle=True, collate_fn=collate_fn)
    crops, plabels, glabels, masks = next(iter(dl2))
    print(f"  crops   : {crops.shape}")
    print(f"  plabels : {plabels.shape}")
    print(f"  glabels : {glabels.shape}")
    print(f"  masks   : {masks.shape}")

    print(f"\nDataset sizes — full: {len(ds)}  crop: {len(ds2)}")
