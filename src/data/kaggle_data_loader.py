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
    from src.data.kaggle_data_loader import VolleyballDataset, collate_fn

Everything else (constructor args, return shapes, collate_fn) is identical.
All dataset logic lives in :mod:`src.data.base_dataset`; this module only
supplies the direct-from-disk storage backend.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image

from configs.path_config import MAIN_DATASET_DIR
from src.data.base_dataset import BaseVolleyballDataset, collate_fn

__all__ = ["VolleyballDataset", "collate_fn"]

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
