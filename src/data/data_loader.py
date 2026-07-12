"""
LMDB-backed volleyball dataset loader supporting all baselines (B1–B8).

Loads annotation data from the master pickle and frame pixel data from
the frames LMDB (produced by ``load_frames_into_lmdb``).  All image
I/O is memory-mapped — no per-frame disk reads during training.

All dataset logic (split filtering, frame selection, cropping, collate)
lives in :mod:`src.data.base_dataset`; this module only supplies the
LMDB storage backend.

Configurable via constructor parameters to return different data shapes:

    full_image=True,  n_frames=1  →  (image, group_label)                [B1]
    full_image=True,  n_frames=9  →  (images, group_label)               [B4]
    crop=True,        n_frames=1  →  (crops, person_labels, group_label) [B3]
    crop=True,        n_frames=9  →  (crops, person_labels, group_label) [B5-B8]
"""

from __future__ import annotations

from PIL import Image

from configs.path_config import FRAMES_LMDB_DIR
from src.data.base_dataset import BaseVolleyballDataset, collate_fn
from src.load_frames_into_lmdb import (
    decode_frame,
    get_all_frame_keys,
    open_frames_lmdb,
)

__all__ = ["VolleyballDataset", "collate_fn"]

# ── Frame index ──────────────────────────────────────────────────────────────

_FRAME_INDEX_CACHE: dict[tuple[str, str], list[str]] | None = None


def _get_frame_index() -> dict[tuple[str, str], list[str]]:
    """Singleton helper to load the frame index mapping from the LMDB key list."""
    global _FRAME_INDEX_CACHE
    if _FRAME_INDEX_CACHE is None:
        index: dict[tuple[str, str], list[str]] = {}
        for key in get_all_frame_keys():
            vid, cid, fname = key.split("/", 2)
            index.setdefault((vid, cid), []).append(fname)

        for v in index.values():
            v.sort()

        _FRAME_INDEX_CACHE = index
    return _FRAME_INDEX_CACHE


# ── Dataset ──────────────────────────────────────────────────────────────────


class VolleyballDataset(BaseVolleyballDataset):
    """
    LMDB-backed volleyball dataset.

    See :class:`src.data.base_dataset.BaseVolleyballDataset` for the full
    parameter and return-shape documentation.
    """

    def __init__(self, *args, **kwargs) -> None:
        if not FRAMES_LMDB_DIR.exists():
            raise FileNotFoundError(
                f"Frames LMDB not found at {FRAMES_LMDB_DIR}. "
                "Run `python -m src.load_frames_into_lmdb` first.",
            )
        super().__init__(*args, **kwargs)

    def _load_frame_index(self) -> dict[tuple[str, str], list[str]]:
        return _get_frame_index()

    def _load_image(self, video_id: str, clip_id: str, frame_name: str) -> Image.Image:
        """Decode a frame directly from the memory-mapped LMDB."""
        key = f"{video_id}/{clip_id}/{frame_name}".encode()

        env = open_frames_lmdb()
        with env.begin() as txn:
            raw = txn.get(key)

        if raw is None:
            raise KeyError(f"Frame not found in LMDB: {key.decode('utf-8')}")

        return decode_frame(raw)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader  # ty:ignore[unresolved-import]
    from torchvision import transforms  # ty:ignore[unresolved-import]

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset_full_image = VolleyballDataset("validation", full_image=True, n_frames=5, transform=tf)
    dataloader_full_image = DataLoader(dataset_full_image, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for images, labels in dataloader_full_image:
        print(images.shape)
        print(labels.shape)
        break

    dataset_crop = VolleyballDataset("validation", full_image=False, crop=True, n_frames=5, transform=tf)
    dataloader_crop = DataLoader(dataset_crop, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for images, labels, group_labels, masks in dataloader_crop:
        print(images.shape)
        print(labels.shape)
        print(group_labels.shape)
        print(masks.shape)
        break

    print(f"Full image dataset size: {len(dataset_full_image)}")
    print(f"Crop dataset size: {len(dataset_crop)}")
