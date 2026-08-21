"""
Export the volleyball frames as an Ultralytics **classification** dataset.

The detector is gone: B9 feeds whole frames to a YOLO classification model and
predicts the clip's group activity, so no boxes are exported — only the frame
and its ``scene_class``.

Ultralytics classification does not read a ``data.yaml``.  It takes a directory
and treats it as an ImageFolder, one subdirectory per class:

    <out_dir>/train/l-pass/1_10535_10535.jpg
    <out_dir>/val/r_spike/0_13286_13286.jpg
    <out_dir>/test/...

so ``src.data.data_loader.VolleyballDataset`` cannot be handed over directly —
this module rewrites the same annotations into that layout instead, using the
project's own train/validation/test video split.

Frames are **warped to a square**, not symlinked, and that is deliberate.
Ultralytics' classification transform is ``Resize(shortest edge)`` +
``CenterCrop`` — on a 1280×720 frame that keeps a 720×720 middle and throws
away 44% of the court width.  The 8 activities come in left/right pairs
(``l_set`` vs ``r_set``), so discarding court width destroys the signal, which
is exactly why ``configs/transforms/default_transforms.yaml`` warps full frames
instead of cropping them.  Exporting square images makes the center crop a
no-op and keeps the whole court.  ``--resize 0`` opts out and symlinks the
originals, accepting the crop.

Usage
-----
    # one image per clip (the labelled centre frame) — ~4.8k images
    python -m utils.yolo_export

    # the 9-frame window every other baseline uses — ~43k images
    python -m utils.yolo_export --frames window --window 9

    # re-check an export that already exists
    python -m utils.yolo_export --summary

Train with ``models/baseline9.py``, which points at the exported root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image  # ty:ignore[import]

from configs.data_split import (
    TEST_VIDEOS_NUMBERS,
    TRAIN_VIDEOS_NUMBERS,
    VALIDATION_VIDEO_NUMBERS,
)
from configs.labels import GROUP_ACTIVITY_TO_IDX
from configs.path_config import (
    JSON_DATA_DIR,
    MAIN_DATASET_DIR,
    VOLLEYBALL_DETECTION_DIR,
    VOLLEYBALL_TRACKING_DIR,
)

#: Ultralytics split directory names, keyed by this project's split names.
SPLIT_DIRS: dict[str, str] = {"train": "train", "validation": "val", "test": "test"}

#: Video ids per split, straight from ``configs.data_split``.
SPLIT_VIDEOS: dict[str, list[int]] = {
    "train": TRAIN_VIDEOS_NUMBERS,
    "validation": VALIDATION_VIDEO_NUMBERS,
    "test": TEST_VIDEOS_NUMBERS,
}

#: Sits beside the other generated data — which is the writable working dir
#: on Kaggle, where DATA_DIR itself is a read-only input mount.
DEFAULT_OUT_DIR: Path = JSON_DATA_DIR.parent / "yolo_cls"

#: Square export size. Match (or exceed) the ``imgsz`` you train at.
DEFAULT_RESIZE = 224


# ── Annotation source ───────────────────────────────────────────────────────


def load_master_annotations() -> dict:
    """
    Return the master annotation dict keyed by ``"video_id/clip_id"``.

    Parses the raw annotation text files (the Kaggle loader's own builder)
    when they are present: it is authoritative, and it skips the ~3 GB of
    Python objects the master pickle unpacks into — of which this export
    needs only ``scene_class``.  Falls back to the pickle otherwise.

    Returns
    -------
    dict
        ``{"video/clip": {"tracking": ..., "actions": ..., "scene_class": ...}}``

    """
    if VOLLEYBALL_TRACKING_DIR.is_dir() or VOLLEYBALL_DETECTION_DIR.is_dir():
        from src.data.kaggle_data_loader import (  # ty:ignore[import]
            _build_annotations_from_disk,
        )

        return _build_annotations_from_disk()

    from src.pickle_dump import load_from_pickle  # ty:ignore[import]

    return load_from_pickle()


def _frame_names(clip_dir: Path, clip_id: str, frames: str, window: int) -> list[str]:
    """
    Pick which frames of a clip to export.

    ``middle`` takes the single annotated target frame (``<clip_id>.jpg``).
    ``window`` takes ``window`` frames centred on it — the same temporal
    window B4–B8 sample, all sharing the clip's activity label.  ``all``
    takes every frame in the clip directory.
    """
    if frames == "middle":
        return [f"{clip_id}.jpg"]

    available = sorted(
        (f.name for f in clip_dir.iterdir() if f.suffix == ".jpg"),
        key=lambda name: int(Path(name).stem),
    )
    if frames == "all":
        return available

    target = f"{clip_id}.jpg"
    if target not in available:
        return available[:window]
    centre = available.index(target)
    half = window // 2
    start = max(0, min(centre - half, len(available) - window))
    return available[start : start + window]


# ── Export ──────────────────────────────────────────────────────────────────


def export_yolo_cls_dataset(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    frames: str = "middle",
    window: int = 9,
    resize: int = DEFAULT_RESIZE,
    frames_root: Path = MAIN_DATASET_DIR,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    verbose: bool = True,
) -> dict:
    """
    Write the ImageFolder-style classification dataset.

    Parameters
    ----------
    out_dir : Path, optional
        Root of the exported tree; pass this to ``model.train(data=...)``.
    frames : {"middle", "window", "all"}, optional
        ``"middle"`` exports the labelled centre frame of each clip (~4.8k
        images) — one sample per activity, no near-duplicates.  ``"window"``
        exports ``window`` frames around it, trading correlated samples for
        volume.  ``"all"`` exports every frame of every clip.
    window : int, optional
        Frames per clip when ``frames="window"``.  Defaults to ``9``, the
        window B4–B8 use.
    resize : int, optional
        Square size to warp each frame to.  ``0`` symlinks the originals
        instead, which leaves Ultralytics free to center-crop them.
    frames_root : Path, optional
        Root of the frame archive (``<root>/<video>/<clip>/<frame>.jpg``).
    splits : tuple of str, optional
        Which of ``train``/``validation``/``test`` to export.
    verbose : bool, optional
        Print per-split counts and a summary.

    Returns
    -------
    dict
        Per-split image counts and class distribution, plus the export root.

    """
    if frames not in {"middle", "window", "all"}:
        raise ValueError(f"frames must be 'middle', 'window' or 'all', got {frames!r}")
    if resize < 0:
        raise ValueError(f"resize must be >= 0, got {resize}")

    master = load_master_annotations()
    if verbose:
        print(f"Loaded {len(master)} clips of annotations")

    stats: dict[str, dict] = {}

    for split in splits:
        split_root = out_dir / SPLIT_DIRS[split]
        # Every class gets a directory even if a split happens to lack it, so
        # the class-index mapping stays identical across train/val/test.
        for name in GROUP_ACTIVITY_TO_IDX:
            (split_root / name).mkdir(parents=True, exist_ok=True)

        wanted = {str(v) for v in SPLIT_VIDEOS[split]}
        per_class: Counter[str] = Counter()
        n_missing = n_unlabelled = 0

        for key, clip_data in master.items():
            video_id, clip_id = key.split("/")
            if video_id not in wanted:
                continue

            scene_class = clip_data.get("scene_class")
            if scene_class not in GROUP_ACTIVITY_TO_IDX:
                n_unlabelled += 1
                continue

            clip_dir = frames_root / video_id / clip_id
            if not clip_dir.is_dir():
                n_missing += 1
                continue

            for frame_name in _frame_names(clip_dir, clip_id, frames, window):
                source = clip_dir / frame_name
                if not source.exists():
                    n_missing += 1
                    continue

                target = split_root / scene_class / f"{video_id}_{clip_id}_{Path(frame_name).stem}.jpg"
                if resize:
                    with Image.open(source) as img:
                        # Exact square warp, matching default_transforms.yaml:
                        # aspect distortion is accepted, losing court width is not.
                        img.convert("RGB").resize((resize, resize), Image.BILINEAR).save(
                            target, quality=95
                        )
                else:
                    target.unlink(missing_ok=True)
                    target.symlink_to(source)

                per_class[scene_class] += 1

        stats[split] = {
            "images": sum(per_class.values()),
            "per_class": dict(sorted(per_class.items())),
            "missing_frames": n_missing,
            "unlabelled_clips": n_unlabelled,
        }
        if verbose:
            print(
                f"  {split:<10} → {sum(per_class.values()):>6} images across "
                f"{len(per_class)} classes"
                + (f", {n_missing} frames missing" if n_missing else "")
                + (f", {n_unlabelled} clips unlabelled" if n_unlabelled else "")
            )

    if verbose:
        print(f"\ndata root : {out_dir}")
        print(f"classes   : {len(GROUP_ACTIVITY_TO_IDX)} ({', '.join(GROUP_ACTIVITY_TO_IDX)})")
        print(
            f"images    : {'warped to ' + str(resize) + '²' if resize else 'symlinked originals'}"
        )
        print(
            f"\nProbe and train with:\n"
            f"    python -m utils.yolo_probe --task classify --imgsz {resize or 224} "
            f"--nc {len(GROUP_ACTIVITY_TO_IDX)}\n"
            f"    python models/baseline9.py"
        )

    return {"splits": stats, "root": str(out_dir), "classes": list(GROUP_ACTIVITY_TO_IDX)}


# ── Verification ────────────────────────────────────────────────────────────


def summarize_export(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    """
    Re-read an export from disk and report what is actually there.

    Counts images per class per split and flags dangling symlinks and any
    video that leaked across splits — the two failures that silently inflate
    a validation score.
    """
    report: dict[str, dict] = {}
    videos_by_split: dict[str, set[str]] = {}

    for split_dir in ("train", "val", "test"):
        root = out_dir / split_dir
        if not root.is_dir():
            continue
        per_class: dict[str, int] = {}
        broken = 0
        videos: set[str] = set()
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            images = list(class_dir.glob("*.jpg"))
            per_class[class_dir.name] = len(images)
            for image in images:
                videos.add(image.name.split("_")[0])
                if not image.resolve().exists():
                    broken += 1
        videos_by_split[split_dir] = videos
        report[split_dir] = {
            "images": sum(per_class.values()),
            "per_class": per_class,
            "broken_links": broken,
        }

    for split, videos in videos_by_split.items():
        leaked = sorted(
            v for other, others in videos_by_split.items() if other != split for v in videos & others
        )
        report[split]["videos"] = len(videos)
        report[split]["videos_shared_with_other_splits"] = leaked
    return report


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the volleyball frames as a YOLO classification dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--frames",
        choices=("middle", "window", "all"),
        default="middle",
        help="labelled centre frame per clip, a window around it, or every frame",
    )
    parser.add_argument("--window", type=int, default=9, help="frames per clip for --frames window")
    parser.add_argument(
        "--resize",
        type=int,
        default=DEFAULT_RESIZE,
        help="square warp size; 0 symlinks originals and accepts a center crop",
    )
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="comma-separated splits to export",
    )
    parser.add_argument("--summary", action="store_true", help="only re-check an existing export")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.summary:
        for split, info in summarize_export(args.out_dir).items():
            print(f"{split}: {info}")
    else:
        export_yolo_cls_dataset(
            out_dir=args.out_dir,
            frames=args.frames,
            window=args.window,
            resize=args.resize,
            splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        )
