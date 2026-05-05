"""
Pre-load all dataset frames into a single pickle file for fast I/O.

Stores **compressed JPEG bytes** (not decoded numpy arrays) to keep the
file size manageable (~50 GB for ~198K frames).  Images are decoded
on-the-fly when accessed.

The pickle structure is a dict keyed by ``"video_id/clip_id/frame.jpg"``
with raw bytes as values::

    {
        "0/3596/3576.jpg": b'\\xff\\xd8\\xff...',
        "0/3596/3577.jpg": b'\\xff\\xd8\\xff...',
        ...
    }

Usage::

    # One-time dump (singleton — skips if file exists)
    python -m src.load_frames_into_pickle

    # Load from any module
    from src.load_frames_into_pickle import load_frames_pickle
    frames = load_frames_pickle()
    img = decode_frame(frames["0/3596/3576.jpg"])

"""

from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path

from PIL import Image

from configs.path_config import FRAMES_DUMP_DIR, MAIN_DATASET_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_frames_to_pickle(
    dataset_dir: Path = MAIN_DATASET_DIR,
    output_path: Path = FRAMES_DUMP_DIR,
) -> None:
    """
    Walk the main dataset and store every .jpg frame's raw bytes in a pickle.

    **Singleton behaviour**: if *output_path* already exists the function
    returns immediately without re-dumping.

    Parameters
    ----------
    dataset_dir : Path
        Root of ``main dataset/`` containing video subdirectories.
    output_path : Path
        Destination pickle file.

    """
    if output_path.exists():
        logger.info(
            "Frames pickle already exists at %s — skipping dump.", output_path,
        )
        return

    frames: dict[str, bytes] = {}
    frame_count = 0

    for video_dir in sorted(dataset_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name

        for clip_dir in sorted(video_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            clip_id = clip_dir.name

            for frame_file in sorted(clip_dir.iterdir()):
                if frame_file.suffix != ".jpg":
                    continue

                key = f"{video_id}/{clip_id}/{frame_file.name}"
                frames[key] = frame_file.read_bytes()
                frame_count += 1

            if frame_count % 10_000 == 0:
                logger.info("  ... loaded %d frames so far", frame_count)

    logger.info("Total frames collected: %d", frame_count)
    logger.info("Writing pickle to %s ...", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = output_path.stat().st_size / (1024 ** 3)
    logger.info("Done. Pickle size: %.2f GB", size_gb)


def load_frames_pickle(path: Path = FRAMES_DUMP_DIR) -> dict[str, bytes]:
    """
    Load the pre-built frames pickle.

    Parameters
    ----------
    path : Path
        Path to the frames pickle file.

    Returns
    -------
    dict[str, bytes]
        Mapping from ``"video_id/clip_id/frame.jpg"`` to raw JPEG bytes.

    Raises
    ------
    FileNotFoundError
        If the pickle does not exist. Run ``dump_frames_to_pickle()`` first.

    """
    if not path.exists():
        raise FileNotFoundError(
            f"Frames pickle not found at {path}. "
            "Run `python -m src.load_frames_into_pickle` first.",
        )

    logger.info("Loading frames pickle from %s ...", path)
    with path.open("rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %d frames.", len(data))
    return data


def decode_frame(raw_bytes: bytes) -> Image.Image:
    """
    Decode raw JPEG bytes into a PIL Image.

    Parameters
    ----------
    raw_bytes : bytes
        JPEG-compressed image data.

    Returns
    -------
    PIL.Image.Image
        Decoded RGB image.

    """
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def get_frame_list_for_clip(
    frames_data: dict[str, bytes], video_id: str, clip_id: str,
) -> list[str]:
    """
    Return sorted frame names available for a given clip in the pickle.

    Parameters
    ----------
    frames_data : dict[str, bytes]
        The loaded frames pickle.
    video_id : str
        e.g. ``"0"``.
    clip_id : str
        e.g. ``"3596"``.

    Returns
    -------
    list[str]
        Sorted frame filenames, e.g. ``["3576.jpg", "3577.jpg", ...]``.

    """
    prefix = f"{video_id}/{clip_id}/"
    return sorted(
        key.split("/", 2)[2]
        for key in frames_data
        if key.startswith(prefix)
    )


# ── Main ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    dump_frames_to_pickle()