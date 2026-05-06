"""
Pre-load all dataset frames into an LMDB database for fast, memory-mapped I/O.

Stores **compressed JPEG bytes** (not decoded numpy arrays) to keep the
database size manageable.  Images are decoded on-the-fly when accessed.

The LMDB structure is a key-value store where keys are UTF-8 encoded
strings of the form ``"video_id/clip_id/frame.jpg"`` and values are raw
JPEG bytes::

    b"0/3596/3576.jpg"  ->  b'\xff\xd8\xff...'
    b"0/3596/3577.jpg"  ->  b'\xff\xd8\xff...'

A special key ``__keys__`` stores a pickle of all frame keys for fast
enumeration without scanning the entire database.

Usage::

    # One-time dump (singleton — skips if database exists)
    python -m src.load_frames_into_lmdb

    # Load from any module
    from src.load_frames_into_lmdb import open_frames_lmdb, decode_frame
    env = open_frames_lmdb()
    with env.begin() as txn:
        raw = txn.get(b"0/3596/3576.jpg")
    img = decode_frame(raw)

"""

from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path

import lmdb  # ty:ignore[unresolved-import]
from PIL import Image

from configs.path_config import FRAMES_LMDB_DIR, MAIN_DATASET_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

_KEYS_META_KEY = b"__keys__"

# Default map_size: 60 GB — LMDB will only use what it needs on disk,
# this is just the upper bound of the virtual address space reservation.
_DEFAULT_MAP_SIZE = 60 * 1024**3


# ── Dump ────────────────────────────────────────────────────────────────────


def dump_frames_to_lmdb(
    dataset_dir: Path = MAIN_DATASET_DIR,
    output_path: Path = FRAMES_LMDB_DIR,
    map_size: int = _DEFAULT_MAP_SIZE,
) -> None:
    """
    Walk the main dataset and store every .jpg frame in an LMDB database.

    **Singleton behaviour**: if *output_path* already exists the function
    returns immediately without re-dumping.

    Parameters
    ----------
    dataset_dir : Path
        Root of ``main dataset/`` containing video subdirectories.
    output_path : Path
        Destination LMDB directory.
    map_size : int
        Maximum size of the LMDB memory map (bytes). Default 60 GB.

    """
    if output_path.exists():
        logger.info(
            "Frames LMDB already exists at %s — skipping dump.",
            output_path,
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(output_path), map_size=map_size)
    all_keys: list[str] = []
    frame_count = 0

    txn = env.begin(write=True)
    try:
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
                    txn.put(key.encode(), frame_file.read_bytes())
                    all_keys.append(key)
                    frame_count += 1

                # Commit every 10 000 frames to keep memory usage low
                if frame_count % 10_000 == 0:
                    logger.info("  ... loaded %d frames so far", frame_count)
                    txn.commit()
                    txn = env.begin(write=True)

        # Store the key index for fast enumeration
        txn.put(
            _KEYS_META_KEY, pickle.dumps(all_keys, protocol=pickle.HIGHEST_PROTOCOL)
        )
        txn.commit()
    except BaseException:
        txn.abort()
        raise

    env.close()

    logger.info("Total frames written: %d", frame_count)
    size_gb = sum(f.stat().st_size for f in output_path.iterdir()) / (1024**3)
    logger.info("Done. LMDB size: %.2f GB", size_gb)


# ── Load ────────────────────────────────────────────────────────────────────

# Module-level cache so repeated calls don't re-open the environment.
_env_cache: lmdb.Environment | None = None


def open_frames_lmdb(path: Path = FRAMES_LMDB_DIR) -> lmdb.Environment:
    """
    Open (or return a cached handle to) the pre-built LMDB environment.

    The environment is opened in **read-only** mode with ``lock=False``
    so multiple processes (DataLoader workers) can share it safely.

    Parameters
    ----------
    path : Path
        Path to the LMDB directory.

    Returns
    -------
    lmdb.Environment

    Raises
    ------
    FileNotFoundError
        If the LMDB directory does not exist.

    """
    global _env_cache  # noqa: PLW0603

    if _env_cache is not None:
        return _env_cache

    if not path.exists():
        raise FileNotFoundError(
            f"Frames LMDB not found at {path}. "
            "Run `python -m src.load_frames_into_lmdb` first.",
        )

    _env_cache = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    logger.info("Opened frames LMDB at %s", path)
    return _env_cache


def get_all_frame_keys(path: Path = FRAMES_LMDB_DIR) -> list[str]:
    """
    Retrieve the cached list of all frame keys from the LMDB.
    This temporarily opens the LMDB and reads the pickled key list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Frames LMDB not found at {path}")

    env = lmdb.open(str(path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            keys_bytes = txn.get(_KEYS_META_KEY)
    finally:
        env.close()

    if not keys_bytes:
        return []
    return pickle.loads(keys_bytes)


def load_frames_lmdb(path: Path = FRAMES_LMDB_DIR) -> dict[str, bytes]:
    """
    Load **all** frames from the LMDB into a plain dict.

    This is a convenience wrapper that provides the same return type as
    the old ``load_frames_pickle()`` function for a smooth migration.
    For large datasets prefer :func:`open_frames_lmdb` and per-key
    lookups instead of loading everything into RAM.

    Parameters
    ----------
    path : Path
        Path to the LMDB directory.

    Returns
    -------
    dict[str, bytes]
        Mapping from ``"video_id/clip_id/frame.jpg"`` to raw JPEG bytes.

    """
    env = open_frames_lmdb(path)
    frames: dict[str, bytes] = {}

    with env.begin() as txn:
        cursor = txn.cursor()
        for key_bytes, value_bytes in cursor:
            if key_bytes == _KEYS_META_KEY:
                continue
            frames[key_bytes.decode()] = bytes(value_bytes)

    logger.info("Loaded %d frames into dict.", len(frames))
    return frames


# ── Helpers ─────────────────────────────────────────────────────────────────


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
    frames_data: dict[str, bytes],
    video_id: str,
    clip_id: str,
) -> list[str]:
    """
    Return sorted frame names available for a given clip.

    Parameters
    ----------
    frames_data : dict[str, bytes]
        The loaded frames dict (from :func:`load_frames_lmdb`).
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
    return sorted(key.split("/", 2)[2] for key in frames_data if key.startswith(prefix))


# ── Main ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    dump_frames_to_lmdb()
