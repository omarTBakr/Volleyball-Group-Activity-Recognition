"""
Download the project's trained checkpoints from Google Drive into
``saved_models/`` (MODEL_SAVE_DIR — resolves correctly both locally and
on Kaggle).

The Drive folder link lives in ``saved_models/modelsDriveLink.txt``.
Requires internet access (on Kaggle: Settings → Internet → On).

Usage::

    python -m utils.download_models              # whole folder
    python -m utils.download_models baseline1_run2.pt   # only files matching
"""

from __future__ import annotations

import sys

from configs.path_config import MODEL_SAVE_DIR

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ktVFPB3j8ZoA5uCcnU1W6LXhzfzKq_aI"


def download_checkpoints(only: list[str] | None = None) -> None:
    """
    Pull checkpoint files from the shared Drive folder into MODEL_SAVE_DIR.

    Parameters
    ----------
    only : list[str] or None
        If given, keep only the downloaded files whose names are in this
        list (the Drive folder API has no per-file filter, so the whole
        folder is fetched and extras are removed afterwards).

    """
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is not installed — run: pip install gdown")

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoints to {MODEL_SAVE_DIR} ...")

    files = gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(MODEL_SAVE_DIR),
        quiet=False,
        use_cookies=False,
    )
    if not files:
        sys.exit(
            "Download failed. Check that the Drive folder is shared as "
            "'Anyone with the link' and that internet access is enabled "
            "(Kaggle: Settings -> Internet -> On).",
        )

    if only:
        from pathlib import Path
        wanted = set(only)
        for f in files:
            p = Path(f)
            if p.name not in wanted and p.suffix == ".pt":
                p.unlink()
                print(f"  removed (not requested): {p.name}")

    kept = sorted(p.name for p in MODEL_SAVE_DIR.glob("*.pt"))
    print(f"Done. Checkpoints in {MODEL_SAVE_DIR}:")
    for name in kept:
        print(f"  {name}")


if __name__ == "__main__":
    download_checkpoints(only=sys.argv[1:] or None)
