"""
Environment-aware path configuration for the volleyball project.

Detects whether we are running on Kaggle or locally and sets all
dataset, model, and output paths accordingly.  Every other module
should import paths from here rather than hardcoding them.
"""

from pathlib import Path

# ── Environment detection ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    KAGGLE_INPUT_DIR  = Path("/kaggle/input/datasets/ahmedmohamed365/volleyball")
    KAGGLE_OUTPUT_DIR = Path("/kaggle/working/Volleyball-Group-Activity-Recognition")

    # ── Source data (read-only on Kaggle) ─────────────────────────────────
    DATA_DIR                   = KAGGLE_INPUT_DIR
    MAIN_DATASET_DIR           = KAGGLE_INPUT_DIR / "volleyball_" / "videos"
    VIDEO_SAMPLE_DIR           = KAGGLE_INPUT_DIR / "videos_sample" / "videos_sample"
    VIDEOS_DIR                 = MAIN_DATASET_DIR
    VOLLEYBALL_DETECTION_DIR   = KAGGLE_INPUT_DIR / "volleyball-detections" / "volleyball-detections"
    VOLLEYBALL_TRACKING_DIR    = KAGGLE_INPUT_DIR / "volleyball_tracking_annotation" / "volleyball_tracking_annotation"
    VOLLEYBALL_ANNOTATIONS_DIR = MAIN_DATASET_DIR

    # ── Generated / cached data → writable working dir ─────────────────
    _GEN_DIR       = KAGGLE_OUTPUT_DIR / "DataSet"

    # ── Output dirs live inside the cloned repo (BASE_DIR) ───────────
    MODEL_SAVE_DIR = BASE_DIR / "saved_models"
    LOGS_DIR       = BASE_DIR / "logs"

else:
    # ── Local paths ──────────────────────────────────────────────────────

    DATA_DIR                   = BASE_DIR / "DataSet"
    MODEL_SAVE_DIR             = BASE_DIR / "saved_models"
    LOGS_DIR                   = BASE_DIR / "logs"
    MAIN_DATASET_DIR           = DATA_DIR / "volleyball_" / "videos"
    VIDEO_SAMPLE_DIR           = DATA_DIR / "videos_sample"
    VIDEOS_DIR                 = MAIN_DATASET_DIR
    VOLLEYBALL_DETECTION_DIR   = DATA_DIR / "volleyball-detections"
    VOLLEYBALL_TRACKING_DIR    = DATA_DIR / "volleyball_tracking_annotation"
    VOLLEYBALL_ANNOTATIONS_DIR = MAIN_DATASET_DIR

    _GEN_DIR = DATA_DIR

# ── Generated files (writable regardless of environment) ─────────────────────
JSON_DATA_DIR   = _GEN_DIR / "volleyball_master.json"
PICKLE_DUMP_DIR = _GEN_DIR / "volleyball_master_pickle.pkl"
FRAMES_DUMP_DIR = _GEN_DIR / "frames_paths_to_images.pkl"
FRAMES_LMDB_DIR = _GEN_DIR / "frames_lmdb"
PLOTS_DIR       = BASE_DIR / "plots"

# Ensure writable directories exist at import time
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
if ON_KAGGLE:
    KAGGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Validation helper ────────────────────────────────────────────────────────

def validate_paths() -> None:
    """Print a quick health-check of all critical dataset paths."""
    paths_to_check = [
        DATA_DIR, MODEL_SAVE_DIR, MAIN_DATASET_DIR,
        VIDEO_SAMPLE_DIR, VIDEOS_DIR,
        VOLLEYBALL_DETECTION_DIR, VOLLEYBALL_TRACKING_DIR,
    ]
    for path in paths_to_check:
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"  [{status}] {path}")


if __name__ == "__main__":
    env = "Kaggle" if ON_KAGGLE else "Local"
    print(f"=== path_config ({env}) ===")
    print(f"  Data source:       {DATA_DIR}")
    print(f"  Main dataset:      {MAIN_DATASET_DIR}")
    print(f"  Model save dir:    {MODEL_SAVE_DIR}")
    print(f"  Detections:        {VOLLEYBALL_DETECTION_DIR}")
    print(f"  Tracking:          {VOLLEYBALL_TRACKING_DIR}")
    print(f"  Master JSON:       {JSON_DATA_DIR}")
    print(f"  Pickle cache:      {PICKLE_DUMP_DIR}")
    print(f"  LMDB frames:       {FRAMES_LMDB_DIR}")
    print(f"  Plots:             {PLOTS_DIR}")
    print(f"  Logs:              {LOGS_DIR}")
    print()
    print("Path validation:")
    validate_paths()
