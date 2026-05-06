from pathlib import Path
import os

# ── Environment detection ──────────────────────────────────────────────────────
ON_KAGGLE = Path('/kaggle/input').exists()

if ON_KAGGLE:
    # !! Set this to your dataset slug (the part after the username in the URL)
    # e.g. for kaggle.com/datasets/ahmedmohamed365/volleyball → slug = "volleyball"
    # Run `!ls /kaggle/input` in your notebook to confirm the exact folder name.
    KAGGLE_DATASET_SLUG = "datasets"

    KAGGLE_INPUT_DIR  = Path('/kaggle/input') / KAGGLE_DATASET_SLUG
    KAGGLE_OUTPUT_DIR = Path('/kaggle/working')

    # ── Source data (read-only on Kaggle) ──────────────────────────────────────
    DATA_DIR                  = KAGGLE_INPUT_DIR
    MAIN_DATASET_DIR          = KAGGLE_INPUT_DIR / "volleyball_" / "videos"
    VIDEO_SAMPLE_DIR          = KAGGLE_INPUT_DIR / "videos_sample"
    VIDEOS_DIR                = MAIN_DATASET_DIR
    VOLLEYBALL_DETECTION_DIR  = KAGGLE_INPUT_DIR / "volleyball-detections"
    VOLLEYBALL_TRACKING_DIR   = KAGGLE_INPUT_DIR / "volleyball_tracking_annotation"
    VOLLEYBALL_ANNOTATIONS_DIR = MAIN_DATASET_DIR

    # ── Generated/cached files → writable working dir ──────────────────────────
    _GEN_DIR           = KAGGLE_OUTPUT_DIR
    MODEL_SAVE_DIR     = KAGGLE_OUTPUT_DIR / "saved_models"

else:
    # ── Local paths (unchanged from original) ──────────────────────────────────
    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR                  = BASE_DIR / "DataSet"
    MODEL_SAVE_DIR            = BASE_DIR / "saved_models"
    MAIN_DATASET_DIR          = DATA_DIR / "volleyball_" / "videos"
    VIDEO_SAMPLE_DIR          = DATA_DIR / "videos_sample"
    VIDEOS_DIR                = MAIN_DATASET_DIR
    VOLLEYBALL_DETECTION_DIR  = DATA_DIR / "volleyball-detections"
    VOLLEYBALL_TRACKING_DIR   = DATA_DIR / "volleyball_tracking_annotation"
    VOLLEYBALL_ANNOTATIONS_DIR = MAIN_DATASET_DIR

    _GEN_DIR = DATA_DIR   # generated files sit inside DataSet/ locally

# ── Generated files (same variable names as before) ───────────────────────────
# These always point to a writable location regardless of environment.
JSON_DATA_DIR    = _GEN_DIR / "volleyball_master.json"
PICKLE_DUMP_DIR  = _GEN_DIR / "volleyball_master_pickle.pkl"
FRAMES_DUMP_DIR  = _GEN_DIR / "frames_paths_to_images.pkl"
FRAMES_LMDB_DIR  = _GEN_DIR / "frames_lmdb"

# Ensure writable dirs exist at import time
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
if ON_KAGGLE:
    KAGGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Video samples (local convenience paths) ────────────────────────────────────
VIDEO_SAMPLE1_DIR = VIDEO_SAMPLE_DIR / "7"  / "38025"
VIDEO_SAMPLE2_DIR = VIDEO_SAMPLE_DIR / "7"  / "51725"
VIDEO_SAMPLE3_DIR = VIDEO_SAMPLE_DIR / "10" / "18360"
VIDEO_SAMPLE4_DIR = VIDEO_SAMPLE_DIR / "10" / "20525"
VIDEO_SAMPLE5_DIR = VIDEO_SAMPLE_DIR / "10" / "20500"

# ── Label mappings (unchanged) ─────────────────────────────────────────────────
GROUP_ACTIVITY_TO_IDX = {
    "Right set":      0,
    "Right spike":    1,
    "Right pass":     2,
    "Right winpoint": 3,
    "Left winpoint":  4,
    "Left pass":      5,
    "Left spike":     6,
    "Left set":       7,
}

PERSON_ACTION_TO_IDX = {
    "Waiting":  0,
    "Setting":  1,
    "Digging":  2,
    "Falling":  3,
    "Spiking":  4,
    "Blocking": 5,
    "Jumping":  6,
    "Moving":   7,
    "Standing": 8,
}

# ── Validation helper ─────────────────────────────────────────────────────────
def validate_paths():
    paths_to_check = [
        DATA_DIR, MODEL_SAVE_DIR, MAIN_DATASET_DIR,
        VIDEO_SAMPLE_DIR, VIDEOS_DIR,
        VIDEO_SAMPLE1_DIR, VIDEO_SAMPLE2_DIR,
        VIDEO_SAMPLE3_DIR, VIDEO_SAMPLE4_DIR,
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
    print()
    print("Path validation:")
    validate_paths()