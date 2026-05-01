from pathlib import Path

import os 
# create a Path object for the current file's directory and Data subdirectory

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "DataSet"
MODEL_SAVE_DIR = BASE_DIR / "saved_models"
MAIN_DATASET_DIR = DATA_DIR / "main dataset"

VIDEO_SAMPLE_DIR = DATA_DIR / "videos_sample"
VIDEOS_DIR = DATA_DIR / "main dataset"

VOLLEYBALL_DETECTION_DIR = DATA_DIR / "volleyball-detections"
VOLLEYBALL_TRACKING_DIR = DATA_DIR / "volleyball_tracking_annotation"
# video samples 
VIDEO_SAMPLE1_DIR = VIDEO_SAMPLE_DIR / "7" / "38025"
VIDEO_SAMPLE2_DIR = VIDEO_SAMPLE_DIR / "7" / "51725"

VIDEO_SAMPLE3_DIR = VIDEO_SAMPLE_DIR / "10" / "18360"
VIDEO_SAMPLE4_DIR = VIDEO_SAMPLE_DIR / "10" / "20525"
VIDEO_SAMPLE5_DIR = VIDEO_SAMPLE_DIR / "10" / "20500"



def validate_paths():
    paths_to_check = [DATA_DIR, MODEL_SAVE_DIR, MAIN_DATASET_DIR, VIDEO_SAMPLE_DIR, VIDEOS_DIR, VIDEO_SAMPLE1_DIR, VIDEO_SAMPLE2_DIR, VIDEO_SAMPLE3_DIR, VIDEO_SAMPLE4_DIR]
    for path in paths_to_check:
        if not path.exists():
            print(f"Warning: Path {path} does not exist.")
        else:
            print(f"Path {path} is valid.")
if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)
    print("Data Directory:", DATA_DIR)
    validate_paths()
