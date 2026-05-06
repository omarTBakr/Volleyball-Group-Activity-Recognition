from pathlib import Path

# create a Path object for the current file's directory and Data subdirectory

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "DataSet"
MODEL_SAVE_DIR = BASE_DIR / "saved_models"
MAIN_DATASET_DIR = DATA_DIR / "volleyball_" / "videos"

VIDEO_SAMPLE_DIR = DATA_DIR / "videos_sample"
VIDEOS_DIR = MAIN_DATASET_DIR

VOLLEYBALL_DETECTION_DIR = DATA_DIR / "volleyball-detections"
VOLLEYBALL_TRACKING_DIR = DATA_DIR / "volleyball_tracking_annotation"
VOLLEYBALL_ANNOTATIONS_DIR = MAIN_DATASET_DIR
# video samples
VIDEO_SAMPLE1_DIR = VIDEO_SAMPLE_DIR / "7" / "38025"
VIDEO_SAMPLE2_DIR = VIDEO_SAMPLE_DIR / "7" / "51725"

VIDEO_SAMPLE3_DIR = VIDEO_SAMPLE_DIR / "10" / "18360"
VIDEO_SAMPLE4_DIR = VIDEO_SAMPLE_DIR / "10" / "20525"
VIDEO_SAMPLE5_DIR = VIDEO_SAMPLE_DIR / "10" / "20500"

JSON_DATA_DIR = DATA_DIR / "volleyball_master.json"
PICKLE_DUMP_DIR = DATA_DIR / "volleyball_master_pickle.pkl"
FRAMES_DUMP_DIR = DATA_DIR / "frames_paths_to_images.pkl"
FRAMES_LMDB_DIR = DATA_DIR / "frames_lmdb"

GROUP_ACTIVITY_TO_IDX = {
    "Right set":0,
    "Right spike":1,
    "Right pass":2,
    "Right winpoint":3,
    "Left winpoint":4,
    "Left pass":5,
    "Left spike":6,
    "Left set":7,
    }

PERSON_ACTION_TO_IDX = {
"Waiting":0,
"Setting":1,
"Digging":2,
"Falling":3,
"Spiking":4,
"Blocking":5,
"Jumping":6,
"Moving":7,
"Standing":8,

}
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
    print("Model Save Directory:", MODEL_SAVE_DIR)
    print("Main Dataset Directory:", MAIN_DATASET_DIR)
    print("Video Sample Directory:", VIDEO_SAMPLE_DIR)
    print("Videos Directory:", VIDEOS_DIR)
    print("Volleyball Detection Directory:", VOLLEYBALL_DETECTION_DIR)
    print("Volleyball Tracking Directory:", VOLLEYBALL_TRACKING_DIR)
    print("Video Sample 1 Directory:", VIDEO_SAMPLE1_DIR)
    print("Video Sample 2 Directory:", VIDEO_SAMPLE2_DIR)
    print("Video Sample 3 Directory:", VIDEO_SAMPLE3_DIR)
    print("Video Sample 4 Directory:", VIDEO_SAMPLE4_DIR)
    print("Video Sample 5 Directory:", VIDEO_SAMPLE5_DIR)
    validate_paths()
