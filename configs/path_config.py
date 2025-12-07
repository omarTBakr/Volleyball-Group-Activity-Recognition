from pathlib import Path


# create a Path object for the current file's directory and Data subdirectory

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "DataSet"
MODEL_SAVE_DIR = BASE_DIR / "saved_models"
MAIN_DATASET_DIR = DATA_DIR / "main dataset"

VIDEO_SAMPLE_DIR = DATA_DIR / "videos_sample"
VIDEOS_DIR = DATA_DIR / "main dataset"
VIDEO_SAMPLE1_DIR = VIDEO_SAMPLE_DIR / "7" / "38025"
VIDEO_SAMPLE2_DIR = VIDEO_SAMPLE_DIR / "10" / "51725"
VIDEO_SAMPLE3_DIR = VIDEO_SAMPLE_DIR / "10" / "18360"
VIDEO_SAMPLE4_DIR = VIDEO_SAMPLE_DIR / "10" / "20525"

# testing the directories

TRAIN_VIDEOS_NUMBERS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VALIDATION_VIDEO_NUMBERS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_VIDEOS_NUMBERS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)
    print("Data Directory:", DATA_DIR)
