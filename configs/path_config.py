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



if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)
    print("Data Directory:", DATA_DIR)
