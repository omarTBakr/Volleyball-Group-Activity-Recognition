from configs.path_config import (
    MAIN_DATASET_DIR,
    VOLLEYBALL_DETECTION_DIR,
    VOLLEYBALL_TRACKING_DIR,
)
from matplotlib import pyplot as plt
import numpy as np


from pathlib import Path
import sys
from collections import defaultdict


def count_directories(directory):
    # make sure the provided path is a directory
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.")

    dir = 0
    for video_folder in directory.iterdir():
        if video_folder.is_dir():
            dir += 1
    return dir


def count_classes_in_directory(directory):
    action_classes = defaultdict(int)
    person_classes = defaultdict(int)

    action_file = directory / "action_detections.txt"
    person_file = directory / "person_detections.txt"

    # print(action_file)
    if action_file.exists():
        # print('file exists and attempting to open it ')
        with open(action_file, "r") as f:
            # print(f'file opened successfully {action_file}')
            for line in f:
                for (
                    entry
                ) in line.strip().split():  # Assuming class_id is the first element
                    try:
                        int(entry)  # Check if entry can be converted to an integer
                    except ValueError:
                        if (
                            ".jpg" not in entry
                            and not entry.replace(".", "").replace("-", "").isnumeric()
                        ):
                            action_classes[entry] += 1

    if person_file.exists():
        with open(person_file, "r") as f:
            for line in f:
                for (
                    entry
                ) in line.strip().split():  # Assuming class_id is the first element
                    try:
                        int(entry)  # Check if entry can be converted to an integer
                    except ValueError:
                        if (
                            ".jpg" not in entry
                            and not entry.replace(".", "").replace("-", "").isnumeric()
                        ):
                            person_classes[entry] += 1

    return action_classes, person_classes


def count_classes_in_dataset(dataset_directory):
    total_action_classes = []
    total_person_classes = []

    for video_folder in dataset_directory.iterdir():
        if video_folder.is_dir():
            for clip_folder in video_folder.iterdir():
                if clip_folder.is_dir():
                    action_classes, person_classes = count_classes_in_directory(
                        clip_folder
                    )
                    total_action_classes.append(action_classes)
                    total_person_classes.append(person_classes)
    return total_action_classes, total_person_classes


def count_classes_in_tracking_dataset(tracking_dataset_directory):
    tracking_classes = defaultdict(int)
    for directory in tracking_dataset_directory.iterdir():
        if directory.is_dir():
            tracking_file = directory / f"{directory.name}.txt"
            # print(tracking_file)
            if tracking_file.exists():
                # print('file exists and attempting to open it ')
                with open(tracking_file, "r") as f:
                    # print(f'file opened successfully {action_file}')
                    for line in f:
                        tracking_classes[ line.strip().split()[-1]] += 1
    return tracking_classes

def count_all_tracking_classes(tracking_dataset_directory):
    tracking_classes = []
    for video_folder in tracking_dataset_directory.iterdir():
        if video_folder.is_dir():
            tracking_classes.append(count_classes_in_tracking_dataset(video_folder))
    return tracking_classes

def plot_class_distribution(classes, label):
    
    # Action classes
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes.keys(), classes.values(), label=label)
    ax.bar_label(bars, padding=3)
    ax.set_title(  f"{label} classes distribution", fontsize=16)
    ax.set_ylabel("number of frames", fontsize=12)
    ax.set_xlabel("classes", fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    plt.tight_layout()
    plt.show()

def merge_dictionaries(dictionaries):
    merged_dict = defaultdict(int)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged_dict[key] += value
    return merged_dict
    
if __name__ == "__main__":
    # video_count = count_directories(MAIN_DATASET_DIR)
    # print(f"Total number of videos in the main dataset: {video_count}")
    # # For each video, count the number of clips (subdirectories)
    # for video_folder in MAIN_DATASET_DIR.iterdir():
    #     if video_folder.is_dir():
    #         clips_count = count_directories(video_folder)
    #         print(f"\tVideo {video_folder.name} has {clips_count} clips.")
    #         # number of images per clip
    #         print(f"\t\tNumber of images in each clip:{len(list( video_folder.iterdir()))}")
    # # Count action and person classes in the dataset
    action_classes, person_classes = count_classes_in_dataset(VOLLEYBALL_DETECTION_DIR)
    # join them into a bigger dictionary
    action_classes = merge_dictionaries(action_classes)
    person_classes = merge_dictionaries(person_classes)

    tracking_classes = count_all_tracking_classes(VOLLEYBALL_TRACKING_DIR)
    tracking_classes = merge_dictionaries(tracking_classes)

    print(f"action class  :{action_classes}\n" )
    print(f"person class  :{person_classes}\n" )
    print(f"tracking class  :{tracking_classes}\n" )
    print(
        f"Total unique action classes in the dataset: {len(action_classes.keys())}"
    )
    print(
        f"Total unique person classes in the dataset: {len(person_classes.keys())}"
    )
    print(
        f"Total unique tracking classes in the dataset: {len(tracking_classes.keys())}"
    )

    plot_class_distribution(action_classes, "action")
    plot_class_distribution(person_classes , "person")
    plot_class_distribution(tracking_classes, "tracking")   