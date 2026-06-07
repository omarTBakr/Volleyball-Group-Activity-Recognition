"""
Dataset split definitions for the volleyball group-activity dataset.

Each list contains the integer video IDs belonging to that split.
The union of all three lists covers every video in the dataset (0–54).
"""

TRAIN_VIDEOS_NUMBERS: list[int] = [
    1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31,
    32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54,
]

VALIDATION_VIDEO_NUMBERS: list[int] = [
    0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51,
]

TEST_VIDEOS_NUMBERS: list[int] = [
    4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47,
]

