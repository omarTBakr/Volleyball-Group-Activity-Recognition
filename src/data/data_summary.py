"""
Dataset statistics and class-distribution analysis.

Counts action, person, tracking, and annotation classes across every
frame in the dataset and plots their distributions.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt  # ty:ignore[unresolved-import]

from configs.path_config import (
    VOLLEYBALL_ANNOTATIONS_DIR,
    VOLLEYBALL_DETECTION_DIR,
    VOLLEYBALL_TRACKING_DIR,
)

# ── Directory Counting ───────────────────────────────────────────────────────


def count_directories(directory: Path) -> int:
    """
    Count the number of immediate subdirectories in *directory*.

    Parameters
    ----------
    directory : Path
        The directory to inspect.

    Returns
    -------
    int
        Number of child directories.

    """
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return 0

    return sum(1 for child in directory.iterdir() if child.is_dir())


# ── Detection Class Counting ────────────────────────────────────────────────


def count_classes_in_directory(directory: Path) -> tuple[defaultdict[str, int], defaultdict[str, int]]:
    """
    Count action and person detection classes in a single clip directory.

    Parameters
    ----------
    directory : Path
        Path to a clip folder containing ``action_detections.txt``
        and/or ``person_detections.txt``.

    Returns
    -------
    tuple[defaultdict, defaultdict]
        ``(action_classes, person_classes)`` count dictionaries.

    """
    action_classes: defaultdict[str, int] = defaultdict(int)
    person_classes: defaultdict[str, int] = defaultdict(int)

    for filepath, target_dict in [
        (directory / "action_detections.txt", action_classes),
        (directory / "person_detections.txt", person_classes),
    ]:
        if not filepath.exists():
            continue
        with filepath.open(mode="r") as f:
            for line in f:
                for entry in line.strip().split():
                    try:
                        int(entry)
                    except ValueError:
                        if ".jpg" not in entry and not entry.replace(".", "").replace("-", "").isnumeric():
                            target_dict[entry] += 1

    return action_classes, person_classes


def count_classes_in_dataset(
    dataset_directory: Path,
) -> tuple[list[defaultdict[str, int]], list[defaultdict[str, int]]]:
    """
    Aggregate action and person class counts across all clips in the dataset.

    Returns
    -------
    tuple[list, list]
        ``(all_action_counts, all_person_counts)`` — one dict per clip.

    """
    total_action_classes: list[defaultdict[str, int]] = []
    total_person_classes: list[defaultdict[str, int]] = []

    for video_folder in dataset_directory.iterdir():
        if not video_folder.is_dir():
            continue
        for clip_folder in video_folder.iterdir():
            if not clip_folder.is_dir():
                continue
            action_classes, person_classes = count_classes_in_directory(clip_folder)
            total_action_classes.append(action_classes)
            total_person_classes.append(person_classes)

    return total_action_classes, total_person_classes


# ── Tracking Class Counting ─────────────────────────────────────────────────


def count_classes_in_tracking_dataset(tracking_dir: Path) -> defaultdict[str, int]:
    """Count action labels from tracking annotation files within a single video."""
    tracking_classes: defaultdict[str, int] = defaultdict(int)

    for directory in tracking_dir.iterdir():
        if not directory.is_dir():
            continue
        tracking_file = directory / f"{directory.name}.txt"
        if not tracking_file.exists():
            continue
        with tracking_file.open(mode="r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    tracking_classes[parts[-1]] += 1

    return tracking_classes


def count_all_tracking_classes(
    tracking_dataset_directory: Path,
) -> list[defaultdict[str, int]]:
    """Aggregate tracking class counts across all videos."""
    return [
        count_classes_in_tracking_dataset(video_folder)
        for video_folder in tracking_dataset_directory.iterdir()
        if video_folder.is_dir()
    ]


# ── Annotation Counting ─────────────────────────────────────────────────────


def count_annotations(directory: Path) -> defaultdict[str, int]:
    """
    Count group-activity and person-action labels from ``annotations.txt`` files.

    Parameters
    ----------
    directory : Path
        The videos dataset directory containing video subdirectories.

    Returns
    -------
    defaultdict[str, int]
        Combined counts of group-activity and person-action labels.

    """
    action_classes: defaultdict[str, int] = defaultdict(int)

    for video_folder in sorted(directory.iterdir()):
        if not video_folder.is_dir():
            continue

        annot_file = video_folder / "annotations.txt"
        if not annot_file.exists():
            continue

        with annot_file.open(mode="r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                action_classes[parts[1]] += 1

                idx = 6
                while idx < len(parts):
                    action_classes[parts[idx]] += 1
                    idx += 5

    return action_classes


# ── Utilities ────────────────────────────────────────────────────────────────


def merge_dictionaries(dictionaries: list[defaultdict[str, int]]) -> defaultdict[str, int]:
    """Merge a list of count dictionaries into a single aggregated dictionary."""
    merged: defaultdict[str, int] = defaultdict(int)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged[key] += value
    return merged


def plot_class_distribution(
    classes: dict[str, int],
    label: str,
    save_path: Path | None = None,
) -> None:
    """
    Plot a bar chart showing the distribution of class counts.

    Parameters
    ----------
    classes : dict[str, int]
        Mapping from class name to count.
    label : str
        Category label used in the chart title.
    save_path : Path or None
        If provided, save the figure to this directory.

    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes.keys(), classes.values(), label=label)
    ax.bar_label(bars, padding=3)
    ax.set_title(f"{label} classes distribution", fontsize=16)
    ax.set_ylabel("Number of frames", fontsize=12)
    ax.set_xlabel("Classes", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path / f"{label}_distribution.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    action_classes, person_classes = count_classes_in_dataset(VOLLEYBALL_DETECTION_DIR)
    action_classes = merge_dictionaries(action_classes)
    person_classes = merge_dictionaries(person_classes)

    tracking_classes = count_all_tracking_classes(VOLLEYBALL_TRACKING_DIR)
    tracking_classes = merge_dictionaries(tracking_classes)
    annotation_classes = count_annotations(VOLLEYBALL_ANNOTATIONS_DIR)

    print(f"Action classes:   {dict(action_classes)}\n")
    print(f"Person classes:   {dict(person_classes)}\n")
    print(f"Tracking classes: {dict(tracking_classes)}\n")
    print(f"Unique action classes:   {len(action_classes)}")
    print(f"Unique person classes:   {len(person_classes)}")
    print(f"Unique tracking classes: {len(tracking_classes)}")

    plot_class_distribution(action_classes, "action")
    plot_class_distribution(person_classes, "person")
    plot_class_distribution(tracking_classes, "tracking")
    plot_class_distribution(annotation_classes, "annotation")
