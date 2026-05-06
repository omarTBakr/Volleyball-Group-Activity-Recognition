"""
Two-stage pipeline for building the volleyball master dataset.

Stage 1 — Player-level parsing (create_master_json):
    Iterates over the volleyball-detections and volleyball_tracking_annotation
    directories, parsing per-frame action detections, person detections, and
    tracking annotations into a single master JSON keyed by "video_id/clip_id".
    Each clip entry contains:
        - "actions":  dict[frame_name -> list of {box, score, label}]  (9 person-action classes)
        - "persons":  dict[frame_name -> list of {box, score, label}]  (person detector output)
        - "tracking": dict[frame_name -> list of {id, box, flags, action}] (tracked players)

Stage 2 — Scene-level enrichment (parse_scene_annotations + merge_dataset_levels):
    Reads each video's annotations.txt from the main dataset directory to extract
    the group-activity label (one of 8 scene classes) per clip, then merges it
    into the master JSON under a new "scene_class" key per clip.

The result is a single dictionary that contains both annotation levels required
by all baseline models (B1–B8).
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from configs.path_config import (
    JSON_DATA_DIR,
    VOLLEYBALL_ANNOTATIONS_DIR,
    VOLLEYBALL_DETECTION_DIR,
    VOLLEYBALL_TRACKING_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_verified_json(json_path: Path) -> dict:
    """
    Load a JSON file after verifying it exists.

    Parameters
    ----------
    json_path : Path
        Absolute path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON contents, or an empty dict if the file is missing.

    """
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found.")
        return {}
    with json_path.open("r") as f:
        return json.load(f)

def parse_scene_annotations(file_path: Path) -> dict[str, str]:
    """
    Parse a video-level annotations.txt to extract group-activity (scene) labels.

    Each line has the format:
        frame_name.jpg  group_activity  x1 y1 w h action  [x1 y1 w h action] ...

    Only the first two columns (frame name and group activity) are extracted here;
    per-player bounding boxes and actions are already captured in Stage 1.

    Parameters
    ----------
    file_path : Path
        Path to a single video's annotations.txt.

    Returns
    -------
    dict[str, str]
        Mapping from frame name (e.g. '13286.jpg') to group-activity label
        (e.g. 'r_winpoint').

    """
    scene_data: dict[str, str] = {}
    if not file_path.exists():
        return scene_data

    with file_path.open("r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 2:
                continue

            img_name = p[0]
            scene_action = p[1]  # e.g., 'r_winpoint', 'l-spike'

            # We only need the scene label here because the
            # player boxes are already in the master JSON
            scene_data[img_name] = scene_action
    return scene_data


def merge_dataset_levels(master_json: dict, scene_labels: dict[str, str]) -> dict:
    """
    Merge Stage 2 scene labels into the Stage 1 master JSON (in-place).

    For every clip in *master_json*, this looks up the clip's middle frame
    in *scene_labels* and adds a top-level ``"scene_class"`` key to the clip
    entry.  Clips whose middle frame is not found in *scene_labels* receive
    ``None``.

    Parameters
    ----------
    master_json : dict
        The master dictionary produced by Stage 1 (``create_master_json``).
    scene_labels : dict[str, str]
        Combined output of ``parse_scene_annotations`` across all videos.

    Returns
    -------
    dict
        The same *master_json* reference, now enriched with ``"scene_class"``.

    """
    merged_count = 0

    for clip_id, content in master_json.items():
        # Check 'actions' dictionary for frame names
        frames = content.get("actions", {}).keys()

        # Find the scene label for this clip by checking its frames
        clip_scene_label = None
        for img_name in frames:
            if img_name in scene_labels:
                clip_scene_label = scene_labels[img_name]
                break  # All frames in a clip share one scene label

        # Add the scene label to the clip level
        master_json[clip_id]["scene_class"] = clip_scene_label
        if clip_scene_label:
            merged_count += 1

    print(f"✅ Merged scene labels for {merged_count} clips.")
    return master_json


def parse_detection_file(file_path: Path) -> dict[str, list[dict]]:
    """
    Parse an action_detections.txt or person_detections.txt file.

    Each line is tab-separated:
        frame_name  num_entries  x1 y1 x2 y2 score label  [x1 y1 x2 y2 score label] ...

    Parameters
    ----------
    file_path : Path
        Path to the detection text file.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from frame name to a list of detection entries.

    """
    data: dict[str, list[dict]] = {}

    if not file_path.exists():
        logger.warning("Detection file not found: %s", file_path)
        return data

    with file_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            img_name = parts[0]
            try:
                num_entries = int(parts[1])
            except ValueError:
                logger.warning("Malformed entry count on line %d in %s", line_num, file_path.name)
                continue

            entries = []
            for i in range(num_entries):
                start = 2 + (i * 6)
                # Need 6 fields: x1, y1, x2, y2, score, label
                if start + 6 > len(parts):
                    logger.debug(
                        "Truncated entry %d on line %d in %s (expected %d fields, got %d)",
                        i, line_num, file_path.name, start + 6, len(parts),
                    )
                    break

                try:
                    entries.append({
                        "box": [
                            int(parts[start]),
                            int(parts[start + 1]),
                            int(parts[start + 2]),
                            int(parts[start + 3]),
                        ],
                        "score": float(parts[start + 4]),
                        "label": parts[start + 5],
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(
                        "Skipping malformed entry %d on line %d in %s: %s",
                        i, line_num, file_path.name, e,
                    )

            data[img_name] = entries

    return data


def parse_tracking_file(file_path: Path) -> dict[str, list[dict]]:
    """
    Parse a tracking annotation file (e.g. 3596.txt).

    Each line is space-separated:
        track_id  x1 y1 x2 y2  frame_number  flag1 flag2 flag3  action

    Parameters
    ----------
    file_path : Path
        Path to the tracking annotation text file.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from frame name (e.g. '3596.jpg') to a list of tracking entries.

    """
    data: dict[str, list[dict]] = defaultdict(list)

    if not file_path.exists():
        logger.warning("Tracking file not found: %s", file_path)
        return data

    with file_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) < 10:
                logger.warning(
                    "Skipping short line %d in %s (got %d fields, expected 10)",
                    line_num, file_path.name, len(parts),
                )
                continue

            try:
                img_name = f"{parts[5]}.jpg"
                data[img_name].append({
                    "id": int(parts[0]),
                    "box": [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])],
                    "flags": [int(parts[6]), int(parts[7]), int(parts[8])],
                    "action": parts[9],
                })
            except (ValueError, IndexError) as e:
                logger.warning("Skipping malformed line %d in %s: %s", line_num, file_path.name, e)

    return dict(data)


def create_master_json(
    detection_dir: Path,
    tracking_dir: Path,
    output_name: str = "volleyball_master.json",
) -> None:
    """
    Build a unified JSON combining detection and tracking data for every clip.

    Parameters
    ----------
    detection_dir : Path
        Root directory containing volleyball-detections (video/clip/action_detections.txt).
    tracking_dir : Path
        Root directory containing volleyball_tracking_annotation (video/clip/clip.txt).
    output_name : str
        Name of the output JSON file (saved under OUTPUT_DIR).

    """
    master_data: dict[str, dict] = {}

    # Iterate over video folders in the detection directory
    for video_folder in sorted(detection_dir.iterdir()):
        if not video_folder.is_dir():
            continue

        for clip_folder in sorted(video_folder.iterdir()):
            if not clip_folder.is_dir():
                continue

            clip_key = f"{video_folder.name}/{clip_folder.name}"

            # Detection files live in detection_dir/video/clip/
            action_file = clip_folder / "action_detections.txt"
            person_file = clip_folder / "person_detections.txt"

            # Tracking file lives in tracking_dir/video/clip/clip.txt
            tracking_file = tracking_dir / video_folder.name / clip_folder.name / f"{clip_folder.name}.txt"

            master_data[clip_key] = {
                "actions": parse_detection_file(action_file),
                "persons": parse_detection_file(person_file),
                "tracking": parse_tracking_file(tracking_file),
            }



    output_path = JSON_DATA_DIR

    with output_path.open("w") as f:
        json.dump(master_data, f, indent=4)

    logger.info("Master JSON saved to: %s (%d clips)", output_path, len(master_data))


def enrich_with_scene_labels(
    annotations_dir: Path = VOLLEYBALL_ANNOTATIONS_DIR,
    json_path: Path = JSON_DATA_DIR,
    save: bool = True,
) -> dict:
    """
    Stage 2 — Load the master JSON and enrich each clip with its scene-level
    group-activity label parsed from each video's annotations.txt.

    Parameters
    ----------
    annotations_dir : Path
        Root directory containing per-video folders, each with an
        ``annotations.txt`` file.
    json_path : Path
        Path to the master JSON produced by Stage 1.
    save : bool
        If True, overwrite *json_path* with the enriched data.

    Returns
    -------
    dict
        The enriched master dictionary with ``"scene_class"`` per clip.

    """
    master_data = load_verified_json(json_path)

    # annotations.txt lives at the *video* level, not per-clip
    scene_labels: dict[str, str] = {}
    for video_folder in sorted(annotations_dir.iterdir()):
        if not video_folder.is_dir():
            continue
        annot_file = video_folder / "annotations.txt"
        video_scenes = parse_scene_annotations(annot_file)
        scene_labels.update(video_scenes)

    logger.info("Collected scene labels for %d frames across all videos.", len(scene_labels))

    merged_data = merge_dataset_levels(master_data, scene_labels)

    if save:
        with json_path.open("w") as f:
            json.dump(merged_data, f, indent=4)
        logger.info("Enriched master JSON saved to: %s", json_path)

    return merged_data


if __name__ == "__main__":
    # ── Stage 1: create the master JSON from detections + tracking ──
    # Uncomment to re-generate (takes a while on the full 60G dataset):
    create_master_json(
        detection_dir=VOLLEYBALL_DETECTION_DIR,
        tracking_dir=VOLLEYBALL_TRACKING_DIR,
    )

    # ── Stage 2: enrich with scene-level group-activity labels ──
    enrich_with_scene_labels()
