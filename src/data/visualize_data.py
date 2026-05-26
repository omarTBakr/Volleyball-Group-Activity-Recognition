"""
Visualization tools for the volleyball dataset.

Provides functions to display sample frames, video sequences,
and fully-annotated clips with bounding boxes and labels.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2  # ty:ignore[unresolved-import]
from matplotlib import pyplot as plt  # ty: ignore[import]  # ty:ignore[unresolved-import]

from configs.path_config import MAIN_DATASET_DIR, VIDEO_SAMPLE_DIR
from src.pickle_dump import load_from_pickle

# ── Video sample directories (used only here) ───────────────────────────────

VIDEO_SAMPLE_DIRS: list[Path] = [
    VIDEO_SAMPLE_DIR / "7"  / "38025",
    VIDEO_SAMPLE_DIR / "7"  / "51725",
    VIDEO_SAMPLE_DIR / "10" / "18360",
    VIDEO_SAMPLE_DIR / "10" / "20525",
    VIDEO_SAMPLE_DIR / "10" / "20500",
]

# ── Color palette for person actions ────────────────────────────────────────

ACTION_COLORS: dict[str, tuple[int, int, int]] = {
    "blocking": (255,   0,   0),   # red
    "digging":  (  0, 200,   0),   # green
    "falling":  (255, 165,   0),   # orange
    "jumping":  (  0,   0, 255),   # blue
    "moving":   (128,   0, 128),   # purple
    "setting":  (  0, 255, 255),   # cyan
    "spiking":  (255, 255,   0),   # yellow
    "standing": (128, 128, 128),   # gray
    "waiting":  (255, 105, 180),   # pink
}
DEFAULT_COLOR = (200, 200, 200)


# ── Dispatcher ───────────────────────────────────────────────────────────────


def visualize_data(n_pictures: int = 5, video: bool = False, n_videos: int = 2) -> None:
    """
    High-level dispatcher: show either images or video sequences.

    Parameters
    ----------
    n_pictures : int
        Number of frames per clip to display (must be odd).
    video : bool
        If True, display video sequences instead of static images.
    n_videos : int
        Number of videos to display when ``video=True``.

    """
    if video:
        visualize_videos(n_videos)
    else:
        visualize_images(n_pictures)


# ── Video Display ────────────────────────────────────────────────────────────


def visualize_videos(n_videos: int = 2) -> None:
    """Encode and display sample video clips via OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 10, (640, 480))

    for directory in VIDEO_SAMPLE_DIRS[:n_videos]:
        for frame in reversed(list(directory.iterdir())):
            if frame.is_file() and frame.suffix == ".jpg":
                img = cv2.imread(str(frame))
                img = cv2.resize(img, (640, 480))
                out.write(img)
                cv2.imshow("Live Encoding", img)

            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

    out.release()
    cv2.destroyAllWindows()


# ── Image Grid ───────────────────────────────────────────────────────────────


def visualize_images(n_pictures: int = 5) -> None:
    """
    Display a grid of frames centered around the middle frame of each sample clip.

    Parameters
    ----------
    n_pictures : int
        Number of frames per row (must be odd).

    """
    assert n_pictures % 2 == 1, "n_pictures must be odd"
    grid: list[list] = []

    for directory in VIDEO_SAMPLE_DIRS:
        all_frames = sorted(directory.glob("*.jpg"), key=lambda x: int(x.stem))
        if not all_frames:
            continue

        mid_idx = len(all_frames) // 2
        offset = n_pictures // 2

        video_sequence = []
        for i in range(mid_idx - offset, mid_idx + offset + 1):
            if 0 <= i < len(all_frames):
                img = cv2.imread(str(all_frames[i]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video_sequence.append(cv2.flip(img, 1))

        if video_sequence:
            grid.append(video_sequence)

    fig, axes = plt.subplots(len(grid), n_pictures, figsize=(15, 2 * len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            axes[i, j].imshow(grid[i][j])
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()


# ── Fully Annotated Visualization ────────────────────────────────────────────


def visualize_video_fully_annotated(
    n_clips: int = 3,
    save_path: str | Path | None = None,
) -> None:
    """
    Display random clips with tracking bounding boxes and group-activity labels.

    Uses the pickle-backed master data and raw ``.jpg`` frames from disk.

    Parameters
    ----------
    n_clips : int
        Number of random clips to visualize.
    save_path : str, Path, or None
        If provided, save the figure instead of showing it.

    """
    master_data = load_from_pickle()
    all_keys = list(master_data.keys())
    chosen_keys = random.sample(all_keys, min(n_clips, len(all_keys)))

    fig, axes = plt.subplots(1, n_clips, figsize=(8 * n_clips, 6))
    if n_clips == 1:
        axes = [axes]

    for ax, clip_key in zip(axes, chosen_keys):
        clip_data = master_data[clip_key]
        video_id, clip_id = clip_key.split("/", 1)

        # Load the middle frame
        frame_path = MAIN_DATASET_DIR / video_id / clip_id / f"{clip_id}.jpg"
        if not frame_path.exists():
            ax.set_title(f"{clip_key} — frame not found")
            ax.axis("off")
            continue

        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes
        middle_frame_name = f"{clip_id}.jpg"
        tracking = clip_data.get("tracking", {})
        persons = tracking.get(middle_frame_name, [])

        if not persons:
            actions = clip_data.get("actions", {})
            persons = actions.get(middle_frame_name, [])

        for person in persons:
            box = person["box"]
            action = person.get("action", person.get("label", "unknown"))
            player_id = person.get("id", "?")
            color = ACTION_COLORS.get(action, DEFAULT_COLOR)

            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label_text = f"{player_id}: {action}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

        scene_class = clip_data.get("scene_class", "unknown")
        ax.imshow(img)
        ax.set_title(f"{clip_key}\nGroup Activity: {scene_class}", fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    visualize_video_fully_annotated(n_clips=3)
