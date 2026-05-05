
import cv2
from matplotlib import pyplot as plt

from configs.path_config import (
    VIDEO_SAMPLE1_DIR,
    VIDEO_SAMPLE2_DIR,
    VIDEO_SAMPLE3_DIR,
    VIDEO_SAMPLE4_DIR,
    VIDEO_SAMPLE5_DIR,
)

videos_dir = [VIDEO_SAMPLE1_DIR , VIDEO_SAMPLE2_DIR , VIDEO_SAMPLE3_DIR , VIDEO_SAMPLE4_DIR , VIDEO_SAMPLE5_DIR]


def visualize_data( n_pictures = 5 , video = False , n_videos = 2):
    """
    Visualizes a sample of the data set by displaying a grid of images with their corresponding labels.

    Args:
        n_pictures (int): The number of images to display in the grid (default is 8)
        video (bool): Whether to display videos instead of images (default is False)
        n_videos (int): The number of videos to display (default is 2)
    
    Note : this is the hirarchachy of the dataset :
    dataset/
    ├── main_dataset/
    ├── video_samples/
    │   ├── 7/
    │   ├── 10/
    │   ├── Info.txt/
    ├── volleyball-detection/
    |   |-0
    |        |- 3596
    |           |-action_detection.txt [number.jpg  count xmin ymin xmax ymax confindence class ... ]
    |           |-person_detection.txt
    ├── volleyball_tracking_annotation/
        |- 0
            |- 3596
                |- 3596.txt

    """
    if video:
        visualize_videos(n_videos)
    else:
        visualize_images(n_pictures)


def visualize_videos(n_videos = 2):
    """
    Visualizes a sample of the videos in the dataset by displaying them.

    Args:
        n_videos (int): The number of videos to display (default is 2)

    """
    # setup the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # VideoWriter(filename, fourcc, fps, frameSize)
    out = cv2.VideoWriter("output.mp4", fourcc, 10, (640, 480))

    for directory in videos_dir[:n_videos]:

        for frame in reversed(list(directory.iterdir())):
            if frame.is_file() and frame.suffix == ".jpg":
                frame = cv2.imread(str(frame))
                frame = cv2.resize(frame, (640, 480))
                out.write(frame)
                cv2.imshow("Live Encoding", frame)

            # Press 'q' to stop early
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

    out.release()
    cv2.destroyAllWindows()

# the below code with parse the data into a big json
# then we will verify the data correctness


# Run the process

def visualize_images(n_pictures=5):
    """
    Visualizes a sample of the images in the dataset by displaying a grid of images with their corresponding labels.

    Args:
        n_pictures (int): The number of images to display in the grid (default is 5) will display 5 * 5 images = 25 images total , 5 from each video_sample directory 
        will choose frames from (2 before the middle  + middle + 2 after middle ) 

    """
    assert n_pictures % 2 == 1, "n_pictures must be odd"
    grid = []

    for directory in videos_dir:
        # We list all jpgs to find the actual count , note this is valid because they are numbered sequentially , so the middle index will be the middle frame
        all_frames = sorted(directory.glob("*.jpg"),key=lambda x: int(x.stem))

        if not all_frames: # if the directory is empty skip it
            continue

        mid_idx = len(all_frames) // 2
        offset = n_pictures // 2

        video_sequence = []
        # Calculate indices based on the middle
        for i in range(mid_idx - offset, mid_idx + offset + 1):
            # Check bounds to ensure index exists
            if 0 <= i < len(all_frames):
                img = cv2.imread(str(all_frames[i]))
                if img is not None:
                    # BGR to RGB for Matplotlib
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Keep your horizontal flip
                    video_sequence.append(cv2.flip(img, 1))

        if video_sequence:
            grid.append(video_sequence)

    # Visualization
    fig, axes = plt.subplots(len(grid), n_pictures, figsize=(15, 2 * len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            axes[i, j].imshow(grid[i][j])
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

from configs.path_config import MAIN_DATASET_DIR
from src.pickle_dump import load_from_pickle

# ── Color palette for person actions ────────────────────────────────────────

ACTION_COLORS = {
    "blocking":  (255,   0,   0),   # red
    "digging":   (  0, 200,   0),   # green
    "falling":   (255, 165,   0),   # orange
    "jumping":   (  0,   0, 255),   # blue
    "moving":    (128,   0, 128),   # purple
    "setting":   (  0, 255, 255),   # cyan
    "spiking":   (255, 255,   0),   # yellow
    "standing":  (128, 128, 128),   # gray
    "waiting":   (255, 105, 180),   # pink
}
DEFAULT_COLOR = (200, 200, 200)


def visualize_video_fully_annotated(n_clips=3, save_path=None):
    """
    Pick random clips and draw fully annotated frames showing:

    - **Tracking bounding boxes** with player ID and action label
    - **Group activity** (scene class) as the figure title

    Uses the pickle-backed master data and raw .jpg frames from disk.

    Parameters
    ----------
    n_clips : int
        Number of random clips to visualise.
    save_path : str or Path, optional
        If provided, save the figure instead of showing it.

    """
    import random

    master_data = load_from_pickle()

    # Pick n_clips random clips
    all_keys = list(master_data.keys())
    chosen_keys = random.sample(all_keys, min(n_clips, len(all_keys)))

    fig, axes = plt.subplots(1, n_clips, figsize=(8 * n_clips, 6))
    if n_clips == 1:
        axes = [axes]

    for ax, clip_key in zip(axes, chosen_keys):
        clip_data = master_data[clip_key]
        video_id, clip_id = clip_key.split("/", 1)

        # Load the middle frame image
        frame_path = MAIN_DATASET_DIR / video_id / clip_id / f"{clip_id}.jpg"
        if not frame_path.exists():
            ax.set_title(f"{clip_key} — frame not found")
            ax.axis("off")
            continue

        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw tracking bounding boxes on the middle frame
        middle_frame_name = f"{clip_id}.jpg"
        tracking = clip_data.get("tracking", {})
        persons = tracking.get(middle_frame_name, [])

        # Fallback to action detections if tracking has no data for this frame
        if not persons:
            actions = clip_data.get("actions", {})
            persons = actions.get(middle_frame_name, [])

        for person in persons:
            box = person["box"]  # [x1, y1, x2, y2] for tracking
            action = person.get("action", person.get("label", "unknown"))
            player_id = person.get("id", "?")
            color = ACTION_COLORS.get(action, DEFAULT_COLOR)

            x1, y1, x2, y2 = box
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label: "ID: action"
            label_text = f"{player_id}: {action}"
            # Background for text
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Title: group activity
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
    # visualize_data(video=True, n_videos=4)
    # visualize_data(video=False, n_pictures=5)
    visualize_video_fully_annotated(n_clips=3)

