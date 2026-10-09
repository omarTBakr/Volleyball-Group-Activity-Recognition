"""Draw B8's prediction on a clip and write it out as a video/GIF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# RGB. Team ids come from B8's rank-by-x split: 0 = left side, 1 = right side.
TEAM_COLORS = {0: (66, 135, 245), 1: (245, 160, 50), -1: (160, 160, 160)}
OK, BAD = (40, 170, 80), (220, 60, 60)


def annotate_clip(
    frames: list[np.ndarray],
    boxes: list[np.ndarray],
    team_ids: np.ndarray,
    pred: str,
    gt: str | None = None,
    max_width: int = 960,
) -> list[np.ndarray]:
    """Return annotated RGB copies of *frames*.

    Boxes are coloured by team (blue = left, orange = right); a banner shows
    the prediction, and the ground truth when given (green if they match).
    """
    team_ids = np.asarray(team_ids).reshape(-1)
    out = []
    for frame, frame_boxes in zip(frames, boxes):
        img = frame.copy()
        for p, (x1, y1, x2, y2) in enumerate(frame_boxes.tolist()):
            team = int(team_ids[p]) if p < len(team_ids) else -1
            cv2.rectangle(img, (x1, y1), (x2, y2), TEAM_COLORS.get(team, TEAM_COLORS[-1]), 3)

        h, w = img.shape[:2]
        scale = max_width / w if w > max_width else 1.0
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        text = f"Pred: {pred}" if gt is None else f"Pred: {pred}   GT: {gt}"
        color = (40, 40, 40) if gt is None else (OK if pred == gt else BAD)
        cv2.rectangle(img, (0, 0), (img.shape[1], 36), color, -1)
        cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(img)
    return out


def write_gif(frames: list[np.ndarray], path: str | Path, fps: float = 3.0) -> Path:
    """Write *frames* as a looping GIF (plays everywhere, incl. browsers)."""
    path = Path(path)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    return path


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: float = 3.0) -> Path:
    """Write *frames* as an mp4, preferring H.264 (browser-playable) over mp4v."""
    path = Path(path)
    h, w = frames[0].shape[:2]
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            return path
    raise RuntimeError("OpenCV could not open a video writer for mp4")
