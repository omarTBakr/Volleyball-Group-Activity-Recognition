"""Sanity check: score B8 on N random validation clips and save a demo GIF.

Run: ``python -m src.inference.check --n 200``. Accuracy should land near
B8's reported numbers — if it doesn't, the inference preprocessing is wrong.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from src.inference.b8 import load_b8, predict_clip
from src.inference.render import annotate_clip, write_gif
from src.inference.sample import get_dataset, load_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="validation clips to score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("demo_clip.gif"))
    args = parser.parse_args()

    model = load_b8(device=args.device)
    ds = get_dataset("validation")
    indices = random.Random(args.seed).sample(range(len(ds)), min(args.n, len(ds)))

    correct = total = 0
    last = None
    for i in indices:
        clip = load_clip(i)
        if clip is None:
            continue
        pred = predict_clip(model, clip.crops, clip.masks, clip.team_ids)
        correct += pred["index"] == clip.gt_index
        total += 1
        last = (clip, pred)
    print(f"B8 validation accuracy on {total} random clips: {correct / total:.4f}")

    clip, pred = last
    frames = annotate_clip(clip.frames, clip.boxes, clip.team_ids[0].numpy(), pred["label"], clip.gt_label)
    print("wrote", write_gif(frames, args.out), f"({clip.video_id}/{clip.clip_id})")


if __name__ == "__main__":
    main()
