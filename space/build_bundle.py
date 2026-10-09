"""Bundle a fixed set of validation clips for the Hugging Face Space.

Run from the repo root: ``python -m space.build_bundle``. For each chosen clip it
copies the 9 original frame JPEGs byte-for-byte (no re-encoding) and writes
``meta.json`` with the player boxes, team ids and ground truth the Space needs,
so the Space reproduces the repo's preprocessing exactly. Clips are stratified
over the 8 activities and drawn from the validation split only.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from src.inference.sample import get_dataset, load_clip

OUT = Path(__file__).resolve().parent / "clips"
N_CLIPS = 20
SEED = 0


def main() -> None:
    ds = get_dataset("validation")
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(ds._records):
        by_class[rec[2]].append(i)

    rng = random.Random(SEED)
    chosen: list[int] = []
    for idxs in by_class.values():             # 2 per class = 16 ...
        chosen += rng.sample(idxs, 2)
    rest = [i for i in range(len(ds)) if i not in chosen]
    chosen += rng.sample(rest, N_CLIPS - len(chosen))   # ... + 4 more at random

    if OUT.exists():
        shutil.rmtree(OUT)
    kept = 0
    for i in chosen:
        clip = load_clip(i)
        if clip is None:
            continue
        video_id, clip_id, _, frame_names, _ = ds._records[i]
        dest = OUT / f"{video_id}_{clip_id}"
        dest.mkdir(parents=True)
        for k, fname in enumerate(frame_names):
            shutil.copyfile(ds.dataset_dir / video_id / clip_id / fname, dest / f"frame_{k}.jpg")
        meta = {
            "video_id": video_id,
            "clip_id": clip_id,
            "ground_truth": clip.gt_label,
            "num_players_padded": int(clip.crops.shape[2]),
            "masks": clip.masks[0].int().tolist(),
            "team_ids": clip.team_ids[0].tolist(),
            "boxes": [b.tolist() for b in clip.boxes],   # per frame, xyxy, aligned with crops
        }
        (dest / "meta.json").write_text(json.dumps(meta))
        kept += 1
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"bundled {kept} clips, {size:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
