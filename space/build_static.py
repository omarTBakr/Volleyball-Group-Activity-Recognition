"""Precompute B8 results on the bundled validation clips for the static Space.

Run from the repo root: ``python space/build_static.py``. Writes
``space_static/results.json`` and one annotated GIF per clip. The Space shows
these stored results; it does not run the model.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import _load_inputs, _render, list_clips, load_model, predict  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "space_static"
CKPT = Path(__file__).resolve().parents[1] / "saved_models" / "baseline8_stage_b_run1.pt"


def main() -> None:
    model = load_model(CKPT)
    (OUT / "gifs").mkdir(parents=True, exist_ok=True)
    results = []
    for name in list_clips():
        out = predict(model, name)
        frames, meta = _load_inputs(name)[3:]
        gif = _render(frames, meta, out["prediction"], width=560)
        shutil.move(gif, OUT / "gifs" / f"{name}.gif")
        results.append({
            "clip": name,
            "video_id": meta["video_id"],
            "clip_id": meta["clip_id"],
            "ground_truth": out["ground_truth"],
            "prediction": out["prediction"],
            "correct": out["prediction"] == out["ground_truth"],
            "probs": out["probs"],
        })
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    n_ok = sum(r["correct"] for r in results)
    print(f"{len(results)} clips, {n_ok} correct -> {OUT}")


if __name__ == "__main__":
    main()
