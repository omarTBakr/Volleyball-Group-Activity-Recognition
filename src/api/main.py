"""FastAPI service: Baseline 8 group-activity prediction on validation clips.

Run: ``uvicorn src.api.main:app --port 8000`` then open ``/docs``.
B8 needs tracked player boxes, so the service serves clips from the dataset's
validation split (boxes included) rather than arbitrary uploaded video.
"""

from __future__ import annotations

import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.inference.b8 import default_ckpt, load_b8, predict_clip
from src.inference.render import annotate_clip, write_gif, write_mp4
from src.inference.sample import ClipSample, find_clip, get_dataset, random_validation_clip

UI_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class PlayerBox(BaseModel):
    box: list[int] = Field(description="[x1, y1, x2, y2] in original frame pixels")
    team: int = Field(description="0 = left side of the court, 1 = right side, -1 = unassigned")


class PredictionResponse(BaseModel):
    video_id: str
    clip_id: str
    ground_truth: str
    prediction: str
    correct: bool
    probabilities: dict[str, float]
    frames: list[list[PlayerBox]] = Field(description="Player boxes for each of the 9 input frames")


class ClipRequest(BaseModel):
    video_id: str = Field(examples=["24"])
    clip_id: str = Field(examples=["12585"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.device = device
    app.state.model = load_b8(default_ckpt(), device)
    app.state.lock = threading.Lock()  # one forward at a time on the shared model
    get_dataset("validation")           # build the annotation index up front
    yield


app = FastAPI(
    title="Volleyball Group Activity Recognition",
    description="Baseline 8 (hierarchical LSTM + team-split pooling) on validation clips.",
    lifespan=lifespan,
)


def _predict(request: Request, clip: ClipSample) -> dict:
    with request.app.state.lock:
        return predict_clip(request.app.state.model, clip.crops, clip.masks, clip.team_ids)


def _response(clip: ClipSample, pred: dict) -> PredictionResponse:
    teams = clip.team_ids[0].tolist()
    frames = [
        [PlayerBox(box=b, team=teams[p] if p < len(teams) else -1) for p, b in enumerate(fb.tolist())]
        for fb in clip.boxes
    ]
    return PredictionResponse(
        video_id=clip.video_id, clip_id=clip.clip_id, ground_truth=clip.gt_label,
        prediction=pred["label"], correct=pred["index"] == clip.gt_index,
        probabilities=pred["probs"], frames=frames,
    )


def _video(clip: ClipSample, pred: dict, fmt: str) -> FileResponse:
    frames = annotate_clip(clip.frames, clip.boxes, clip.team_ids[0].numpy(), pred["label"], clip.gt_label)
    suffix = ".gif" if fmt == "gif" else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
    (write_gif if fmt == "gif" else write_mp4)(frames, path)
    return FileResponse(path, media_type="image/gif" if fmt == "gif" else "video/mp4",
                        filename=f"{clip.video_id}_{clip.clip_id}{suffix}",
                        content_disposition_type="inline")  # show in the browser, don't download


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """Basic browser UI over the endpoints below."""
    return FileResponse(UI_PAGE, media_type="text/html")


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model": "baseline8", "device": request.app.state.device,
            "validation_clips": len(get_dataset("validation"))}


@app.get("/predict/random-validation", response_model=PredictionResponse)
def predict_random(request: Request, seed: int | None = Query(None, description="Fix for a reproducible clip")):
    clip = random_validation_clip(seed)
    return _response(clip, _predict(request, clip))


@app.get("/predict/random-validation/video")
def predict_random_video(
    request: Request,
    seed: int | None = None,
    format: str = Query("gif", pattern="^(gif|mp4)$"),
):
    clip = random_validation_clip(seed)
    return _video(clip, _predict(request, clip), format)


@app.post("/predict/clip", response_model=PredictionResponse)
def predict_specific(request: Request, body: ClipRequest):
    try:
        clip = find_clip(body.video_id, body.clip_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from e
    return _response(clip, _predict(request, clip))


@app.get("/predict/clip/video")
def predict_specific_video(
    request: Request,
    video_id: str,
    clip_id: str,
    format: str = Query("gif", pattern="^(gif|mp4)$"),
):
    try:
        clip = find_clip(video_id, clip_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from e
    return _video(clip, _predict(request, clip), format)
