"""Run Baseline 8 on a bundled clip and render the annotated result (no Gradio dependency)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from b8_model import NUM_GROUP_ACTIVITIES, NUM_PERSON_ACTIONS, GroupTemporalClassifier, PersonTemporalLSTM, _group_dims, _person_dims

CLASSES = ["l-pass", "r-pass", "l-spike", "r_spike", "l_set", "r_set", "l_winpoint", "r_winpoint"]
CLIPS_DIR = Path(__file__).resolve().parent / "clips"
MODEL_REPO = "OmarTBakr/volleyball-activity-b8"
CKPT_NAME = "baseline8_stage_b_run1.pt"

TEAM_COLORS = {0: (66, 135, 245), 1: (245, 160, 50)}  # left = blue, right = orange
_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def list_clips() -> list[str]:
    return sorted(p.name for p in CLIPS_DIR.iterdir() if (p / "meta.json").is_file())


def load_model(ckpt_path: str | os.PathLike | None = None) -> GroupTemporalClassifier:
    """Load B8 from a local path, ``$VB_B8_CKPT``, or the Hugging Face model repo."""
    ckpt_path = ckpt_path or os.environ.get("VB_B8_CKPT")
    if not ckpt_path:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(MODEL_REPO, CKPT_NAME)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    h1, layers = _person_dims(state, prefix="person.")
    h2, pool, T, head = _group_dims(state)
    person = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS, backbone_name="resnet50", checkpoint=None,
        lstm_hidden=h1, lstm_layers=layers, pretrained_backbone=False,
    )
    model = GroupTemporalClassifier(
        person_model=person, num_classes=NUM_GROUP_ACTIVITIES,
        lstm2_hidden=h2, pool=pool, T=T, hidden_dim=head,
    )
    model.load_state_dict(state)
    return model.eval()


def _load_inputs(clip_name: str):
    d = CLIPS_DIR / clip_name
    meta = json.loads((d / "meta.json").read_text())
    frames = [Image.open(d / f"frame_{k}.jpg").convert("RGB") for k in range(len(meta["boxes"]))]
    P = meta["num_players_padded"]

    per_frame = []
    for img, boxes in zip(frames, meta["boxes"]):
        crops = torch.stack([_TF(img.crop(tuple(b))) for b in boxes]) if boxes else torch.zeros(0, 3, 224, 224)
        if crops.shape[0] < P:  # zero-pad frames with fewer players, as the repo's loader does
            crops = torch.cat([crops, crops.new_zeros(P - crops.shape[0], 3, 224, 224)])
        per_frame.append(crops)
    return (
        torch.stack(per_frame)[None],                              # (1, T, P, 3, 224, 224)
        torch.tensor(meta["masks"], dtype=torch.bool)[None],       # (1, P)
        torch.tensor(meta["team_ids"], dtype=torch.long)[None],    # (1, P)
        frames, meta,
    )


@torch.no_grad()
def predict(model: GroupTemporalClassifier, clip_name: str) -> dict:
    crops, masks, team_ids, frames, meta = _load_inputs(clip_name)
    probs = F.softmax(model(crops, masks, team_ids).float(), dim=1)[0]
    pred = CLASSES[int(probs.argmax())]
    return {
        "probs": {c: float(p) for c, p in zip(CLASSES, probs)},
        "prediction": pred,
        "ground_truth": meta["ground_truth"],
        "gif": _render(frames, meta, pred),
        "meta": meta,
    }


def _render(frames, meta, pred: str, width: int = 800) -> str:
    gt = meta["ground_truth"]
    teams = meta["team_ids"]
    out = []
    for img, boxes in zip(frames, meta["boxes"]):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        for p, b in enumerate(boxes):
            draw.rectangle(b, outline=TEAM_COLORS.get(teams[p] if p < len(teams) else -1, (160, 160, 160)), width=3)
        banner = (40, 170, 80) if pred == gt else (220, 60, 60)
        draw.rectangle([0, 0, img.width, 40], fill=banner)
        draw.text((10, 12), f"Pred: {pred}    GT: {gt}", fill=(255, 255, 255))
        out.append(img.resize((width, int(img.height * width / img.width))))
    path = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
    out[0].save(path, save_all=True, append_images=out[1:], duration=350, loop=0)
    return path
