"""Gradio demo: Baseline 8 volleyball group-activity recognition on bundled validation clips."""

from __future__ import annotations

import random

import gradio as gr

from predictor import list_clips, load_model, predict

MODEL = load_model()
CLIPS = list_clips()


def run(clip_name: str):
    out = predict(MODEL, clip_name)
    verdict = "✅ correct" if out["prediction"] == out["ground_truth"] else "❌ wrong"
    summary = f"**Predicted:** {out['prediction']}  \n**Ground truth:** {out['ground_truth']}  \n{verdict}"
    return out["gif"], out["probs"], summary


def random_clip():
    name = random.choice(CLIPS)
    return (name, *run(name))


with gr.Blocks(title="Volleyball Group Activity Recognition") as demo:
    gr.Markdown(
        "# Volleyball Group Activity Recognition — Baseline 8\n"
        "Hierarchical player-LSTM → team-split pooling → scene-LSTM. It reads **9 frames and the tracked player boxes** "
        "(blue = left team, orange = right team) and predicts one of 8 group activities. "
        "The clips below are held-out **validation** clips (video_clip ids). "
        "Reported test accuracy: **85.64%** (1,337 clips)."
    )
    with gr.Row():
        clip = gr.Dropdown(CLIPS, value=CLIPS[0], label="Validation clip")
        rand = gr.Button("🎲 Random clip")
        go = gr.Button("Predict", variant="primary")
    with gr.Row():
        video = gr.Image(label="Annotated clip", type="filepath")
        with gr.Column():
            label = gr.Label(num_top_classes=8, label="Class probabilities")
            summary = gr.Markdown()
    go.click(run, clip, [video, label, summary])
    clip.change(run, clip, [video, label, summary])
    rand.click(random_clip, None, [clip, video, label, summary])
    demo.load(run, clip, [video, label, summary])

if __name__ == "__main__":
    demo.launch()
