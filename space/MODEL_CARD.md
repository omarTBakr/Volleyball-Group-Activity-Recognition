---
license: mit
library_name: pytorch
pipeline_tag: video-classification
tags: [volleyball, group-activity-recognition, lstm, resnet50, sports]
datasets: [volleyball]
metrics: [accuracy, f1]
---

# Volleyball Group Activity Recognition — Baseline 8

Hierarchical temporal model for the Volleyball dataset (Ibrahim et al., CVPR 2016): a frozen ResNet-50 feature
extractor per player crop → **player LSTM** over 9 frames → pooling **separately per team** (left/right, split by
box x-position) → **scene LSTM** with skip connections → 8-way group-activity classifier.

## Results (test split, 1,337 clips, 16 videos)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| B1 single-frame ResNet-50 | 62.60% | 0.630 |
| B7 hierarchical, side-blind pooling | 73.75% | 0.701 |
| **B8 (this model), team-split pooling** | **85.64%** | **0.855** |
| B9 YOLO26-x full-frame classifier, scored per clip | 80.33% | 0.817 |

(The paper reports 81.9% for its hierarchical model.)

## Usage

Needs `b8_model.py` from the Space/repo. Input: `crops (1, 9, P, 3, 224, 224)` (ImageNet-normalised player crops),
`masks (1, P)`, `team_ids (1, P)` (0 = left, 1 = right). Output: logits over
`l-pass, r-pass, l-spike, r_spike, l_set, r_set, l_winpoint, r_winpoint`.
Try it in the [Space](https://huggingface.co/spaces/OmarTBakr/volleyball-activity-demo).

## Limitations

- **Requires tracked player boxes.** There is no detector or tracker in the model; on new video you must supply them,
  and accuracy with predicted (rather than ground-truth) boxes has not been measured.
- Trained and evaluated on one dataset (broadcast-style volleyball, 55 videos); no claim of generalisation.
- Per-clip prediction from 9 frames centred on the annotated frame.
