---
title: Volleyball Group Activity Recognition
emoji: 🏐
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: mit
---

# Volleyball Group Activity Recognition (Baseline 8)

Demo of a hierarchical temporal model (player LSTM → team-split pooling → scene LSTM) that recognises
8 volleyball group activities from 9 frames and tracked player boxes. Weights live in
[`OmarTBakr/volleyball-activity-b8`](https://huggingface.co/OmarTBakr/volleyball-activity-b8);
code and full write-up in the project repository.

The 20 bundled clips come from the dataset's **validation** split.
