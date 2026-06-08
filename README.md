# Volleyball Group Activity Recognition

A deep learning pipeline for **group activity recognition** in volleyball videos, based on the [CVPR 2016 paper](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf) by Mostafa S. Ibrahim et al.

[![Sample clip — click to play](reports/figures/output_thumbnail.jpg)](output.mp4)

> GitHub's Markdown does not auto-embed `.mp4` files. The thumbnail above is a static frame; **click it** to open the full clip in GitHub's player. For true inline playback, drag-and-drop `output.mp4` into a GitHub issue/PR to upload it to the attachment CDN, then paste the resulting `user-attachments.githubusercontent.com/...` URL on its own line in place of the thumbnail.

---

## At a Glance

- **Dataset**: 55 volleyball videos, 4,830 clips, two annotation levels — 8 group activities (scene-level) and 9 person actions (player-level).
- **Baselines**: 8 progressively complex models (B1–B8) sharing a single data loader. **B1 and B3 are complete; B4–B8 are pending.**
- **Current test scores**: B1 macro-F1 0.196 / B3 macro-F1 0.178 (preliminary — see [Results](#results); B3's number predates recent loss / pooling fixes).
- **Stack**: PyTorch + Hydra config + TensorBoard logging; multi-GPU via `nn.DataParallel`; Kaggle dual-T4 ready.
- **Paper**: Ibrahim et al., *A Hierarchical Deep Temporal Model for Group Activity Recognition*, CVPR 2016.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Baseline Models](#baseline-models)
- [Results](#results)
- [Dataset](#dataset)
- [Data Pipeline](#data-pipeline)
- [Data Loader API](#data-loader-api)
- [Project Structure](#project-structure)
- [References](#references)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset (one-time)

```bash
# Step 1: build master JSON from detections + tracking and enrich with scene labels
python -m src.json_parser

# Step 2: dump annotations to a pickle cache for fast metadata loading
python -m src.pickle_dump

# Step 3: pack raw .jpg frames into a memory-mapped LMDB database
python -m src.load_frames_into_lmdb
```

All three scripts are singletons — they skip work if the output already exists.

### 3. Verify the Loader

```bash
python -m src.data.data_loader
```

Smoke-tests the dataset by pulling a few batches in full-image and crop mode.

### 4. Train a Baseline

```bash
python -m models.baseline1   # B1: Two-stage fine-tuned ResNet-50
python -m models.baseline3   # B3: Person-then-group crop classifier
```

Hyperparameters live in `configs/baseline{1,3}.yaml` (Hydra). Per-epoch metrics are written to `logs/<baseline>/runN.json` and a TensorBoard event file under `logs/<baseline>/tensorboard/runN/`.

### 5. Evaluate a Trained Checkpoint

```bash
uv run python -m utils.evaluate --model baseline1_run1.pt           --baseline baseline1
uv run python -m utils.evaluate --model baseline3_stage_b_run2.pt   --baseline baseline3
```

Produces confusion matrix, classification report, precision–recall curves, and mAP under `plots/<baseline>/`. The Baseline 3 evaluator **auto-detects the saved pool mode** (`max` vs `concat`) from the checkpoint's classifier shape, so legacy and current checkpoints both load without YAML edits.

---

## Baseline Models

| Baseline | Status | Input | Temporal | Player-Level | Scene-Level |
|----------|--------|-------|----------|--------------|-------------|
| **B1** | ✅ Done | Middle frame (full) | ✗ | ✗ | Image classifier (8 classes) |
| **B3** | ✅ Done | Middle frame (crops) | ✗ | Crop classifier (9 classes) | Max+mean concat pool over players → MLP (8 classes) |
| **B4** | 🔲 Pending | 9 frames (full) | LSTM on frame features | ✗ | LSTM → 8 classes |
| **B5** | 🔲 Pending | 9 frames (crops) | LSTM per player | Max-pool players | NN (8 classes) |
| **B6** | 🔲 Pending | 9 frames (crops) | LSTM on pooled frames | Max-pool per frame | LSTM → 8 classes |
| **B7** | 🔲 Pending | 9 frames (crops) | LSTM₁ per player + LSTM₂ | Max-pool per frame | LSTM₂ → 8 classes |
| **B8** | 🔲 Pending | 9 frames (crops) | LSTM₁ per player + LSTM₂ | Team-split pool (6+6) | Concat teams → LSTM₂ |

---

## Results

### Baseline 1 — Single-Frame Image Classifier

Training uses a two-stage strategy: a linear probe (head-only) followed by full fine-tuning with differential learning rates and cosine annealing.

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 (pretrained) |
| Stage 1 (linear probe) | 5 epochs, lr = 1e-3 |
| Stage 2 (full fine-tune) | 50 epochs, backbone lr = 1e-4, head lr = 3 × 1e-4 |
| LR Scheduler | CosineAnnealingLR |
| Label Smoothing | 0.1 |
| Weight Decay | 0.05 |
| Early Stopping Patience | 10 epochs |

#### Test Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 20.8% |
| Macro F1 | .196 |
| Loss | 2.4 |

| Confusion Matrix | Classification Report |
|:---:|:---:|
| ![Confusion Matrix](plots/baseline1/Confusion%20Matrix.png) | ![Classification Report](plots/baseline1/Classification%20Report.png) |
| **Precision-Recall Curves** | **mAP & F1 per Class** |
| ![Precision-Recall Curves](plots/baseline1/Precision-Recall%20Curves.png) | ![mAP & F1](plots/baseline1/mAP%20%26%20F1%20Score%20per%20Class.png) |

---

### Baseline 3 — Per-Player Crops → Frozen Backbone → Concat-Pool → MLP

A two-stage architecture. **Stage A** trains a ResNet-50 end-to-end on individual player crops to classify the 9 person actions (`blocking`, `digging`, …, `waiting`). **Stage B** freezes that backbone (with `fc = Identity`), passes the per-player crops of a clip through it to produce one `(P, 2048)` feature matrix per clip, applies concatenated max- and mean-pool across the player dimension to get a `(2 × 2048)`-wide vector, and trains a small MLP head to predict the 8 group activities.

The concat pool gives the head two complementary signals: max captures *"is any player exhibiting feature k strongly?"* and mean captures *"what's the typical team level of feature k?"*. Class-weighted CrossEntropy is used in both stages to counter the heavy `standing` skew in Stage A (~70% of all crops) and the rare `l/r_winpoint` classes (~2.5× rarer than `spike/pass/set`) in Stage B.

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 (pretrained) — `cfg.model.name` switchable to `resnet101` |
| Stage A (person-action) | up to 100 epochs, lr = 1e-3, full backbone, class-weighted CE |
| Stage B (group-activity head) | up to 100 epochs, lr = 1e-3, frozen backbone, MLP head |
| Stage B pool | `concat` (max + mean), classifier in = 2 × 2048 |
| MLP head | `Linear(4096, 512) → ReLU → Dropout(0.4) → Linear(512, 8)` |
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 5e-4 |
| LR Scheduler (Stage B) | CosineAnnealingLR |
| Label Smoothing | 0.01 |
| Class-weighted loss | inverse-frequency (`w_k = N / (K · n_k)`), both stages |
| Multi-GPU | `nn.DataParallel` when `n_gpus > 1` (Kaggle dual-T4 ready) |
| Early Stopping Patience | Stage A 25, Stage B 25 |

#### Test Metrics (run2 — preliminary, pre-class-weighted-loss baseline)

| Metric | Value |
|--------|-------|
| Accuracy | 21.6% |
| Macro F1 | .178 |
| Loss | 2.00 |

The `run2` numbers above were produced with the legacy max-only pool and no class weighting, before the architectural fixes for per-video memorization and the standing-majority bias landed. The current code (concat pool + class-weighted CE in both stages) has not been retrained yet; expect a meaningful gain on rare-class recall once it is.

| Confusion Matrix | Classification Report |
|:---:|:---:|
| ![Confusion Matrix](plots/baseline3/Confusion%20Matrix.png) | ![Classification Report](plots/baseline3/Classification%20Report.png) |
| **Precision-Recall Curves** | **mAP & F1 per Class** |
| ![Precision-Recall Curves](plots/baseline3/Precision-Recall%20Curves.png) | ![mAP & F1](plots/baseline3/mAP%20%26%20F1%20Score%20per%20Class.png) |

---

## Dataset

### Class Labels

**8 Group Activities (scene-level)**

| Index | Activity | Index | Activity |
|-------|----------|-------|----------|
| 0 | `l-pass` | 4 | `l_set` |
| 1 | `r-pass` | 5 | `r_set` |
| 2 | `l-spike` | 6 | `l_winpoint` |
| 3 | `r_spike` | 7 | `r_winpoint` |

**9 Person Actions (player-level)**

| Index | Action | Index | Action |
|-------|--------|-------|--------|
| 0 | `blocking` | 5 | `setting` |
| 1 | `digging` | 6 | `spiking` |
| 2 | `falling` | 7 | `standing` |
| 3 | `jumping` | 8 | `waiting` |
| 4 | `moving` | | |

### Splits

| Split | Videos | Clips |
|-------|--------|-------|
| **Train** | 24 | 2,152 |
| **Validation** | 15 | 1,341 |
| **Test** | 16 | 1,337 |
| **Total** | 55 | 4,830 |

Split definitions live in `configs/data_split.py`.

### Raw Directory Layout

```
DataSet/
├── volleyball_/videos/                    # Raw video frames + annotations
│   ├── 0/                                 # Video 0
│   │   ├── annotations.txt                # Group activity + person boxes per clip
│   │   ├── 3596/                          # Clip (middle frame = 3596)
│   │   │   ├── 3576.jpg                   # 41 frames per clip
│   │   │   ├── 3577.jpg
│   │   │   ├── ...
│   │   │   └── 3616.jpg
│   │   └── 13286/
│   │       └── ...
│   ├── 1/
│   └── ... (55 videos total, ~4,830 clips)
│
├── volleyball-detections/                 # Pre-computed detections
│   └── {video_id}/{clip_id}/
│       ├── action_detections.txt          # Tab-separated: frame  N  [x y w h score label] × N
│       └── person_detections.txt          # Tab-separated: frame  N  [x y w h score label] × N
│
├── volleyball_tracking_annotation/        # Player tracking with IDs
│   └── {video_id}/{clip_id}/
│       └── {clip_id}.txt                  # Space-separated: id x1 y1 x2 y2 frame f1 f2 f3 action
│
├── volleyball_master.json                 # Stage 1+2 unified output
└── volleyball_master_pickle.pkl           # Fast-load cache
```

---

## Data Pipeline

The raw dataset contains three separate annotation sources. The pipeline unifies them in two parsing stages, then caches in two fast-loading formats. End-to-end flow:

```mermaid
graph TB
    subgraph "Raw Dataset (60GB)"
        A[volleyball-detections/] -->|action_detections.txt<br>person_detections.txt| P
        B[volleyball_tracking_annotation/] -->|clip_id.txt| P
        C[volleyball_/videos/] -->|annotations.txt| E
        C -->|.jpg frames| DL
    end

    subgraph "Two-Stage Parsing Pipeline"
        P["Stage 1: json_parser.py<br>create_master_json()"] -->|volleyball_master.json| D
        D[Master JSON] --> E["Stage 2: json_parser.py<br>enrich_with_scene_labels()"]
        E -->|enriched JSON| F["pickle_dump.py<br>dump_to_pickle()"]
        F -->|volleyball_master_pickle.pkl| G[Fast Pickle Cache]
    end

    subgraph "PyTorch Data Loading"
        G -->|load_from_pickle| DL["data_loader.py<br>VolleyballDataset"]
        DL --> H{Mode?}
        H -->|full_image=True| I["Full Frames<br>(B1, B4)"]
        H -->|crop=True| J["Person Crops<br>(B3, B5-B8)"]
    end

    subgraph "Model Training"
        I --> M[Baseline Models]
        J --> M
        M --> R[Results]
    end
```

### Stage 1 — Player-Level Parsing

```mermaid
flowchart LR
    subgraph "Per Clip (e.g. 0/3596)"
        AD[action_detections.txt] -->|parse_detection_file| MJ
        PD[person_detections.txt] -->|parse_detection_file| MJ
        TK[3596.txt] -->|parse_tracking_file| MJ
    end
    MJ["Master JSON Entry<br>{actions, persons, tracking}"]
```

`create_master_json()` iterates over all 55 videos × ~90 clips each, parsing:

| Source File | Parser | Output per Frame |
|---|---|---|
| `action_detections.txt` | `parse_detection_file()` | `{box: [x,y,w,h], score, label}` |
| `person_detections.txt` | `parse_detection_file()` | `{box: [x,y,w,h], score, label}` |
| `clip_id.txt` (tracking) | `parse_tracking_file()` | `{id, box: [x1,y1,x2,y2], flags, action}` |

### Stage 2 — Scene-Level Enrichment

```mermaid
flowchart LR
    AN[annotations.txt<br>per video] -->|parse_scene_annotations| SL["scene_labels<br>{frame.jpg → group_activity}"]
    SL -->|merge_dataset_levels| MJ["Master JSON<br>+ scene_class per clip"]
```

`enrich_with_scene_labels()` reads each video's `annotations.txt` to extract the **group-activity label** (one of 8 scene classes) and attaches it to each clip as `"scene_class"`.

### Pickle & LMDB Caching

To avoid severe I/O bottlenecks and RAM exhaustion during training, the dataset is cached in two high-performance formats:

1. **Annotations (Pickle)**: The enriched JSON (~1.6 GB) is dumped to pickle (`volleyball_master_pickle.pkl`, ~247 MB) for instantaneous metadata loading.
2. **Frames (LMDB)**: The raw `.jpg` frames (~50 GB) are packed into a memory-mapped LMDB database (`frames_lmdb`) to allow lightning-fast lazy loading of image bytes on the fly.

Both build scripts are **singletons** — they skip execution if the database already exists.

```mermaid
flowchart LR
    JSON["volleyball_master.json<br>(1.6 GB)"] -->|src.pickle_dump| PKL["volleyball_master_pickle.pkl<br>(247 MB)"]
    RAW["Raw .jpg frames<br>(~50 GB)"] -->|src.load_frames_into_lmdb| LMDB["frames_lmdb<br>(Memory-mapped)"]

    PKL --> DS[VolleyballDataset]
    LMDB --> DS
```

---

## Data Loader API

`VolleyballDataset` is a **generic** PyTorch `Dataset` that loads from the pickle cache and supports all baselines through constructor flags:

```python
from src.data.data_loader import VolleyballDataset, collate_fn

# B1: Full image, middle frame only → (image, group_label)
ds = VolleyballDataset(mode="train", n_frames=1, full_image=True, transform=transform)

# B3: Cropped persons, middle frame → (crops [P,C,H,W], person_labels [P], group_label)
ds = VolleyballDataset(mode="train", n_frames=1, crop=True, transform=transform)

# B4: Full image, 9-frame sequence → (images [9,C,H,W], group_label)
ds = VolleyballDataset(mode="train", n_frames=9, full_image=True, transform=transform)

# B5-B8: Cropped persons, 9-frame sequence → (crops [9,P,C,H,W], person_labels [P], group_label)
ds = VolleyballDataset(mode="train", n_frames=9, crop=True, transform=transform)
```

### Return Shapes by Configuration

```mermaid
graph TD
    DS[VolleyballDataset] --> FI{full_image?}
    DS --> CR{crop?}

    FI -->|n_frames=1| F1["(image, group_label)<br>B1"]
    FI -->|n_frames=9| F9["(images [T,C,H,W], group_label)<br>B4"]

    CR -->|n_frames=1| C1["(crops [P,C,H,W], person_labels [P], group_label)<br>B3"]
    CR -->|n_frames=9| C9["(crops [T,P,C,H,W], person_labels [P], group_label)<br>B5-B8"]
```

### Collate Function

`collate_fn` handles **variable player counts** across clips by padding the player dimension to the batch maximum and returning a boolean mask:

```python
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
# Crop mode returns: (crops_batch, person_labels_batch, group_labels_batch, masks_batch)
```

### Per-Baseline Batch Unpackers

Each baseline that uses crop mode pairs the loader with a tiny `batch_unpack` callable that surfaces the right `(model_inputs, target)` from the 4-tuple. These slot into the shared training/eval driver in `utils.utility` (see `train_one_epoch`'s `batch_unpack` kwarg) and into `utils.evaluate`. Adding a new baseline = one model class + one unpacker; no changes to the loader or the training loop.

---

## Project Structure

```
Project1/
├── configs/
│   ├── __init__.py              # Package exports
│   ├── path_config.py           # All dataset/output paths
│   ├── data_split.py            # Train/val/test video IDs
│   ├── labels.py                # Label-to-index mappings (8 group + 9 person)
│   ├── baseline1.yaml           # Hydra config for B1
│   ├── baseline3.yaml           # Hydra config for B3
│   └── transforms/
│       └── default_transforms.yaml
│
├── src/
│   ├── json_parser.py           # Two-stage parsing pipeline
│   ├── pickle_dump.py           # Singleton pickle dump/load
│   ├── load_frames_into_lmdb.py # Pack frames into LMDB
│   ├── load_frames_into_pickle.py
│   └── data/
│       ├── data_loader.py       # Original data loader
│       ├── kaggle_data_loader.py# Kaggle-compatible data loader
│       ├── data_summary.py      # Statistics and class distributions
│       └── visualize_data.py    # Dataset visualization
│
├── models/
│   ├── baseline1.py             # B1: Two-stage fine-tuned ResNet50 (✅ done)
│   └── baseline3.py             # B3: Person-then-group crop classifier (✅ done)
│
├── utils/
│   ├── utility.py               # Training/eval loop helpers + class-weight tools
│   ├── evaluate.py              # Post-training evaluation + plots
│   ├── plotting.py              # Confusion matrix, PR curves, mAP
│   └── load_model_config.py     # Hydra config → transforms/scheduler builders
│
├── reports/
│   ├── report.tex               # LaTeX report
│   └── figures/                 # Report figures (incl. demo thumbnail)
│
├── DataSet/                     # Raw data (not tracked in git)
├── saved_models/                # Model checkpoints (.pt)
├── runs/                        # Hydra run outputs
├── logs/                        # Per-baseline TensorBoard + JSON metric logs
└── plots/                       # Evaluation visualizations
    ├── baseline1/               # B1 evaluation plots
    │   ├── Confusion Matrix.png
    │   ├── Classification Report.png
    │   ├── Precision-Recall Curves.png
    │   └── mAP & F1 Score per Class.png
    └── baseline3/               # B3 evaluation plots (same four)
```

---

## References

- Ibrahim, M. S., Muralidharan, S., Deng, Z., Vahdat, A., & Mori, G. (2016). *A Hierarchical Deep Temporal Model for Group Activity Recognition*. **CVPR 2016**. [PDF](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf).
- Volleyball dataset repository: [github.com/mostafa-saad/deep-activity-rec](https://github.com/mostafa-saad/deep-activity-rec).
