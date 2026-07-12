# Bug Report — Corrupted Group-Activity Labels (62% of the dataset)

**Date:** 2026-07-13
**Severity:** Critical — silent training-label corruption
**Status:** Fixed, regenerated, and verified

---

## TL;DR

**3,001 of 4,830 clips (62.1%) had the wrong `scene_class` label** in
`DataSet/volleyball_master.json` and in the pickle
(`volleyball_master_pickle.pkl`) that every training run loads.
The model was effectively being trained on shuffled labels, which is why no
amount of learning-rate tuning, regularization, dropout, schedulers, or
freezing strategies could make it learn: the *ceiling* on achievable accuracy
was set by the label noise, not the optimization.

The suspicion that the bug was "in the loader" was close — the loader is where
the wrong labels entered the model — but the loaders themselves were reading
their input faithfully. The corruption happened one step earlier, in the
annotation-merge step of `src/json_parser.py`.

---

## 1. Root cause

### Where

`src/json_parser.py` — Stage 2 of the dataset pipeline
(`enrich_with_scene_labels()` + `merge_dataset_levels()`).

### What went wrong

Two compounding flaws:

**Flaw A — labels flattened into a global, per-frame-name dict.**
`enrich_with_scene_labels()` parsed each video's `annotations.txt`
(which maps *middle-frame name → group activity*, e.g. `13286.jpg → l-spike`)
and merged all 55 videos into **one flat dict keyed by frame name only**:

```python
scene_labels: dict[str, str] = {}
for video_folder in ...:
    scene_labels.update(parse_scene_annotations(annot_file))   # ← collision!
```

Frame names are **not unique across videos** (they are frame numbers of the
source video). Any two videos containing a clip with the same frame number
collide, and the later video silently overwrites the earlier one's label.

**Flaw B — label looked up via *any* frame of the clip, first match wins.**
`merge_dataset_levels()` then iterated over **all ~41 frames** of each clip
and assigned the first frame name found in that global dict:

```python
for img_name in content.get("actions", {}).keys():   # ~41 frames per clip
    if img_name in scene_labels:
        clip_scene_label = scene_labels[img_name]
        break
```

A clip's 41-frame window (`clip_id − 20 … clip_id + 20`) frequently contains
the *middle frame of a neighboring clip* — and via Flaw A, the middle frame of
clips from **completely different videos**. Since the frames iterate in
ascending order, the earliest colliding frame won, which was usually **not**
the clip's own middle frame.

### Effect

Each clip received the label of whatever *other* clip's middle-frame number
happened to fall earliest inside its frame window. The result is
label assignment that is heavily shuffled but still produces a plausible-looking
class distribution — which is exactly why it was so hard to spot:

| Check | Result (before fix) |
|---|---|
| Clips with a label | 4,830 / 4,830 ✓ (no missing labels, no errors) |
| Label strings valid | all 8 classes, plausible distribution ✓ |
| **Labels actually correct** | **1,829 / 4,830 (37.9%)** ✗ |

Verified by re-parsing every video's `annotations.txt` keyed by
`(video_id, frame_name)` and comparing against the stored `scene_class`
per clip. Examples of corrupted entries:

```
clip 0/13456   true = l-spike    stored = r_set
clip 0/18706   true = r_set      stored = r-pass
clip 0/18756   true = r_spike    stored = l_set
clip 0/18931   true = l-pass     stored = r_spike
```

### Why training behaved the way it did

- With ~62% wrong labels distributed roughly uniformly, the Bayes-optimal
  classifier tops out near the noise floor — the model *cannot* fit the data
  signal because there barely is one.
- Train loss can still decrease (a big ResNet memorizes noise), while
  validation F1 stays near chance and never tracks train — the classic
  symptom pattern you observed across runs.
- Regularization / LR changes shift *where* on the memorization curve you
  land, producing small, inconsistent metric changes that look like
  hyper-parameter sensitivity. None of it could fix the labels.

### A contributing factor: the loader hid the problem

`data_loader.py` / `kaggle_data_loader.py` mapped labels with a **silent
fallback to class 0**:

```python
group_label = GROUP_ACTIVITY_TO_IDX.get(scene_class, 0) if scene_class else 0
```

Any unknown or missing label silently became `l-pass`. No exception, no
warning — nothing ever forced the corrupted annotations to surface.

---

## 2. The fix

### 2.1 `src/json_parser.py` (root cause)

- `enrich_with_scene_labels()` now keeps scene labels **keyed per video**:
  `dict[video_id][frame_name] → label`. No cross-video collisions possible.
- `merge_dataset_levels()` now looks up **exactly the clip's own middle
  frame** (`f"{clip_id}.jpg"` — the clip folder is named after its annotated
  middle frame) **within its own video**, instead of scanning all 41 frames
  against a global dict. Clips with no annotation get `None` plus a logged
  warning.
- The `__main__` block no longer unconditionally re-runs the expensive
  Stage 1 detections parse (the comment said it was opt-in, but the call was
  live); Stage 1 is now skipped when the master JSON already exists.

### 2.2 Regenerated artifacts

- `DataSet/volleyball_master.json` — Stage 2 re-run with the fixed merge.
- `DataSet/volleyball_master_pickle.pkl` — stale pickle deleted and re-dumped
  (note: `dump_to_pickle()` is a skip-if-exists singleton, so deleting the old
  file was required).

### 2.3 Loader hardening (fail loud, stay compatible)

`BaseVolleyballDataset._group_label()` now:

- **raises `KeyError`** on a `scene_class` string not present in
  `GROUP_ACTIVITY_TO_IDX` (this is what would have exposed the bug on day one
  had the labels been malformed rather than merely wrong);
- logs a **warning** (instead of silently using 0) when `scene_class` is
  `None`, telling you to re-run the enrichment.

---

## 3. Verification (after fix)

| Check | Result |
|---|---|
| JSON labels vs. per-video ground truth | **4,830 / 4,830 correct** |
| Pickle labels vs. per-video ground truth | **4,830 / 4,830 correct** |
| Split sizes (train / val / test) | 2,152 / 1,341 / 1,337 = 4,830 |
| Per-split label check through the actual `VolleyballDataset` objects | 0 mismatches in all three splits |
| Disk loader: full-image `n_frames=1` batch | `(4, 3, 224, 224)`, labels `(4,)` ✓ |
| Disk loader: full-image `n_frames=5` batch | `(2, 5, 3, 224, 224)` ✓ |
| Disk loader: crop `n_frames=1` batch | crops `(4, 12, 3, 224, 224)`, labels `(4, 12)`, masks ✓ |
| Disk loader: crop `n_frames=3` (temporal) batch | `(2, 3, 12, 3, 96, 96)` ✓ |
| LMDB loader: full-image + crop batches | ✓ |

**Expected impact:** Baseline 1 (ResNet-50 on the middle frame) should now
train normally — published results for this exact setup on this dataset reach
roughly 70–80% test accuracy, versus near-chance behavior before the fix.
Any previously saved checkpoints and logged runs were trained on corrupted
labels and should be discarded / re-run.

---

## 4. Code organization & repetitiveness (also fixed)

### 4.1 Duplicated loaders → shared base class

`src/data/data_loader.py` (LMDB) and `src/data/kaggle_data_loader.py` (disk)
duplicated ~90% of their code — dataset filtering, frame-window selection,
person cropping, `__getitem__`, and two diverging copies of `collate_fn`.
They had already drifted apart (the Kaggle copy had crop-aware middle-frame
search, boundary padding, and variable-length collate handling that the LMDB
copy lacked).

New structure:

```
src/data/base_dataset.py        ← all shared logic + the single collate_fn
    BaseVolleyballDataset       (split filter, frame selection, crops, labels)
src/data/data_loader.py        ← LMDB backend only (~60 lines of real code)
    _load_frame_index()  from LMDB key list
    _load_image()        memory-mapped LMDB read
src/data/kaggle_data_loader.py ← disk backend only (~60 lines of real code)
    _load_frame_index()  one-time directory walk (cached)
    _load_image()        PIL read from disk
```

**Backward compatibility is fully preserved:**

- `from src.data.data_loader import VolleyballDataset, collate_fn` — unchanged
- `from src.data.kaggle_data_loader import VolleyballDataset, collate_fn` — unchanged
- Constructor signatures unchanged (the Kaggle variant keeps its extra
  `dataset_dir` parameter)
- Return types / tensor shapes / collate output contracts unchanged
- `models/baseline1.py`, `models/baseline3.py`, and the notebook need **no
  changes**

Both loaders also gained the *better* of the two divergent behaviors
(crop-aware middle-frame selection, boundary frame padding to guarantee
`n_frames`, variable-length collate padding), so the LMDB path is now
strictly more robust than before.

### 4.2 Latent crash fixed in temporal crop mode

In both original loaders, `crop=True, n_frames>1` called
`torch.stack(valid_frames)` over per-frame crop tensors whose person count
`P` can differ between frames (tracking dropouts / detection fallback) —
a guaranteed `RuntimeError` on such clips. Frames are now zero-padded to the
clip-max `P` before stacking.

### 4.3 Smaller cleanups

- `kaggle_data_loader.py`'s docstring told users to import
  `src.data.data_loader_kaggle`, a module that doesn't exist — corrected to
  `src.data.kaggle_data_loader`.
- `json_parser.py` `__main__` no longer silently re-runs the multi-hour
  Stage 1 parse (see §2.1).

---

## 5. If you retrain today

1. Nothing to regenerate — JSON and pickle are already fixed in place.
   (If you ever rebuild from scratch: `python -m src.json_parser` then
   delete the old pickle and `python -m src.pickle_dump`.)
2. On Kaggle, the same fixed `json_parser.py` must be used to build the
   working-dir JSON/pickle — the corruption would otherwise be reproduced
   there. If a pre-built pickle is uploaded as a Kaggle dataset, replace it
   with the regenerated one.
3. Discard old checkpoints in `saved_models/` and treat all runs in `logs/`
   as invalid baselines — their labels were corrupted.
