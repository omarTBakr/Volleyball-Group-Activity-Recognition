"""
Baseline 9 — YOLO classification backbone on full volleyball frames.

B9 feeds the whole frame to an Ultralytics YOLO *classification* model and
predicts the clip's group activity — the same task and the same 8 classes as
B1, with a YOLO backbone in place of the torchvision ResNet.  No boxes are
involved: the detector head is not used, and nothing here reads the player
annotations.

Single stage: there is no probe/finetune split and no Stage A/Stage B like the
hierarchical baselines: one ``model.train()`` call fine-tunes the whole
network on the 8 group activities.

Ultralytics classification takes a directory, not a ``data.yaml``, and treats
it as an ImageFolder (``<root>/<split>/<class>/*.jpg``), so it cannot consume
``src.data.data_loader.VolleyballDataset``.  ``utils.yolo_export`` writes that
layout from the same annotations, honouring the project's video split:

    python -m utils.yolo_export

The model scale and batch size are measured, not guessed.  ``utils.yolo_probe``
walks the scale ladder (extra-large → large → medium → small → nano) to find
the largest architecture this card can train, then ladders the batch size for
that scale.  Run it once, with the task it will actually train:

    python -m utils.yolo_probe --task classify --imgsz 224 --nc 8

then paste its two numbers into ``MODEL`` and ``BATCH`` below.  If you leave
them empty, this script reads ``logs/yolo_probe.json`` (written by the probe)
and uses the values from the last probe run, so a plain

    python models/baseline9.py

works right after probing.  Anything you type into the constants wins over the
report — the report is only the fallback.

Two Ultralytics defaults are overridden below because this dataset breaks
them, both for the reason ``configs/transforms/default_transforms.yaml``
already documents: the 8 activities come in left/right pairs, so a horizontal
flip relabels the image, and a center crop discards the court width that
distinguishes the pair.

Logging matches the rest of the project.  Ultralytics owns the epoch loop, so
``utils.trainer.Trainer`` cannot run this baseline — but it is still the log
sink: :meth:`Trainer.log_epoch` and :meth:`Trainer.record_test` are called
from Ultralytics' own callbacks, so B9's scalars, its
``<LOG_ROOT>/baseline9/<run_id>.json`` history and its hparams card come out
in the same shape as B1–B8 and sit on the same ``tensorboard --logdir``.
Per-epoch accuracy and macro-F1 come from ``sklearn`` over the validator's raw
predictions — the same functions ``utils.utility`` scores the other baselines
with, so the numbers are comparable rather than merely similar.

Weights land under ``PROJECT/RUN_NAME`` in the usual Ultralytics layout
(``weights/best.pt``, ``weights/last.pt``, curves, confusion matrix), which
:data:`OUT_ROOT` redirects off the repo's external drive — see the comment
there.
"""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms import RandomErasing
from ultralytics import YOLO  # ty:ignore[import]  # ty:ignore[unresolved-import]

from configs.labels import NUM_GROUP_ACTIVITIES
from configs.path_config import LOGS_DIR, MODEL_SAVE_DIR
from utils.trainer import Trainer
from utils.yolo_export import DEFAULT_OUT_DIR
from utils.yolo_probe import load_probe_report

# ═════════════════════════════════════════════════════════════════════════════
# ══ 1. CONFIG ══
# ═════════════════════════════════════════════════════════════════════════════

# ── Fill MODEL / BATCH from the probe, or leave empty to inherit ────────────

MODEL: str = "yolo26x-cls.pt"    # e.g. "yolo26m-cls.pt" — from `python -m utils.yolo_probe --task classify`
BATCH: int | None = 64           # e.g. 64               — from the same probe run

#: Root of the ImageFolder tree written by ``utils.yolo_export``.
DATA_ROOT: Path = DEFAULT_OUT_DIR

#: Keep this equal to the exporter's ``--resize``: the frames are already
#: square, so Ultralytics' Resize + CenterCrop becomes a no-op and the full
#: court survives. Raising it means re-exporting at the same size.
IMG_SIZE = 224

EPOCHS = 100
PATIENCE = 25            # early stopping, in epochs without val improvement
WORKERS = 8
DEVICE = 0
NUM_CLASSES = NUM_GROUP_ACTIVITIES  # 8 group activities

#: Random-erasing probability — Ultralytics' own knob, passed to model.train().
#: Ultralytics defaults to 0.4; a frame whose label rests on ONE acting player
#: should not be punched out that often, so only a fifth of them are.
ERASE_P = 0.015

#: Erased area, as a fraction of the frame. Ultralytics exposes the probability
#: above but NOT this, and torchvision's default is (0.02, 0.33) — up to a
#: third of the court blanked, which routinely deleted the acting player and
#: left the sample carrying a label nothing in the image supports.
#:
#: The scale that matters is the player: a box of ~100x200 px in a 1280x720
#: frame warps to ~17x62 px in the exported 224 square, i.e. ~2% of the area.
#: This range tops out at exactly that — the hole can hide at most one player,
#: usually half of one — because the 8 classes are defined by what a SINGLE
#: player is doing, so anything larger risks erasing the evidence and leaving
#: the frame labelled for an action it no longer shows. Occlusion noise this
#: mild is a light regularizer, not the main defence against overfitting;
#: that job belongs to PATIENCE, the RandomResizedCrop above, and weight decay.
#:
#: Patched onto the transform by make_erasing_shrinker() below.
ERASE_SCALE = (0.005, 0.015)

RUN_NAME = "baseline9_yolo_cls"
BASELINE = "baseline9"

# ── Output location: redirected off the external drive ─────────────────────
#
# The repo lives on an external NTFS disk (ntfs-3g over USB) that started
# returning EIO mid-training: Ultralytics writes a ~227 MB checkpoint on every
# improving epoch, one of those writes left a ``best.pt`` that can be neither
# read nor unlinked, and the kernel log shows real media errors
# (``Buffer I/O error on dev sda1, logical block 199886507``). A 100-epoch run
# should not depend on that disk, so every output this script produces goes to
# the internal SSD instead. Move the results back beside the repo by hand once
# the drive has been through ``chkdsk``.
#
# Set OUT_ROOT to None to restore the original, repo-local layout:
#     PROJECT  = MODEL_SAVE_DIR / "runs"     →  <repo>/saved_models/runs/
#     LOG_ROOT = LOGS_DIR                    →  <repo>/logs/
# (Do that on Kaggle too, where those paths already point at the working dir.)

#: SSD mirror of the repo's output tree, or ``None`` to write beside the repo.
OUT_ROOT: Path | None = Path.home() / "volleyball_out"

#: Ultralytics writes <PROJECT>/<RUN_NAME>/weights/best.pt
PROJECT: Path = (OUT_ROOT / "saved_models" / "runs") if OUT_ROOT else MODEL_SAVE_DIR / "runs"

#: Root for this baseline's TensorBoard scalars and per-epoch JSON.
LOG_ROOT: Path = (OUT_ROOT / "logs") if OUT_ROOT else LOGS_DIR


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. CONFIG RESOLUTION ══
# ═════════════════════════════════════════════════════════════════════════════


def resolve_model_and_batch() -> tuple[str, int]:
    """
    Return the ``(model, batch)`` this run should use.

    The constants above take precedence.  Whatever is left empty is filled
    from the last ``utils.yolo_probe`` report so the two scripts chain
    without copy-paste.

    Returns
    -------
    tuple
        Model weights spec (e.g. ``"yolo26m-cls.pt"``) and batch size.

    Raises
    ------
    SystemExit
        If a value is neither set here nor available from a probe report.

    """
    model, batch = MODEL.strip(), BATCH
    report = load_probe_report() if (not model or batch is None) else None

    if report and report.get("task") != "classify":
        # A detection probe measured a different architecture under a much
        # heavier loss; both its model name and its batch size are wrong here.
        print(
            f"⚠  ignoring the probe report — it measured task="
            f"{report.get('task')!r}, not 'classify'."
        )
        report = None

    if report:
        source = f"probe report ({report['gpu']}, imgsz={report['imgsz']})"
        if not model and report.get("model"):
            model = report["model"]
            print(f"MODEL not set — using {model} from {source}")
        if batch is None and report.get("batch"):
            batch = report["batch"]
            print(f"BATCH not set — using {batch} from {source}")
        if report.get("imgsz") not in (None, IMG_SIZE):
            print(
                f"⚠  probe measured imgsz={report['imgsz']} but this run trains at "
                f"{IMG_SIZE}; re-probe if you hit OOM."
            )

    if not model or batch is None:
        raise SystemExit(
            "MODEL and BATCH are unset and no usable probe report was found.\n"
            "Run the capacity probe first:\n"
            f"    python -m utils.yolo_probe --task classify --imgsz {IMG_SIZE} "
            f"--nc {NUM_CLASSES}\n"
            "then set MODEL / BATCH at the top of this file."
        )
    return model, batch


# ═════════════════════════════════════════════════════════════════════════════
# ══ 3. ULTRALYTICS → TRAINER CALLBACKS ══
# ═════════════════════════════════════════════════════════════════════════════
#
# Ultralytics drives the loop and calls these; they forward the numbers into
# the project's ``Trainer`` log sink. ``on_val_end`` fires before
# ``on_fit_epoch_end`` on every validation pass, so the scores below are always
# the ones belonging to the epoch being logged — and the last pass of all is
# the final evaluation, which is where ``train_test`` reads them from.


def make_val_scorer(scores: dict) -> callable:
    """
    Build the ``on_val_end`` callback that scores a validation pass.

    Ultralytics reports top-1/top-5 only; the rest of this project selects and
    compares on macro-F1, so both are recomputed with the same ``sklearn``
    calls :func:`utils.utility._run_one_epoch` uses, over the validator's raw
    top-1 predictions.

    Parameters
    ----------
    scores : dict
        Mutable holder the callback overwrites with ``{"acc", "f1"}`` — read
        by :func:`make_epoch_logger` and, after the final pass, by
        :func:`train_test`.

    """
    def on_val_end(validator) -> None:
        # validator.pred rows are the top-5 class indices, best first.
        y_pred = torch.cat(validator.pred)[:, 0].cpu().numpy()
        y_true = torch.cat(validator.targets).cpu().numpy()
        scores["acc"] = float(accuracy_score(y_true, y_pred))
        scores["f1"] = float(f1_score(y_true, y_pred, average="macro"))

    return on_val_end


def make_epoch_logger(logger: Trainer, scores: dict) -> callable:
    """
    Build the ``on_fit_epoch_end`` callback that logs one epoch.

    Parameters
    ----------
    logger : Trainer
        The log sink — only its :meth:`~utils.trainer.Trainer.log_epoch` is
        used, since Ultralytics owns the training loop.
    scores : dict
        The holder :func:`make_val_scorer` fills for this same epoch.

    """
    def on_fit_epoch_end(trainer) -> None:
        logger.global_epoch = trainer.epoch + 1
        record = {
            "epoch": logger.global_epoch,
            "stage": "",
            # {"train/loss": x} — one entry for classification.
            "train_loss": next(iter(trainer.label_loss_items(trainer.tloss).values())),
            # val/loss exists only while training: a standalone val() has no loss.
            "val_loss": trainer.metrics.get("val/loss"),
            "val_acc": scores.get("acc"),
            "val_top5": trainer.metrics.get("metrics/accuracy_top5"),
            "val_f1": scores.get("f1"),
            "learning_rate": next(iter(trainer.lr.values()), None),
        }
        logger.log_epoch({k: v for k, v in record.items() if v is not None})

        # Ultralytics' own epoch table reports top-1/top-5 only; macro-F1 is
        # what this project selects and compares baselines on, so echo the
        # epoch in the same "Val ->" format Trainer.run_stage prints for B1–B8.
        shown = (("Loss", record["val_loss"]), ("Acc", record["val_acc"]),
                 ("Top-5", record["val_top5"]), ("F1", record["val_f1"]))
        print("Val   -> " + ", ".join(
            f"{name}: {value:.4f}" for name, value in shown if value is not None))

    return on_fit_epoch_end


def make_erasing_shrinker(scale: tuple[float, float]) -> callable:
    """
    Build the ``on_train_start`` callback that shrinks RandomErasing.

    ``model.train(erasing=...)`` sets only how *often* a frame is punched out;
    how *much* comes from torchvision's ``RandomErasing(scale=(0.02, 0.33))``,
    which Ultralytics builds internally and never exposes. This reaches into
    the dataset's transform pipeline and narrows that range to *scale* — see
    :data:`ERASE_SCALE` for why a third of the frame is not a regularizer here.

    Runs at ``on_train_start``: the loader exists by then, and no batch has
    been pulled yet, so the change is in place before the dataloader workers
    fork.

    Parameters
    ----------
    scale : tuple
        ``(min, max)`` erased area as a fraction of the frame.

    """
    def on_train_start(trainer) -> None:
        transforms = getattr(trainer.train_loader.dataset, "torch_transforms", None)
        # Located by type rather than by index: if Ultralytics ever reorders
        # its pipeline, this must degrade to the warning below, never silently
        # rewrite the scale of some other transform.
        erasing = next(
            (t for t in getattr(transforms, "transforms", []) if isinstance(t, RandomErasing)),
            None,
        )
        if erasing is None:
            print("⚠  no RandomErasing in the train transforms — scale left untouched.")
            return
        print(f"  erasing: scale {tuple(erasing.scale)} → {tuple(scale)} (p={erasing.p})")
        erasing.scale = scale

    return on_train_start


# ═════════════════════════════════════════════════════════════════════════════
# ══ 4. MAIN ENTRYPOINT ══
# ═════════════════════════════════════════════════════════════════════════════


def train_test() -> None:
    """Fine-tune the probed YOLO classifier on full frames, then evaluate it."""
    if not (DATA_ROOT / "train").is_dir():
        raise SystemExit(
            f"Classification dataset not found under {DATA_ROOT}\n"
            "Export the frames first:\n"
            "    python -m utils.yolo_export"
        )

    model_spec, batch = resolve_model_and_batch()

    # ── Logging ──────────────────────────────────────────────────────────
    run_id = Trainer.next_run_id(BASELINE, LOG_ROOT)
    logger = Trainer(BASELINE, run_id, log_root=LOG_ROOT)
    scores: dict[str, float] = {}

    print(f"\n{'='*60}")
    print(f"  BASELINE 9: fine-tuning {model_spec} ({EPOCHS} epochs, single stage)")
    print(f"  Data   : {DATA_ROOT}")
    print(f"  Imgsz  : {IMG_SIZE}   batch: {batch}   classes: {NUM_CLASSES}")
    print(f"  Weights: {PROJECT / RUN_NAME}")
    print(f"  Logs   : {logger.log_dir / 'tensorboard' / run_id}")
    print(f"{'='*60}")

    model = YOLO(model_spec)
    model.add_callback("on_val_end", make_val_scorer(scores))
    model.add_callback("on_fit_epoch_end", make_epoch_logger(logger, scores))
    model.add_callback("on_train_start", make_erasing_shrinker(ERASE_SCALE))

    model.train(
        data=str(DATA_ROOT),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=batch,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        # A mirrored frame is a DIFFERENT class here (l_set ↔ r_set), so the
        # 0.5 default would mislabel half the training set.
        fliplr=0.0,
        flipud=0.0,
        # RandomResizedCrop range = (1 - scale, 1.0). The default 0.5 lets a
        # crop keep half the frame, which can cut off the acting side of the
        # court; 0.2 keeps the zoom mild enough to stay class-preserving.
        scale=0.2,
        # How often a frame gets a hole punched in it; how big that hole is
        # comes from ERASE_SCALE via make_erasing_shrinker().
        erasing=ERASE_P,
    )

    # ── Test ─────────────────────────────────────────────────────────────
    # The other baselines report the held-out test split, so B9 does too —
    # falling back to val only if the export predates the test split.
    split = "test" if (DATA_ROOT / "test").is_dir() else "val"
    print(f"\n--- {BASELINE} — evaluating best model on {split} ---")
    metrics = model.val(split=split)

    best_val_f1 = max(
        (epoch["val_f1"] for epoch in logger.metrics_history if "val_f1" in epoch),
        default=scores["f1"],
    )
    logger.record_test(
        # A standalone val() reports no loss — only accuracy metrics — so the
        # summary card carries nan here rather than an invented number.
        test_loss=float("nan"),
        test_acc=scores["acc"],
        test_f1=scores["f1"],
        hparam_dict={
            "baseline": BASELINE,
            "model": model_spec,
            "batch": batch,
            "imgsz": IMG_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "fliplr": 0.0,
            "scale": 0.2,
            "erasing": ERASE_P,
            "erase_scale": str(ERASE_SCALE),
            "data": str(DATA_ROOT),
            "eval_split": split,
        },
        best_val_f1=best_val_f1,
        extra={"split": split, "test_top5": float(metrics.top5)},
    )
    logger.close()

    print(f"Final {split.title()} -> Acc: {scores['acc']:.4f}, "
          f"Top-5: {metrics.top5:.4f}, F1: {scores['f1']:.4f}")
    print(f"  best : {PROJECT / RUN_NAME / 'weights' / 'best.pt'}")
    print(f"  logs : {logger.json_path}")


if __name__ == "__main__":
    train_test()
