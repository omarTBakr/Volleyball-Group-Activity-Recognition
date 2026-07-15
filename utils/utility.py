"""
Training, validation, and testing loops plus model checkpoint I/O.

This module provides the core epoch-level training primitives used by
all baseline scripts.  Visualization helpers have been moved to
``utils.plotting``.
"""

from __future__ import annotations

import numpy as np
import torch     # ty:ignore[import]  # ty:ignore[unresolved-import]
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader  # ty:ignore[import]
from torch.utils.tensorboard import SummaryWriter  # ty:ignore[import]
from tqdm import tqdm  # ty:ignore[import]

from configs.path_config import MODEL_SAVE_DIR


# ── Device Selection ────────────────────────────────────────────────────────


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Return a verified compute device, falling back to CPU if needed.

    ``torch.cuda.is_available()`` can return ``True`` even when the
    installed PyTorch build lacks kernel support for the physical GPU
    (e.g. Tesla P100 with torch ≥ 2.9).  This helper runs a tiny
    tensor operation on the GPU to confirm it truly works.

    Parameters
    ----------
    preferred : str, optional
        Desired device string (``"cuda"`` or ``"cpu"``).  Defaults to
        ``"cuda"``.

    Returns
    -------
    torch.device
        A device that is guaranteed to execute tensor operations.

    """
    if preferred == "cuda" and torch.cuda.is_available():
        try:
            # Smoke-test: run a trivial operation on the GPU
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            print(
                "⚠  CUDA is available but the current GPU is not "
                "compatible with this PyTorch build. Falling back to CPU."
            )
    return torch.device("cpu")

# ── Model Checkpoint I/O ────────────────────────────────────────────────────


def save_model(
    model_name: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: float,
    class_to_idx: dict[str, int] | None = None,
) -> None:
    """
    Save a training checkpoint to disk.

    Parameters
    ----------
    model_name : str
        Filename (without extension) for the checkpoint.
    epoch : int
        Current epoch number.
    model : nn.Module
        The model whose weights to save.
    optimizer : optim.Optimizer
        The optimizer whose state to save.
    loss : float
        The loss value at this checkpoint.
    class_to_idx : dict or None
        Optional label mapping to persist alongside the weights.

    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "class_to_idx": class_to_idx,
    }
    save_path = MODEL_SAVE_DIR / model_name
    torch.save(checkpoint, save_path)


def load_model(
    model_name: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> tuple[nn.Module, optim.Optimizer | None, int, float, dict[str, int] | None]:
    """
    Load a training checkpoint from disk.

    Parameters
    ----------
    model_name : str
        Filename used when saving.
    model : nn.Module
        An *unloaded* model instance with the correct architecture.
    optimizer : optim.Optimizer or None
        If provided, its state will be restored from the checkpoint.

    Returns
    -------
    tuple
        ``(model, optimizer, epoch, loss, class_to_idx)``

    """
    checkpoint = torch.load(MODEL_SAVE_DIR / model_name, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)
    class_to_idx = checkpoint.get("class_to_idx", None)

    return model, optimizer, epoch, loss, class_to_idx


# ── Generic Epoch Driver ─────────────────────────────────────────────────────
#
# A single implementation behind train_one_epoch / validate_one_epoch /
# test_one_epoch. The legacy 2-tuple `(data, target)` → `model(data)` contract
# is preserved by the default `batch_unpack`. Baselines whose data loaders
# return more than two items per batch (e.g. baseline3's crop-mode 4-tuple of
# ``(crops, person_labels, group_labels, masks)``) or whose models take more
# than one positional input (e.g. baseline3's GroupActivityModel takes both
# crops and masks) plug in a custom unpacker without modifying the shared loop.

from collections.abc import Callable
from typing import Any

# A batch-unpacker takes a raw batch off the dataloader and returns either
# `None` (skip this batch) or a 2-tuple `(model_inputs, target)` where
# `model_inputs` is itself a tuple of tensors expanded into ``model(*inputs)``.
BatchUnpack = Callable[[Any], tuple[tuple[torch.Tensor, ...], torch.Tensor] | None]


def _default_batch_unpack(batch: Any) -> tuple[tuple[torch.Tensor, ...], torch.Tensor] | None:
    """Legacy contract: 2-tuple ``(data, target)`` → single-input model call."""
    if batch is None or len(batch) < 2:
        return None
    data, target = batch[0], batch[1]
    return (data,), target


def _run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    *,
    batch_unpack: BatchUnpack | None = None,
    num_classes: int | None = None,
    accumulate_grad_batches: int = 1,
    desc: str = "Epoch",
) -> tuple[float, float, float, np.ndarray]:
    """
    Run one pass over *dataloader* — training when *optimizer* is provided,
    evaluation when it is None.

    Parameters
    ----------
    model, dataloader, criterion, device
        Standard PyTorch training inputs.
    optimizer : optim.Optimizer or None
        Provide an optimizer to train; pass ``None`` to run in eval / no-grad
        mode (used by validate_one_epoch / test_one_epoch).
    batch_unpack : BatchUnpack, optional
        Callable that converts a raw batch into ``(model_inputs, target)``,
        where ``model_inputs`` is a tuple expanded into ``model(*inputs)``.
        Default unpacker keeps the legacy 2-tuple ``(data, target)`` contract,
        so existing baselines need no change.
    num_classes : int, optional
        If provided, macro F1 and the confusion matrix are computed over
        ``labels=range(num_classes)``, guaranteeing the metric/matrix shape
        is stable even when a class happens to be absent from an epoch.
        Pass ``None`` (default) for legacy behavior (sklearn auto-detection).
    accumulate_grad_batches : int, default 1
        Gradient accumulation: the optimizer steps once every N loader
        batches, with each batch's loss scaled by 1/N — so the effective
        batch size is ``loader_batch_size × N`` while per-step memory stays
        at the loader batch size. Leftover gradients at the end of the
        epoch are flushed with a final step. Ignored in eval mode.
    desc : str
        tqdm progress-bar description.

    Returns
    -------
    (loss, accuracy, f1, confusion_matrix)
        Mean per-batch loss, accuracy, macro-F1, and confusion matrix
        accumulated over every processed batch. Batches that the unpacker
        rejects (returns None) are skipped and don't contribute to any
        statistic.
    """
    is_train = optimizer is not None
    unpack = batch_unpack or _default_batch_unpack
    accum = max(1, accumulate_grad_batches) if is_train else 1

    y_true: list[int] = []
    y_pred: list[int] = []
    running_loss = 0.0
    n_steps = 0
    pending_grads = 0  # micro-batches backward-ed since the last optimizer step

    model.train(is_train)
    if is_train:
        optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=desc, unit="batch", dynamic_ncols=True, leave=True)
    for batch in pbar:
        if not batch:
            continue
        unpacked = unpack(batch)
        if unpacked is None:
            continue
        inputs, target = unpacked
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if target.numel() == 0:
            continue

        inputs = tuple(t.to(device, non_blocking=True) for t in inputs)
        target = target.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            output = model(*inputs)
            loss = criterion(output, target)

        if is_train:
            # Scale so the accumulated gradient matches one big-batch step.
            (loss / accum).backward()
            pending_grads += 1
            if pending_grads == accum:
                optimizer.step()
                optimizer.zero_grad()
                pending_grads = 0

        running_loss += loss.item()
        n_steps += 1

        y_true.extend(target.detach().cpu().tolist())
        y_pred.extend(output.argmax(dim=1).detach().cpu().tolist())

    # Flush a leftover partial accumulation window at the end of the epoch.
    if is_train and pending_grads > 0:
        optimizer.step()
        optimizer.zero_grad()

    loss_epoch = running_loss / max(n_steps, 1)

    if y_true:
        acc_epoch = float(accuracy_score(y_true, y_pred))
        if num_classes is not None:
            f1_epoch = float(f1_score(
                y_true, y_pred,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            ))
            conf_mat = confusion_matrix(
                y_true, y_pred, labels=list(range(num_classes)),
            )
        else:
            f1_epoch = float(f1_score(y_true, y_pred, average="macro"))
            conf_mat = confusion_matrix(y_true, y_pred)
    else:
        acc_epoch = 0.0
        f1_epoch = 0.0
        n = num_classes if num_classes is not None else 1
        conf_mat = np.zeros((n, n), dtype=int)

    return loss_epoch, acc_epoch, f1_epoch, conf_mat


# ── Training Loop ────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    batch_unpack: BatchUnpack | None = None,
    num_classes: int | None = None,
    accumulate_grad_batches: int = 1,
    desc: str = "Training",
) -> tuple[float, float, float, np.ndarray]:
    """
    Train the model for one epoch and return ``(loss, acc, f1, conf_mat)``.

    Legacy callers (``train_one_epoch(model, loader, criterion, opt, device)``)
    continue to work unchanged: the default unpacker handles the standard
    ``(data, target)`` batch contract and dispatches to ``model(data)``.

    For batches with extra items or multi-input models (e.g. baseline3's
    ``(crops, person_labels, group_labels, masks)`` with
    ``GroupActivityModel.forward(crops, masks)``), pass a ``batch_unpack``
    callable that returns ``(model_inputs_tuple, target_tensor)``.
    """
    return _run_one_epoch(
        model, dataloader, criterion, optimizer, device,
        batch_unpack=batch_unpack, num_classes=num_classes,
        accumulate_grad_batches=accumulate_grad_batches, desc=desc,
    )


# ── Validation Loop ──────────────────────────────────────────────────────────


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    batch_unpack: BatchUnpack | None = None,
    num_classes: int | None = None,
    desc: str = "Validation",
) -> tuple[float, float, float, np.ndarray]:
    """Validate (no gradients) for one epoch. Same kwargs as ``train_one_epoch``."""
    return _run_one_epoch(
        model, dataloader, criterion, None, device,
        batch_unpack=batch_unpack, num_classes=num_classes, desc=desc,
    )


# ── Test Loop ────────────────────────────────────────────────────────────────


def test_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    batch_unpack: BatchUnpack | None = None,
    num_classes: int | None = None,
    desc: str = "Testing",
) -> tuple[float, float, float, np.ndarray]:
    """Run a single evaluation pass on the test set. Same kwargs as ``train_one_epoch``."""
    return _run_one_epoch(
        model, dataloader, criterion, None, device,
        batch_unpack=batch_unpack, num_classes=num_classes, desc=desc,
    )


# ── TensorBoard Experiment Summary ───────────────────────────────────────────


def log_experiment_summary(
    writer: SummaryWriter,
    run_id: str,
    hparam_dict: dict,
    test_f1: float,
    test_acc: float,
    test_loss: float,
    best_val_f1: float,
) -> None:
    """Log a self-contained experiment summary to TensorBoard.

    Writes two complementary entries into *writer*:

    * **TEXT tab** — a markdown table of every hyperparameter and final metric.
      Renders without TensorFlow and is visible immediately.
    * **HPARAMS tab** — the structured hparam/metric mapping used by the
      parallel-coordinates and scatter-matrix views (requires TensorFlow).

    Call this once per run, just before ``writer.close()``.
    All baseline scripts should use this function to keep the TensorBoard
    output format consistent across experiments.

    Parameters
    ----------
    writer : SummaryWriter
        The open TensorBoard writer for this run.
    run_id : str
        Human-readable run identifier (e.g. ``"run7"``).
    hparam_dict : dict
        Flat dict of hyperparameter names → scalar or string values.
        Must include at minimum a ``"baseline"`` key.
    test_f1 : float
        Final macro F1 score on the test set.
    test_acc : float
        Final accuracy on the test set.
    test_loss : float
        Final loss on the test set.
    best_val_f1 : float
        Best validation F1 achieved during training.
    """
    metric_dict = {
        "hparam/test_f1":     test_f1,
        "hparam/test_acc":    test_acc,
        "hparam/test_loss":   test_loss,
        "hparam/best_val_f1": best_val_f1,
    }

    # ── TEXT tab (no TensorFlow required) ────────────────────────────────
    hparam_rows = "\n".join(f"| `{k}` | {v} |" for k, v in hparam_dict.items())
    metric_rows = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in metric_dict.items())
    summary_text = (
        f"## {run_id}\n\n"
        f"### Hyperparameters\n\n"
        f"| Parameter | Value |\n|---|---|\n{hparam_rows}\n\n"
        f"### Final Metrics\n\n"
        f"| Metric | Value |\n|---|---|\n{metric_rows}"
    )
    writer.add_text("Experiment Summary", summary_text, global_step=0)

    # ── HPARAMS tab (requires TensorFlow for full rendering) ─────────────
    writer.add_hparams(hparam_dict, metric_dict)


# ── Class-Weight Helpers ─────────────────────────────────────────────────────
#
# Shared across baselines that need a class-balanced CrossEntropyLoss against
# the volleyball dataset's heavy per-class skew (e.g. person actions dominated
# by ``standing``, group activities with rare ``*_winpoint`` classes).
#
# The label-counting helpers take ``samples`` — VolleyballDataset.samples, the
# in-memory list of ``(video_id, clip_id, clip_data)`` tuples — so no image
# I/O happens. They replicate the loader's exact label-derivation logic so
# counts match what training will actually see.


def inverse_freq_weights(counts: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency CrossEntropyLoss weights: ``w_k = N / (K · n_k)``.

    Mean weight is 1, so the overall loss magnitude is preserved while rare
    classes get a proportionally larger gradient. Clamps to avoid div-by-zero
    on absent classes.

    Parameters
    ----------
    counts : torch.Tensor
        Shape ``(num_classes,)``, integer per-class sample counts.
    num_classes : int
        K — the number of classes.

    Returns
    -------
    torch.Tensor
        Shape ``(num_classes,)`` float weights ready to pass as
        ``nn.CrossEntropyLoss(weight=...)``.
    """
    total = counts.sum().clamp_min(1).float()
    return total / (counts.float().clamp_min(1) * num_classes)


def person_action_label_counts(samples, num_classes: int) -> torch.Tensor:
    """
    Count per-frame person-action labels across a list of dataset samples
    (e.g. ``VolleyballDataset.samples``). Mirrors the loader's
    tracking→actions fallback so counts match what training will actually see.

    Parameters
    ----------
    samples : iterable of ``(video_id, clip_id, clip_data)`` tuples
    num_classes : int
        Should be ``NUM_PERSON_ACTIONS`` (9 for this dataset).

    Returns
    -------
    torch.Tensor
        Shape ``(num_classes,)`` integer counts.
    """
    from configs.labels import PERSON_ACTION_TO_IDX

    counts = torch.zeros(num_classes, dtype=torch.long)
    for _video_id, clip_id, clip_data in samples:
        middle_name = f"{clip_id}.jpg"
        persons = clip_data.get("tracking", {}).get(middle_name, [])
        if not persons:
            persons = clip_data.get("actions", {}).get(middle_name, [])
        for p in persons:
            action = p.get("action", "standing")
            idx = PERSON_ACTION_TO_IDX.get(action, PERSON_ACTION_TO_IDX["standing"])
            counts[idx] += 1
    return counts


def group_activity_label_counts(samples, num_classes: int) -> torch.Tensor:
    """
    Count per-clip group-activity labels across a list of dataset samples.
    Mirrors the loader's exact mapping
    ``GROUP_ACTIVITY_TO_IDX.get(scene_class, 0) if scene_class else 0`` —
    so a clip with missing/None ``scene_class`` contributes to class 0 just
    as it does in training.

    Parameters
    ----------
    samples : iterable of ``(video_id, clip_id, clip_data)`` tuples
    num_classes : int
        Should be ``NUM_GROUP_ACTIVITIES`` (8 for this dataset).

    Returns
    -------
    torch.Tensor
        Shape ``(num_classes,)`` integer counts.
    """
    from configs.labels import GROUP_ACTIVITY_TO_IDX

    counts = torch.zeros(num_classes, dtype=torch.long)
    for _video_id, _clip_id, clip_data in samples:
        scene_class = clip_data.get("scene_class")
        idx = GROUP_ACTIVITY_TO_IDX.get(scene_class, 0) if scene_class else 0
        counts[idx] += 1
    return counts
