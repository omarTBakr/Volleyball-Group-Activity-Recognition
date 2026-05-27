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


# ── Training Loop ────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray]:
    """
    Train the model for a single epoch.

    Iterates over every batch in *dataloader*, performing forward pass,
    loss computation, back-propagation, and optimizer step.  Epoch-level
    metrics are computed from the accumulated predictions.

    Parameters
    ----------
    model : nn.Module
        The model to train.  Will be set to ``model.train()`` mode.
    dataloader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function (e.g. ``nn.CrossEntropyLoss()``).
    optimizer : optim.Optimizer
        Optimizer instance used to update model weights.
    device : torch.device
        Target device (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    tuple[float, float, float, np.ndarray]
        ``(loss, accuracy, f1_score, confusion_matrix)`` where loss is
        the mean batch loss, accuracy and F1 are macro-averaged, and
        confusion_matrix has shape ``(n_classes, n_classes)``.

    """
    y_true: list[int] = []
    y_pred: list[int] = []
    running_loss = 0.0

    model.train()

    pbar = tqdm(dataloader, desc="Training", unit="batch", dynamic_ncols=True, leave=True)

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        y_true.extend(target.cpu().numpy())
        y_pred.extend(output.argmax(dim=1).cpu().numpy())

    loss_epoch = running_loss / len(dataloader)
    acc_epoch = float(accuracy_score(y_true, y_pred))
    f1_epoch = float(f1_score(y_true, y_pred, average="macro"))
    conf_mat = confusion_matrix(y_true, y_pred)

    return loss_epoch, acc_epoch, f1_epoch, conf_mat


# ── Validation Loop ──────────────────────────────────────────────────────────


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray]:
    """
    Validate the model for a single epoch (no gradient computation).

    Identical to :func:`train_one_epoch` except that the model is placed
    in ``eval()`` mode and all forward passes run inside
    ``torch.no_grad()``.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    dataloader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[float, float, float, np.ndarray]
        ``(loss, accuracy, f1_score, confusion_matrix)``.

    """
    y_true: list[int] = []
    y_pred: list[int] = []
    running_loss = 0.0

    model.eval()

    pbar = tqdm(dataloader, desc="Validation", unit="batch", dynamic_ncols=True, leave=True)

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

        y_true.extend(target.cpu().numpy())
        y_pred.extend(output.argmax(dim=1).cpu().numpy())

    loss_epoch = running_loss / len(dataloader)
    acc_epoch = float(accuracy_score(y_true, y_pred))
    f1_epoch = float(f1_score(y_true, y_pred, average="macro"))
    conf_mat = confusion_matrix(y_true, y_pred)

    return loss_epoch, acc_epoch, f1_epoch, conf_mat


# ── Test Loop ────────────────────────────────────────────────────────────────


def test_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray]:
    """
    Evaluate the model on the test set.

    Functionally identical to :func:`validate_one_epoch` but kept as a
    separate function so callers can distinguish validation from final
    test evaluation in logs and progress bars.

    Parameters
    ----------
    model : nn.Module
        The trained model to evaluate.
    dataloader : DataLoader
        Test data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[float, float, float, np.ndarray]
        ``(loss, accuracy, f1_score, confusion_matrix)``.

    """
    y_true: list[int] = []
    y_pred: list[int] = []
    running_loss = 0.0

    model.eval()

    pbar = tqdm(dataloader, desc="Testing", unit="batch", dynamic_ncols=True, leave=True)

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

        y_true.extend(target.cpu().numpy())
        y_pred.extend(output.argmax(dim=1).cpu().numpy())

    loss_epoch = running_loss / len(dataloader)
    acc_epoch = float(accuracy_score(y_true, y_pred))
    f1_epoch = float(f1_score(y_true, y_pred, average="macro"))
    conf_mat = confusion_matrix(y_true, y_pred)

    return loss_epoch, acc_epoch, f1_epoch, conf_mat
