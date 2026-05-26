"""
Visualization utilities for training metrics and evaluation results.

All plot functions save figures to disk (no interactive ``plt.show()`` calls)
so they work seamlessly in headless environments like Kaggle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # ty:ignore[unresolved-import]
import numpy as np

from configs.path_config import PLOTS_DIR

# ── Confusion Matrix ─────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path = PLOTS_DIR,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and save a normalized confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Raw confusion matrix (counts).
    class_names : list[str]
        Ordered list of class label strings.
    save_path : Path
        Directory where the figure will be saved.
    title : str
        Title used for both the figure and the filename.

    """
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate each cell with its normalized value
    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── Metric Curves ────────────────────────────────────────────────────────────


def plot_loss(
    train_loss: list[float],
    val_loss: list[float],
    save_path: Path = PLOTS_DIR,
    title: str = "Training & Validation Loss",
) -> None:
    """Plot training and validation loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Training Loss", marker="o")
    ax.plot(epochs, val_loss, label="Validation Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_accuracy(
    train_acc: list[float],
    val_acc: list[float],
    save_path: Path = PLOTS_DIR,
    title: str = "Training & Validation Accuracy",
) -> None:
    """Plot training and validation accuracy curves over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_acc) + 1)
    ax.plot(epochs, train_acc, label="Training Accuracy", marker="o")
    ax.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_f1_score(
    train_f1: list[float],
    val_f1: list[float],
    save_path: Path = PLOTS_DIR,
    title: str = "Training & Validation F1 Score",
) -> None:
    """Plot training and validation F1 score curves over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_f1) + 1)
    ax.plot(epochs, train_f1, label="Training F1 Score", marker="o")
    ax.plot(epochs, val_f1, label="Validation F1 Score", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # Example usage
    cm = np.array([[10, 2, 3], [4, 15, 1], [0, 1, 20]])
    class_names = ["Action1", "Action2", "Action3"]
    plot_confusion_matrix(cm, class_names)

    train_loss = [1.0, 0.8, 0.6, 0.4, 0.2]
    val_loss = [1.2, 1.0, 0.9, 0.8, 0.7]
    plot_loss(train_loss, val_loss)

    train_acc = [0.7, 0.75, 0.8, 0.85, 0.9]
    val_acc = [0.65, 0.7, 0.75, 0.78, 0.82]
    plot_accuracy(train_acc, val_acc)

    train_f1 = [0.7, 0.75, 0.8, 0.85, 0.9]
    val_f1 = [0.65, 0.7, 0.75, 0.78, 0.82]
    plot_f1_score(train_f1, val_f1)

    