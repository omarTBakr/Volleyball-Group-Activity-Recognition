"""
Visualization utilities for training metrics and evaluation results.

All plot functions save figures to disk (no interactive ``plt.show()`` calls)
so they work seamlessly in headless environments like Kaggle.

Every function accepts an optional ``baseline`` parameter.  When provided,
plots are saved into ``PLOTS_DIR / baseline /`` (the sub-directory is
created automatically).  This keeps outputs from different experiments
neatly separated (e.g. ``plots/B1/``, ``plots/B2/``).

Functions
---------
plot_confusion_matrix
    Normalized confusion-matrix heatmap.
plot_loss
    Training & validation loss curves over epochs.
plot_accuracy
    Training & validation accuracy curves over epochs.
plot_f1_score
    Training & validation F1-score curves over epochs.
plot_precision_recall_auc
    Per-class Precision–Recall curves with AUC values.
plot_classification_report
    Grouped horizontal bar chart of per-class precision / recall / F1
    with support counts and aggregate metrics in the subtitle.
plot_map_f1
    Bar chart comparing Mean Average Precision and macro-F1 per class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # ty:ignore[unresolved-import]
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from configs.path_config import PLOTS_DIR


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_save_dir(save_path: Path, baseline: str | None) -> Path:
    """
    Return the final save directory, optionally nested under *baseline*.

    If *baseline* is not ``None``, the returned path is
    ``save_path / baseline``.  The directory is created (with parents)
    if it does not already exist.

    Parameters
    ----------
    save_path : Path
        Root output directory (usually ``PLOTS_DIR``).
    baseline : str or None
        Optional experiment / baseline name to create a sub-folder.

    Returns
    -------
    Path
        The directory where the figure should be saved.
    """
    if baseline is not None:
        save_path = save_path / baseline
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


# ── Confusion Matrix ─────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path = PLOTS_DIR,
    title: str = "Confusion Matrix",
    baseline: str | None = None,
) -> None:
    """
    Plot and save a normalized confusion matrix as a heatmap.

    Each row of the raw count matrix is divided by its row-sum so that
    cell values represent the fraction of true samples predicted as each
    class.  The figure is saved as a high-resolution PNG.

    Parameters
    ----------
    cm : np.ndarray
        Raw confusion matrix of shape ``(n_classes, n_classes)`` containing
        integer counts.
    class_names : list[str]
        Ordered list of human-readable class label strings.  Length must
        match the dimension of *cm*.
    save_path : Path, default ``PLOTS_DIR``
        Root directory for saving.  Combined with *baseline* to form the
        final output directory.
    title : str, default ``"Confusion Matrix"``
        Title used for both the figure heading and the output filename.
    baseline : str or None, default ``None``
        If given, a sub-directory with this name is created under
        *save_path* and the figure is saved there.

    Returns
    -------
    None
        The figure is written to ``<save_dir> / f"{title}.png"``.

    Examples
    --------
    >>> cm = np.array([[50, 2], [5, 43]])
    >>> plot_confusion_matrix(cm, ["cat", "dog"], baseline="B1")
    """
    out_dir = _resolve_save_dir(save_path, baseline)
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
    fig.savefig(out_dir / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── Metric Curves ────────────────────────────────────────────────────────────


def plot_loss(
    train_loss: list[float],
    val_loss: list[float],
    save_path: Path = PLOTS_DIR,
    title: str = "Training & Validation Loss",
    baseline: str | None = None,
) -> None:
    """
    Plot training and validation loss curves over epochs.

    Both curves are drawn on the same axes so that over-fitting (diverging
    curves) is easy to spot visually.

    Parameters
    ----------
    train_loss : list[float]
        Per-epoch training loss values.
    val_loss : list[float]
        Per-epoch validation loss values.  Must have the same length as
        *train_loss*.
    save_path : Path, default ``PLOTS_DIR``
        Root directory for saving.
    title : str, default ``"Training & Validation Loss"``
        Title for the figure and the output filename.
    baseline : str or None, default ``None``
        Optional baseline sub-directory name.

    Returns
    -------
    None
        The figure is written to ``<save_dir> / f"{title}.png"``.

    Examples
    --------
    >>> plot_loss([1.0, 0.8, 0.6], [1.2, 1.0, 0.9], baseline="B3")
    """
    out_dir = _resolve_save_dir(save_path, baseline)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Training Loss", marker="o")
    ax.plot(epochs, val_loss, label="Validation Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(out_dir / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_accuracy(
    train_acc: list[float],
    val_acc: list[float],
    save_path: Path = PLOTS_DIR,
    title: str = "Training & Validation Accuracy",
    baseline: str | None = None,
) -> None:
    """
    Plot training and validation accuracy curves over epochs.

    Accuracy values are expected in the range ``[0, 1]``.

    Parameters
    ----------
    train_acc : list[float]
        Per-epoch training accuracy values.
    val_acc : list[float]
        Per-epoch validation accuracy values.  Must have the same length
        as *train_acc*.
    save_path : Path, default ``PLOTS_DIR``
        Root directory for saving.
    title : str, default ``"Training & Validation Accuracy"``
        Title for the figure and the output filename.
    baseline : str or None, default ``None``
        Optional baseline sub-directory name.

    Returns
    -------
    None
        The figure is written to ``<save_dir> / f"{title}.png"``.

    Examples
    --------
    >>> plot_accuracy([0.7, 0.8, 0.9], [0.65, 0.75, 0.82], baseline="B1")
    """
    out_dir = _resolve_save_dir(save_path, baseline)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_acc) + 1)
    ax.plot(epochs, train_acc, label="Training Accuracy", marker="o")
    ax.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(out_dir / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_f1_score(
    train_f1: list[float],
    val_f1: list[float],
    save_path: Path = PLOTS_DIR,
    title: str = "Training & Validation F1 Score",
    baseline: str | None = None,
) -> None:
    """
    Plot training and validation macro-F1 score curves over epochs.

    F1 values are expected in the range ``[0, 1]``.

    Parameters
    ----------
    train_f1 : list[float]
        Per-epoch training F1 score values.
    val_f1 : list[float]
        Per-epoch validation F1 score values.  Must have the same length
        as *train_f1*.
    save_path : Path, default ``PLOTS_DIR``
        Root directory for saving.
    title : str, default ``"Training & Validation F1 Score"``
        Title for the figure and the output filename.
    baseline : str or None, default ``None``
        Optional baseline sub-directory name.

    Returns
    -------
    None
        The figure is written to ``<save_dir> / f"{title}.png"``.

    Examples
    --------
    >>> plot_f1_score([0.7, 0.8, 0.9], [0.65, 0.75, 0.82])
    """
    out_dir = _resolve_save_dir(save_path, baseline)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_f1) + 1)
    ax.plot(epochs, train_f1, label="Training F1 Score", marker="o")
    ax.plot(epochs, val_f1, label="Validation F1 Score", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(out_dir / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── Precision-Recall & AUC ───────────────────────────────────────────────────


def plot_precision_recall_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    save_path: Path = PLOTS_DIR,
    title: str = "Precision-Recall Curves",
    baseline: str | None = None,
) -> dict[str, float]:
    """
    Plot per-class Precision–Recall curves and annotate each with its AUC.

    The input ``y_score`` should contain the **raw softmax probabilities**
    (or logits passed through softmax) produced by the model for every
    sample, with shape ``(n_samples, n_classes)``.

    For each class the curve is computed in a *one-vs-rest* fashion using
    :func:`sklearn.metrics.precision_recall_curve`.  The area under each
    curve (AUC-PR) is computed with
    :func:`sklearn.metrics.average_precision_score`.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels of shape ``(n_samples,)``.
    y_score : np.ndarray
        Predicted probability matrix of shape ``(n_samples, n_classes)``.
        Each row must sum to 1 (softmax output).
    class_names : list[str]
        Human-readable class names, length must equal ``n_classes``.
    save_path : Path, default ``PLOTS_DIR``
        Root directory for saving.
    title : str, default ``"Precision-Recall Curves"``
        Title for the figure and the output filename.
    baseline : str or None, default ``None``
        Optional baseline sub-directory name.

    Returns
    -------
    dict[str, float]
        Mapping of class name → Average Precision (AUC-PR) value.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 0, 1])
    >>> y_score = np.array([
    ...     [0.9, 0.05, 0.05],
    ...     [0.1, 0.8, 0.1],
    ...     [0.1, 0.1, 0.8],
    ...     [0.8, 0.1, 0.1],
    ...     [0.2, 0.7, 0.1],
    ... ])
    >>> auc_dict = plot_precision_recall_auc(y_true, y_score, ["A", "B", "C"])
    """
    out_dir = _resolve_save_dir(save_path, baseline)
    n_classes = len(class_names)

    # One-hot encode the true labels
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=int)
    for idx, label in enumerate(y_true):
        y_true_bin[idx, label] = 1

    fig, ax = plt.subplots(figsize=(10, 7))
    ap_scores: dict[str, float] = {}

    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], y_score[:, i]
        )
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        ap_scores[name] = float(ap)

        ax.plot(recall, precision, lw=2, label=f"{name}  (AUC = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return ap_scores


# ── Classification Report Heatmap ────────────────────────────────────────────


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path = PLOTS_DIR,
    title: str = "Classification Report",
    baseline: str | None = None,
) -> dict:
    """
    Render the sklearn classification report as a grouped horizontal bar chart.

    One row per class with three bars — **precision**, **recall**, and
    **F1** — so per-class strengths and weaknesses can be compared at a
    glance (unlike the raw text table). Each class label carries its
    support count, bar values are annotated at the bar ends, and the
    aggregate metrics (accuracy, macro-F1, weighted-F1) appear as a
    subtitle rather than as pseudo-rows.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels of shape ``(n_samples,)``.
    y_pred : np.ndarray
        Predicted integer labels of shape ``(n_samples,)``.
    class_names : list[str]
        Human-readable class names whose order matches the label integers.
    save_path : Path, default ``PLOTS_DIR``
        Root directory for saving.
    title : str, default ``"Classification Report"``
        Title for the figure and the output filename.
    baseline : str or None, default ``None``
        Optional baseline sub-directory name.

    Returns
    -------
    dict
        The raw classification report dictionary as returned by
        :func:`sklearn.metrics.classification_report` with
        ``output_dict=True``.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_pred = np.array([0, 1, 1, 1, 2, 0])
    >>> report = plot_classification_report(
    ...     y_true, y_pred, ["cat", "dog", "bird"], baseline="B2"
    ... )
    """
    out_dir = _resolve_save_dir(save_path, baseline)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    n_classes = len(class_names)
    precision = [report[c]["precision"] for c in class_names]
    recall = [report[c]["recall"] for c in class_names]
    f1 = [report[c]["f1-score"] for c in class_names]
    support = [int(report[c]["support"]) for c in class_names]

    # ── Grouped horizontal bars (first class on top) ──
    y = np.arange(n_classes)
    bar_h = 0.26
    fig_h = max(6, n_classes * 1.0 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Offsets chosen so that, with the inverted y-axis, bars appear
    # top-to-bottom as Precision / Recall / F1 — matching the legend order.
    bar_groups = (
        (precision, "Precision", "#4C72B0", -bar_h),
        (recall,    "Recall",    "#55A868", 0.0),
        (f1,        "F1 Score",  "#DD8452", +bar_h),
    )
    for values, label, color, offset in bar_groups:
        bars = ax.barh(
            y + offset, values, bar_h,
            label=label, color=color, edgecolor="white",
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}",
                ha="left", va="center", fontsize=8.5, fontweight="bold",
            )

    # Macro-F1 reference line
    macro_f1 = report["macro avg"]["f1-score"]
    ax.axvline(macro_f1, color="#DD8452", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"Macro-F1 = {macro_f1:.3f}", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{name}\n(n={n})" for name, n in zip(class_names, support)],
        fontsize=11,
    )
    ax.invert_yaxis()  # first class at the top, like the text report
    ax.set_xlim(0.0, 1.12)
    ax.set_xlabel("Score", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    acc = report["accuracy"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    total = int(report["macro avg"]["support"])
    ax.set_title(
        f"{title}\n"
        f"accuracy {acc:.3f}  ·  macro F1 {macro_f1:.3f}  ·  "
        f"weighted F1 {weighted_f1:.3f}  ·  N = {total}",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return report


# ── Mean Average Precision & F1 ──────────────────────────────────────────────


def plot_map_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path = PLOTS_DIR,
    title: str = "mAP & F1 Score per Class",
    baseline: str | None = None,
) -> tuple[float, dict[str, dict[str, float]]]:
    """
    Compute and plot per-class Average Precision (AP) and F1.

    This computes the metrics alongside the overall Mean Average Precision (mAP)
    and macro-F1. A grouped bar chart is produced with two bars per class
    (AP and F1), plus horizontal reference lines for the mAP and macro-F1.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels of shape ``(n_samples,)``.
    y_score : np.ndarray
        Predicted probability matrix of shape ``(n_samples, n_classes)``.
        Each row should be a softmax distribution over classes.
    y_pred : np.ndarray
        Predicted integer labels of shape ``(n_samples,)`` (used to
        compute the F1 score).
    class_names : list[str]
        Human-readable class names whose order matches the label integers.
    save_path : Path, default ``PLOTS_DIR``
        Root directory for saving.
    title : str, default ``"mAP & F1 Score per Class"``
        Title for the figure and the output filename.
    baseline : str or None, default ``None``
        Optional baseline sub-directory name.

    Returns
    -------
    tuple[float, dict[str, dict[str, float]]]
        A 2-tuple of:

        * **mAP** (*float*) – the mean of the per-class AP values.
        * **per_class** (*dict*) – mapping of class name to a dict with
          keys ``"ap"`` and ``"f1"``.

    Examples
    --------
    >>> y_true  = np.array([0, 1, 2, 0, 1])
    >>> y_score = np.array([
    ...     [0.9, 0.05, 0.05],
    ...     [0.1, 0.8, 0.1],
    ...     [0.1, 0.1, 0.8],
    ...     [0.8, 0.1, 0.1],
    ...     [0.2, 0.7, 0.1],
    ... ])
    >>> y_pred = np.array([0, 1, 2, 0, 1])
    >>> mAP, per_class = plot_map_f1(
    ...     y_true, y_score, y_pred, ["A", "B", "C"], baseline="B5"
    ... )
    """
    out_dir = _resolve_save_dir(save_path, baseline)
    n_classes = len(class_names)

    # One-hot encode true labels
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=int)
    for idx, label in enumerate(y_true):
        y_true_bin[idx, label] = 1

    # ── Per-class metrics ──
    report: Any = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {}
    ap_values: list[float] = []
    f1_values: list[float] = []

    for i, name in enumerate(class_names):
        ap = float(average_precision_score(y_true_bin[:, i], y_score[:, i]))
        f1 = report[name]["f1-score"]
        per_class[name] = {"ap": ap, "f1": f1}
        ap_values.append(ap)
        f1_values.append(f1)

    mAP = float(np.mean(ap_values))
    macro_f1 = float(np.mean(f1_values))

    # ── Grouped bar chart ──
    x = np.arange(n_classes)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.2), 6))

    bars_ap = ax.bar(
        x - bar_width / 2, ap_values, bar_width,
        label="Average Precision (AP)", color="#4C72B0", edgecolor="white",
    )
    bars_f1 = ax.bar(
        x + bar_width / 2, f1_values, bar_width,
        label="F1 Score", color="#DD8452", edgecolor="white",
    )

    # Reference lines
    ax.axhline(mAP, color="#4C72B0", linestyle="--", linewidth=1.2,
               label=f"mAP = {mAP:.3f}")
    ax.axhline(macro_f1, color="#DD8452", linestyle="--", linewidth=1.2,
               label=f"Macro-F1 = {macro_f1:.3f}")

    # Value labels on bars
    for bar in bars_ap:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return mAP, per_class


# ── Demo / Smoke-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEMO_BASELINE = "demo"

    # ── Confusion matrix ──
    cm = np.array([[10, 2, 3], [4, 15, 1], [0, 1, 20]])
    class_names = ["Action1", "Action2", "Action3"]
    plot_confusion_matrix(cm, class_names, baseline=DEMO_BASELINE)

    # ── Epoch curves ──
    train_loss = [1.0, 0.8, 0.6, 0.4, 0.2]
    val_loss = [1.2, 1.0, 0.9, 0.8, 0.7]
    plot_loss(train_loss, val_loss, baseline=DEMO_BASELINE)

    train_acc = [0.7, 0.75, 0.8, 0.85, 0.9]
    val_acc = [0.65, 0.7, 0.75, 0.78, 0.82]
    plot_accuracy(train_acc, val_acc, baseline=DEMO_BASELINE)

    train_f1 = [0.7, 0.75, 0.8, 0.85, 0.9]
    val_f1 = [0.65, 0.7, 0.75, 0.78, 0.82]
    plot_f1_score(train_f1, val_f1, baseline=DEMO_BASELINE)

    # ── New evaluation plots ──
    rng = np.random.default_rng(42)
    n_samples = 200
    y_true = rng.integers(0, 3, size=n_samples)
    # Simulate softmax probabilities
    logits = rng.standard_normal((n_samples, 3))
    logits[np.arange(n_samples), y_true] += 2.0  # boost correct class
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_score = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_pred = y_score.argmax(axis=1)

    auc_dict = plot_precision_recall_auc(
        y_true, y_score, class_names, baseline=DEMO_BASELINE,
    )
    print("AUC-PR per class:", auc_dict)

    report = plot_classification_report(
        y_true, y_pred, class_names, baseline=DEMO_BASELINE,
    )
    print("Classification report dict keys:", list(report.keys()))

    mAP, per_class = plot_map_f1(
        y_true, y_score, y_pred, class_names, baseline=DEMO_BASELINE,
    )
    print(f"mAP = {mAP:.4f}")
    print("Per-class AP & F1:", per_class)
