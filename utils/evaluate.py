"""
Post-training evaluation script for baseline models.

Loads a saved model checkpoint, runs inference on the test split,
and generates all standard visualisation plots (confusion matrix,
classification report, precision-recall AUC, mAP vs F1).

Usage
-----
    uv run python -m utils.evaluate --model baseline1_run1.pt
    uv run python -m utils.evaluate --model baseline1_run1.pt --baseline baseline3
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.labels import IDX_TO_GROUP_ACTIVITY, NUM_GROUP_ACTIVITIES
from configs.path_config import BASE_DIR
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from utils.load_model_config import build_transforms
from utils.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_map_f1,
    plot_precision_recall_auc,
)
from utils.utility import get_device, load_model

# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    IDX_TO_GROUP_ACTIVITY[i] for i in range(NUM_GROUP_ACTIVITIES)
]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Model & Dataset Factory
# ═════════════════════════════════════════════════════════════════════════════


def _build_model(baseline_name: str, cfg: DictConfig) -> nn.Module:
    """Instantiate the correct model architecture for *baseline_name*.

    This function is the single place to extend when adding new baselines.
    """
    if baseline_name == "baseline1":
        from models.baseline1 import Model as Baseline1Model

        return Baseline1Model(
            num_classes=NUM_GROUP_ACTIVITIES,
            backbone_name=cfg.model.name,
            dropout=cfg.model.get("dropout", 0.0),
        )

    if baseline_name == "baseline3":
        from models.baseline3 import build_model as build_baseline3

        return build_baseline3(cfg, num_classes=NUM_GROUP_ACTIVITIES)

    raise ValueError(
        f"Evaluation not implemented for baseline: '{baseline_name}'"
    )


def _dataset_kwargs_for(baseline_name: str) -> dict[str, Any]:
    """Return the ``VolleyballDataset`` keyword arguments for *baseline_name*."""
    if baseline_name == "baseline1":
        return {"full_image": True, "n_frames": 1}

    if baseline_name == "baseline3":
        return {"crop": True, "full_image": False}

    raise ValueError(
        f"Dataset config not defined for baseline: '{baseline_name}'"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Data Loading
# ═════════════════════════════════════════════════════════════════════════════


def _build_test_loader(
    cfg: DictConfig, dataset_kwargs: dict[str, Any]
) -> DataLoader:
    """Build and return the test-split ``DataLoader``."""
    transforms_dict = build_transforms(cfg)

    test_dataset = VolleyballDataset(
        mode="test",
        transform=transforms_dict["test"],
        **dataset_kwargs,
    )
    return DataLoader(
        test_dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Checkpoint Loading
# ═════════════════════════════════════════════════════════════════════════════


def _load_checkpoint(
    model: nn.Module, filename: str, device: torch.device
) -> nn.Module:
    """Load saved weights into *model*, move to *device*, and set eval mode."""
    print(f"Loading checkpoint '{filename}' ...")
    try:
        model, _, _, _, _ = load_model(filename, model)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        print(
            "Make sure the filename is correct (e.g. 'baseline1_run1.pt') "
            "and it exists in your saved_models/ folder."
        )
        sys.exit(1)

    model.to(device)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 4. Inference
# ═════════════════════════════════════════════════════════════════════════════


def _unpack_batch(batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (data, target) from a batch regardless of collate format.

    Full-image mode returns ``(images, labels)`` while crop mode returns
    ``(crops, person_labels, group_labels, masks)``.
    """
    if len(batch) == 2:
        return batch[0], batch[1]
    if len(batch) == 4:
        return batch[0], batch[2]  # crops, group_labels
    raise ValueError(f"Unexpected batch format (len={len(batch)})")


def _run_inference(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return ``(y_true, y_pred, y_score)``."""
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_score_all: list[np.ndarray] = []

    print(f"\nEvaluating on {device} ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            data, target = _unpack_batch(batch)
            data = data.to(device)

            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            y_true_all.extend(target.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            y_score_all.extend(probs.cpu().numpy())

    return (
        np.array(y_true_all),
        np.array(y_pred_all),
        np.array(y_score_all),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. Plot Generation
# ═════════════════════════════════════════════════════════════════════════════


def _generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    baseline_name: str,
) -> None:
    """Generate and save all evaluation plots."""
    print("\nGenerating plots ...")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, baseline=baseline_name)
    print("  ✓ Confusion matrix")

    plot_classification_report(y_true, y_pred, class_names, baseline=baseline_name)
    print("  ✓ Classification report")

    plot_precision_recall_auc(y_true, y_score, class_names, baseline=baseline_name)
    print("  ✓ Precision-Recall AUC curves")

    plot_map_f1(y_true, y_score, y_pred, class_names, baseline=baseline_name)
    print("  ✓ mAP & F1 bar chart")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Public Entry Point
# ═════════════════════════════════════════════════════════════════════════════


def evaluate(model_filename: str, baseline_name: str) -> None:
    """Evaluate a saved checkpoint and generate all metric plots.

    Parameters
    ----------
    model_filename : str
        Filename of the saved checkpoint (e.g. ``"baseline1_run1.pt"``).
    baseline_name : str
        Baseline identifier — used to select the model architecture,
        load the matching Hydra config, and name the output sub-folder
        under ``plots/``.
    """
    device = get_device()

    # ── Config ────────────────────────────────────────────────────────
    configs_dir = str(BASE_DIR / "configs")
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(config_name=baseline_name)

    # ── Pipeline ──────────────────────────────────────────────────────
    model = _build_model(baseline_name, cfg)
    dataset_kwargs = _dataset_kwargs_for(baseline_name)
    test_loader = _build_test_loader(cfg, dataset_kwargs)

    model = _load_checkpoint(model, model_filename, device)

    y_true, y_pred, y_score = _run_inference(model, test_loader, device)

    _generate_plots(y_true, y_pred, y_score, CLASS_NAMES, baseline_name)

    print(
        f"\n✓ Done! All plots saved to 'plots/{baseline_name}/'."
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model and plot all metrics.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Filename of the saved model (e.g. 'baseline1_run1.pt')",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline1",
        help="Baseline identifier (defaults to 'baseline1')",
    )
    args = parser.parse_args()

    evaluate(args.model, args.baseline)
