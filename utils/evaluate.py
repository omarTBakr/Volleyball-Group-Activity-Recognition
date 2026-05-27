"""
Post-training evaluation script for baseline models.

Loads a saved model checkpoint, runs inference on the test split,
and generates all standard visualisation plots (confusion matrix,
classification report, precision-recall AUC, mAP vs F1).

Usage
-----
    uv run python -m utils.evaluate --model baseline1_run1.pt
    uv run python -m utils.evaluate --model baseline1_run1.pt --baseline B1
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from sklearn.metrics import confusion_matrix
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


def _setup_for_baseline(
    baseline_name: str, cfg: Any
) -> tuple[torch.nn.Module, dict[str, Any]]:
    
    
    """Return the unloaded model instance and dataset kwargs for a given baseline."""
    # this function is to be extended to include all baselines
    
    
    if baseline_name == "baseline1":
        from models.baseline1 import Model as Baseline1Model

        model = Baseline1Model(num_classes=NUM_GROUP_ACTIVITIES)
        dataset_kwargs = {"full_image": True, "n_frames": 1}
    elif baseline_name == "baseline3":
        from models.baseline3 import build_model as build_baseline3

        model = build_baseline3(cfg, num_classes=NUM_GROUP_ACTIVITIES)
        dataset_kwargs = {"crop": True, "full_image": False}
    else:
        raise ValueError(f"Evaluation not implemented for baseline: '{baseline_name}'")

    return model, dataset_kwargs


def _get_data_loader(baseline_name: str, dataset_kwargs: dict[str, Any]) -> DataLoader:
    """Initialize and return the test DataLoader."""
    print("Loading test dataset...")

    configs_dir = str(BASE_DIR / "configs")
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(config_name=baseline_name)

    transforms_dict = build_transforms(cfg)
    test_transform = transforms_dict["test"]

    test_dataset = VolleyballDataset(
        mode="test", transform=test_transform, **dataset_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return test_loader


def _load_model_weights(
    model: torch.nn.Module, model_filename: str, device: torch.device
) -> torch.nn.Module:
    """Load model weights from disk and move to the target device."""
    print(f"Loading model '{model_filename}'...")
    try:
        model, _, _, _, _ = load_model(model_filename, model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Make sure the model name is correct (e.g., 'baseline1_run1.pt') "
            "and it exists in your saved_models folder."
        )
        sys.exit(1)

    model.to(device)
    model.eval()
    return model


def _run_inference(
    model: torch.nn.Module, test_loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a full inference pass over the test dataset."""
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    y_score_list: list[np.ndarray] = []

    print(f"\nEvaluating on {device}...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if len(batch) == 2:
                data, target = batch
            elif len(batch) == 4:
                # crops, person_labels, group_labels, masks
                data, _, target, _ = batch
            else:
                raise ValueError("Unexpected batch format from DataLoader")

            data = data.to(device)
            output = model(data)

            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            y_true_list.extend(target.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            y_score_list.extend(probs.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_score = np.array(y_score_list)

    return y_true, y_pred, y_score


def _generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    baseline_name: str,
) -> None:
    """Generate all evaluation plots and save them to disk."""
    print("\nGenerating plots...")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, baseline=baseline_name)
    print("  ✓ Confusion matrix saved")

    # Classification Report Heatmap
    plot_classification_report(y_true, y_pred, class_names, baseline=baseline_name)
    print("  ✓ Classification report saved")

    # Precision-Recall AUC curves per class
    plot_precision_recall_auc(y_true, y_score, class_names, baseline=baseline_name)
    print("  ✓ Precision-Recall AUC curves saved")

    # mAP & F1 Grouped Bar Chart
    plot_map_f1(y_true, y_score, y_pred, class_names, baseline=baseline_name)
    print("  ✓ mAP & F1 bar chart saved")


def evaluate(model_filename: str, baseline_name: str) -> None:
    """
    Evaluate a saved model checkpoint and generate all metric plots.

    Parameters
    ----------
    model_filename : str
        Filename of the saved checkpoint (e.g. ``"baseline1_run1.pt"``).
    baseline_name : str
        Sub-directory name under ``plots/`` to save the generated figures.
    """
    device = get_device()

    # Load configuration to get model architecture and dataset specs
    configs_dir = str(BASE_DIR / "configs")
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(config_name=baseline_name)

    model, dataset_kwargs = _setup_for_baseline(baseline_name, cfg)

    # 1. Setup DataLoader
    test_loader = _get_data_loader(baseline_name, dataset_kwargs)
    class_names = [IDX_TO_GROUP_ACTIVITY[i] for i in range(NUM_GROUP_ACTIVITIES)]

    # 2. Load Model Weights
    model = _load_model_weights(model, model_filename, device)

    # 3. Run Inference
    y_true, y_pred, y_score = _run_inference(model, test_loader, device)

    # 4. Generate All Plots
    _generate_plots(y_true, y_pred, y_score, class_names, baseline_name)

    print(
        f"\n✓ Done! All metrics have been plotted and saved to "
        f"the 'plots/{baseline_name}/' folder."
    )


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
        help="Folder name to save plots into (defaults to 'baseline1')",
    )
    args = parser.parse_args()

    evaluate(args.model, args.baseline)
