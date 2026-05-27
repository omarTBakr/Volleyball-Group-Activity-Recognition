import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.labels import IDX_TO_GROUP_ACTIVITY, NUM_GROUP_ACTIVITIES
from models.baseline1 import Model as Baseline1Model
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from utils.load_model_config import build_transforms
from utils.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_map_f1,
    plot_precision_recall_auc,
)
from utils.utility import get_device, load_model


def evaluate(model_filename: str, baseline_name: str) -> None:
    device = get_device()
    
    # ── 1. Setup Data ──
    print("Loading test dataset...")
    
    # Initialize Hydra and load config
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="baseline1")
    
    transforms_dict = build_transforms(cfg)
    test_transform = transforms_dict["test"]
    
    test_dataset = VolleyballDataset(split="test", transform=test_transform, n_frames=1)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=False
    )
    class_names = [IDX_TO_GROUP_ACTIVITY[i] for i in range(NUM_GROUP_ACTIVITIES)]

    # ── 2. Load Model ──
    print(f"Loading model '{model_filename}'...")
    model = Baseline1Model(num_classes=NUM_GROUP_ACTIVITIES)
    try:
        model, _, _, _, _ = load_model(model_filename, model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model name is correct (e.g., 'baseline1_run1.pt') and it exists in your saved_models folder.")
        sys.exit(1)
        
    model.to(device)
    model.eval()

    # ── 3. Inference Loop ──
    y_true_list = []
    y_pred_list = []
    y_score_list = []

    print(f"\nEvaluating on {device}...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            output = model(data)
            
            # Get probabilities (scores) using Softmax
            probs = F.softmax(output, dim=1)
            # Get absolute predictions
            preds = output.argmax(dim=1)
            
            y_true_list.extend(target.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            y_score_list.extend(probs.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_score = np.array(y_score_list)

    # ── 4. Generate All Plots ──
    print("\nGenerating plots...")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, baseline=baseline_name)
    
    # Classification Report Heatmap
    plot_classification_report(y_true, y_pred, class_names, baseline=baseline_name)
    
    # Precision-Recall AUC curves per class
    plot_precision_recall_auc(y_true, y_score, class_names, baseline=baseline_name)
    
    # mAP & F1 Grouped Bar Chart
    plot_map_f1(y_true, y_pred, y_score, class_names, baseline=baseline_name)

    print(f"\n✓ Done! All metrics have been plotted and saved to the 'plots/{baseline_name}' folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved model and plot all metrics.")
    parser.add_argument("--model", type=str, required=True, 
                        help="Exact filename of the saved model (e.g., 'baseline1_run1.pt')")
    parser.add_argument("--baseline", type=str, default="baseline1", 
                        help="Folder name to save plots into (defaults to 'baseline1')")
    args = parser.parse_args()
    
    evaluate(args.model, args.baseline)
