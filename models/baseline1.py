"""
Baseline 1 — Single-frame group-activity classification.

Fine-tunes a ResNet-50 on the middle frame of each clip to predict
the group activity (one of 8 scene-level classes).

Uses:
    - Full image mode, ``n_frames=1`` (middle frame only)
    - Standard cross-entropy loss
    - Config-driven via Hydra (``configs/baseline1.yaml``)

Class names and ``num_classes`` are sourced from
:mod:`configs.labels` so that the YAML only holds hyper-parameters.
"""

from __future__ import annotations

import json
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from configs.labels import (
    GROUP_ACTIVITY_TO_IDX,
    IDX_TO_GROUP_ACTIVITY,
    NUM_GROUP_ACTIVITIES,
)
from configs.path_config import LOGS_DIR
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from utils.load_model_config import build_model, build_scheduler, build_transforms
from utils.utility import (
    get_device,
    load_model,
    save_model,
    test_one_epoch,
    train_one_epoch,
    validate_one_epoch,
)

# ═════════════════════════════════════════════════════════════════════════════
# ══ 1. MODEL CLASS ══
# ═════════════════════════════════════════════════════════════════════════════


class Model(nn.Module):
    """Simple ResNet-50 wrapper for single-frame classification."""

    def __init__(self, num_classes: int = NUM_GROUP_ACTIVITIES) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet-50 backbone."""
        return self.backbone(x)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. MAIN TRAINING LOOP ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline1", version_base=None)
def train_test(cfg: DictConfig) -> None:
    """Run the full train → validate → test pipeline for Baseline 1."""
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Logging Setup ────────────────────────────────────────────────────
    run_log_dir = LOGS_DIR / "baseline1"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_count = len(list(run_log_dir.glob("*.json"))) + 1
    run_id = f"run{run_count}"

    writer = SummaryWriter(log_dir=run_log_dir / "tensorboard" / run_id)
    metrics_history = []

    # Class metadata comes from the labels module, not from the YAML.
    num_classes = NUM_GROUP_ACTIVITIES
    class_names = list(GROUP_ACTIVITY_TO_IDX.keys())

 
    # ── Data ─────────────────────────────────────────────────────────────
    tf = build_transforms(cfg)

    train_dataset = VolleyballDataset(
        mode="train", full_image=True, n_frames=1, transform=tf["train"],
    )
    val_dataset = VolleyballDataset(
        mode="validation", full_image=True, n_frames=1, transform=tf["validation"],
    )
    test_dataset = VolleyballDataset(
        mode="test", full_image=True, n_frames=1, transform=tf["test"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, collate_fn=collate_fn,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = Model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    best_f1 = 0.0
    patience = cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0
    global_epoch = 0

    # ═════════════════════════════════════════════════════════════════════
    # Stage 1: Linear probe — freeze backbone, train head only
    # ═════════════════════════════════════════════════════════════════════
    warmup_epochs = cfg.get("warmup_epochs", 5)
    warmup_lr = cfg.get("warmup_lr", 1e-3)
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Linear Probe ({warmup_epochs} epochs, lr={warmup_lr})")
    print(f"{'='*60}")

    # Freeze entire backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    optimizer_s1 = optim.AdamW(
        model.backbone.fc.parameters(),
        lr=warmup_lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    for epoch in range(warmup_epochs):
        global_epoch += 1
        print(f"\n--- Stage 1 · Epoch {epoch + 1}/{warmup_epochs} ---")

        train_loss, train_acc, train_f1, _ = train_one_epoch(
            model, train_loader, criterion, optimizer_s1, device,
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            model, val_loader, criterion, device,
        )

        writer.add_scalar("Loss/train", train_loss, global_epoch)
        writer.add_scalar("Loss/val", val_loss, global_epoch)
        writer.add_scalar("F1_Score/train", train_f1, global_epoch)
        writer.add_scalar("F1_Score/val", val_f1, global_epoch)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        epoch_metrics = {
            "epoch": global_epoch,
            "stage": 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "learning_rate": warmup_lr,
        }
        metrics_history.append(epoch_metrics)

        log_path = run_log_dir / f"{run_id}.json"
        with log_path.open("w") as f:
            json.dump({"epochs": metrics_history}, f, indent=4)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_model(f"baseline1_{run_id}.pt", global_epoch, model, optimizer_s1, val_loss)
            print(f"  ✓ New best model saved (F1: {best_f1:.4f})")

    # ═════════════════════════════════════════════════════════════════════
    # Stage 2: Full fine-tune — unfreeze backbone, differential LR
    # ═════════════════════════════════════════════════════════════════════
    finetune_epochs = cfg.num_epochs
    head_mult = cfg.get("head_lr_multiplier", 3)
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Full Fine-tune ({finetune_epochs} epochs)")
    print(f"  Backbone lr={cfg.learning_rate}, Head lr={cfg.learning_rate * head_mult}")
    print(f"{'='*60}")

    # Unfreeze all parameters
    for param in model.backbone.parameters():
        param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = list(model.backbone.fc.parameters())

    optimizer_s2 = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.learning_rate},
            {"params": head_params, "lr": cfg.learning_rate * head_mult},
        ],
        weight_decay=cfg.get("weight_decay", 0.05),
    )
    scheduler = build_scheduler(optimizer_s2, cfg)

    epochs_without_improvement = 0  # reset for stage 2

    for epoch in range(finetune_epochs):
        global_epoch += 1
        print(f"\n--- Stage 2 · Epoch {epoch + 1}/{finetune_epochs} ---")

        train_loss, train_acc, train_f1, _ = train_one_epoch(
            model, train_loader, criterion, optimizer_s2, device,
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            model, val_loader, criterion, device,
        )

        if scheduler:
            scheduler.step()
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], global_epoch)

        writer.add_scalar("Loss/train", train_loss, global_epoch)
        writer.add_scalar("Loss/val", val_loss, global_epoch)
        writer.add_scalar("F1_Score/train", train_f1, global_epoch)
        writer.add_scalar("F1_Score/val", val_f1, global_epoch)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # ── JSON Logging ─────────────────────────────────────────────────
        epoch_metrics = {
            "epoch": global_epoch,
            "stage": 2,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
        if scheduler:
            epoch_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        metrics_history.append(epoch_metrics)

        log_path = run_log_dir / f"{run_id}.json"
        with log_path.open("w") as f:
            json.dump({"epochs": metrics_history}, f, indent=4)

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            save_model(f"baseline1_{run_id}.pt", global_epoch, model, optimizer_s2, val_loss)
            print(f"  ✓ New best model saved (F1: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  ⏹ Early stopping — no improvement for {patience} epochs.")
                break

    writer.close()

    # ── Test Best Model ──────────────────────────────────────────────────
    print("\n--- Testing Best Model ---")
    best_model = Model(num_classes=num_classes)
    best_model, _, _, _, _ = load_model(f"baseline1_{run_id}.pt", best_model)
    best_model.to(device)

    test_loss, test_acc, test_f1, _ = test_one_epoch(
        best_model, test_loader, criterion, device,
    )
    print(f"Final Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    # Log test results
    with (run_log_dir / f"{run_id}.json").open("r+") as f:
        data = json.load(f)
        data["test"] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
        }
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


if __name__ == "__main__":
    train_test()

