"""
Baseline 1 — Single-frame group-activity classification.

Fine-tunes a ResNet-50 on the middle frame of each clip to predict
the group activity (one of 8 scene-level classes).

Uses:
    - Full image mode, ``n_frames=1`` (middle frame only)
    - Standard cross-entropy loss
    - Config-driven via Hydra (``configs/baseline1.yaml``)
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.labels import NUM_GROUP_ACTIVITIES
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from utils.load_model_config import build_model, build_scheduler, build_transforms
from utils.utility import (
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
        from torchvision import models

        self.num_classes = num_classes
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. MAIN TRAINING LOOP ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline1", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

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
    num_classes = len(cfg.class_names)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = build_scheduler(optimizer, cfg)

    best_f1 = 0.0

    # ── Training Loop ────────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{cfg.num_epochs} ---")

        train_loss, train_acc, train_f1, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            model, val_loader, criterion, device,
        )

        if scheduler:
            scheduler.step()
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1_Score/train", train_f1, epoch)
        writer.add_scalar("F1_Score/val", val_f1, epoch)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_model("baseline1", epoch, model, optimizer, val_loss)
            print(f"  ✓ New best model saved (F1: {best_f1:.4f})")

    writer.close()

    # ── Test Best Model ──────────────────────────────────────────────────
    print("\n--- Testing Best Model ---")
    best_model = build_model(cfg)
    best_model, _, _, _, _ = load_model("baseline1", best_model)
    best_model.to(device)

    test_loss, test_acc, test_f1, _ = test_one_epoch(
        best_model, test_loader, criterion, device,
    )
    print(f"Final Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    train_test()
