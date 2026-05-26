"""
Baseline 3 — Temporal group-activity classification with per-player crops.

Fine-tunes a ResNet-50 wrapped in a ``SequenceResNet`` that handles
5D temporal inputs ``[B, T, C, H, W]`` by processing frames individually
and averaging logits (late fusion).

Uses:
    - Crop mode with ``n_frames=9`` (temporal window)
    - Frame chunking to limit peak GPU memory
    - Config-driven via Hydra (``configs/baseline3.yaml``)
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from src.data.data_loader import VolleyballDataset, collate_fn
from utils.load_model_config import build_scheduler, build_transforms
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


class SequenceResNet(nn.Module):
    """
    Wraps a ResNet so it can accept either 4D ``[B, C, H, W]``
    or 5D ``[B, T, C, H, W]`` inputs.

    For 5D inputs, frames are processed individually (or in chunks)
    and the resulting logits are averaged over the time dimension
    (late fusion).

    Parameters
    ----------
    backbone : nn.Module
        A classification backbone (e.g. ResNet-50).
    frame_chunk_size : int or None
        If set, process at most this many frames per forward pass
        to limit GPU memory usage.

    """

    def __init__(self, backbone: nn.Module, frame_chunk_size: int | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.frame_chunk_size = frame_chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Process all frames at once if chunking isn't needed
            if self.frame_chunk_size is None or self.frame_chunk_size >= T:
                x = x.reshape(B * T, C, H, W)
                logits = self.backbone(x).reshape(B, T, -1)
                return logits.mean(dim=1)

            # Process in chunks to limit peak memory
            chunks = []
            for start in range(0, T, self.frame_chunk_size):
                end = min(T, start + self.frame_chunk_size)
                chunk = x[:, start:end].reshape(B * (end - start), C, H, W)
                logits_chunk = self.backbone(chunk).reshape(B, end - start, -1)
                chunks.append(logits_chunk)

            return torch.cat(chunks, dim=1).mean(dim=1)

        # Standard 4D input
        return self.backbone(x)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. MODEL BUILDER ══
# ═════════════════════════════════════════════════════════════════════════════


def build_model(cfg: DictConfig, num_classes: int) -> SequenceResNet:
    """Build a SequenceResNet from config."""
    pretrained = cfg.get("pretrained", True)
    frame_chunk_size = cfg.get("frame_chunk_size", None)

    if pretrained:
        base_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        base_resnet = models.resnet50(weights=None)

    base_resnet.fc = nn.Linear(base_resnet.fc.in_features, num_classes)
    return SequenceResNet(base_resnet, frame_chunk_size=frame_chunk_size)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 3. MAIN TRAINING LOOP ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline3", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    # ── Data ─────────────────────────────────────────────────────────────
    tf = build_transforms(cfg)

    train_dataset = VolleyballDataset(mode="train", transform=tf["train"], crop=True)
    val_dataset = VolleyballDataset(mode="validation", transform=tf["validation"], crop=True)
    test_dataset = VolleyballDataset(mode="test", transform=tf["test"], crop=True)

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

    # ── Class Mapping ────────────────────────────────────────────────────
    class_to_idx: dict[str, int] = {}
    for i, name in enumerate(cfg.class_names):
        class_to_idx[name] = i
        class_to_idx[name.replace("_", "-")] = i
        class_to_idx[name.replace("-", "_")] = i

    num_classes = len(cfg.class_names)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(cfg, num_classes=num_classes).to(device)
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
            save_model("baseline3", epoch, model, optimizer, val_loss, class_to_idx)
            print(f"  ✓ New best model saved (F1: {best_f1:.4f})")

    writer.close()

    # ── Test Best Model ──────────────────────────────────────────────────
    print("\n--- Testing Best Model ---")
    best_model = build_model(cfg, num_classes=num_classes)
    best_model, _, _, _, loaded_idx = load_model("baseline3", best_model)
    best_model.to(device)

    test_loss, test_acc, test_f1, _ = test_one_epoch(
        best_model, test_loader, criterion, device,
    )
    print(f"Final Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    train_test()
