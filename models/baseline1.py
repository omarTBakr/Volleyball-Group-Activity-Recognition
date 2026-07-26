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

# Force a non-interactive matplotlib backend BEFORE any other import.
# On Kaggle, MPLBACKEND is preset to "module://matplotlib_inline.backend_inline"
# which the venv's matplotlib rejects; tensorboard pulls in TF→keras→pyplot at
# import time and would crash on that lookup.
import os
os.environ["MPLBACKEND"] = "Agg"

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from configs.labels import (
    GROUP_ACTIVITY_TO_IDX,
    NUM_GROUP_ACTIVITIES,
)
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from utils.load_model_config import build_scheduler, build_transforms
from utils.trainer import Trainer
from utils.utility import (
    get_device,
    load_model,
)

# ═════════════════════════════════════════════════════════════════════════════
# ══ 1. MODEL CLASS ══
# ═════════════════════════════════════════════════════════════════════════════


class Model(nn.Module):
    """ResNet wrapper for single-frame classification with optional head dropout."""

    def __init__(
        self,
        num_classes: int = NUM_GROUP_ACTIVITIES,
        backbone_name: str = "resnet50",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet101":
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    run_id = Trainer.next_run_id("baseline1")
    trainer = Trainer("baseline1", run_id, device=device)
    ckpt_name = f"baseline1_{run_id}.pt"

    # Class metadata comes from the labels module, not from the YAML.
    num_classes = NUM_GROUP_ACTIVITIES

 
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
    model = Model(
        num_classes=num_classes,
        backbone_name=cfg.model.name,
        dropout=cfg.model.get("dropout", 0.0),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    # ═════════════════════════════════════════════════════════════════════
    # Stage 1: Linear probe — freeze backbone, train head only
    # ═════════════════════════════════════════════════════════════════════
    warmup_epochs = cfg.get("warmup_epochs", 5)
    warmup_lr = cfg.get("warmup_lr", 1e-3)

    # Freeze entire backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    # Keep BatchNorm layers whose weights are frozen in eval mode, so their
    # running stats don't drift even when train_one_epoch flips model.train().
    # Stage 1: all backbone BN frozen → all eval. Stage 2: only conv1/bn1/layer1/
    # layer2 BN are frozen → those eval, layer3/layer4 BN train normally.
    _orig_train = model.train

    def _train_with_frozen_bn(mode: bool = True):
        _orig_train(mode)
        if mode:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                    m.eval()
        return model

    model.train = _train_with_frozen_bn

    # optimizer_s1 = optim.SGD(
    #     model.backbone.fc.parameters(),
    #     lr=warmup_lr,
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=cfg.get("weight_decay", 5e-4),
    # )
    # AdamW equivalent — no momentum/nesterov (betas default (0.9, 0.999)); use
    # an AdamW-scale warmup_lr from the config (1e-3 head probe).
    optimizer_s1 = optim.AdamW(
        model.backbone.fc.parameters(),
        lr=warmup_lr,
        weight_decay=cfg.get("weight_decay", 5e-4),
    )

    # Probe: no scheduler and no early stopping (runs all warmup epochs),
    # matching the original Stage-1 behavior.
    trainer.run_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s1,
        num_classes=num_classes,
        num_epochs=warmup_epochs,
        checkpoint_name=ckpt_name,
        class_to_idx=GROUP_ACTIVITY_TO_IDX,
        stage="probe",
        desc="B1-probe",
    )

    # ═════════════════════════════════════════════════════════════════════
    # Stage 2: Full fine-tune — unfreeze backbone, differential LR
    # ═════════════════════════════════════════════════════════════════════
    finetune_epochs = cfg.num_epochs
    head_mult = cfg.get("head_lr_multiplier", 3)

    # Partial unfreeze: keep conv1/bn1/layer1/layer2 frozen (generic low-level
    # features), only train layer3, layer4, and the head. Restricts the surface
    # area available for per-video memorization.
    for param in model.backbone.parameters():
        param.requires_grad = False
    for name in ("layer3", "layer4", "fc"):
        for param in getattr(model.backbone, name).parameters():
            param.requires_grad = True

    # The frozen-BN train() override from Stage 1 still applies — keeps the
    # still-frozen early-layer BN stats from drifting.

    backbone_params = [
        p for n, p in model.named_parameters()
        if "fc" not in n and p.requires_grad
    ]
    head_params = list(model.backbone.fc.parameters())

    # optimizer_s2 = optim.SGD(
    #     [
    #         {"params": backbone_params, "lr": cfg.learning_rate},
    #         {"params": head_params, "lr": cfg.learning_rate * head_mult},
    #     ],
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=cfg.get("weight_decay", 5e-4),
    # )
    # AdamW equivalent — differential LR groups preserved; AdamW-scale backbone
    # lr from the config (1e-4 fine-tune), head at ×head_mult.
    optimizer_s2 = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.learning_rate},
            {"params": head_params, "lr": cfg.learning_rate * head_mult},
        ],
        weight_decay=cfg.get("weight_decay", 5e-4),
    )
    scheduler = build_scheduler(optimizer_s2, cfg)

    # Fine-tune shares the checkpoint (and thus the best-F1) with the probe:
    # it only overwrites when it beats the probe's best.
    best_f1 = trainer.run_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s2,
        scheduler=scheduler,
        num_classes=num_classes,
        num_epochs=finetune_epochs,
        checkpoint_name=ckpt_name,
        class_to_idx=GROUP_ACTIVITY_TO_IDX,
        patience=cfg.get("early_stopping_patience", 0),
        stage="finetune",
        desc="B1-finetune",
    )

    # ── Test Best Model ──────────────────────────────────────────────────
    best_model = Model(
        num_classes=num_classes,
        backbone_name=cfg.model.name,
        dropout=cfg.model.get("dropout", 0.0),
    )
    best_model, _, _, _, _ = load_model(ckpt_name, best_model)
    best_model.to(device)

    trainer.run_test(
        model=best_model,
        test_loader=test_loader,
        criterion=criterion,
        num_classes=num_classes,
        best_val_f1=best_f1,
        hparam_dict={
            "baseline":                "baseline1",
            "batch_size":              cfg.batch_size,
            "warmup_epochs":           cfg.warmup_epochs,
            "warmup_lr":               cfg.warmup_lr,
            "num_epochs":              cfg.num_epochs,
            "learning_rate":           cfg.learning_rate,
            "weight_decay":            cfg.weight_decay,
            "head_lr_multiplier":      cfg.get("head_lr_multiplier", 1),
            "label_smoothing":         cfg.get("label_smoothing", 0.0),
            "early_stopping_patience": cfg.get("early_stopping_patience", 0),
            "scheduler":               cfg.lr_scheduler.name if cfg.get("lr_scheduler") else "none",
            "backbone":                cfg.model.name,
            "dropout":                 float(cfg.model.get("dropout", 0.0)),
        },
    )

    trainer.close()


if __name__ == "__main__":
    train_test()
