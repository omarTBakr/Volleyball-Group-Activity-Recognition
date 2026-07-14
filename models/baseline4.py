"""
Baseline 4 — Temporal full-image classification.

For each clip, the 9-frame window around the middle frame is pushed through
a frozen ResNet-50 feature extractor (Baseline 1's fine-tuned backbone by
default, plain ImageNet weights as fallback), producing a (T, 2048) feature
sequence per clip. An LSTM consumes the sequence and its final hidden state
is classified into the 8 group activities by a small MLP head.

Uses:
    - Full image mode, ``n_frames=9`` (temporal window)
    - Frozen backbone → only the LSTM + head train
    - Class-weighted cross-entropy (rare ``l/r_winpoint`` classes)
    - Config-driven via Hydra (``configs/baseline4.yaml``)
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend BEFORE any other import.
# On Kaggle, MPLBACKEND is preset to "module://matplotlib_inline.backend_inline"
# which the venv's matplotlib rejects; tensorboard pulls in TF→keras→pyplot at
# import time and would crash on that lookup.
import os
os.environ["MPLBACKEND"] = "Agg"

import json

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.labels import GROUP_ACTIVITY_TO_IDX, NUM_GROUP_ACTIVITIES
from configs.path_config import LOGS_DIR
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from utils.featureExtractor import FeatureExtractor
from utils.load_model_config import build_scheduler, build_transforms
from utils.utility import (
    get_device,
    group_activity_label_counts,
    inverse_freq_weights,
    load_model,
    log_experiment_summary,
    save_model,
    test_one_epoch,
    train_one_epoch,
    validate_one_epoch,
)

# ═════════════════════════════════════════════════════════════════════════════
# ══ 1. MODEL CLASS ══
# ═════════════════════════════════════════════════════════════════════════════


class TemporalImageClassifier(nn.Module):
    """
    Frozen CNN features per frame → LSTM over time → 8-class group activity.

    Parameters
    ----------
    num_classes : int
        Number of group-activity classes (8).
    backbone_name : str
        Feature-extractor backbone ("resnet50" or "resnet101").
    checkpoint : str or None
        Project checkpoint to load the backbone from (e.g. Baseline 1's
        ``baseline1_run2.pt``). ``None`` → ImageNet weights.
    lstm_hidden : int
        LSTM hidden size.
    lstm_layers : int
        Number of stacked LSTM layers (dropout applies between layers).
    dropout : float
        Dropout inside the MLP head (and between LSTM layers if > 1).

    """

    def __init__(
        self,
        num_classes: int = NUM_GROUP_ACTIVITIES,
        backbone_name: str = "resnet50",
        checkpoint: str | None = None,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Frozen — stays in eval mode and produces no-grad features.
        self.extractor = FeatureExtractor(model_name=backbone_name, checkpoint=checkpoint)

        self.lstm = nn.LSTM(
            input_size=self.extractor.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C, H, W) — T full frames per clip, already transformed.
        returns : (B, num_classes) logits
        """
        B, T, C, H, W = x.shape

        # Spatial: all frames through the frozen CNN in one pass
        feats = self.extractor(x.view(B * T, C, H, W))   # (B*T, D)
        feats = feats.view(B, T, -1)                     # (B, T, D)

        # Temporal: LSTM over the frame sequence, classify the final state
        _, (h_n, _) = self.lstm(feats)
        return self.classifier(h_n[-1])                  # (B, num_classes)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. MAIN ENTRYPOINT ══
# ═════════════════════════════════════════════════════════════════════════════
#
# Full-image mode batches are plain ``(images, labels)`` 2-tuples, so the
# shared epoch driver's default unpacker applies — no custom batch_unpack.


@hydra.main(config_path="../configs", config_name="baseline4", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Logging ──────────────────────────────────────────────────────────
    run_log_dir = LOGS_DIR / "baseline4"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_count = len(list(run_log_dir.glob("*.json"))) + 1
    run_id = f"run{run_count}"
    writer = SummaryWriter(log_dir=run_log_dir / "tensorboard" / run_id)
    metrics_history: list[dict] = []

    ckpt_name = f"baseline4_{run_id}.pt"

    # ── Data (full-image mode, 9-frame temporal window) ──────────────────
    tf = build_transforms(cfg)

    train_dataset = VolleyballDataset(
        mode="train", full_image=True, n_frames=cfg.n_frames, transform=tf["train"],
    )
    val_dataset = VolleyballDataset(
        mode="validation", full_image=True, n_frames=cfg.n_frames, transform=tf["validation"],
    )
    test_dataset = VolleyballDataset(
        mode="test", full_image=True, n_frames=cfg.n_frames, transform=tf["test"],
    )

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BASELINE 4: Temporal Image Classifier ({cfg.num_epochs} epochs, lr={cfg.learning_rate})")
    print(f"  Backbone: {cfg.model.name} (frozen, checkpoint={cfg.model.get('checkpoint')})")
    print(f"  Target: {NUM_GROUP_ACTIVITIES} classes — {list(GROUP_ACTIVITY_TO_IDX.keys())}")
    print(f"{'='*60}")

    model = TemporalImageClassifier(
        num_classes=NUM_GROUP_ACTIVITIES,
        backbone_name=cfg.model.name,
        checkpoint=cfg.model.get("checkpoint"),
        lstm_hidden=cfg.lstm.hidden_dim,
        lstm_layers=cfg.lstm.num_layers,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    # Unwrapped reference for checkpoint I/O (DataParallel prefixes keys).
    model_inner = model

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_dp = n_gpus > 1 and cfg.get("data_parallel", True)
    if use_dp:
        print(f"  DataParallel across {n_gpus} GPUs (each sees batch_size/{n_gpus})")
        model = nn.DataParallel(model)

    # Inverse-frequency class weights: l/r_winpoint are ~2.5× rarer than the
    # spike/pass/set classes — same rationale as baseline3's Stage B.
    if cfg.get("class_weighted_loss", True):
        counts = group_activity_label_counts(train_dataset.samples, NUM_GROUP_ACTIVITIES)
        cw = inverse_freq_weights(counts, NUM_GROUP_ACTIVITIES)
        idx_to_name = {v: k for k, v in GROUP_ACTIVITY_TO_IDX.items()}
        print("  Per-class group-activity stats (count → weight):")
        for i in range(NUM_GROUP_ACTIVITIES):
            print(f"    {idx_to_name[i]:<12s}  n={int(counts[i]):>5d}  w={float(cw[i]):.3f}")
        criterion = nn.CrossEntropyLoss(
            weight=cw.to(device),
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    # Only the LSTM + head train; the frozen extractor is excluded.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        trainable_params,
        lr=cfg.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=cfg.get("weight_decay", 5e-4),
    )
    scheduler = build_scheduler(optimizer, cfg)

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable parameters: {n_trainable:,}")

    # ── Training loop ────────────────────────────────────────────────────
    best_f1 = 0.0
    patience = cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    for epoch in range(cfg.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{cfg.num_epochs} ---")

        train_loss, train_acc, train_f1, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            num_classes=NUM_GROUP_ACTIVITIES,
            desc="Train[B4]",
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            model, val_loader, criterion, device,
            num_classes=NUM_GROUP_ACTIVITIES,
            desc="Val[B4]",
        )

        if scheduler:
            scheduler.step()
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch + 1)

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("F1_Score/train", train_f1, epoch + 1)
        writer.add_scalar("F1_Score/val", val_f1, epoch + 1)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "learning_rate": scheduler.get_last_lr()[0] if scheduler else cfg.learning_rate,
        }
        metrics_history.append(entry)
        with (run_log_dir / f"{run_id}.json").open("w") as f:
            json.dump({"epochs": metrics_history}, f, indent=4)

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            save_model(ckpt_name, epoch + 1, model_inner, optimizer, val_loss,
                       class_to_idx=GROUP_ACTIVITY_TO_IDX)
            print(f"  ✓ New best model saved (F1: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  ⏹ Early stopping — no improvement for {patience} epochs.")
                break

    # ── Test best model ──────────────────────────────────────────────────
    print("\n--- Testing best model ---")
    best_model = TemporalImageClassifier(
        num_classes=NUM_GROUP_ACTIVITIES,
        backbone_name=cfg.model.name,
        checkpoint=cfg.model.get("checkpoint"),
        lstm_hidden=cfg.lstm.hidden_dim,
        lstm_layers=cfg.lstm.num_layers,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    best_model, _, _, _, _ = load_model(ckpt_name, best_model)
    if use_dp:
        best_model = nn.DataParallel(best_model)

    test_loss, test_acc, test_f1, _ = test_one_epoch(
        best_model, test_loader, criterion, device,
        num_classes=NUM_GROUP_ACTIVITIES,
        desc="Test[B4]",
    )
    print(f"Final Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    with (run_log_dir / f"{run_id}.json").open("r+") as f:
        data = json.load(f)
        data["test"] = {"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1}
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    log_experiment_summary(
        writer=writer,
        run_id=run_id,
        hparam_dict={
            "baseline":                "baseline4",
            "batch_size":              cfg.batch_size,
            "n_frames":                cfg.n_frames,
            "num_epochs":              cfg.num_epochs,
            "learning_rate":           cfg.learning_rate,
            "weight_decay":            cfg.get("weight_decay", 0.0),
            "lstm_hidden":             cfg.lstm.hidden_dim,
            "lstm_layers":             cfg.lstm.num_layers,
            "dropout":                 float(cfg.get("dropout", 0.0)),
            "label_smoothing":         cfg.get("label_smoothing", 0.0),
            "early_stopping_patience": cfg.get("early_stopping_patience", 0),
            "scheduler":               cfg.lr_scheduler.name if cfg.get("lr_scheduler") else "none",
            "backbone":                cfg.model.name,
            "backbone_checkpoint":     str(cfg.model.get("checkpoint")),
        },
        test_f1=test_f1,
        test_acc=test_acc,
        test_loss=test_loss,
        best_val_f1=best_f1,
    )

    writer.close()


if __name__ == "__main__":
    train_test()
