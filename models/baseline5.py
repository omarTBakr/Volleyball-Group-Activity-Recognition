"""
Baseline 5 — Temporal person-level classification.

For each clip, every player's 9-crop sequence is pushed through a frozen
person-feature extractor (Baseline 3's Stage-A backbone, trained on the
9 person actions), producing a (T, 2048) feature sequence per player.
One LSTM — shared across players — summarizes each player's sequence into
its final hidden state; the player summaries are then pooled across the
player dimension (masked, so padded slots can't contribute) and a small
MLP head classifies the 8 group activities.

Uses:
    - Crop mode, ``n_frames=9`` (temporal window of per-player crops)
    - Frozen backbone → only the LSTM + head train
    - Masked player pooling (max / mean / concat, from ``cfg.pool``)
    - Class-weighted cross-entropy (rare ``l/r_winpoint`` classes)
    - Config-driven via Hydra (``configs/baseline5.yaml``)
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


class TemporalPersonClassifier(nn.Module):
    """
    Frozen person features per crop → shared LSTM per player → masked pool
    across players → 8-class group activity.

    Parameters
    ----------
    num_classes : int
        Number of group-activity classes (8).
    backbone_name : str
        Feature-extractor backbone ("resnet50" or "resnet101").
    checkpoint : str or None
        Project checkpoint to load the backbone from (Baseline 3's Stage-A
        person-action backbone). ``None`` → ImageNet weights.
    lstm_hidden : int
        LSTM hidden size.
    lstm_layers : int
        Number of stacked LSTM layers (dropout applies between layers).
    dropout : float
        Dropout on the frozen features, the player summaries, and inside
        the MLP head.
    pool : {"max", "mean", "concat"}
        Aggregation across the player dimension. ``"concat"`` doubles the
        classifier input width (max ‖ mean).

    """

    _VALID_POOLS = ("max", "mean", "concat")

    def __init__(
        self,
        num_classes: int = NUM_GROUP_ACTIVITIES,
        backbone_name: str = "resnet50",
        checkpoint: str | None = None,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        pool: str = "max",
    ) -> None:
        super().__init__()

        if pool not in self._VALID_POOLS:
            raise ValueError(f"Unsupported pool '{pool}'. Use one of {self._VALID_POOLS}.")
        self.pool = pool

        # Frozen — stays in eval mode and produces no-grad features.
        self.extractor = FeatureExtractor(model_name=backbone_name, checkpoint=checkpoint)
        self.feature_dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=self.extractor.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # LayerNorm, not BatchNorm: training uses gradient accumulation with
        # small micro-batches (4 clips, 2 per GPU under DataParallel), where
        # batch statistics are meaningless noise. LayerNorm normalizes per
        # sample and is batch-size independent.
        classifier_in = 2 * lstm_hidden if pool == "concat" else lstm_hidden
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(128, num_classes),
        )

    def forward(self, crops: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        crops : (B, T, P, C, H, W) — per-player crop sequences, already transformed.
        masks : (B, P) bool — True for real players, False for padded slots.
        returns : (B, num_classes) logits
        """
        B, T, P, C, H, W = crops.shape
        D = self.extractor.feature_dim

        # 1. Spatial: all crops through the frozen CNN in one pass
        feats = self.extractor(crops.view(B * T * P, C, H, W))   # (B·T·P, D)
        feats = self.feature_dropout(feats)

        # 2. One temporal sequence per player: (B, T, P, D) → (B·P, T, D)
        feats = feats.view(B, T, P, D).permute(0, 2, 1, 3)       # (B, P, T, D)
        feats = feats.reshape(B * P, T, D)

        # 3. Shared LSTM over each player's frame sequence
        _, (h_n, _) = self.lstm(feats)
        player_summaries = h_n[-1].view(B, P, -1)                # (B, P, H)
        player_summaries = self.feature_dropout(player_summaries)

        # 4. Masked pooling across players — padded slots must not contribute
        mask_3d = masks.unsqueeze(-1).expand_as(player_summaries)

        if self.pool in ("max", "concat"):
            pooled_max = player_summaries.masked_fill(~mask_3d, float("-inf")).max(dim=1)[0]
            # Clips with zero valid players → all -inf; sanitize to 0.
            pooled_max = torch.where(
                torch.isinf(pooled_max), torch.zeros_like(pooled_max), pooled_max,
            )
        if self.pool in ("mean", "concat"):
            valid = masks.sum(dim=1).clamp_min(1).unsqueeze(-1).float()      # (B, 1)
            pooled_mean = player_summaries.masked_fill(~mask_3d, 0.0).sum(dim=1) / valid

        if self.pool == "max":
            team_summary = pooled_max
        elif self.pool == "mean":
            team_summary = pooled_mean
        else:
            team_summary = torch.cat([pooled_max, pooled_mean], dim=-1)      # (B, 2H)

        # 5. Classify
        return self.classifier(team_summary)                     # (B, num_classes)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. BATCH UNPACKER ══
# ═════════════════════════════════════════════════════════════════════════════
#
# Crop-mode collate returns 4-tuples ``(crops, person_labels, group_labels,
# masks)``. B5 consumes crops + masks and targets the GROUP labels — the
# per-person labels are ignored (they are unreliable on detection-fallback
# frames anyway).


def temporal_crop_unpack(batch):
    """``(crops, person_labels, group_labels, masks)`` → ``((crops, masks), group_labels)``."""
    if not batch or len(batch) < 4:
        return None
    crops, _person_labels, group_labels, masks = batch
    if crops.dim() != 6 or crops.numel() == 0:
        return None
    return (crops, masks), group_labels


# ═════════════════════════════════════════════════════════════════════════════
# ══ 3. MAIN ENTRYPOINT ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline5", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Logging ──────────────────────────────────────────────────────────
    run_log_dir = LOGS_DIR / "baseline5"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_count = len(list(run_log_dir.glob("*.json"))) + 1
    run_id = f"run{run_count}"
    writer = SummaryWriter(log_dir=run_log_dir / "tensorboard" / run_id)
    metrics_history: list[dict] = []

    ckpt_name = f"baseline5_{run_id}.pt"

    # ── Gradient accumulation ─────────────────────────────────────────────
    # cfg.batch_size is the EFFECTIVE batch; the loader carries only
    # micro_batch_size clips at a time and gradients accumulate between
    # optimizer steps — keeps per-step RAM/VRAM at micro-batch level.
    effective_batch = cfg.batch_size
    micro_batch = cfg.get("micro_batch_size", effective_batch)
    if effective_batch % micro_batch != 0:
        raise ValueError(
            f"batch_size ({effective_batch}) must be divisible by "
            f"micro_batch_size ({micro_batch}).",
        )
    accum_steps = effective_batch // micro_batch

    # ── Data (crop mode, 9-frame temporal window of player crops) ────────
    tf = build_transforms(cfg)

    train_dataset = VolleyballDataset(
        mode="train", full_image=False, crop=True, n_frames=cfg.n_frames, transform=tf["train"],
    )
    val_dataset = VolleyballDataset(
        mode="validation", full_image=False, crop=True, n_frames=cfg.n_frames, transform=tf["validation"],
    )
    test_dataset = VolleyballDataset(
        mode="test", full_image=False, crop=True, n_frames=cfg.n_frames, transform=tf["test"],
    )

    loader_kwargs = dict(
        batch_size=micro_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    if cfg.num_workers > 0:
        # Each queued B5 micro-batch is ~260 MB of crop tensors in shared
        # memory; keep only one per worker in flight instead of the default 2.
        loader_kwargs["prefetch_factor"] = cfg.get("prefetch_factor", 1)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BASELINE 5: Temporal Person Classifier ({cfg.num_epochs} epochs, lr={cfg.learning_rate})")
    print(f"  Backbone: {cfg.model.name} (frozen, checkpoint={cfg.model.get('checkpoint')})")
    print(f"  Player pool: {cfg.get('pool', 'max')}")
    print(f"  Batch: effective {effective_batch} = micro {micro_batch} × {accum_steps} accumulation steps")
    print(f"  Target: {NUM_GROUP_ACTIVITIES} classes — {list(GROUP_ACTIVITY_TO_IDX.keys())}")
    print(f"{'='*60}")

    model_kwargs = dict(
        num_classes=NUM_GROUP_ACTIVITIES,
        backbone_name=cfg.model.name,
        checkpoint=cfg.model.get("checkpoint"),
        lstm_hidden=cfg.lstm.hidden_dim,
        lstm_layers=cfg.lstm.num_layers,
        dropout=cfg.get("dropout", 0.3),
        pool=cfg.get("pool", "max"),
    )
    model = TemporalPersonClassifier(**model_kwargs).to(device)

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
            batch_unpack=temporal_crop_unpack,
            num_classes=NUM_GROUP_ACTIVITIES,
            accumulate_grad_batches=accum_steps,
            desc="Train[B5]",
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            model, val_loader, criterion, device,
            batch_unpack=temporal_crop_unpack,
            num_classes=NUM_GROUP_ACTIVITIES,
            desc="Val[B5]",
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
    best_model = TemporalPersonClassifier(**model_kwargs).to(device)
    best_model, _, _, _, _ = load_model(ckpt_name, best_model)
    if use_dp:
        best_model = nn.DataParallel(best_model)

    test_loss, test_acc, test_f1, _ = test_one_epoch(
        best_model, test_loader, criterion, device,
        batch_unpack=temporal_crop_unpack,
        num_classes=NUM_GROUP_ACTIVITIES,
        desc="Test[B5]",
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
            "baseline":                "baseline5",
            "batch_size":              effective_batch,
            "micro_batch_size":        micro_batch,
            "accumulation_steps":      accum_steps,
            "n_frames":                cfg.n_frames,
            "num_epochs":              cfg.num_epochs,
            "learning_rate":           cfg.learning_rate,
            "weight_decay":            cfg.get("weight_decay", 0.0),
            "lstm_hidden":             cfg.lstm.hidden_dim,
            "lstm_layers":             cfg.lstm.num_layers,
            "pool":                    cfg.get("pool", "max"),
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
