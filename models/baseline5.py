"""
Baseline 5 — Two-stage temporal person-level classification.

Stage A (person-action temporal pretraining, 9 classes):
    Each player's 9-crop sequence goes through a frozen person-feature
    extractor (Baseline 3's Stage-A backbone) into a shared LSTM; the
    LSTM's final hidden state is classified into the 9 person actions.
    Only the LSTM + action head train.

Stage B (group-activity fine-tune, 8 classes):
    Load Stage A's best LSTM, freeze it. Each clip's per-player LSTM
    summaries are pooled across the player dimension (masked max / mean /
    concat) and a small MLP head classifies the 8 group activities.
    Only the MLP head trains.

Uses:
    - Crop mode, ``n_frames=9`` (temporal window of per-player crops)
    - Gradient accumulation (effective batch = micro batch × accum steps)
    - Class-weighted cross-entropy in both stages
    - Config-driven via Hydra (``configs/baseline5.yaml``)
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend BEFORE any other import.
# On Kaggle, MPLBACKEND is preset to "module://matplotlib_inline.backend_inline"
# which the venv's matplotlib rejects; tensorboard pulls in TF→keras→pyplot at
# import time and would crash on that lookup.
import os
os.environ["MPLBACKEND"] = "Agg"

import gc
import json

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.labels import (
    GROUP_ACTIVITY_TO_IDX,
    NUM_GROUP_ACTIVITIES,
    NUM_PERSON_ACTIONS,
    PERSON_ACTION_TO_IDX,
)
from configs.path_config import LOGS_DIR
from src.data.kaggle_data_loader import (
    VolleyballDataset,
    collate_fn,
    free_annotation_cache,
)
from src.pickle_dump import free_master_data_cache
from utils.featureExtractor import FeatureExtractor
from utils.load_model_config import build_scheduler, build_transforms
from utils.utility import (
    get_device,
    group_activity_label_counts,
    inverse_freq_weights,
    load_model,
    log_experiment_summary,
    person_action_label_counts,
    save_model,
    test_one_epoch,
    train_one_epoch,
    validate_one_epoch,
)

# ═════════════════════════════════════════════════════════════════════════════
# ══ 1. MODEL CLASSES ══
# ═════════════════════════════════════════════════════════════════════════════


class PersonTemporalLSTM(nn.Module):
    """
    Stage A: frozen person features per crop → shared LSTM → 9-class action.

    Consumes one crop SEQUENCE per player: ``(N, T, C, H, W)`` where N is
    a flat batch of players (padded slots are filtered out by the Stage-A
    unpacker before they reach the model).

    Parameters
    ----------
    num_actions : int
        Number of person-action classes (9).
    backbone_name : str
        Feature-extractor backbone ("resnet50" or "resnet101").
    checkpoint : str or None
        Project checkpoint for the backbone (Baseline 3's Stage-A
        person-action backbone). ``None`` → ImageNet weights.
    lstm_hidden : int
        LSTM hidden size.
    lstm_layers : int
        Number of stacked LSTM layers (dropout applies between layers).
    dropout : float
        Dropout on the frozen features and the LSTM summary.

    """

    def __init__(
        self,
        num_actions: int = NUM_PERSON_ACTIONS,
        backbone_name: str = "resnet50",
        checkpoint: str | None = None,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Frozen — stays in eval mode and produces no-grad features.
        self.extractor = FeatureExtractor(model_name=backbone_name, checkpoint=checkpoint)
        self.feature_dropout = nn.Dropout(p=dropout)

        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(
            input_size=self.extractor.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.action_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden, lstm_hidden//2),
            nn.LayerNorm(lstm_hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden//2, num_actions),
        )


    def forward_summaries(self, seqs: torch.Tensor) -> torch.Tensor:
        """``(N, T, C, H, W)`` player sequences → ``(N, lstm_hidden)`` summaries."""
        N, T, C, H, W = seqs.shape
        
        feats = self.extractor(seqs.reshape(N * T, C, H, W))    # (N·T, D)
        feats = self.feature_dropout(feats).view(N, T, -1)      # (N, T, D)
        
        _, (h_n, _) = self.lstm(feats)
        return h_n[-1]                                          # (N, H)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """``(N, T, C, H, W)`` → ``(N, num_actions)`` logits."""
        return self.action_head(self.forward_summaries(seqs))


class GroupTemporalClassifier(nn.Module):
    """
    Stage B: frozen Stage-A person LSTM → masked pool across players → MLP → 8.

    Parameters
    ----------
    person_model : PersonTemporalLSTM
        Already-trained Stage A model; frozen entirely here.
    num_classes : int
        Number of group-activity classes (8).
    hidden_dim : int
        Width of the MLP head's first hidden layer.
    dropout : float
        Dropout inside the MLP head.
    pool : {"max", "mean", "concat"}
        Aggregation across players; "concat" doubles the classifier input.

    """

    _VALID_POOLS = ("max", "mean", "concat")

    def __init__(
        self,
        person_model: PersonTemporalLSTM,
        num_classes: int = NUM_GROUP_ACTIVITIES,
        hidden_dim: int = 512,
        dropout: float = 0.4,
        pool: str = "max",
    ) -> None:
        super().__init__()

        if pool not in self._VALID_POOLS:
            raise ValueError(f"Unsupported pool '{pool}'. Use one of {self._VALID_POOLS}.")
        self.pool = pool

        self.person = person_model
        # Freeze the whole Stage-A model — only the MLP head trains.
        for p in self.person.parameters():
            p.requires_grad = False

        # LayerNorm, not BatchNorm: training uses gradient accumulation with
        # small micro-batches where batch statistics are meaningless noise.
        classifier_in = (
            2 * person_model.lstm_hidden if pool == "concat" else person_model.lstm_hidden
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def train(self, mode: bool = True):
        # Keep the frozen Stage-A model in eval mode (LSTM dropout off,
        # deterministic summaries) even when the head trains.
        super().train(mode)
        self.person.eval()
        return self

    def forward(self, crops: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        crops : (B, T, P, C, H, W) — per-player crop sequences.
        masks : (B, P) bool — True for real players, False for padded slots.
        returns : (B, num_classes) logits
        """
        B, T, P, C, H, W = crops.shape

        seqs = crops.permute(0, 2, 1, 3, 4, 5).reshape(B * P, T, C, H, W)
        with torch.no_grad():   # Stage-A model is frozen
            summaries = self.person.forward_summaries(seqs)     # (B·P, H)
        
        summaries = summaries.view(B, P, -1)                    # (B, P, H)

        # Masked pooling across players — padded slots must not contribute
        mask_3d = masks.unsqueeze(-1).expand_as(summaries)

        if self.pool in ("max", "concat"):
            pooled_max = summaries.masked_fill(~mask_3d, float("-inf")).max(dim=1)[0]
            pooled_max = torch.where(
                torch.isinf(pooled_max), torch.zeros_like(pooled_max), pooled_max,
            )
        if self.pool in ("mean", "concat"):
            valid = masks.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
            pooled_mean = summaries.masked_fill(~mask_3d, 0.0).sum(dim=1) / valid

        if self.pool == "max":
            team = pooled_max
        elif self.pool == "mean":
            team = pooled_mean
        else:
            team = torch.cat([pooled_max, pooled_mean], dim=-1)

        return self.classifier(team)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. BATCH UNPACKERS ══
# ═════════════════════════════════════════════════════════════════════════════
#
# Crop-mode collate yields ``(crops, person_labels, group_labels, masks)``.
# Stage A flattens (B, P) player sequences and drops padded slots via the
# mask; Stage B keeps crops+masks together and targets the group labels.


def stage_a_unpack(batch):
    """4-tuple → ``((player_sequences,), person_action_labels)`` for valid players."""
    if not batch or len(batch) < 4:
        return None
    crops, person_labels, _group_labels, masks = batch
    if crops.dim() != 6 or crops.numel() == 0:
        return None

    B, T, P = crops.shape[:3]
    seqs = crops.permute(0, 2, 1, 3, 4, 5).reshape(B * P, T, *crops.shape[3:])
    labels = person_labels.reshape(B * P)
    flat_mask = masks.reshape(B * P)

    valid = flat_mask.nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return None
    return (seqs[valid],), labels[valid]


def temporal_crop_unpack(batch):
    """4-tuple → ``((crops, masks), group_labels)`` (Stage B / evaluation)."""
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
    # run_count = 2 
    run_id = f"run{run_count}"
    writer = SummaryWriter(log_dir=run_log_dir / "tensorboard" / run_id)
    metrics_history: list[dict] = []

    stage_a_ckpt = f"baseline5_stage_a_{run_id}.pt"
    stage_b_ckpt = f"baseline5_stage_b_{run_id}.pt"

    # ── Gradient accumulation ─────────────────────────────────────────────
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

    # All three datasets have copied what they need into compact records —
    # release the master annotation dict BEFORE any DataLoader workers
    # fork, so neither the main process nor the 3 × num_workers forked
    # workers carry (and gradually copy-on-write duplicate) it. This is what
    # kept Kaggle at the RAM ceiling regardless of batch/micro-batch size.
    # Both caches are covered: the disk-built one and the pickle fallback.
    free_annotation_cache()
    free_master_data_cache()
    gc.collect()

    loader_kwargs = dict(
        batch_size=micro_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    if cfg.num_workers > 0:
        # Each queued micro-batch is ~260 MB of crop tensors in shared memory.
        loader_kwargs["prefetch_factor"] = cfg.get("prefetch_factor", 1)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_dp = n_gpus > 1 and cfg.get("data_parallel", True)

    # ═════════════════════════════════════════════════════════════════════
    # STAGE A — person-action temporal pretraining (9 classes)
    # ═════════════════════════════════════════════════════════════════════
    stage_a_cfg = cfg.stage_a
    print(f"\n{'='*60}")
    print(f"  STAGE A: Person-Action Temporal LSTM ({stage_a_cfg.num_epochs} epochs, lr={stage_a_cfg.learning_rate})")
    print(f"  Backbone: {cfg.model.name} (frozen, checkpoint={cfg.model.get('checkpoint')})")
    print(f"  Batch: effective {effective_batch} = micro {micro_batch} × {accum_steps} accumulation steps")
    print(f"  Target: {NUM_PERSON_ACTIONS} classes — {list(PERSON_ACTION_TO_IDX.keys())}")
    print(f"{'='*60}")

    person_model = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=cfg.model.get("checkpoint"),
        lstm_hidden=cfg.lstm.hidden_dim,
        lstm_layers=cfg.lstm.num_layers,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    person_inner = person_model

    if use_dp:
        print(f"  DataParallel across {n_gpus} GPUs")
        person_model = nn.DataParallel(person_model)

    if stage_a_cfg.get("class_weighted_loss", True):
        counts = person_action_label_counts(train_dataset.samples, NUM_PERSON_ACTIONS)
        cw = inverse_freq_weights(counts, NUM_PERSON_ACTIONS)
        idx_to_name = {v: k for k, v in PERSON_ACTION_TO_IDX.items()}
        print("  Per-class person-action stats (count → weight):")
        for i in range(NUM_PERSON_ACTIONS):
            print(f"    {idx_to_name[i]:<10s}  n={int(counts[i]):>6d}  w={float(cw[i]):.3f}")
        criterion_a = nn.CrossEntropyLoss(
            weight=cw.to(device), label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        criterion_a = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    trainable_a = [p for p in person_model.parameters() if p.requires_grad]
    optimizer_a = optim.SGD(
        trainable_a,
        lr=stage_a_cfg.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=stage_a_cfg.get("weight_decay", 5e-4),
    )
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_a):,}")

    best_f1_a = 0.0
    patience_a = stage_a_cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0
    global_epoch = 0

    for epoch in range(stage_a_cfg.num_epochs):
        global_epoch += 1
        print(f"\n--- Stage A · Epoch {epoch + 1}/{stage_a_cfg.num_epochs} ---")

        train_loss, train_acc, train_f1, _ = train_one_epoch(
            person_model, train_loader, criterion_a, optimizer_a, device,
            batch_unpack=stage_a_unpack,
            num_classes=NUM_PERSON_ACTIONS,
            accumulate_grad_batches=accum_steps,
            desc="Train[B5-A]",
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            person_model, val_loader, criterion_a, device,
            batch_unpack=stage_a_unpack,
            num_classes=NUM_PERSON_ACTIONS,
            desc="Val[B5-A]",
        )

        writer.add_scalar("StageA/Loss/train", train_loss, global_epoch)
        writer.add_scalar("StageA/Loss/val", val_loss, global_epoch)
        writer.add_scalar("StageA/F1/train", train_f1, global_epoch)
        writer.add_scalar("StageA/F1/val", val_f1, global_epoch)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        metrics_history.append({
            "epoch": global_epoch, "stage": "A",
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "learning_rate": stage_a_cfg.learning_rate,
        })
        with (run_log_dir / f"{run_id}.json").open("w") as f:
            json.dump({"epochs": metrics_history}, f, indent=4)

        if val_f1 > best_f1_a:
            best_f1_a = val_f1
            epochs_without_improvement = 0
            save_model(stage_a_ckpt, global_epoch, person_inner, optimizer_a, val_loss,
                       class_to_idx=PERSON_ACTION_TO_IDX)
            print(f"  ✓ New best Stage-A model saved (person F1: {best_f1_a:.4f})")
        else:
            epochs_without_improvement += 1
            if patience_a > 0 and epochs_without_improvement >= patience_a:
                print(f"  ⏹ Stage-A early stopping — no improvement for {patience_a} epochs.")
                break

    # ═════════════════════════════════════════════════════════════════════
    # STAGE B — group-activity head (8 classes) on the frozen Stage-A LSTM
    # ═════════════════════════════════════════════════════════════════════
    stage_b_cfg = cfg.stage_b
    print(f"\n{'='*60}")
    print(f"  STAGE B: Group-Activity Head ({stage_b_cfg.num_epochs} epochs, lr={stage_b_cfg.learning_rate})")
    print(f"  Player pool: {stage_b_cfg.get('pool', 'max')}")
    print(f"  Target: {NUM_GROUP_ACTIVITIES} classes — {list(GROUP_ACTIVITY_TO_IDX.keys())}")
    print(f"{'='*60}")

    # Reload best Stage-A weights so B always starts from the best checkpoint.
    reloaded = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=None,
        lstm_hidden=cfg.lstm.hidden_dim,
        lstm_layers=cfg.lstm.num_layers,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    reloaded, _, _, _, _ = load_model(stage_a_ckpt, reloaded)

    group_model = GroupTemporalClassifier(
        person_model=reloaded,
        num_classes=NUM_GROUP_ACTIVITIES,
        hidden_dim=stage_b_cfg.get("hidden_dim", 256),
        dropout=stage_b_cfg.get("dropout", 0.3),
        pool=stage_b_cfg.get("pool", "max"),
    ).to(device)
    group_inner = group_model

    if use_dp:
        print(f"  DataParallel across {n_gpus} GPUs")
        group_model = nn.DataParallel(group_model)

    if stage_b_cfg.get("class_weighted_loss", True):
        counts_b = group_activity_label_counts(train_dataset.samples, NUM_GROUP_ACTIVITIES)
        cw_b = inverse_freq_weights(counts_b, NUM_GROUP_ACTIVITIES)
        idx_to_name = {v: k for k, v in GROUP_ACTIVITY_TO_IDX.items()}
        print("  Per-class group-activity stats (count → weight):")
        for i in range(NUM_GROUP_ACTIVITIES):
            print(f"    {idx_to_name[i]:<12s}  n={int(counts_b[i]):>5d}  w={float(cw_b[i]):.3f}")
        criterion_b = nn.CrossEntropyLoss(
            weight=cw_b.to(device), label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        criterion_b = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    trainable_b = [p for p in group_model.parameters() if p.requires_grad]
    optimizer_b = optim.SGD(
        trainable_b,
        lr=stage_b_cfg.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=stage_b_cfg.get("weight_decay", 5e-4),
    )
    scheduler_b = build_scheduler(optimizer_b, cfg)
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_b):,}")

    best_f1_b = 0.0
    patience_b = stage_b_cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    for epoch in range(stage_b_cfg.num_epochs):
        global_epoch += 1
        print(f"\n--- Stage B · Epoch {epoch + 1}/{stage_b_cfg.num_epochs} ---")

        train_loss, train_acc, train_f1, _ = train_one_epoch(
            group_model, train_loader, criterion_b, optimizer_b, device,
            batch_unpack=temporal_crop_unpack,
            num_classes=NUM_GROUP_ACTIVITIES,
            accumulate_grad_batches=accum_steps,
            desc="Train[B5-B]",
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            group_model, val_loader, criterion_b, device,
            batch_unpack=temporal_crop_unpack,
            num_classes=NUM_GROUP_ACTIVITIES,
            desc="Val[B5-B]",
        )

        if scheduler_b:
            scheduler_b.step()
            writer.add_scalar("StageB/LR", scheduler_b.get_last_lr()[0], global_epoch)

        writer.add_scalar("StageB/Loss/train", train_loss, global_epoch)
        writer.add_scalar("StageB/Loss/val", val_loss, global_epoch)
        writer.add_scalar("StageB/F1/train", train_f1, global_epoch)
        writer.add_scalar("StageB/F1/val", val_f1, global_epoch)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        entry = {
            "epoch": global_epoch, "stage": "B",
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "learning_rate": scheduler_b.get_last_lr()[0] if scheduler_b else stage_b_cfg.learning_rate,
        }
        metrics_history.append(entry)
        with (run_log_dir / f"{run_id}.json").open("w") as f:
            json.dump({"epochs": metrics_history}, f, indent=4)

        if val_f1 > best_f1_b:
            best_f1_b = val_f1
            epochs_without_improvement = 0
            save_model(stage_b_ckpt, global_epoch, group_inner, optimizer_b, val_loss,
                       class_to_idx=GROUP_ACTIVITY_TO_IDX)
            print(f"  ✓ New best Stage-B model saved (group F1: {best_f1_b:.4f})")
        else:
            epochs_without_improvement += 1
            if patience_b > 0 and epochs_without_improvement >= patience_b:
                print(f"  ⏹ Stage-B early stopping — no improvement for {patience_b} epochs.")
                break

    # ── Test best Stage-B model ──────────────────────────────────────────
    print("\n--- Testing best Stage-B model ---")
    fresh_person = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=None,
        lstm_hidden=cfg.lstm.hidden_dim,
        lstm_layers=cfg.lstm.num_layers,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    best_group = GroupTemporalClassifier(
        person_model=fresh_person,
        num_classes=NUM_GROUP_ACTIVITIES,
        hidden_dim=stage_b_cfg.get("hidden_dim", 256),
        dropout=stage_b_cfg.get("dropout", 0.3),
        pool=stage_b_cfg.get("pool", "max"),
    ).to(device)
    best_group, _, _, _, _ = load_model(stage_b_ckpt, best_group)
    if use_dp:
        best_group = nn.DataParallel(best_group)

    test_loss, test_acc, test_f1, _ = test_one_epoch(
        best_group, test_loader, criterion_b, device,
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
            "stage_a_epochs":          stage_a_cfg.num_epochs,
            "stage_a_lr":              stage_a_cfg.learning_rate,
            "stage_a_patience":        stage_a_cfg.get("early_stopping_patience", 0),
            "stage_b_epochs":          stage_b_cfg.num_epochs,
            "stage_b_lr":              stage_b_cfg.learning_rate,
            "stage_b_hidden_dim":      stage_b_cfg.get("hidden_dim", 256),
            "stage_b_pool":            stage_b_cfg.get("pool", "max"),
            "stage_b_patience":        stage_b_cfg.get("early_stopping_patience", 0),
            "lstm_hidden":             cfg.lstm.hidden_dim,
            "lstm_layers":             cfg.lstm.num_layers,
            "dropout":                 float(cfg.get("dropout", 0.0)),
            "label_smoothing":         cfg.get("label_smoothing", 0.0),
            "scheduler":               cfg.lr_scheduler.name if cfg.get("lr_scheduler") else "none",
            "backbone":                cfg.model.name,
            "backbone_checkpoint":     str(cfg.model.get("checkpoint")),
            "best_stage_a_val_f1":     best_f1_a,
        },
        test_f1=test_f1,
        test_acc=test_acc,
        test_loss=test_loss,
        best_val_f1=best_f1_b,
    )

    writer.close()


if __name__ == "__main__":
    train_test()
