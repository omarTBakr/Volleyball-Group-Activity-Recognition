"""
Baseline 6 — Scene-level temporal model (pool players per frame → LSTM over
time) with a skip-connection and Conv1d temporal fusion.

Core flow (course B6: the LSTM is applied at the image/scene level only):
    Each frame's player crops go through a frozen person-feature extractor
    (Baseline 3's Stage-A backbone) and are masked-pooled across players
    (max / mean / concat) into one vector per frame → a (B, T, D) scene
    sequence.  An LSTM consumes it; its T hidden states are concatenated
    along the time axis with a linear projection of the pooled per-frame
    features (skip connection), giving (B, 2T, lstm_hidden).  A two-stage
    Conv1d (global kernel 2T) collapses this into a (B, lstm_hidden//4)
    clip summary.

Stage A (person-action temporal pretraining, 9 classes):
    The same machinery run with P=1 — each player's track is an independent
    "clip" (pooling is an identity), and the summary is classified into the
    9 person actions.  Only LSTM + projection + Conv1d + action head train.

Stage B (group-activity fine-tune, 8 classes):
    Load Stage A's best model, freeze it.  Full clips (all players) produce
    lstm_hidden//4 summaries; a small MLP head classifies the 8 group
    activities.  Only the MLP head trains.

Uses:
    - Crop mode, ``n_frames=9`` (temporal window of per-player crops)
    - Gradient accumulation (effective batch = micro batch × accum steps)
    - Class-weighted cross-entropy in both stages
    - Config-driven via Hydra (``configs/baseline6.yaml``)
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
from configs.path_config import LOGS_DIR, MODEL_SAVE_DIR
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
    Scene-level temporal model: frozen per-crop features → masked player pool
    per frame → LSTM over time → skip-connection + Conv1d fusion → summary.

    Consumes clips ``(B, T, P, C, H, W)`` with player masks ``(B, P)``.
    Stage A feeds P=1 single-player tracks (9-action head); Stage B feeds
    full clips and reads ``forward_summaries``.

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
    pretrained_backbone : bool
        Only consulted when ``checkpoint`` is None. ``False`` leaves the
        extractor randomly initialized — for callers that immediately
        restore the whole model from a saved checkpoint (evaluation).
    pool : {"max", "mean", "concat"}
        Per-frame aggregation across players, applied BEFORE the LSTM.
        "concat" (max‖mean) doubles the LSTM's input width.
    T : int
        Frames per clip; fixes the Conv1d global kernel (``2*T``).

    """

    def __init__(
        self,
        num_actions: int = NUM_PERSON_ACTIONS,
        backbone_name: str = "resnet50",
        checkpoint: str | None = None,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
        pool: str = "concat",
        T: int = 9,
    ) -> None:
        super().__init__()
        self.T = T
        self.pool = pool
        # Frozen — stays in eval mode and produces no-grad features.
        self.extractor = FeatureExtractor(
            model_name=backbone_name, checkpoint=checkpoint,
            pretrained=pretrained_backbone,
        )
        self.feature_dropout = nn.Dropout(p=dropout)

        if pool not in ("max", "mean", "concat"):
            raise ValueError(f"Unsupported pool '{pool}'. Use 'max', 'mean' or 'concat'.")

        self.lstm_hidden = lstm_hidden
        # Players are pooled per frame BEFORE the LSTM, so the LSTM consumes
        # one pooled scene vector per timestep. "concat" (max‖mean) doubles
        # that vector's width; the LSTM and the skip projection must match it.
        lstm2_input_size = (
            2 * self.extractor.feature_dim if pool == "concat"
            else self.extractor.feature_dim
        )
        self.lstm1 = nn.LSTM(
            input_size=self.extractor.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm2_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Project the pooled per-frame features to LSTM hidden size so they can
        # be concatenated with the LSTM output along the time axis (skip
        # connection). Result after concat: (B, 2*T, lstm_hidden)
        self.project = nn.Linear(lstm_input_size, lstm_hidden)

        # Conv1d collapses the (B, lstm_hidden, 2*T) tensor to (B, lstm_hidden//4).
        # kernel_size=2*T is a global temporal kernel — exactly one output position.

        self.conv_projection = nn.Sequential(
            nn.Conv1d(in_channels=lstm_hidden, out_channels=lstm_hidden//2, kernel_size=2*T, padding=0),
            nn.BatchNorm1d(lstm_hidden//2),   # ← works on (B, C, L)
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=lstm_hidden//2, out_channels=lstm_hidden//4, kernel_size=1, padding=0),  # ← no padding
            nn.BatchNorm1d(lstm_hidden//4),
            nn.Flatten()
)
        
        self.action_head = nn.Sequential(
            
            nn.Linear(lstm_hidden//4, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_actions),
        )

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, C, H, W)`` → ``(B, T, P, D)`` backbone features (no LSTM)."""
        B, T, P, C, H, W = x.shape
        with torch.no_grad():
            x = self.extractor(x.reshape(B * T * P, C, H, W))
            return x.view(B, T, P, -1)

    def forward_summaries(self, seqs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, C, H, W)`` + ``(B, P)`` masks → ``(B, lstm_hidden//4)`` summaries.

        Players are pooled per frame BEFORE the LSTM, so the LSTM models
        scene-level dynamics. Stage A calls this with P=1 (a single player's
        track, pooling is an identity); Stage B with the full player set.
        """
        B, T, P, C, H, W = seqs.shape
        feats = self.feature_extractor(seqs)      # (B, T, P, D)
        feats = feats.permute(0, 2, 1, 3)         # (B, P, T, D)
        feats = self.feature_dropout(feats)
        


        out1 , (_ , _) = self.lstm1(feats)              # (B, P, T, D)
        
        # Masked pooling across players (dim=2) — padded slots must not
        # contribute. masks (B, P) broadcasts over time and feature dims.
        mask4 = masks[:, None, :, None]           # (B, 1, P, 1)

        if self.pool in ("max", "concat"):
            pooled_max = out1.masked_fill(~mask4, float("-inf")).max(dim=2)[0]
            pooled_max = torch.where(
                torch.isinf(pooled_max), torch.zeros_like(pooled_max), pooled_max,
            )                                     # (B, T, D)
        if self.pool in ("mean", "concat"):
            valid = masks.sum(dim=1).clamp_min(1).view(B, 1, 1).float()
            pooled_mean = out1.masked_fill(~mask4, 0.0).sum(dim=2) / valid  # (B, T, D)

        if self.pool == "max":
            # pyrefly: ignore [unbound-name]
            team = pooled_max
        elif self.pool == "mean":
            # pyrefly: ignore [unbound-name]
            team = pooled_mean
        else:
            # pyrefly: ignore [unbound-name]
            team = torch.cat([pooled_max, pooled_mean], dim=-1)   # (B, T, 2D)

        out2, (_ , _ ) = self.lstm2(team)                  # (B, T, lstm_hidden)
        team_projected = self.project(feats)       # (B, T, lstm_hidden)

        # Skip connection along TIME dim → (B, 2T, lstm_hidden)
        combined = torch.cat([out1, out2], dim=1)  # (B, 2T, lstm_hidden)
        combined = combined.permute(0, 2, 1)      # (B, lstm_hidden, 2T)
        combined = torch.cat([combined, team_projected], dim=1)

        return self.conv_projection(combined)     # (B, lstm_hidden//4)

    def forward(self, seqs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, C, H, W)`` + ``(B, P)`` → ``(B, num_actions)`` logits."""
        return self.action_head(self.forward_summaries(seqs, masks))



class GroupTemporalClassifier(nn.Module):
    """
    Stage B: frozen Stage-A scene model → clip summary → MLP → 8.

    Player pooling happens inside ``person_model.forward_summaries`` (before
    its LSTM), so the summary handed to the classifier is always
    ``lstm_hidden // 4`` wide regardless of the pooling mode.

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

    """

    def __init__(
        self,
        person_model: PersonTemporalLSTM,
        num_classes: int = NUM_GROUP_ACTIVITIES,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.person = person_model
        # Freeze the whole Stage-A model — only the MLP head trains.
        for p in self.person.parameters():
            p.requires_grad = False

        classifier_in = person_model.lstm_hidden // 4
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
        with torch.no_grad():   # Stage-A model is frozen
            summaries = self.person.forward_summaries(crops, masks)  # (B, lstm_hidden//4)

        return self.classifier(summaries)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 2. BATCH UNPACKERS ══
# ═════════════════════════════════════════════════════════════════════════════
#
# Crop-mode collate yields ``(crops, person_labels, group_labels, masks)``.
# Stage A flattens (B, P) player sequences and drops padded slots via the
# mask; Stage B keeps crops+masks together and targets the group labels.


def stage_a_unpack(batch):
    """4-tuple → ``((tracks, track_masks), person_action_labels)`` for valid players.

    Each valid player becomes an independent P=1 "clip" ``(N, T, 1, C, H, W)``
    with an all-ones ``(N, 1)`` mask: the model's per-frame player pooling is
    then an identity and the LSTM sees that one person's track.
    """
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
    tracks = seqs[valid].unsqueeze(2)                      # (N, T, 1, C, H, W)
    track_masks = torch.ones(
        tracks.shape[0], 1, dtype=torch.bool, device=tracks.device,
    )
    return (tracks, track_masks), labels[valid]


def temporal_crop_unpack(batch):
    """4-tuple → ``((crops, masks), group_labels)`` (Stage B / evaluation)."""
    if not batch or len(batch) < 4:
        return None
    crops, _person_labels, group_labels, masks = batch
    if crops.dim() != 6 or crops.numel() == 0:
        return None
    return (crops, masks), group_labels


# ═════════════════════════════════════════════════════════════════════════════
# ══ 3. CHECKPOINT-ARCHITECTURE INFERENCE ══
# ═════════════════════════════════════════════════════════════════════════════
#
# Reloads build models from the CHECKPOINT's dimensions, not the config's:
# hyperparameters (lstm.hidden_dim, stage_b.hidden_dim, pool) routinely
# change between iterations, and an already-trained run must keep loading
# regardless of what the yaml says today.


def _checkpoint_state(ckpt_name: str) -> dict:
    """Return the model state dict stored in a saved checkpoint."""
    ckpt = torch.load(MODEL_SAVE_DIR / ckpt_name, map_location="cpu", weights_only=False)
    return ckpt.get("model_state_dict", ckpt)


def _person_dims(
    state: dict, prefix: str = "", cfg_pool: str = "max",
) -> tuple[int, int, int, str]:
    """``(lstm_hidden, lstm_layers, T, pool)`` stored in a PersonTemporalLSTM state.

    ``T`` (number of frames) is recovered from the first Conv1d kernel size:
    ``conv_projection.0.weight`` has shape ``(out, lstm_hidden, 2*T)``.
    ``pool`` is recovered from the LSTM input width: 2× the backbone feature
    dim (2048 for resnet50/101) ⇒ "concat". "max" and "mean" share a width,
    so the config's choice is kept for those.
    """
    hidden = state[f"{prefix}lstm.weight_hh_l0"].shape[1]
    stem = f"{prefix}lstm.weight_ih_l"
    layers = sum(1 for k in state if k.startswith(stem) and k[len(stem):].isdigit())
    conv_w = state[f"{prefix}conv_projection.0.weight"]  # (out, lstm_hidden, 2*T)
    T = int(conv_w.shape[-1]) // 2
    input_size = state[f"{prefix}lstm.weight_ih_l0"].shape[1]
    if input_size == 2 * 2048:
        pool = "concat"
    else:
        pool = cfg_pool if cfg_pool in ("max", "mean") else "max"
    return int(hidden), layers, T, pool


def _group_dims(state: dict) -> int:
    """MLP ``hidden_dim`` (first classifier Linear's out_features) in a saved state."""
    linears = sorted(
        (int(k.split(".")[1]), k)
        for k, v in state.items()
        if k.startswith("classifier.") and k.endswith(".weight") and v.dim() == 2
    )
    hidden_dim, _classifier_in = state[linears[0][1]].shape
    return int(hidden_dim)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 4. MAIN ENTRYPOINT ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline6", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Logging ──────────────────────────────────────────────────────────
    run_log_dir = LOGS_DIR / "baseline6"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_count = len(list(run_log_dir.glob("*.json"))) + 1
    run_id = f"run{run_count}"
    writer = SummaryWriter(log_dir=run_log_dir / "tensorboard" / run_id)
    metrics_history: list[dict] = []

    stage_a_ckpt = f"baseline6_stage_a_{run_id}.pt"
    stage_b_ckpt = f"baseline6_stage_b_{run_id}.pt"

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
        pool=cfg.get("pool", "max"),    # per-frame player pooling, sets LSTM input width
        T=cfg.n_frames,                 # kernel_size=2*T — must match the data
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
            desc="Train[B6-A]",
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            person_model, val_loader, criterion_a, device,
            batch_unpack=stage_a_unpack,
            num_classes=NUM_PERSON_ACTIONS,
            desc="Val[B6-A]",
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
    print(f"  Player pool (per frame, inside scene LSTM): {cfg.get('pool', 'max')}")
    print(f"  Target: {NUM_GROUP_ACTIVITIES} classes — {list(GROUP_ACTIVITY_TO_IDX.keys())}")
    print(f"{'='*60}")

    # Reload best Stage-A weights so B always starts from the best checkpoint.
    # LSTM dimensions come from the checkpoint itself (see section 3) — the
    # config may have been retuned since that Stage A was trained.
    state_a = _checkpoint_state(stage_a_ckpt)
    lstm_hidden_a, lstm_layers_a, T_a, pool_a = _person_dims(
        state_a, cfg_pool=cfg.get("pool", "max"),
    )
    if (lstm_hidden_a, lstm_layers_a) != (cfg.lstm.hidden_dim, cfg.lstm.num_layers):
        print(
            f"  ⚠ Config LSTM ({cfg.lstm.hidden_dim}×{cfg.lstm.num_layers}) ≠ "
            f"checkpoint '{stage_a_ckpt}' ({lstm_hidden_a}×{lstm_layers_a}) — "
            "using the checkpoint's dimensions.",
        )
    reloaded = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=None,
        lstm_hidden=lstm_hidden_a,
        lstm_layers=lstm_layers_a,
        dropout=cfg.get("dropout", 0.3),
        pool=pool_a,                    # recovered from checkpoint's LSTM input width
        T=T_a,                          # recovered from checkpoint's Conv1d kernel
    ).to(device)
    reloaded, _, _, _, _ = load_model(stage_a_ckpt, reloaded)

    group_model = GroupTemporalClassifier(
        person_model=reloaded,
        num_classes=NUM_GROUP_ACTIVITIES,
        hidden_dim=stage_b_cfg.get("hidden_dim", 256),
        dropout=stage_b_cfg.get("dropout", 0.3),
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
            desc="Train[B6-B]",
        )
        val_loss, val_acc, val_f1, _ = validate_one_epoch(
            group_model, val_loader, criterion_b, device,
            batch_unpack=temporal_crop_unpack,
            num_classes=NUM_GROUP_ACTIVITIES,
            desc="Val[B6-B]",
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
    # Rebuilt from the SAVED checkpoint's architecture: within this run the
    # Stage-A reload above may already differ from the config, and the same
    # holds when testing a checkpoint from an older iteration.
    print("\n--- Testing best Stage-B model ---")
    state_b = _checkpoint_state(stage_b_ckpt)
    lstm_hidden_b, lstm_layers_b, T_b, pool_b = _person_dims(
        state_b, prefix="person.", cfg_pool=cfg.get("pool", "max"),
    )
    hidden_b = _group_dims(state_b)
    fresh_person = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=None,
        lstm_hidden=lstm_hidden_b,
        lstm_layers=lstm_layers_b,
        dropout=cfg.get("dropout", 0.3),
        pool=pool_b,                    # recovered from checkpoint's LSTM input width
        T=T_b,                          # recovered from checkpoint's Conv1d kernel
    ).to(device)
    best_group = GroupTemporalClassifier(
        person_model=fresh_person,
        num_classes=NUM_GROUP_ACTIVITIES,
        hidden_dim=hidden_b,
        dropout=stage_b_cfg.get("dropout", 0.3),
    ).to(device)
    best_group, _, _, _, _ = load_model(stage_b_ckpt, best_group)
    if use_dp:
        best_group = nn.DataParallel(best_group)

    test_loss, test_acc, test_f1, _ = test_one_epoch(
        best_group, test_loader, criterion_b, device,
        batch_unpack=temporal_crop_unpack,
        num_classes=NUM_GROUP_ACTIVITIES,
        desc="Test[B6]",
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
            "baseline":                "baseline6",
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
            "pool":                    cfg.get("pool", "max"),
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
