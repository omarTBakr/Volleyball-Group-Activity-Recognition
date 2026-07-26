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

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader

from configs.labels import (
    GROUP_ACTIVITY_TO_IDX,
    NUM_GROUP_ACTIVITIES,
    NUM_PERSON_ACTIONS,
    PERSON_ACTION_TO_IDX,
)
from configs.path_config import MODEL_SAVE_DIR
from src.data.kaggle_data_loader import (
    VolleyballDataset,
    collate_fn,
    free_annotation_cache,
)
from src.data.unpackers import group_crop_unpack, person_seq_unpack
from src.pickle_dump import free_master_data_cache
from utils.featureExtractor import FeatureExtractor
from utils.load_model_config import build_scheduler, build_transforms
from utils.trainer import Trainer
from utils.utility import (
    get_device,
    group_activity_label_counts,
    inverse_freq_weights,
    load_model,
    person_action_label_counts,
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
    pretrained_backbone : bool
        Only consulted when ``checkpoint`` is None. ``False`` leaves the
        extractor randomly initialized — for callers that immediately
        restore the whole model from a saved checkpoint (evaluation).

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
    ) -> None:
        super().__init__()

        # Frozen — stays in eval mode and produces no-grad features.
        self.extractor = FeatureExtractor(
            model_name=backbone_name, checkpoint=checkpoint,
            pretrained=pretrained_backbone,
        )
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


# ══ 2. BATCH UNPACKERS ══
#
# Canonical unpackers live in src.data.unpackers. B5's Stage A feeds each
# player's whole (T,C,H,W) sequence as a single input (its forward takes just
# the sequence — person_seq); Stage B passes crops+masks (group_crop).
# Re-exported under the historical names so utils.evaluate keeps resolving them.
stage_a_unpack = person_seq_unpack
temporal_crop_unpack = group_crop_unpack


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


def _person_dims(state: dict, prefix: str = "") -> tuple[int, int]:
    """``(lstm_hidden, lstm_layers)`` as stored in a PersonTemporalLSTM state."""
    hidden = state[f"{prefix}lstm.weight_hh_l0"].shape[1]
    stem = f"{prefix}lstm.weight_ih_l"
    layers = sum(1 for k in state if k.startswith(stem) and k[len(stem):].isdigit())
    return int(hidden), layers


def _group_dims(state: dict, cfg_pool: str) -> tuple[int, str]:
    """``(hidden_dim, pool)`` consistent with a GroupTemporalClassifier state.

    ``hidden_dim`` is the first classifier Linear's out_features. The pool
    is recovered from that Linear's in_features: 2×lstm_hidden ⇒ "concat".
    "max" and "mean" are indistinguishable by shape, so the config's choice
    is kept when it names one of those two, else "max".
    """
    lstm_hidden, _ = _person_dims(state, prefix="person.")
    linears = sorted(
        (int(k.split(".")[1]), k)
        for k, v in state.items()
        if k.startswith("classifier.") and k.endswith(".weight") and v.dim() == 2
    )
    hidden_dim, classifier_in = state[linears[0][1]].shape
    if classifier_in == 2 * lstm_hidden:
        pool = "concat"
    else:
        pool = cfg_pool if cfg_pool in ("max", "mean") else "max"
    return int(hidden_dim), pool


# ═════════════════════════════════════════════════════════════════════════════
# ══ 4. MAIN ENTRYPOINT ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline5", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Logging ──────────────────────────────────────────────────────────
    run_id = Trainer.next_run_id("baseline5")
    trainer = Trainer("baseline5", run_id, device=device)
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
    # optimizer_a = optim.SGD(
    #     trainable_a,
    #     lr=stage_a_cfg.learning_rate,
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=stage_a_cfg.get("weight_decay", 5e-4),
    # )
    # AdamW equivalent — trains a fresh player LSTM on frozen backbone features.
    optimizer_a = optim.AdamW(
        trainable_a,
        lr=stage_a_cfg.learning_rate,
        weight_decay=stage_a_cfg.get("weight_decay", 5e-4),
    )
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_a):,}")

    best_f1_a = trainer.run_stage(
        model=person_model,
        save_ref=person_inner,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_a,
        optimizer=optimizer_a,
        num_classes=NUM_PERSON_ACTIONS,
        num_epochs=stage_a_cfg.num_epochs,
        checkpoint_name=stage_a_ckpt,
        class_to_idx=PERSON_ACTION_TO_IDX,
        batch_unpack=stage_a_unpack,
        accum_steps=accum_steps,
        patience=stage_a_cfg.get("early_stopping_patience", 0),
        stage="A",
        desc="B5-A",
    )

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
    # LSTM dimensions come from the checkpoint itself (see section 3) — the
    # config may have been retuned since that Stage A was trained.
    state_a = _checkpoint_state(stage_a_ckpt)
    lstm_hidden_a, lstm_layers_a = _person_dims(state_a)
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
    # optimizer_b = optim.SGD(
    #     trainable_b,
    #     lr=stage_b_cfg.learning_rate,
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=stage_b_cfg.get("weight_decay", 5e-4),
    # )
    # AdamW equivalent — trains the MLP head on frozen Stage-A summaries.
    optimizer_b = optim.AdamW(
        trainable_b,
        lr=stage_b_cfg.learning_rate,
        weight_decay=stage_b_cfg.get("weight_decay", 5e-4),
    )
    scheduler_b = build_scheduler(optimizer_b, cfg)
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_b):,}")

    best_f1_b = trainer.run_stage(
        model=group_model,
        save_ref=group_inner,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_b,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        num_classes=NUM_GROUP_ACTIVITIES,
        num_epochs=stage_b_cfg.num_epochs,
        checkpoint_name=stage_b_ckpt,
        class_to_idx=GROUP_ACTIVITY_TO_IDX,
        batch_unpack=temporal_crop_unpack,
        accum_steps=accum_steps,
        patience=stage_b_cfg.get("early_stopping_patience", 0),
        stage="B",
        desc="B5-B",
    )

    # ── Test best Stage-B model ──────────────────────────────────────────
    # Rebuilt from the SAVED checkpoint's architecture: within this run the
    # Stage-A reload above may already differ from the config, and the same
    # holds when testing a checkpoint from an older iteration.
    state_b = _checkpoint_state(stage_b_ckpt)
    lstm_hidden_b, lstm_layers_b = _person_dims(state_b, prefix="person.")
    hidden_b, pool_b = _group_dims(state_b, stage_b_cfg.get("pool", "max"))
    if pool_b != stage_b_cfg.get("pool", "max"):
        print(
            f"  ⚠ Config pool '{stage_b_cfg.get('pool', 'max')}' is incompatible with "
            f"checkpoint '{stage_b_ckpt}' — using '{pool_b}'.",
        )
    fresh_person = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=None,
        lstm_hidden=lstm_hidden_b,
        lstm_layers=lstm_layers_b,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    best_group = GroupTemporalClassifier(
        person_model=fresh_person,
        num_classes=NUM_GROUP_ACTIVITIES,
        hidden_dim=hidden_b,
        dropout=stage_b_cfg.get("dropout", 0.3),
        pool=pool_b,
    ).to(device)
    best_group, _, _, _, _ = load_model(stage_b_ckpt, best_group)
    if use_dp:
        best_group = nn.DataParallel(best_group)

    trainer.run_test(
        model=best_group,
        test_loader=test_loader,
        criterion=criterion_b,
        num_classes=NUM_GROUP_ACTIVITIES,
        batch_unpack=temporal_crop_unpack,
        best_val_f1=best_f1_b,
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
    )

    trainer.close()


if __name__ == "__main__":
    train_test()
