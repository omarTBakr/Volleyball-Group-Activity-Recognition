"""
Baseline 8 — Hierarchical two-LSTM model with TEAM-SPLIT pooling (player
LSTM 1 → pool each team per frame → scene LSTM 2) with skip connections at
both levels.

B8 = B7 plus the paper's group-style pooling. The only architectural change
over B7 is how players are aggregated into the per-frame scene vector: instead
of pooling all ~12 players together (side-blind), each team's players are
pooled SEPARATELY using ``team_ids`` and the two team vectors are concatenated.
That keeps *which side did what* — the signal B7's pooling erases, and the
direct fix for the left/right winpoint/pass/set confusion. Both of B7's skip
connections are kept unchanged.

Core flow:
    Each player's 9-crop track goes through a frozen person-feature extractor
    (Baseline 3's Stage-A backbone) into a shared **LSTM 1** over time. The
    player-level skip concatenates LSTM 1's per-timestep output with a linear
    projection of the raw backbone features along the FEATURE axis (the
    paper's fc7 ‖ hidden trick) → one (2·H1)-wide vector per player per frame.
    Per frame, the two teams are masked-pooled separately and concatenated
    into a scene sequence, which **LSTM 2** consumes. The scene-level skip is
    B6's recipe: LSTM 2's T hidden states are concatenated along the TIME axis
    with a projection of the pooled scene features → (B, 2T, H2), and a
    two-stage Conv1d (global kernel 2T) collapses this into (B, H2//4).

    The player skip must be feature-axis (the time axis has to survive for
    LSTM 2); the time-axis-concat + Conv1d fusion lives at the scene level,
    where collapsing time is the goal.

Team ids come from the loader (``with_teams=True`` → collate emits a
``(B, P)`` team_ids tensor, 0 = left court side, 1 = right, -1 for padding),
derived once per clip from box center-x. Stage A is team-agnostic (per-player
action) and ignores them.

Stage A (person-action pretraining of LSTM 1, 9 classes):
    Each valid player is an independent P=1 track. LSTM 1 + projection +
    action head train on the 9 person actions from the track's last-timestep
    representation. No pooling is involved at this stage.

Stage B (group-activity, 8 classes) — two phases, mirroring B6:
    Load Stage A's best model. Phase 1 ("probe"): the player model (LSTM 1 +
    projection) stays frozen while the fresh scene modules (LSTM 2, scene
    projection, Conv1d fusion, MLP head) train at ``stage_b.warmup_lr``.
    Phase 2 (joint fine-tune): ``unfreeze_player_temporal()`` opens LSTM 1 +
    the player projection at the low ``stage_b.learning_rate`` while the scene
    modules continue at ``learning_rate × head_lr_multiplier``. The ResNet
    extractor and Stage A's action head stay frozen throughout.

Uses:
    - Crop mode, ``n_frames=9``, ``with_teams=True`` (temporal per-player crops)
    - Gradient accumulation (effective batch = micro batch × accum steps)
    - Class-weighted cross-entropy in both stages
    - Config-driven via Hydra (``configs/baseline8.yaml``)
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
from src.data.unpackers import group_team_unpack, person_track_unpack
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
    Stage A: frozen per-crop features → shared player LSTM 1 over time →
    feature-axis skip (LSTM output ‖ projected features) → 9-class action.

    Consumes clips ``(B, T, P, C, H, W)``.  Stage A feeds P=1 single-player
    tracks; Stage B calls ``forward_player_sequences`` on full clips to get
    per-player per-frame representations for its own pooling + LSTM 2.

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
        LSTM 1 hidden size (H1). Per-player representations are 2·H1 wide
        (LSTM output ‖ projected features).
    lstm_layers : int
        Number of stacked LSTM 1 layers (dropout applies between layers).
    dropout : float
        Dropout on the frozen features and inside the action head.
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
        self.lstm1 = nn.LSTM(
            input_size=self.extractor.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Player-level skip: project raw backbone features to H1 so each
        # timestep's representation is [lstm1 output ‖ projected features]
        # → 2·H1 wide. Feature-axis concat keeps the time axis intact for
        # Stage B's LSTM 2.
        self.project = nn.Linear(self.extractor.feature_dim, lstm_hidden)

        self.action_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden, num_actions),
        )

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, C, H, W)`` → ``(B, T, P, D)`` backbone features (no LSTM)."""
        B, T, P, C, H, W = x.shape
        with torch.no_grad():
            x = self.extractor(x.reshape(B * T * P, C, H, W))
            return x.view(B, T, P, -1)

    def forward_player_sequences(self, seqs: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, C, H, W)`` → ``(B, T, P, 2·H1)`` per-player representations.

        LSTM 1 runs over TIME independently for each player (weights shared,
        players folded into the batch). No pooling here — Stage B owns that.
        """
        B, T, P, C, H, W = seqs.shape
        feats = self.feature_extractor(seqs)                    # (B, T, P, D)
        feats = self.feature_dropout(feats)

        # Fold players into the batch so the LSTM sequence axis is time.
        per_player = feats.permute(0, 2, 1, 3).reshape(B * P, T, -1)  # (B·P, T, D)

        out1, (_, _) = self.lstm1(per_player)                   # (B·P, T, H1)
        proj = self.project(per_player)                         # (B·P, T, H1)
        repr_ = torch.cat([out1, proj], dim=-1)                 # (B·P, T, 2·H1)

        return repr_.view(B, P, T, -1).permute(0, 2, 1, 3)      # (B, T, P, 2·H1)

    def forward(self, seqs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, C, H, W)`` + ``(B, P)`` → ``(B·P, num_actions)`` logits.

        Stage A path: each player's track is classified from its LAST
        timestep's representation. The unpacker feeds P=1 tracks, so the
        output lines up with the flattened per-player action labels.
        ``masks`` is accepted for interface symmetry with Stage B; padded
        slots are already filtered out by the unpacker.
        """
        B, T, P, _, _, _ = seqs.shape
        repr_ = self.forward_player_sequences(seqs)             # (B, T, P, 2·H1)
        last = repr_[:, -1].reshape(B * P, -1)                  # (B·P, 2·H1)
        return self.action_head(last)                           # (B·P, 9)


class GroupTemporalClassifier(nn.Module):
    """
    Stage B: player model → TEAM-SPLIT pool per frame → scene LSTM 2 →
    time-axis skip + Conv1d fusion (B6/B7 recipe) → MLP → 8.

    This is B8's defining change over B7: instead of pooling all players into
    one scene vector, each team's players are pooled separately (using
    ``team_ids``) and the two team vectors are concatenated. That preserves
    *which side did what* — the signal B7's side-blind pooling erases — so the
    scene vector doubles in width. Both of B7's skip connections are kept
    unchanged: the player-level feature-axis skip lives in ``person_model``,
    and the scene-level time-axis skip + Conv1d fusion is below.

    Training is two-phase (mirroring B6): constructed with the Stage-A player
    model frozen (phase 1 — only the fresh scene modules train), then
    ``unfreeze_player_temporal()`` opens LSTM 1 + the player projection for
    joint fine-tuning. The ResNet extractor and Stage A's 9-way action head
    stay frozen throughout.

    Parameters
    ----------
    person_model : PersonTemporalLSTM
        Already-trained Stage A model; frozen at construction.
    num_classes : int
        Number of group-activity classes (8).
    lstm2_hidden : int
        Scene LSTM 2 hidden size (H2). Clip summary is H2 // 4 wide.
    pool : {"max", "mean", "concat"}
        Per-team aggregation across that team's players. A team pools to
        2·H1 (max/mean) or 4·H1 (concat, max ‖ mean); the two teams are then
        concatenated, so LSTM 2's input is twice that: 4·H1 or 8·H1.
    T : int
        Frames per clip; fixes the Conv1d global kernel (``2*T``).
    hidden_dim : int
        Width of the MLP head's first hidden layer.
    dropout : float
        Dropout inside the MLP head.

    """

    def __init__(
        self,
        person_model: PersonTemporalLSTM,
        num_classes: int = NUM_GROUP_ACTIVITIES,
        lstm2_hidden: int = 512,
        pool: str = "max",
        T: int = 9,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        if pool not in ("max", "mean", "concat"):
            raise ValueError(f"Unsupported pool '{pool}'. Use 'max', 'mean' or 'concat'.")
        self.pool = pool
        self.T = T
        self.lstm2_hidden = lstm2_hidden

        self.person = person_model
        # Start fully frozen (probe phase) — only the fresh scene modules
        # train until unfreeze_player_temporal() is called.
        for p in self.person.parameters():
            p.requires_grad = False
        self.player_trainable = False

        player_repr = 2 * person_model.lstm_hidden          # 2·H1
        # One team pools to team_width; two teams concatenate → 2·team_width.
        team_width = 2 * player_repr if pool == "concat" else player_repr
        lstm2_input = 2 * team_width

        self.lstm2 = nn.LSTM(
            input_size=lstm2_input,
            hidden_size=lstm2_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Scene-level skip: project the pooled scene sequence to H2 so it can
        # be concatenated with LSTM 2's outputs along the TIME axis
        # → (B, 2T, H2), then collapsed by the global-kernel Conv1d.
        self.scene_project = nn.Linear(lstm2_input, lstm2_hidden)

        self.conv_projection = nn.Sequential(
            nn.Conv1d(lstm2_hidden, lstm2_hidden // 2, kernel_size=2 * T),
            nn.BatchNorm1d(lstm2_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(lstm2_hidden // 2, lstm2_hidden // 4, kernel_size=1),
            nn.BatchNorm1d(lstm2_hidden // 4),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm2_hidden // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def unfreeze_player_temporal(self) -> list[torch.nn.Parameter]:
        """Open the pretrained player machinery (LSTM 1 + projection) for
        joint fine-tuning and return its parameters for a low-LR optimizer
        group. The ResNet extractor and the 9-way action head stay frozen.
        """
        player_params: list[torch.nn.Parameter] = []
        for module in (self.person.lstm1, self.person.project):
            for p in module.parameters():
                p.requires_grad = True
                player_params.append(p)
        self.player_trainable = True
        return player_params

    def train(self, mode: bool = True):
        # Probe phase: keep the frozen Stage-A model in eval mode (LSTM 1
        # dropout off, deterministic representations). After
        # unfreeze_player_temporal(), the player parts follow train mode; the
        # ResNet extractor still forces itself to eval (see FeatureExtractor).
        super().train(mode)
        if not self.player_trainable:
            self.person.eval()
        return self

    def _pool_players(self, repr_: torch.Tensor, player_mask: torch.Tensor) -> torch.Tensor:
        """Masked pool ``repr_ (B, T, P, 2·H1)`` over the players selected by
        ``player_mask (B, P)`` → ``(B, T, team_width)``.

        ``team_width`` is 2·H1 for max/mean, 4·H1 for concat (max ‖ mean). A
        frame/clip with zero selected players yields a zero vector (max's -inf
        is sanitized; mean divides by clamp_min(1) over a zero sum).
        """
        B = repr_.shape[0]
        mask4 = player_mask[:, None, :, None]                     # (B, 1, P, 1)

        if self.pool in ("max", "concat"):
            pooled_max = repr_.masked_fill(~mask4, float("-inf")).max(dim=2)[0]
            pooled_max = torch.where(
                torch.isinf(pooled_max), torch.zeros_like(pooled_max), pooled_max,
            )                                                     # (B, T, 2·H1)
        if self.pool in ("mean", "concat"):
            valid = player_mask.sum(dim=1).clamp_min(1).view(B, 1, 1).float()
            pooled_mean = repr_.masked_fill(~mask4, 0.0).sum(dim=2) / valid

        if self.pool == "max":
            # pyrefly: ignore [unbound-name]
            return pooled_max
        if self.pool == "mean":
            # pyrefly: ignore [unbound-name]
            return pooled_mean
        # pyrefly: ignore [unbound-name]
        return torch.cat([pooled_max, pooled_mean], dim=-1)       # (B, T, 4·H1)

    def forward(
        self, crops: torch.Tensor, masks: torch.Tensor, team_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        crops    : (B, T, P, C, H, W) — per-player crop sequences.
        masks    : (B, P) bool — True for real players, False for padded slots.
        team_ids : (B, P) long — 0 = left court side, 1 = right, -1 for padding.
        returns  : (B, num_classes) logits
        """
        # No no_grad here: gradients must reach LSTM 1 + projection once they
        # are unfrozen. While frozen, requires_grad=False keeps it cheap.
        repr_ = self.person.forward_player_sequences(crops)       # (B, T, P, 2·H1)

        # TEAM-SPLIT pooling: pool each team's players separately (padded slots
        # are excluded by masks; team_ids == -1 there is harmless), then
        # concatenate. The scene vector keeps which side did what.
        left_mask = masks & (team_ids == 0)                       # (B, P)
        right_mask = masks & (team_ids == 1)                      # (B, P)
        left = self._pool_players(repr_, left_mask)               # (B, T, team_width)
        right = self._pool_players(repr_, right_mask)             # (B, T, team_width)
        scene = torch.cat([left, right], dim=-1)                  # (B, T, 2·team_width)

        out2, (_, _) = self.lstm2(scene)                          # (B, T, H2)
        scene_projected = self.scene_project(scene)               # (B, T, H2)

        # Scene skip along TIME dim → (B, 2T, H2) → (B, H2, 2T)
        combined = torch.cat([out2, scene_projected], dim=1).permute(0, 2, 1)

        summary = self.conv_projection(combined)                  # (B, H2//4)
        return self.classifier(summary)


# ══ 2. BATCH UNPACKERS ══
#
# Canonical unpackers live in src.data.unpackers. Stage A routes each player as
# a P=1 track (person_track — team-agnostic, ignores the extra team_ids the
# team-mode loader adds). Stage B uses group_team: it forwards team_ids so the
# model pools each team separately, and REQUIRES the loader built with
# with_teams=True. Re-exported under the historical names so utils.evaluate
# keeps resolving them from this module.
stage_a_unpack = person_track_unpack
temporal_crop_unpack = group_team_unpack


# ═════════════════════════════════════════════════════════════════════════════
# ══ 3. CHECKPOINT-ARCHITECTURE INFERENCE ══
# ═════════════════════════════════════════════════════════════════════════════
#
# Reloads build models from the CHECKPOINT's dimensions, not the config's:
# hyperparameters (lstm hidden sizes, stage_b.hidden_dim, pool) routinely
# change between iterations, and an already-trained run must keep loading
# regardless of what the yaml says today.


def _checkpoint_state(ckpt_name: str) -> dict:
    """Return the model state dict stored in a saved checkpoint."""
    ckpt = torch.load(MODEL_SAVE_DIR / ckpt_name, map_location="cpu", weights_only=False)
    return ckpt.get("model_state_dict", ckpt)


def _person_dims(state: dict, prefix: str = "") -> tuple[int, int]:
    """``(lstm1_hidden, lstm1_layers)`` stored in a PersonTemporalLSTM state."""
    hidden = state[f"{prefix}lstm1.weight_hh_l0"].shape[1]
    stem = f"{prefix}lstm1.weight_ih_l"
    layers = sum(1 for k in state if k.startswith(stem) and k[len(stem):].isdigit())
    return int(hidden), layers


def _group_dims(state: dict, cfg_pool: str = "max") -> tuple[int, str, int, int]:
    """``(lstm2_hidden, pool, T, head_hidden)`` from a GroupTemporalClassifier state.

    Team-split (B8): LSTM 2's input width is ``2 × team_width`` where a team
    pools to ``player_repr`` (max/mean) or ``2·player_repr`` (concat). So
    ``4·player_repr`` ⇒ "concat" and ``2·player_repr`` ⇒ max/mean (same width,
    config's choice kept). ``T`` comes from the first Conv1d kernel (``2*T``);
    ``head_hidden`` from the classifier's first 2-D Linear.
    """
    lstm1_hidden, _ = _person_dims(state, prefix="person.")
    player_repr = 2 * lstm1_hidden

    lstm2_hidden = state["lstm2.weight_hh_l0"].shape[1]
    lstm2_input = state["lstm2.weight_ih_l0"].shape[1]
    if lstm2_input == 4 * player_repr:
        pool = "concat"
    else:  # 2 * player_repr
        pool = cfg_pool if cfg_pool in ("max", "mean") else "max"

    T = int(state["conv_projection.0.weight"].shape[-1]) // 2

    linears = sorted(
        (int(k.split(".")[1]), k)
        for k, v in state.items()
        if k.startswith("classifier.") and k.endswith(".weight") and v.dim() == 2
    )
    head_hidden, _ = state[linears[0][1]].shape
    return int(lstm2_hidden), pool, T, int(head_hidden)


# ═════════════════════════════════════════════════════════════════════════════
# ══ 4. MAIN ENTRYPOINT ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline8", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Logging ──────────────────────────────────────────────────────────
    run_id = Trainer.next_run_id("baseline8")
    trainer = Trainer("baseline8", run_id, device=device)
    stage_a_ckpt = f"baseline8_stage_a_{run_id}.pt"
    stage_b_ckpt = f"baseline8_stage_b_{run_id}.pt"

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


    # the return form this will be (image_tensor, mask, label, bboxes, team_ids)
    train_dataset = VolleyballDataset(
        mode="train", full_image=False, crop=True, n_frames=cfg.n_frames, with_teams=True, transform=tf["train"],
    )
    val_dataset = VolleyballDataset(
        mode="validation", full_image=False, crop=True, n_frames=cfg.n_frames, with_teams=True, transform=tf["validation"],
    )
    test_dataset = VolleyballDataset(
        mode="test", full_image=False, crop=True, n_frames=cfg.n_frames, with_teams=True, transform=tf["test"],
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

    # (crops_batch, person_labels_batch, group_labels_batch, masks_batch, team_ids_batch)
 
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_dp = n_gpus > 1 and cfg.get("data_parallel", True)

    # ═════════════════════════════════════════════════════════════════════
    # STAGE A — person-action pretraining of LSTM 1 (9 classes)
    # ═════════════════════════════════════════════════════════════════════
    stage_a_cfg = cfg.stage_a
    print(f"\n{'='*60}")
    print(f"  STAGE A: Player LSTM 1 pretrain ({stage_a_cfg.num_epochs} epochs, lr={stage_a_cfg.learning_rate})")
    print(f"  Backbone: {cfg.model.name} (frozen, checkpoint={cfg.model.get('checkpoint')})")
    print(f"  Batch: effective {effective_batch} = micro {micro_batch} × {accum_steps} accumulation steps")
    print(f"  Target: {NUM_PERSON_ACTIONS} classes — {list(PERSON_ACTION_TO_IDX.keys())}")
    print(f"{'='*60}")

    person_model = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=cfg.model.get("checkpoint"),
        lstm_hidden=cfg.lstm1.hidden_dim,
        lstm_layers=cfg.lstm1.num_layers,
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
    optimizer_a = optim.AdamW(
        trainable_a,
        lr=stage_a_cfg.learning_rate,
        weight_decay=stage_a_cfg.get("weight_decay", 5e-4),
    )
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_a):,}")

    # Stage A selects its best checkpoint on validation ACCURACY (not F1).
    best_acc_a = trainer.run_stage(
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
        desc="B8-A",
        select_metric="acc",
    )

    # ═════════════════════════════════════════════════════════════════════
    # STAGE B — scene LSTM 2 + fusion + head (8 classes), player model frozen
    # ═════════════════════════════════════════════════════════════════════
    stage_b_cfg = cfg.stage_b
    print(f"\n{'='*60}")
    print(f"  STAGE B: Scene LSTM 2 + head ({stage_b_cfg.get('warmup_epochs', 10)} probe + "
          f"{stage_b_cfg.num_epochs} fine-tune epochs)")
    print(f"  Player pool (per frame): {cfg.get('pool', 'max')}")
    print(f"  Target: {NUM_GROUP_ACTIVITIES} classes — {list(GROUP_ACTIVITY_TO_IDX.keys())}")
    print(f"{'='*60}")

    # Reload best Stage-A weights so B always starts from the best checkpoint.
    # LSTM 1 dimensions come from the checkpoint itself (see section 3) — the
    # config may have been retuned since that Stage A was trained.
    state_a = _checkpoint_state(stage_a_ckpt)
    lstm1_hidden_a, lstm1_layers_a = _person_dims(state_a)
    if (lstm1_hidden_a, lstm1_layers_a) != (cfg.lstm1.hidden_dim, cfg.lstm1.num_layers):
        print(
            f"  ⚠ Config LSTM 1 ({cfg.lstm1.hidden_dim}×{cfg.lstm1.num_layers}) ≠ "
            f"checkpoint '{stage_a_ckpt}' ({lstm1_hidden_a}×{lstm1_layers_a}) — "
            "using the checkpoint's dimensions.",
        )
    reloaded = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=None,
        lstm_hidden=lstm1_hidden_a,
        lstm_layers=lstm1_layers_a,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    reloaded, _, _, _, _ = load_model(stage_a_ckpt, reloaded)

    group_model = GroupTemporalClassifier(
        person_model=reloaded,
        num_classes=NUM_GROUP_ACTIVITIES,
        lstm2_hidden=cfg.lstm2.hidden_dim,
        pool=cfg.get("pool", "max"),
        T=cfg.n_frames,
        hidden_dim=stage_b_cfg.get("hidden_dim", 512),
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

    # Two-phase Stage B (mirrors B6): first train the fresh scene modules with
    # the player model frozen, then joint fine-tuning with differential LRs.
    # Both phases write the same stage_b_ckpt, so the Trainer shares one best
    # (on val accuracy) across them.
    warmup_epochs = stage_b_cfg.get("warmup_epochs", 10)
    warmup_lr = stage_b_cfg.get("warmup_lr", 1e-3)
    head_mult = stage_b_cfg.get("head_lr_multiplier", 10)
    weight_decay_b = stage_b_cfg.get("weight_decay", 5e-4)
    patience_b = stage_b_cfg.get("early_stopping_patience", 0)

    stage_b_kwargs = dict(
        model=group_model,
        save_ref=group_inner,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_b,
        num_classes=NUM_GROUP_ACTIVITIES,
        checkpoint_name=stage_b_ckpt,
        class_to_idx=GROUP_ACTIVITY_TO_IDX,
        batch_unpack=temporal_crop_unpack,
        accum_steps=accum_steps,
        patience=patience_b,
        tb_prefix="StageB",
        select_metric="acc",
    )

    # ── Phase 1: probe — player model frozen, fresh scene modules only ───
    scene_params = [p for p in group_model.parameters() if p.requires_grad]
    # optimizer_probe = optim.SGD(
    #     scene_params,
    #     lr=warmup_lr,
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=weight_decay_b,
    # )
    # AdamW equivalent — probe trains the fresh scene modules (AdamW-scale warmup_lr).
    optimizer_probe = optim.AdamW(
        scene_params,
        lr=warmup_lr,
        weight_decay=weight_decay_b,
    )
    print(f"  Phase 1 — probe: {warmup_epochs} epochs, scene-module lr={warmup_lr}")
    print(f"  Trainable parameters: {sum(p.numel() for p in scene_params):,}")
    trainer.run_stage(
        optimizer=optimizer_probe, num_epochs=warmup_epochs,
        stage="B-probe", desc="B8-B-probe", **stage_b_kwargs,
    )

    # ── Phase 2: joint fine-tune — unfreeze LSTM 1 + player projection ───
    player_params = group_inner.unfreeze_player_temporal()
    # optimizer_ft = optim.SGD(
    #     [
    #         {"params": player_params, "lr": stage_b_cfg.learning_rate},
    #         {"params": scene_params, "lr": stage_b_cfg.learning_rate * head_mult},
    #     ],
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=weight_decay_b,
    # )
    # AdamW equivalent — differential LR groups preserved; AdamW-scale player
    # lr (1e-4 fine-tune of the pretrained LSTM 1), scene modules at ×head_mult.
    optimizer_ft = optim.AdamW(
        [
            {"params": player_params, "lr": stage_b_cfg.learning_rate},
            {"params": scene_params, "lr": stage_b_cfg.learning_rate * head_mult},
        ],
        weight_decay=weight_decay_b,
    )
    scheduler_ft = build_scheduler(optimizer_ft, cfg)
    print(f"\n  Phase 2 — joint fine-tune: {stage_b_cfg.num_epochs} epochs, "
          f"player lr={stage_b_cfg.learning_rate}, "
          f"scene lr={stage_b_cfg.learning_rate * head_mult}")
    print(f"  Trainable parameters: "
          f"{sum(p.numel() for p in player_params) + sum(p.numel() for p in scene_params):,}")
    best_acc_b = trainer.run_stage(
        optimizer=optimizer_ft, scheduler=scheduler_ft, num_epochs=stage_b_cfg.num_epochs,
        stage="B-ft", desc="B8-B-ft", **stage_b_kwargs,
    )

    # ── Test best Stage-B model ──────────────────────────────────────────
    # Rebuilt from the SAVED checkpoint's architecture: within this run the
    # Stage-A reload above may already differ from the config, and the same
    # holds when testing a checkpoint from an older iteration.
    print("\n--- Testing best Stage-B model ---")
    state_b = _checkpoint_state(stage_b_ckpt)
    lstm1_hidden_b, lstm1_layers_b = _person_dims(state_b, prefix="person.")
    lstm2_hidden_b, pool_b, T_b, head_hidden_b = _group_dims(
        state_b, cfg_pool=cfg.get("pool", "max"),
    )
    fresh_person = PersonTemporalLSTM(
        num_actions=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        checkpoint=None,
        lstm_hidden=lstm1_hidden_b,
        lstm_layers=lstm1_layers_b,
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    best_group = GroupTemporalClassifier(
        person_model=fresh_person,
        num_classes=NUM_GROUP_ACTIVITIES,
        lstm2_hidden=lstm2_hidden_b,
        pool=pool_b,
        T=T_b,
        hidden_dim=head_hidden_b,
        dropout=stage_b_cfg.get("dropout", 0.3),
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
        best_val_f1=best_acc_b,
        hparam_dict={
            "baseline":                "baseline8",
            "batch_size":              effective_batch,
            "micro_batch_size":        micro_batch,
            "accumulation_steps":      accum_steps,
            "n_frames":                cfg.n_frames,
            "stage_a_epochs":          stage_a_cfg.num_epochs,
            "stage_a_lr":              stage_a_cfg.learning_rate,
            "stage_a_patience":        stage_a_cfg.get("early_stopping_patience", 0),
            "stage_b_epochs":          stage_b_cfg.num_epochs,
            "stage_b_lr":              stage_b_cfg.learning_rate,
            "stage_b_hidden_dim":      stage_b_cfg.get("hidden_dim", 512),
            "pool":                    cfg.get("pool", "max"),
            "stage_b_patience":        stage_b_cfg.get("early_stopping_patience", 0),
            "lstm1_hidden":            cfg.lstm1.hidden_dim,
            "lstm1_layers":            cfg.lstm1.num_layers,
            "lstm2_hidden":            cfg.lstm2.hidden_dim,
            "dropout":                 float(cfg.get("dropout", 0.0)),
            "label_smoothing":         cfg.get("label_smoothing", 0.0),
            "scheduler":               cfg.lr_scheduler.name if cfg.get("lr_scheduler") else "none",
            "backbone":                cfg.model.name,
            "backbone_checkpoint":     str(cfg.model.get("checkpoint")),
            "best_stage_a_val_acc":    best_acc_a,
        },
    )

    trainer.close()


if __name__ == "__main__":
    train_test()
