"""Standalone Baseline 8 model (no training/data dependencies).

Extracted verbatim from ``models/baseline8.py`` and ``utils/featureExtractor.py``
of the project repo so the Space can run without Hydra, LMDB or the trainer.
Only the checkpoint-loading helper of FeatureExtractor was dropped: the Stage-B
checkpoint already contains the backbone weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

NUM_PERSON_ACTIONS = 9
NUM_GROUP_ACTIVITIES = 8

_SUPPORTED_BACKBONES = {
    "resnet50": models.ResNet50_Weights,
    "resnet101": models.ResNet101_Weights,
}


class FeatureExtractor(nn.Module):
    """
    Frozen ResNet feature extractor: images → feature vectors.

    Parameters
    ----------
    model_name : str
        Backbone name — one of ``"resnet50"``, ``"resnet101"``.
        Unknown names raise (no silent fallback).
    checkpoint : str or None
        Optional checkpoint filename under ``MODEL_SAVE_DIR`` (as saved by
        ``utils.utility.save_model``). Loads the backbone weights from it,
        e.g. Baseline 1's fine-tuned ResNet. ``None`` → ImageNet weights
        (``Weights.DEFAULT``, same as the training scripts).
    pretrained : bool
        Only consulted when ``checkpoint`` is None. ``False`` skips the
        ImageNet download and leaves the backbone randomly initialized —
        for callers that restore every weight from a full-model checkpoint
        right after construction (e.g. utils.evaluate), where loading
        ImageNet weights would be wasted work.

    """

    def __init__(
        self,
        model_name: str = "resnet50",
        checkpoint: str | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if model_name not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{model_name}'. "
                f"Choose from {sorted(_SUPPORTED_BACKBONES)}.",
            )

        weights = (
            _SUPPORTED_BACKBONES[model_name].DEFAULT
            if pretrained and not checkpoint else None
        )
        backbone = getattr(models, model_name)(weights=weights)

        self.feature_dim = backbone.fc.in_features

        if checkpoint:
            self._load_backbone_from_checkpoint(backbone, checkpoint)
            print(f"  [FeatureExtractor] {model_name} weights loaded from checkpoint: {checkpoint}")
        elif pretrained:
            print(f"  [FeatureExtractor] {model_name} using generic ImageNet weights "
                  "(no project checkpoint given)")
        else:
            print(f"  [FeatureExtractor] {model_name} left uninitialized — "
                  "weights expected from a full-model checkpoint load")
        
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Permanently frozen: no grads, BN in eval mode.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()


    def train(self, mode: bool = True):
        # Stay in eval mode even if a parent module calls .train() —
        # frozen BN running stats must not switch to batch statistics.
        super().train(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) — already transformed/normalized.
        returns : (B, feature_dim)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D batch (B, C, H, W), got shape {tuple(x.shape)}")
        # no_grad, not inference_mode: inference tensors cannot be saved for
        # backward, which breaks training any trainable layers (LSTM/MLP)
        # stacked on top of these features.
        with torch.no_grad():
            return self.backbone(x)



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

