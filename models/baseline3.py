"""
Baseline 3 — Two-stage person-then-group classification on per-player crops.

Stage A (person-action pretraining, 9 classes):
    ResNet-50 backbone trained on individual player crops to predict one of
    9 person-action labels (blocking, digging, falling, jumping, moving,
    setting, spiking, standing, waiting). Pure per-crop classification —
    the group label is ignored here.

Stage B (group-activity fine-tune, 8 classes):
    Load Stage A's ResNet, drop its 9-way head, freeze the entire backbone.
    For each clip the per-player crops are pushed through the frozen
    backbone, the resulting [P, 2048] features are max-pooled across the
    player dimension (with a validity mask so padded players can't win the
    max), and a small MLP classifies the pooled vector into one of 8 group
    activities.

Only this file and configs/baseline3.yaml are modified — all data loading,
checkpoint I/O, and metric helpers are reused as-is.
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
    NUM_PERSON_ACTIONS,
    PERSON_ACTION_TO_IDX,
)
from src.data.kaggle_data_loader import VolleyballDataset, collate_fn
from src.data.unpackers import group_crop_unpack, person_frame_unpack
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


class PersonActionResNet(nn.Module):
    """ResNet → 9-class person-action classifier (Stage A).

    Backbone is selectable via ``backbone_name`` (``"resnet50"`` or
    ``"resnet101"``). Wired up from ``cfg.model.name`` in the YAML.
    """

    def __init__(
        self,
        num_classes: int = NUM_PERSON_ACTIONS,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif backbone_name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                "Use 'resnet50' or 'resnet101'."
            )

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GroupActivityModel(nn.Module):
    """
    Stage B: frozen ResNet feature extractor → pool across players → MLP → 8-class.

    The aggregation across the player dimension is selectable:
      - ``pool="max"``    → classifier input is ``feature_dim``. (Legacy.)
      - ``pool="mean"``   → classifier input is ``feature_dim``.
      - ``pool="concat"`` → classifier input is ``2 * feature_dim``. Max captures
        "is some player exhibiting feature k strongly?", mean captures "what's
        the typical team level of feature k?" — together they encode both
        signals.

    Parameters
    ----------
    person_model : PersonActionResNet
        Already-trained Stage A model. Its fc is replaced with ``Identity`` so the
        backbone returns the feature_dim-wide vector before classification.
    num_classes : int
        Number of group-activity classes (8).
    hidden_dim : int
        Width of the MLP hidden layer.
    dropout : float
        Dropout applied inside the MLP head.
    pool : {"max", "mean", "concat"}
        Player-dim aggregation. Default ``"concat"``. Pass ``"max"`` to match
        checkpoints saved before the concat-pool change.
    """

    _VALID_POOLS = ("max", "mean", "concat")

    def __init__(
        self,
        person_model: PersonActionResNet,
        num_classes: int = NUM_GROUP_ACTIVITIES,
        hidden_dim: int = 512,
        dropout: float = 0.4,
        pool: str = "concat",
    ) -> None:
        super().__init__()

        if pool not in self._VALID_POOLS:
            raise ValueError(
                f"Unsupported pool '{pool}'. Use one of {self._VALID_POOLS}."
            )
        self.pool = pool

        self.feature_dim = person_model.feature_dim
        self.backbone = person_model.backbone
        self.backbone.fc = nn.Identity()  # output feature_dim-wide vector per crop

        # Freeze the entire ResNet — only the MLP head trains in Stage B.
        for p in self.backbone.parameters():
            p.requires_grad = False

        classifier_in = 2 * self.feature_dim if pool == "concat" else self.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval mode so BN running stats don't drift.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, crops: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        crops : (B, P, C, H, W)
        masks : (B, P) bool — True for real players, False for padding
        returns: (B, num_classes) logits
        """
        B, P, C, H, W = crops.shape
        flat = crops.view(B * P, C, H, W)
        feats = self.backbone(flat).view(B, P, self.feature_dim)  # (B, P, D)

        mask_3d = masks.unsqueeze(-1).expand_as(feats)

        if self.pool in ("max", "concat"):
            # Max pool: drive padded slots to -inf so they cannot win the max.
            feats_for_max = feats.masked_fill(~mask_3d, float("-inf"))
            pooled_max, _ = feats_for_max.max(dim=1)  # (B, D)
            # Clips with zero valid players → all -inf; sanitize to 0.
            pooled_max = torch.where(
                torch.isinf(pooled_max), torch.zeros_like(pooled_max), pooled_max,
            )

        if self.pool in ("mean", "concat"):
            # Mean pool: zero out padded slots, divide by valid count per clip.
            feats_for_mean = feats.masked_fill(~mask_3d, 0.0)
            valid_count = masks.sum(dim=1).clamp_min(1).unsqueeze(-1).float()  # (B, 1)
            pooled_mean = feats_for_mean.sum(dim=1) / valid_count  # (B, D)

        if self.pool == "max":
            pooled = pooled_max
        elif self.pool == "mean":
            pooled = pooled_mean
        else:  # "concat"
            pooled = torch.cat([pooled_max, pooled_mean], dim=-1)  # (B, 2D)

        return self.classifier(pooled)


# ══ 2. BATCH UNPACKERS ══
#
# The canonical unpackers live in src.data.unpackers (shared across baselines).
# B3 works on single middle-frame crops: Stage A is per-player action
# classification (single-input model), Stage B is group classification via
# forward(crops, masks). Re-exported under the historical names so external
# importers (utils.evaluate) keep resolving them from this module.
stage_a_unpack = person_frame_unpack
stage_b_unpack = group_crop_unpack


# ═════════════════════════════════════════════════════════════════════════════
# ══ 4. MAIN ENTRYPOINT ══
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="baseline3", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Logging ──────────────────────────────────────────────────────────
    run_id = Trainer.next_run_id("baseline3")
    trainer = Trainer("baseline3", run_id, device=device)
    stage_a_ckpt = f"baseline3_stage_a_{run_id}.pt"
    stage_b_ckpt = f"baseline3_stage_b_{run_id}.pt"

    # ── Data (crop mode, single middle frame per clip) ───────────────────
    tf = build_transforms(cfg)

    train_dataset = VolleyballDataset(
        mode="train", full_image=False, crop=True, n_frames=1, transform=tf["train"],
    )
    val_dataset = VolleyballDataset(
        mode="validation", full_image=False, crop=True, n_frames=1, transform=tf["validation"],
    )
    test_dataset = VolleyballDataset(
        mode="test", full_image=False, crop=True, n_frames=1, transform=tf["test"],
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

    # ═════════════════════════════════════════════════════════════════════
    # STAGE A — person-action pretraining (9 classes)
    # ═════════════════════════════════════════════════════════════════════
    stage_a_cfg = cfg.stage_a
    print(f"\n{'='*60}")
    print(f"  STAGE A: Person-Action Pretrain ({stage_a_cfg.num_epochs} epochs, lr={stage_a_cfg.learning_rate})")
    print(f"  Target: {NUM_PERSON_ACTIONS} classes — {list(PERSON_ACTION_TO_IDX.keys())}")
    print(f"{'='*60}")

    person_model = PersonActionResNet(
        num_classes=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        pretrained=cfg.model.get("pretrained", True),
    ).to(device)

    # Keep an unwrapped reference for checkpoint I/O (DataParallel prefixes
    # state-dict keys with "module."; saving the inner module keeps checkpoints
    # round-trippable into either wrapped or unwrapped models).
    person_inner = person_model

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_dp = n_gpus > 1 and cfg.get("data_parallel", True)
    if use_dp:
        print(f"  DataParallel across {n_gpus} GPUs (each sees batch_size/{n_gpus})")
        person_model = nn.DataParallel(person_model)

    # Inverse-frequency class weights for the (heavily skewed) 9 person-action
    # labels. Without this the backbone collapses to "always predict standing"
    # for the first ~10 epochs and never recovers a useful representation of
    # the minority actions (spiking/setting/passing) — exactly the actions that
    # carry signal for Stage B's group classification.
    use_class_weights = stage_a_cfg.get("class_weighted_loss", True)
    if use_class_weights:
        counts = person_action_label_counts(train_dataset.samples, NUM_PERSON_ACTIONS)
        cw = inverse_freq_weights(counts, NUM_PERSON_ACTIONS)
        idx_to_name = {v: k for k, v in PERSON_ACTION_TO_IDX.items()}
        print("  Per-class person-action stats (count → weight):")
        for i in range(NUM_PERSON_ACTIONS):
            print(f"    {idx_to_name[i]:<10s}  n={int(counts[i]):>6d}  w={float(cw[i]):.3f}")
        criterion_a = nn.CrossEntropyLoss(
            weight=cw.to(device),
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        criterion_a = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    # optimizer_a = optim.SGD(
    #     person_model.parameters(),
    #     lr=stage_a_cfg.learning_rate,
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=stage_a_cfg.get("weight_decay", 5e-4),
    # )
    # AdamW equivalent — Stage A fine-tunes the full ResNet, so use an
    # AdamW-scale lr from the config (1e-4; 1e-3 would wreck pretrained features).
    optimizer_a = optim.AdamW(
        person_model.parameters(),
        lr=stage_a_cfg.learning_rate,
        weight_decay=stage_a_cfg.get("weight_decay", 5e-4),
    )

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
        patience=stage_a_cfg.get("early_stopping_patience", 0),
        stage="A",
        desc="B3-A",
    )

    # ═════════════════════════════════════════════════════════════════════
    # STAGE B — group-activity fine-tune (8 classes)
    # ═════════════════════════════════════════════════════════════════════
    stage_b_cfg = cfg.stage_b
    print(f"\n{'='*60}")
    print(f"  STAGE B: Group-Activity (frozen backbone) ({stage_b_cfg.num_epochs} epochs)")
    print(f"  MLP head: {person_inner.feature_dim} → {stage_b_cfg.hidden_dim} → {NUM_GROUP_ACTIVITIES}")
    print(f"{'='*60}")

    # Reload best Stage-A weights into a fresh (unwrapped) ResNet so we always start
    # B from the best person-action checkpoint, not the last epoch's.
    reloaded = PersonActionResNet(
        num_classes=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        pretrained=False,
    ).to(device)
    reloaded, _, _, _, _ = load_model(stage_a_ckpt, reloaded)

    group_model = GroupActivityModel(
        person_model=reloaded,
        num_classes=NUM_GROUP_ACTIVITIES,
        hidden_dim=stage_b_cfg.hidden_dim,
        dropout=stage_b_cfg.get("dropout", 0.4),
        pool=stage_b_cfg.get("pool", "concat"),
    ).to(device)
    group_inner = group_model

    if use_dp:
        print(f"  DataParallel across {n_gpus} GPUs")
        group_model = nn.DataParallel(group_model)

    # Inverse-frequency class weights for the 8 group-activity classes.
    # l/r_winpoint are about 2.5× rarer than the spike/pass/set classes in this
    # split; without weighting the head learns to suppress winpoint predictions
    # and the model's biggest confusion (l-* vs r-*) gets even worse on the
    # minority side.
    use_class_weights_b = stage_b_cfg.get("class_weighted_loss", True)
    if use_class_weights_b:
        counts_b = group_activity_label_counts(train_dataset.samples, NUM_GROUP_ACTIVITIES)
        cw_b = inverse_freq_weights(counts_b, NUM_GROUP_ACTIVITIES)
        idx_to_name = {v: k for k, v in GROUP_ACTIVITY_TO_IDX.items()}
        print("  Per-class group-activity stats (count → weight):")
        for i in range(NUM_GROUP_ACTIVITIES):
            print(f"    {idx_to_name[i]:<12s}  n={int(counts_b[i]):>5d}  w={float(cw_b[i]):.3f}")
        criterion_b = nn.CrossEntropyLoss(
            weight=cw_b.to(device),
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        criterion_b = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    # optimizer_b = optim.SGD(
    #     group_inner.classifier.parameters(),
    #     lr=stage_b_cfg.learning_rate,
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=stage_b_cfg.get("weight_decay", 5e-4),
    # )
    # AdamW equivalent — Stage B trains only the MLP head on frozen features.
    optimizer_b = optim.AdamW(
        group_inner.classifier.parameters(),
        lr=stage_b_cfg.learning_rate,
        weight_decay=stage_b_cfg.get("weight_decay", 5e-4),
    )
    scheduler_b = build_scheduler(optimizer_b, cfg)

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
        batch_unpack=stage_b_unpack,
        patience=stage_b_cfg.get("early_stopping_patience", 0),
        stage="B",
        desc="B3-B",
    )

    # ── Test best Stage-B model ──────────────────────────────────────────
    fresh_person = PersonActionResNet(
        num_classes=NUM_PERSON_ACTIONS,
        backbone_name=cfg.model.name,
        pretrained=False,
    ).to(device)
    best_group = GroupActivityModel(
        person_model=fresh_person,
        num_classes=NUM_GROUP_ACTIVITIES,
        hidden_dim=stage_b_cfg.hidden_dim,
        dropout=stage_b_cfg.get("dropout", 0.4),
        pool=stage_b_cfg.get("pool", "concat"),
    ).to(device)
    best_group, _, _, _, _ = load_model(stage_b_ckpt, best_group)
    if use_dp:
        best_group = nn.DataParallel(best_group)

    trainer.run_test(
        model=best_group,
        test_loader=test_loader,
        criterion=criterion_b,
        num_classes=NUM_GROUP_ACTIVITIES,
        batch_unpack=stage_b_unpack,
        best_val_f1=best_f1_b,
        hparam_dict={
            "baseline":                "baseline3",
            "batch_size":              cfg.batch_size,
            "stage_a_epochs":          stage_a_cfg.num_epochs,
            "stage_a_lr":              stage_a_cfg.learning_rate,
            "stage_a_weight_decay":    stage_a_cfg.get("weight_decay", 0.0),
            "stage_a_patience":        stage_a_cfg.get("early_stopping_patience", 0),
            "stage_b_epochs":          stage_b_cfg.num_epochs,
            "stage_b_lr":              stage_b_cfg.learning_rate,
            "stage_b_weight_decay":    stage_b_cfg.get("weight_decay", 0.0),
            "stage_b_hidden_dim":      stage_b_cfg.hidden_dim,
            "stage_b_dropout":         float(stage_b_cfg.get("dropout", 0.0)),
            "stage_b_patience":        stage_b_cfg.get("early_stopping_patience", 0),
            "label_smoothing":         cfg.get("label_smoothing", 0.0),
            "scheduler":               cfg.lr_scheduler.name if cfg.get("lr_scheduler") else "none",
            "backbone":                cfg.model.name,
            "best_stage_a_val_f1":     best_f1_a,
        },
    )

    trainer.close()


if __name__ == "__main__":
    train_test()
