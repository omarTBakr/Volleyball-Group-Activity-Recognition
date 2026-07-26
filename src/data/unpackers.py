"""
Canonical batch unpackers for the volleyball baselines.

An *unpacker* turns a raw batch off the DataLoader into
``(model_inputs_tuple, target_tensor)`` — the contract the shared epoch driver
in :func:`utils.utility._run_one_epoch` expands into ``model(*inputs)``. It
returns ``None`` to skip a batch (e.g. an all-padding batch with no real
players).

Why a separate module (and not ``Dataset.__getitem__``)
------------------------------------------------------
Unpacking happens **after** ``collate_fn`` — it reshapes the *batched,
padded* tensors ``(crops, person_labels, group_labels, masks[, team_ids])``,
folds players into the batch, and drops padded slots via the mask. A
``Dataset`` only ever sees one clip at a time, so this logic cannot live in
``__getitem__``; it is inherently a per-batch transform. Centralizing it here
removes the near-identical copies that used to live in every crop-mode
baseline.

An unpacker encodes three things at once: the collate output shape, the
model's ``forward`` signature, and which label is the target. That coupling is
stable for the two canonical tasks below — if a model's ``forward`` signature
changes, its unpacker is what must change with it.

Tasks
-----
``person_frame``  single middle-frame crops → per-player action (single-input
                  model). Used by B3 Stage A.
``person_seq``    temporal crops → per-player action; each valid player's whole
                  ``(T,C,H,W)`` sequence is one *single* input. Used by B5
                  Stage A (its ``forward`` takes just the sequence).
``person_track``  temporal crops → per-player action; each valid player as an
                  independent P=1 track for a ``forward(seqs, masks)`` model.
                  Used by B6/B7 Stage A.
``group_crop``    crops (single-frame OR temporal) → group activity
                  (``forward(crops, masks)``; the model pools internally).
                  Used by B3/B5/B6/B7 Stage B.
``group_team``    crops + masks + team_ids → group activity
                  (``forward(crops, masks, team_ids)``; the model pools each
                  team separately). Used by B8 Stage B — needs the loader
                  built with ``with_teams=True`` so collate emits team_ids.

``person_seq`` vs ``person_track`` differ only in the model contract they feed:
B5 pools nothing at Stage A and takes the raw sequence, while B6/B7 route each
track through their per-frame pooling (an identity at P=1) and so need the mask.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

# Mirrors utils.utility.BatchUnpack; redeclared here to keep this module free of
# a utils.utility import (and any import-cycle risk with the data package).
BatchUnpack = Callable[[Any], tuple[tuple[torch.Tensor, ...], torch.Tensor] | None]


def person_frame_unpack(batch):
    """Single-frame crops → per-player action classification.

    ``(crops[B,P,C,H,W], person_labels[B,P], group_labels, masks[B,P])``
    → ``((valid_crops[N,C,H,W],), valid_person_labels[N])`` for a single-input
    person-action model. Padded players are dropped via the mask.
    """
    if not batch or len(batch) < 4:
        return None
    crops, person_labels, _group_labels, masks = batch[0], batch[1], batch[2], batch[3]
    if crops.dim() < 5 or crops.numel() == 0:
        return None

    b, p = crops.shape[:2]
    flat = crops.reshape(b * p, *crops.shape[2:])
    labels = person_labels.reshape(b * p)
    valid = masks.reshape(b * p).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return None
    return (flat[valid],), labels[valid]


def person_seq_unpack(batch):
    """Temporal crops → per-player action, each player's sequence a single input.

    ``(crops[B,T,P,C,H,W], person_labels[B,P], group_labels, masks[B,P])``
    → ``((valid_sequences[N,T,C,H,W],), valid_person_labels[N])`` for a model
    whose ``forward`` takes just the sequence (B5's Stage A pools nothing).
    Padded players are dropped via the mask.
    """
    if not batch or len(batch) < 4:
        return None
    crops, person_labels, _group_labels, masks = batch[0], batch[1], batch[2], batch[3]
    if crops.dim() != 6 or crops.numel() == 0:
        return None

    b, t, p = crops.shape[:3]
    seqs = crops.permute(0, 2, 1, 3, 4, 5).reshape(b * p, t, *crops.shape[3:])
    labels = person_labels.reshape(b * p)
    valid = masks.reshape(b * p).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return None
    return (seqs[valid],), labels[valid]


def person_track_unpack(batch):
    """Temporal crops → per-player action classification on P=1 tracks.

    ``(crops[B,T,P,C,H,W], person_labels[B,P], group_labels, masks[B,P])``
    → ``((tracks[N,T,1,C,H,W], track_masks[N,1]), valid_person_labels[N])``.

    Each valid player becomes an independent P=1 "clip" with an all-ones mask,
    so a model's per-frame player pooling is an identity and its LSTM sees that
    one person's track. For models whose ``forward`` is ``(seqs, masks)``.
    """
    if not batch or len(batch) < 4:
        return None
    crops, person_labels, _group_labels, masks = batch[0], batch[1], batch[2], batch[3]
    if crops.dim() != 6 or crops.numel() == 0:
        return None

    b, t, p = crops.shape[:3]
    seqs = crops.permute(0, 2, 1, 3, 4, 5).reshape(b * p, t, *crops.shape[3:])
    labels = person_labels.reshape(b * p)
    valid = masks.reshape(b * p).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return None
    tracks = seqs[valid].unsqueeze(2)                        # (N, T, 1, C, H, W)
    track_masks = torch.ones(
        tracks.shape[0], 1, dtype=torch.bool, device=tracks.device,
    )
    return (tracks, track_masks), labels[valid]


def group_crop_unpack(batch):
    """Crops (single-frame OR temporal) → group-activity classification.

    ``(crops, person_labels, group_labels, masks[, team_ids])``
    → ``((crops, masks), group_labels)`` for models whose ``forward`` is
    ``(crops, masks)`` and that pool players internally. Handles single-frame
    ``(B,P,…)`` and temporal ``(B,T,P,…)`` crops alike, and ignores any extra
    collate elements (e.g. ``team_ids`` when the loader has ``with_teams``).
    """
    if not batch or len(batch) < 4:
        return None
    crops, group_labels, masks = batch[0], batch[2], batch[3]
    if crops.dim() < 5 or crops.numel() == 0:
        return None
    return (crops, masks), group_labels


def group_team_unpack(batch):
    """Crops + masks + team_ids → group-activity classification (team split).

    ``(crops, person_labels, group_labels, masks, team_ids)``
    → ``((crops, masks, team_ids), group_labels)`` for models whose ``forward``
    is ``(crops, masks, team_ids)`` and that pool each team separately (B8).

    Requires the loader built with ``with_teams=True`` — its collate emits the
    5th ``team_ids`` tensor ``(B, P)`` (0 = left court side, 1 = right, -1 for
    padded slots). Returns ``None`` if team_ids is absent (loader not in team
    mode), which surfaces the misconfiguration instead of silently pooling
    all players.
    """
    if not batch or len(batch) < 5:
        return None
    crops, group_labels, masks, team_ids = batch[0], batch[2], batch[3], batch[4]
    if crops.dim() < 5 or crops.numel() == 0:
        return None
    return (crops, masks, team_ids), group_labels


# Task name → unpacker. The factory lets a caller select an unpacker by a
# parameter (mirroring how the loader is configured by mode) instead of
# importing the function directly.
_UNPACKERS: dict[str, BatchUnpack] = {
    "person_frame": person_frame_unpack,
    "person_seq": person_seq_unpack,
    "person_track": person_track_unpack,
    "group_crop": group_crop_unpack,
    "group_team": group_team_unpack,
}


def get_unpacker(task: str) -> BatchUnpack:
    """Return the unpacker for ``task`` (see module docstring for the names)."""
    try:
        return _UNPACKERS[task]
    except KeyError:
        raise ValueError(
            f"Unknown unpacker task '{task}'. Choose from {sorted(_UNPACKERS)}.",
        ) from None
