"""
Shared implementation for the volleyball dataset loaders.

``BaseVolleyballDataset`` holds every piece of logic that does not depend
on the physical frame storage: split filtering, frame-window selection,
person cropping, label mapping, ``__getitem__`` routing, and the collate
function.  Concrete subclasses only provide:

    _load_frame_index() -> dict[(video_id, clip_id), list[frame_name]]
    _load_image(video_id, clip_id, frame_name) -> PIL.Image

Two subclasses exist:

    src.data.data_loader.VolleyballDataset         — frames from LMDB
    src.data.kaggle_data_loader.VolleyballDataset  — frames from disk

Both keep their original public interface (constructor args, return
shapes, ``collate_fn``), so existing training scripts work unchanged.

Supported configurations (all baselines B1–B8):

    full_image=True,  n_frames=1  →  (image, group_label)                [B1]
    full_image=True,  n_frames=9  →  (images, group_label)               [B4]
    crop=True,        n_frames=1  →  (crops, person_labels, group_label) [B3]
    crop=True,        n_frames=9  →  (crops, person_labels, group_label) [B5-B8]
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch  # ty:ignore[unresolved-import]
from PIL import Image
from torch.utils.data import Dataset  # ty:ignore[unresolved-import]
from torchvision.transforms import ToTensor  # ty:ignore[unresolved-import]

from configs.data_split import (
    TEST_VIDEOS_NUMBERS,
    TRAIN_VIDEOS_NUMBERS,
    VALIDATION_VIDEO_NUMBERS,
)
from configs.labels import GROUP_ACTIVITY_TO_IDX, PERSON_ACTION_TO_IDX
from src.pickle_dump import load_from_pickle

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_video_ids_for_mode(mode: str) -> list[int]:
    """Return the video IDs for the given split ("train"/"validation"/"test")."""
    mode_map = {
        "train": TRAIN_VIDEOS_NUMBERS,
        "validation": VALIDATION_VIDEO_NUMBERS,
        "test": TEST_VIDEOS_NUMBERS,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {list(mode_map.keys())}.")
    return mode_map[mode]


# ── Dataset ──────────────────────────────────────────────────────────────────


class BaseVolleyballDataset(Dataset):
    """
    Storage-agnostic volleyball group-activity dataset.

    Parameters
    ----------
    mode : str
        Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
    n_frames : int
        Number of frames to sample per clip. Must be a positive odd number.
    full_image : bool
        If True, return full-resolution frames. Default True.
    crop : bool
        If True, return per-person crops from bounding boxes. Default False.
    transform : callable or None
        A torchvision transform applied to each image / crop.

    """

    def __init__(
        self,
        mode: str = "train",
        n_frames: int = 1,
        full_image: bool = True,
        crop: bool = False,
        transform: Callable | None = None,
        with_teams: bool = False,
    ) -> None:
        super().__init__()

        if crop and full_image:
            raise ValueError("crop and full_image cannot both be True. Choose one.")
        if n_frames % 2 == 0 or n_frames <= 0:
            raise ValueError(
                "n_frames must be a positive odd number to have a clear middle frame.",
            )
        if with_teams and not crop:
            raise ValueError("with_teams=True requires crop=True (team ids describe players).")

        self.mode = mode
        self.n_frames = n_frames
        self.full_image = full_image
        self.crop = crop
        self.transform = transform
        # Opt-in per-player team ids (0 = left court side, 1 = right), aligned
        # with the person axis. OFF by default so every existing baseline sees
        # the unchanged 3-tuple item / 4-tuple batch contract. B8's team-split
        # pooling turns this on. Team membership is derived from the box
        # center-x of the label-source frame, split at the median — NOT from
        # the player ordering, which is track-id-sorted (see
        # _get_persons_for_frame) and carries no court-side information.
        self.with_teams = with_teams

        self._master_data: dict = self._load_master_data()

        # The 'persons' detections are never read by the loader (crops come
        # from tracking, with 'actions' as fallback) yet hold ~40% of the
        # annotation RAM (~1.2 GB). Drop them so the dict forked into every
        # DataLoader worker is as small as possible. Idempotent — the pickle
        # cache is shared across train/val/test datasets.
        for clip in self._master_data.values():
            clip.pop("persons", None)

        self._frame_index: dict[tuple[str, str], list[str]] = self._load_frame_index()

        self.samples: list[tuple[str, str, dict]] = []
        self._build_samples()

        # Everything __getitem__ needs is precomputed here into compact
        # numpy/str records (~20 MB). DataLoader workers therefore never
        # traverse the multi-GB annotation dict: forked copy-on-write pages
        # stay shared instead of being gradually duplicated per worker as
        # sampling touches them (which is what OOM-kills workers mid-epoch).
        self._records: list[tuple] = []
        self._precompute_records()

        # After precompute, nothing in this object needs the multi-GB master
        # dict anymore. Swap each sample's clip_data for a slim stand-in that
        # keeps exactly what external consumers (the label-count helpers in
        # utils.utility) read, and drop our reference to the dict itself so
        # callers can free the process-wide cache (see
        # src.pickle_dump.free_master_data_cache) before workers fork.
        self._slim_samples()
        self._master_data = {}

    def _slim_samples(self) -> None:
        """Replace clip_data refs with minimal dicts (scene_class + middle frame)."""
        slim_samples: list[tuple[str, str, dict]] = []
        for video_id, clip_id, clip_data in self.samples:
            middle_name = f"{clip_id}.jpg"
            slim: dict = {"scene_class": clip_data.get("scene_class")}
            tracking_mid = clip_data.get("tracking", {}).get(middle_name)
            if tracking_mid is not None:
                slim["tracking"] = {middle_name: tracking_mid}
            actions_mid = clip_data.get("actions", {}).get(middle_name)
            if actions_mid is not None:
                slim["actions"] = {middle_name: actions_mid}
            slim_samples.append((video_id, clip_id, slim))
        self.samples = slim_samples

    def _precompute_records(self) -> None:
        """Resolve labels, frame selection, and person boxes per sample."""
        for video_id, clip_id, clip_data in self.samples:
            group_label = self._group_label(video_id, clip_id, clip_data)
            frame_names = tuple(self._select_frame_names(video_id, clip_id, clip_data))

            persons_per_frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if not self.full_image:
                for fname in frame_names:
                    boxes: list[tuple[int, int, int, int]] = []
                    labels: list[int] = []
                    for person in self._get_persons_for_frame(clip_data, fname):
                        box = person["box"]
                        # Tracking uses [x1,y1,x2,y2]; detections use [x,y,w,h]
                        if "id" in person:
                            x1, y1, x2, y2 = box
                        else:
                            x, y, w, h = box
                            x1, y1, x2, y2 = x, y, x + w, y + h
                        boxes.append((x1, y1, x2, y2))
                        action = person.get("action", "standing")
                        labels.append(
                            PERSON_ACTION_TO_IDX.get(action, PERSON_ACTION_TO_IDX["standing"]),
                        )
                    persons_per_frame[fname] = (
                        np.asarray(boxes, dtype=np.int32).reshape(-1, 4),
                        np.asarray(labels, dtype=np.int64),
                    )

            self._records.append(
                (video_id, clip_id, group_label, frame_names, persons_per_frame),
            )

    # ── Storage hooks (implemented by subclasses) ────────────────────────

    def _load_master_data(self) -> dict:
        """
        Return the master annotation dict keyed by ``"video_id/clip_id"``.

        Default implementation reads the pickled master JSON (shared,
        process-cached). Subclasses may override to build the same
        structure from another source (e.g. the raw annotation text files
        on disk — see ``kaggle_data_loader``). Each clip entry must offer
        ``"scene_class"`` plus per-frame ``"tracking"``/``"actions"``.
        """
        return load_from_pickle()

    def _load_frame_index(self) -> dict[tuple[str, str], list[str]]:
        """Return ``(video_id, clip_id) -> sorted list of frame filenames``."""
        raise NotImplementedError

    def _load_image(self, video_id: str, clip_id: str, frame_name: str) -> Image.Image:
        """Load one frame as a PIL RGB image."""
        raise NotImplementedError

    # ── Index building ────────────────────────────────────────────────────

    def _build_samples(self) -> None:
        """Filter the master pickle by the video IDs in the current split."""
        video_ids = _get_video_ids_for_mode(self.mode)
        valid_prefixes = {str(v) for v in video_ids}

        for clip_key, clip_data in self._master_data.items():
            # clip_key is "video_id/clip_id", e.g. "0/13286"
            video_id, clip_id = clip_key.split("/", 1)
            if video_id in valid_prefixes:
                self.samples.append((video_id, clip_id, clip_data))

    # ── Label mapping ─────────────────────────────────────────────────────

    def _group_label(self, video_id: str, clip_id: str, clip_data: dict) -> int:
        """
        Map the clip's ``scene_class`` string to its integer label.

        Fails loudly on labels missing from ``GROUP_ACTIVITY_TO_IDX`` —
        a silent fallback here once masked a corrupted annotation merge
        that mislabeled 62% of the dataset.
        """
        scene_class = clip_data.get("scene_class")
        if scene_class is None:
            logger.warning(
                "Clip %s/%s has no scene_class; defaulting to label 0. "
                "Re-run `python -m src.json_parser` to enrich the master JSON.",
                video_id, clip_id,
            )
            return 0
        if scene_class not in GROUP_ACTIVITY_TO_IDX:
            raise KeyError(
                f"Unknown scene_class '{scene_class}' for clip {video_id}/{clip_id}. "
                f"Expected one of {list(GROUP_ACTIVITY_TO_IDX)}. "
                "The master JSON/pickle is likely corrupted — regenerate it with "
                "`python -m src.json_parser` then `python -m src.pickle_dump`.",
            )
        return GROUP_ACTIVITY_TO_IDX[scene_class]

    # ── Frame selection ───────────────────────────────────────────────────

    def _find_valid_middle_index(
        self, all_frames: list[str], clip_id: str, clip_data: dict,
    ) -> int:
        """Find the index of the best middle frame, ensuring annotations exist if cropping."""
        middle_name = f"{clip_id}.jpg"
        if middle_name in all_frames:
            best_idx = all_frames.index(middle_name)
        else:
            best_idx = len(all_frames) // 2

        # If we need crops, ensure the selected middle frame actually has person annotations
        if self.crop:
            search_offsets = [0]
            for i in range(1, len(all_frames)):
                search_offsets.extend([i, -i])

            for offset in search_offsets:
                idx = best_idx + offset
                if 0 <= idx < len(all_frames):
                    fname = all_frames[idx]
                    if self._get_persons_for_frame(clip_data, fname):
                        return idx
        return best_idx

    def _pad_frame_sequence(
        self, selected: list[str], all_frames: list[str], start: int, end: int,
    ) -> list[str]:
        """Pad a sequence of frames symmetrically at boundaries to ensure length == n_frames."""
        while len(selected) < self.n_frames:
            if start > 0:
                start -= 1
                selected.insert(0, all_frames[start])
            elif end < len(all_frames):
                selected.append(all_frames[end])
                end += 1
            elif len(selected) > 0:
                # Duplicate extreme boundary frame if completely out of physical frames
                selected.append(selected[-1])
            else:
                break
        return selected

    def _select_frame_names(
        self, video_id: str, clip_id: str, clip_data: dict,
    ) -> list[str]:
        """Select exactly ``n_frames`` filenames centered on the best valid middle frame."""
        all_frames = self._frame_index.get((video_id, clip_id), [])
        if not all_frames:
            return []

        mid_idx = self._find_valid_middle_index(all_frames, clip_id, clip_data)

        half = self.n_frames // 2
        start = max(0, mid_idx - half)
        end = min(len(all_frames), mid_idx + half + 1)

        selected = all_frames[start:end]
        return self._pad_frame_sequence(selected, all_frames, start, end)

    # ── Person annotations ────────────────────────────────────────────────

    def _get_persons_for_frame(self, clip_data: dict, frame_name: str) -> list[dict]:
        """
        Return person entries for a specific frame.

        Prefers tracking data (which carries consistent player IDs) and
        falls back to action detections when tracking is unavailable.
        """
        persons = clip_data.get("tracking", {}).get(frame_name, [])
        if persons:
            # Sort by track id so player index p refers to the SAME person in
            # every frame of a clip — required by per-player temporal models
            # (B5+); harmless for order-invariant pooling baselines (B3).
            return sorted(persons, key=lambda p: p.get("id", 0))
        return clip_data.get("actions", {}).get(frame_name, [])

    @staticmethod
    def _crop_boxes(
        image: Image.Image, boxes: np.ndarray, labels: np.ndarray,
    ) -> tuple[list[Image.Image], list[int], list[float]]:
        """
        Crop person regions from the image given precomputed boxes.

        Boxes are ``(P, 4)`` int arrays in ``[x1, y1, x2, y2]`` format
        (detections were converted from ``[x, y, w, h]`` at precompute
        time); they are clamped to image bounds here, and degenerate
        boxes are skipped together with their labels.

        Returns the kept crops, their labels, and each kept box's clamped
        center-x (``(x1 + x2) / 2``). The center-x list stays aligned with
        crops/labels through the same skip filtering, so team assignment
        (see ``_team_ids``) lines up with the person axis exactly.
        """
        crops: list[Image.Image] = []
        kept_labels: list[int] = []
        kept_cx: list[float] = []
        img_w, img_h = image.size

        for (x1, y1, x2, y2), label in zip(boxes.tolist(), labels.tolist()):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            kept_labels.append(label)
            kept_cx.append((x1 + x2) / 2.0)

        return crops, kept_labels, kept_cx

    @staticmethod
    def _team_ids(center_x: Sequence[float]) -> torch.Tensor:
        """Rank-based left/right team split from box center-x → int64 ``(P,)``.

        The paper's recipe: order players by x, first half = one team. The
        ``P // 2`` leftmost players get team 0, the rest team 1. Ties break
        by stable sort; an odd count puts the extra player on the right
        (team 1). Returned in the ORIGINAL player order (track-id order), so
        it aligns with the crops and person labels — not sorted by x.
        """
        p = len(center_x)
        teams = torch.ones(p, dtype=torch.long)  # default right; left half overwritten
        if p <= 1:
            teams.zero_()  # a lone player can't be split — call it team 0
            return teams
        order = np.argsort(np.asarray(center_x, dtype=np.float64), kind="stable")
        teams[order[: p // 2].tolist()] = 0
        return teams

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Return a sample depending on the configured mode.

        Returns
        -------
        For ``full_image=True, n_frames=1``:
            ``(image_tensor, group_label)``

        For ``full_image=True, n_frames>1``:
            ``(images_tensor [T,C,H,W], group_label)``

        For ``crop=True, n_frames=1``:
            ``(crops_tensor [P,C,H,W], person_labels [P], group_label)``

        For ``crop=True, n_frames>1``:
            ``(crops_tensor [T,P,C,H,W], person_labels [P], group_label)``

        """
        # Read ONLY the precomputed record — never the master annotation dict
        # (workers must not dirty its copy-on-write pages; see __init__).
        video_id, clip_id, group_label, frame_names, persons_per_frame = self._records[index]
        frame_names = list(frame_names)

        if self.full_image:
            return self._getitem_full_image(video_id, clip_id, frame_names, group_label)
        return self._getitem_crop(
            video_id, clip_id, frame_names, persons_per_frame, group_label,
        )

    def _getitem_full_image(
        self,
        video_id: str,
        clip_id: str,
        frame_names: list[str],
        group_label: int,
    ) -> tuple[torch.Tensor, int]:
        """Load full frame(s) and return as tensor(s)."""
        if not frame_names:
            raise RuntimeError(
                f"No frames found for video={video_id} clip={clip_id}. "
                "Check that the frame storage (LMDB / dataset directory) is complete.",
            )

        images = []
        for fname in frame_names:
            img = self._load_image(video_id, clip_id, fname)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        if self.n_frames == 1:
            return images[0], group_label
        return torch.stack(images, dim=0), group_label

    def _empty_crop(self, group_label: int):
        """Return the mode-appropriate empty-crop item (teams-aware)."""
        empty, empty_l = torch.empty(0), torch.empty(0, dtype=torch.long)
        if self.with_teams:
            return empty, empty_l, group_label, torch.empty(0, dtype=torch.long)
        return empty, empty_l, group_label

    def _pack_crop(self, crops_t, labels_t, group_label: int, center_x):
        """Assemble a crop item, appending team ids when ``with_teams``.

        ``center_x`` is the per-player box center-x of the SAME frame that
        produced ``labels_t``, so the team split aligns with the person axis.
        """
        if not self.with_teams:
            return crops_t, labels_t, group_label
        teams = self._team_ids(center_x) if len(center_x) else torch.empty(0, dtype=torch.long)
        return crops_t, labels_t, group_label, teams

    def _getitem_crop(
        self,
        video_id: str,
        clip_id: str,
        frame_names: list[str],
        persons_per_frame: dict[str, tuple[np.ndarray, np.ndarray]],
        group_label: int,
    ) -> tuple:
        """Load cropped person images and return as tensor(s)."""
        _tf = self.transform or ToTensor()

        if self.n_frames == 1:
            middle_frame = frame_names[len(frame_names) // 2] if frame_names else None
            if middle_frame is None:
                return self._empty_crop(group_label)

            img = self._load_image(video_id, clip_id, middle_frame)
            boxes, labels = persons_per_frame[middle_frame]
            crops, person_labels, center_x = self._crop_boxes(img, boxes, labels)
            crops = [_tf(c) for c in crops]

            if not crops:
                return self._empty_crop(group_label)

            return self._pack_crop(
                torch.stack(crops, dim=0),                       # (P, C, H, W)
                torch.tensor(person_labels, dtype=torch.long),   # (P,)
                group_label,
                center_x,
            )

        # Temporal crops — use tracking data per frame for accurate boxes
        all_frame_crops = []
        all_frame_labels: list[list[int] | None] = []
        all_frame_cx: list[list[float] | None] = []

        for fname in frame_names:
            img = self._load_image(video_id, clip_id, fname)
            boxes, labels = persons_per_frame[fname]
            crops, person_labels, center_x = self._crop_boxes(img, boxes, labels)
            crops = [_tf(c) for c in crops]

            if crops:
                all_frame_crops.append(torch.stack(crops, dim=0))  # (P, C, H, W)
                all_frame_labels.append(person_labels)
                all_frame_cx.append(center_x)
            else:
                all_frame_crops.append(None)
                all_frame_labels.append(None)
                all_frame_cx.append(None)

        valid_frames = [f for f in all_frame_crops if f is not None]
        if not valid_frames:
            return self._empty_crop(group_label)

        # Person labels (and team ids) come from the frame nearest to the
        # MIDDLE (annotated target) frame that has crops — actions change
        # within a clip, and the middle frame is what the annotations describe.
        mid = len(frame_names) // 2
        label_off = next(
            off
            for off in sorted(range(-mid, len(frame_names) - mid), key=abs)
            if 0 <= mid + off < len(all_frame_labels) and all_frame_labels[mid + off] is not None
        )
        person_labels = all_frame_labels[mid + label_off]
        center_x = all_frame_cx[mid + label_off]

        # Frames can carry different person counts (tracking dropouts /
        # detection fallback) — zero-pad each frame to the clip max so the
        # temporal stack is rectangular instead of crashing.
        max_p = max(f.shape[0] for f in valid_frames)
        if any(f.shape[0] != max_p for f in valid_frames):
            valid_frames = [
                torch.cat([f, f.new_zeros(max_p - f.shape[0], *f.shape[1:])], dim=0)
                if f.shape[0] < max_p else f
                for f in valid_frames
            ]

        return self._pack_crop(
            torch.stack(valid_frames, dim=0),                    # (T, P, C, H, W)
            torch.tensor(person_labels, dtype=torch.long),       # (P,)
            group_label,
            center_x,
        )


# ── Collate function ─────────────────────────────────────────────────────────


def _pad_and_stack_crops(
    crops_list: Sequence[torch.Tensor],
    person_labels_list: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-pad batch crops (temporal or single) up to maximum player/time counts."""
    max_players = max(
        (c.shape[0] if c.dim() == 4 else c.shape[1] for c in crops_list if c.numel() > 0),
        default=0,
    )

    if max_players == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)

    batch_size = len(crops_list)
    sample_shape = next(c for c in crops_list if c.numel() > 0).shape

    if len(sample_shape) == 4:
        # Single-frame crops (P, C, H, W)
        _, C, H, W = sample_shape
        padded_crops = torch.zeros(batch_size, max_players, C, H, W)
        padded_labels = torch.zeros(batch_size, max_players, dtype=torch.long)
        masks = torch.zeros(batch_size, max_players, dtype=torch.bool)

        for i, (crops, plabels) in enumerate(zip(crops_list, person_labels_list)):
            if crops.numel() == 0:
                continue
            n = crops.shape[0]
            padded_crops[i, :n] = crops
            padded_labels[i, :n] = plabels
            masks[i, :n] = True

    elif len(sample_shape) == 5:
        # Temporal crops (T, P, C, H, W)
        max_T = max((c.shape[0] for c in crops_list if c.numel() > 0), default=0)
        _, _, C, H, W = sample_shape
        padded_crops = torch.zeros(batch_size, max_T, max_players, C, H, W)
        padded_labels = torch.zeros(batch_size, max_players, dtype=torch.long)
        masks = torch.zeros(batch_size, max_players, dtype=torch.bool)

        for i, (crops, plabels) in enumerate(zip(crops_list, person_labels_list)):
            if crops.numel() == 0:
                continue
            t, n = crops.shape[0], crops.shape[1]
            padded_crops[i, :t, :n] = crops
            # Labels come from the clip's middle frame, which can carry fewer
            # persons than the clip-max player count n; only slots that have
            # BOTH a crop track and a label are marked valid.
            k = min(n, plabels.shape[0])
            padded_labels[i, :k] = plabels[:k]
            masks[i, :k] = True
    else:
        raise ValueError(f"Unexpected crop shape: {sample_shape}")

    return padded_crops, padded_labels, masks


def collate_fn(batch: list[tuple[Any, ...]]) -> tuple[torch.Tensor, ...]:
    """
    Custom collate function for variable numbers of player crops per clip.

    Handles batches from full-image mode (2-tuple items), crop mode
    (3-tuple items), and crop mode with team ids (4-tuple items, when the
    dataset was built with ``with_teams=True``).  For crop mode, pads the
    player dimension to the maximum count in the batch and returns a mask
    indicating valid players.

    Returns
    -------
    For full-image mode:
        ``(images_batch, labels_batch)``

    For crop mode:
        ``(crops_batch, person_labels_batch, group_labels_batch, masks_batch)``

    For crop mode with team ids (``with_teams=True``):
        ``(..., masks_batch, team_ids_batch)`` — one extra tensor, shape
        ``(B, max_players)``, values 0/1 for real players and -1 for padded
        slots (aligned with ``masks_batch``). The first four elements are
        byte-for-byte the same as the no-teams contract, so existing
        4-tuple unpackers keep working; only team-aware code reads the 5th.

    """
    if not batch:
        return ()

    if len(batch[0]) == 2:
        # Full-image mode
        images, labels = zip(*batch)
        shapes = [img.shape for img in images]

        if all(s == shapes[0] for s in shapes):
            return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)

        # Variable temporal size (n_frames > 1 with missing physical frames)
        max_T = max(s[0] for s in shapes)
        _, C, H, W = shapes[0]
        padded_images = torch.zeros(len(images), max_T, C, H, W)
        for i, img in enumerate(images):
            padded_images[i, : img.shape[0]] = img

        return padded_images, torch.tensor(labels, dtype=torch.long)

    # Crop mode — variable number of players. Items are 3-tuples normally,
    # or 4-tuples (…, team_ids) when the dataset has with_teams=True.
    with_teams = len(batch[0]) == 4
    if with_teams:
        crops_list, person_labels_list, group_labels, teams_list = zip(*batch)
    else:
        crops_list, person_labels_list, group_labels = zip(*batch)

    padded_crops, padded_labels, masks = _pad_and_stack_crops(crops_list, person_labels_list)

    out = (
        padded_crops,
        padded_labels,
        torch.tensor(group_labels, dtype=torch.long),
        masks,
    )
    if not with_teams:
        return out

    # Pad team ids onto the mask grid: -1 for padded/invalid slots, and the
    # per-item real players (masks[i].sum() of them) filled from teams[:k].
    # Same k the label padding uses, so team ids line up with person labels.
    if masks.dim() == 2:
        padded_teams = torch.full(masks.shape, -1, dtype=torch.long)
        for i, teams in enumerate(teams_list):
            k = int(masks[i].sum())
            if k > 0 and teams.numel() > 0:
                padded_teams[i, :k] = teams[:k]
    else:
        padded_teams = torch.empty(0, dtype=torch.long)

    return out + (padded_teams,)
