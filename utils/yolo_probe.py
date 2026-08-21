"""
Two-stage YOLO capacity probe: *which model scale fits this card*, then
*which batch size fits that scale*.

Fine-tuning a YOLO detector on a consumer GPU fails in two different ways,
and they need to be diagnosed in that order:

1. **The model is too big for the card.**  No batch size rescues an
   extra-large model on 8 GB — the weights, gradients and optimizer state
   alone eat the budget.  Stage 1 walks the scale ladder
   ``x → l → m → s → n`` (extra-large / large / medium / small / nano-tiny)
   at a fixed reference batch and keeps the largest scale that still fits.
2. **The batch is too big for the model.**  Once the scale is settled,
   Stage 2 ladders the batch size for *that* scale and reports the largest
   one that fits, optionally refining between the last success and the first
   failure so you get e.g. ``batch=12`` instead of a blunt ``batch=8``.

Each probe runs a genuine training step — forward, detection loss, backward,
optimizer step — on synthetic images and boxes, so the measured peak includes
activations, gradients, AMP scaler and optimizer state.  No dataset is needed
and, by default, no pretrained weights are downloaded either: memory depends
on architecture, not on the values inside the tensors, so scales are built
from their YAML configs.

Usage
-----
    # from the project root
    python -m utils.yolo_probe --imgsz 640 --nc 1 --max-objects 12

    # or from Python
    from utils.yolo_probe import probe_yolo_capacity
    report = probe_yolo_capacity(family="yolo26", imgsz=640, nc=1)
    print(report["model"], report["batch"])

The run writes ``logs/yolo_probe.json`` so training scripts (see
``models/baseline9.py``) can pick the result up with ``load_probe_report()``.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch  # ty:ignore[import]  # ty:ignore[unresolved-import]

from configs.path_config import LOGS_DIR

# ── Scale ladder ────────────────────────────────────────────────────────────

#: Ultralytics compound-scaling suffixes, largest → smallest.
SCALE_ORDER: tuple[str, ...] = ("x", "l", "m", "s", "n")

#: Human-readable name for each suffix (what people mean by "card size").
SCALE_NAMES: dict[str, str] = {
    "x": "extra-large",
    "l": "large",
    "m": "medium",
    "s": "small",
    "n": "nano (tiny)",
}

#: Rule-of-thumb starting point per card, used only for the printed hint;
#: the actual pick always comes from the measured probe.
VRAM_HINTS: tuple[tuple[float, str], ...] = (
    (22.0, "x"),
    (15.0, "l"),
    (10.0, "m"),
    (7.0, "s"),
    (0.0, "n"),
)

#: Where the probe result is cached for training scripts to read back.
REPORT_PATH: Path = LOGS_DIR / "yolo_probe.json"

_GB = 1024 ** 3


@dataclass
class ProbeResult:
    """Outcome of a single (scale, batch) training-step probe.

    Attributes
    ----------
    scale : str
        Compound-scaling suffix that was probed (``"x"``…``"n"``).
    batch : int
        Batch size that was probed.
    imgsz : int
        Square input resolution used for the probe.
    status : str
        ``"fits"``, ``"over_budget"``, ``"oom"`` or ``"error"``.
    peak_gb : float | None
        Peak CUDA memory allocated during the step, in GiB.  ``None`` when
        the probe never completed a step.
    budget_gb : float
        Memory the probe was allowed to use (free VRAM × safety fraction).
    detail : str
        Free-form note — the exception message for failed probes.

    """

    scale: str
    batch: int
    imgsz: int
    status: str
    task: str = "detect"
    peak_gb: float | None = None
    budget_gb: float = 0.0
    detail: str = ""

    @property
    def fits(self) -> bool:
        """True when the step completed *and* stayed inside the budget."""
        return self.status == "fits"


# ── GPU inspection ──────────────────────────────────────────────────────────


def get_gpu_info(device: int = 0) -> dict:
    """
    Describe the CUDA device the probe will run on.

    Parameters
    ----------
    device : int, optional
        CUDA device index.  Defaults to ``0``.

    Returns
    -------
    dict
        ``name``, ``total_gb`` and ``free_gb`` (free memory as reported by
        the driver, i.e. after the CUDA context and any other process).

    Raises
    ------
    SystemExit
        If no usable CUDA device is present — probing on CPU is meaningless.

    """
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. The capacity probe needs an NVIDIA GPU "
            "with a CUDA-enabled PyTorch build."
        )
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "name": props.name,
        "total_gb": total_bytes / _GB,
        "free_gb": free_bytes / _GB,
    }


def suggest_scale_for_vram(total_gb: float) -> str:
    """Return the scale suffix usually sensible for a card of this size."""
    for threshold, scale in VRAM_HINTS:
        if total_gb >= threshold:
            return scale
    return "n"


# ── Input normalisation ─────────────────────────────────────────────────────


def normalize_imgsz(imgsz: int, stride: int = 32) -> int:
    """
    Round ``imgsz`` up to a multiple of the model stride, as training does.

    A YOLO backbone downsamples by 32 and concatenates feature maps on the way
    back up, so a resolution that is not a multiple of 32 makes those maps
    disagree by a pixel and the forward pass dies with an unhelpful
    "Sizes of tensors must match" error.  The trainer never hits this because
    ``check_imgsz`` silently rounds first — 720 becomes 736 — so the probe has
    to round identically or it would both fail on valid input and measure a
    resolution training would never use.

    Parameters
    ----------
    imgsz : int
        Requested square resolution.
    stride : int, optional
        Maximum model stride.  32 for every current YOLO detection model.

    Returns
    -------
    int
        The resolution training will actually use.

    """
    from ultralytics.utils.checks import check_imgsz  # ty:ignore[import]

    return int(check_imgsz(imgsz, stride=stride, max_dim=1))


# ── Single probe ────────────────────────────────────────────────────────────


def model_spec(family: str, scale: str, task: str, suffix: str) -> str:
    """Return the Ultralytics spec for a scale, e.g. ``yolo26m-cls.pt``."""
    return f"{family}{scale}{'-cls' if task == 'classify' else ''}{suffix}"


def _build_model(
    family: str, scale: str, nc: int, pretrained: bool, task: str = "detect"
) -> torch.nn.Module:
    """
    Build a model for ``family`` + ``scale`` ready for a training step.

    Random-initialised (``pretrained=False``) is the default: it needs no
    download and occupies essentially as much memory as the fine-tuned model.
    """
    from ultralytics.cfg import get_cfg  # ty:ignore[import]
    from ultralytics.nn.tasks import ClassificationModel, DetectionModel  # ty:ignore[import]
    from ultralytics.utils import DEFAULT_CFG  # ty:ignore[import]

    if pretrained:
        from ultralytics import YOLO  # ty:ignore[import]

        net = YOLO(model_spec(family, scale, task, ".pt")).model.float()
        # A checkpoint arrives inference-ready: half precision with grads
        # switched off. The trainer re-enables them, and so must the probe or
        # the backward pass has nothing to differentiate.
        for param in net.parameters():
            param.requires_grad_(True)
    else:
        build = ClassificationModel if task == "classify" else DetectionModel
        net = build(model_spec(family, scale, task, ".yaml"), nc=nc, verbose=False)

    # The loss reads training hyper-parameters as attributes (``args.box``,
    # ``args.cls``, ``args.dfl``).  A YAML-built model has no ``args`` at all,
    # and a checkpoint carries them as a plain dict — neither works until the
    # trainer normally installs a config namespace, so do it here.
    if not hasattr(getattr(net, "args", None), "box"):
        net.args = get_cfg(DEFAULT_CFG)
    return net


def _synthetic_batch(
    batch: int, imgsz: int, nc: int, max_objects: int, device: torch.device, task: str = "detect"
) -> dict:
    """
    Build a fake training batch in Ultralytics' format for ``task``.

    Classification needs only images and one class index each.  Detection
    additionally needs boxes as normalized ``xywh``; there, ``max_objects``
    matters because the task-aligned assigner allocates
    ``(batch, objects, anchors)`` buffers — a crowded scene costs materially
    more than a sparse one.
    """
    if task == "classify":
        return {
            "img": torch.rand(batch, 3, imgsz, imgsz, device=device),
            "cls": torch.randint(0, max(nc, 1), (batch,), device=device),
        }

    n = batch * max_objects
    return {
        "img": torch.rand(batch, 3, imgsz, imgsz, device=device),
        "cls": torch.randint(0, max(nc, 1), (n, 1), device=device).float(),
        "bboxes": torch.cat(
            [
                torch.rand(n, 2, device=device) * 0.6 + 0.2,   # cx, cy
                torch.rand(n, 2, device=device) * 0.2 + 0.05,  # w, h
            ],
            dim=1,
        ),
        "batch_idx": torch.arange(batch, device=device).repeat_interleave(max_objects).float(),
    }


def probe_step(
    family: str,
    scale: str,
    batch: int,
    imgsz: int = 640,
    *,
    nc: int = 80,
    max_objects: int = 12,
    amp: bool = True,
    pretrained: bool = False,
    steps: int = 3,
    fraction: float = 0.85,
    device: int = 0,
    task: str = "detect",
    verbose: bool = True,
) -> ProbeResult:
    """
    Run real training steps at one (scale, batch) point and measure the peak.

    Parameters
    ----------
    family : str
        Model family prefix, e.g. ``"yolo26"`` or ``"yolov8"``.
    scale : str
        Compound-scaling suffix, one of :data:`SCALE_ORDER`.
    batch : int
        Batch size to probe.
    imgsz : int, optional
        Square training resolution.  Defaults to ``640``.
    nc : int, optional
        Number of classes in your dataset — the classification loss holds
        ``(batch, anchors, nc)`` buffers, so this affects the measurement.
    max_objects : int, optional
        Objects per image to simulate.  Defaults to ``12`` (a volleyball
        frame's worth of players).
    amp : bool, optional
        Probe under mixed precision, as Ultralytics trains by default.
    pretrained : bool, optional
        Download and use the real ``.pt`` weights instead of building the
        architecture from YAML.  Same memory, slower start.
    steps : int, optional
        Training steps to run.  Three is enough: the first allocates
        gradients, the second the optimizer state, the third is steady state.
    fraction : float, optional
        Share of *free* VRAM the probe is allowed to occupy before the point
        is called ``over_budget``.  Leaves headroom for dataloader pinning,
        validation and fragmentation.
    device : int, optional
        CUDA device index.
    task : {"detect", "classify"}, optional
        Which head to probe.  ``"classify"`` builds the ``-cls`` variant and
        a cross-entropy step; it uses far less memory than detection, whose
        loss dominates the peak, so probing the wrong task badly misestimates
        the batch size.
    verbose : bool, optional
        Print a one-line result per probe.

    Returns
    -------
    ProbeResult
        Measurement for this point; never raises on OOM.

    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    imgsz = normalize_imgsz(imgsz)

    torch_device = torch.device(f"cuda:{device}")
    if not torch.cuda.is_initialized():
        # The memory-stat calls below reject a device on an uninitialised
        # context, which bites when probe_step() is called on its own.
        torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch_device)
    free_gb = torch.cuda.mem_get_info(device)[0] / _GB
    budget_gb = free_gb * fraction

    net = optimizer = scaler = None
    result = ProbeResult(
        scale=scale, batch=batch, imgsz=imgsz, status="error", task=task, budget_gb=budget_gb
    )
    try:
        net = _build_model(family, scale, nc, pretrained, task).to(torch_device).train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)  # worst-case state
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        for _ in range(steps):
            sample = _synthetic_batch(batch, imgsz, nc, max_objects, torch_device, task)
            with torch.amp.autocast("cuda", enabled=amp):
                loss, _ = net(sample)
            scaler.scale(loss.sum()).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        peak_gb = torch.cuda.max_memory_allocated(torch_device) / _GB
        result.peak_gb = peak_gb
        result.status = "fits" if peak_gb <= budget_gb else "over_budget"
    except torch.OutOfMemoryError as exc:
        result.status = "oom"
        result.detail = str(exc).splitlines()[0]
    except RuntimeError as exc:
        # Some OOMs surface as a plain RuntimeError depending on where they hit.
        oom = "out of memory" in str(exc).lower()
        result.status = "oom" if oom else "error"
        result.detail = str(exc).splitlines()[0]
    except Exception as exc:  # noqa: BLE001 - a broken scale must not kill the sweep
        result.status = "error"
        result.detail = f"{type(exc).__name__}: {exc}"
    finally:
        del net, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()

    if verbose:
        peak = f"{result.peak_gb:.2f} GB peak" if result.peak_gb is not None else "—"
        note = f"  ({result.detail})" if result.detail else ""
        print(
            f"  {model_spec(family, scale, task, '')} @ batch {batch:>3} : {result.status:<11} "
            f"{peak} / {budget_gb:.2f} GB budget{note}"
        )
    return result


# ── Stage 1: model scale ────────────────────────────────────────────────────


def find_largest_scale(
    family: str = "yolo26",
    *,
    ref_batch: int = 4,
    scales: tuple[str, ...] = SCALE_ORDER,
    **probe_kwargs,
) -> tuple[str | None, list[ProbeResult]]:
    """
    Find the largest model scale that trains at a small reference batch.

    Walks ``scales`` from largest to smallest and stops at the first one that
    fits.  The reference batch is deliberately small (``4``): Stage 1 asks
    "does this *architecture* fit at all", Stage 2 then pushes the batch.

    Parameters
    ----------
    family : str, optional
        Model family prefix.  Defaults to ``"yolo26"``.
    ref_batch : int, optional
        Batch size used to compare scales.  Defaults to ``4``.
    scales : tuple of str, optional
        Ladder to walk, largest first.  Defaults to :data:`SCALE_ORDER`.
    **probe_kwargs
        Forwarded to :func:`probe_step` (``imgsz``, ``nc``, ``max_objects``,
        ``amp``, ``fraction``, ``device``…).

    Returns
    -------
    tuple
        ``(scale, results)`` — the chosen suffix (``None`` if nothing fits)
        and every measurement taken.

    """
    results: list[ProbeResult] = []
    chosen: str | None = None
    for scale in scales:
        result = probe_step(family, scale, ref_batch, **probe_kwargs)
        results.append(result)
        if result.fits:
            chosen = scale
            break
    return chosen, results


# ── Stage 2: batch size ─────────────────────────────────────────────────────


def find_max_batch(
    family: str,
    scale: str,
    *,
    candidates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    refine: bool = True,
    **probe_kwargs,
) -> tuple[int | None, list[ProbeResult]]:
    """
    Find the largest batch size that trains for a fixed model scale.

    Ladders ``candidates`` upward and stops at the first failure, so the cost
    is one probe per rung rather than a full sweep.  With ``refine=True`` it
    then binary-searches the gap between the last success and the first
    failure, which typically buys a non-power-of-two batch (e.g. 12 or 24).

    Parameters
    ----------
    family, scale : str
        Model to probe, e.g. ``"yolo26"`` + ``"m"``.
    candidates : tuple of int, optional
        Ascending batch ladder.
    refine : bool, optional
        Binary-search between the last fit and the first failure.
    **probe_kwargs
        Forwarded to :func:`probe_step`.

    Returns
    -------
    tuple
        ``(batch, results)`` — the largest batch that fit (``None`` if even
        the smallest candidate failed) and every measurement taken.

    """
    results: list[ProbeResult] = []
    best: int | None = None
    first_fail: int | None = None

    for batch in sorted(b for b in candidates if b >= 1):
        result = probe_step(family, scale, batch, **probe_kwargs)
        results.append(result)
        if result.fits:
            best = batch
        else:
            first_fail = batch
            break

    if refine and best is not None and first_fail is not None:
        low, high = best, first_fail
        while high - low > 1:
            mid = (low + high) // 2
            result = probe_step(family, scale, mid, **probe_kwargs)
            results.append(result)
            if result.fits:
                low = mid
            else:
                high = mid
        best = low

    return best, results


# ── Orchestration ───────────────────────────────────────────────────────────


def probe_yolo_capacity(
    family: str = "yolo26",
    *,
    imgsz: int = 640,
    nc: int = 80,
    max_objects: int = 12,
    ref_batch: int = 4,
    scales: tuple[str, ...] = SCALE_ORDER,
    candidates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    task: str = "detect",
    amp: bool = True,
    pretrained: bool = False,
    fraction: float = 0.85,
    refine: bool = True,
    device: int = 0,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run both stages and report the model + batch this card can train.

    Stage 1 settles the scale (extra-large → nano), Stage 2 settles the batch
    size for that scale.  Splitting it this way avoids the classic mistake of
    concluding "batch 4 OOMs, my GPU is too small" when the real answer is
    "this scale is too big — the next one down trains at batch 16".

    Parameters
    ----------
    family : str, optional
        Model family prefix.  Defaults to ``"yolo26"``.
    imgsz : int, optional
        Training resolution to probe at.  Halving it is the cheapest way to
        buy memory if nothing fits.
    nc : int, optional
        Class count of your dataset.
    max_objects : int, optional
        Objects per image to simulate.
    ref_batch : int, optional
        Batch used while comparing scales in Stage 1.
    scales, candidates : tuple, optional
        Scale ladder and batch ladder.
    task : {"detect", "classify"}, optional
        Which head to measure — must match what you intend to train.
    amp, pretrained, fraction, refine, device : optional
        See :func:`probe_step` and :func:`find_max_batch`.
    save : bool, optional
        Write the report to :data:`REPORT_PATH`.
    verbose : bool, optional
        Print progress and a final summary.

    Returns
    -------
    dict
        ``gpu``, ``total_gb``, ``free_gb``, ``imgsz``, ``nc``,
        ``max_objects``, ``amp``, ``scale``, ``model``, ``batch``,
        ``peak_gb``, plus the raw ``stage1``/``stage2`` measurements.

    """
    gpu = get_gpu_info(device)
    requested_imgsz, imgsz = imgsz, normalize_imgsz(imgsz)
    probe_kwargs = dict(
        imgsz=imgsz,
        nc=nc,
        max_objects=max_objects,
        amp=amp,
        pretrained=pretrained,
        fraction=fraction,
        device=device,
        task=task,
        verbose=verbose,
    )

    if verbose:
        hint = suggest_scale_for_vram(gpu["total_gb"])
        print(f"GPU: {gpu['name']} | {gpu['total_gb']:.1f} GB total, {gpu['free_gb']:.1f} GB free")
        if imgsz != requested_imgsz:
            print(
                f"imgsz {requested_imgsz} is not a multiple of stride 32 — probing at "
                f"{imgsz}, which is what training would round it to"
            )
        objects = f", {max_objects} objects/image" if task == "detect" else ""
        print(f"Probing {task} at imgsz={imgsz}, nc={nc}{objects}, amp={amp}")
        print(
            f"Rule of thumb for this card: {model_spec(family, hint, task, '')} "
            f"({SCALE_NAMES[hint]}) — verifying by measurement\n"
        )
        print(f"=== Stage 1: largest model scale that trains (batch {ref_batch}) ===")

    scale, stage1 = find_largest_scale(
        family, ref_batch=ref_batch, scales=scales, **probe_kwargs
    )

    batch: int | None = None
    stage2: list[ProbeResult] = []
    if scale is not None:
        if verbose:
            print(
                f"\n=== Stage 2: largest batch size for {model_spec(family, scale, task, '')} ==="
            )
        batch, stage2 = find_max_batch(
            family, scale, candidates=candidates, refine=refine, **probe_kwargs
        )

    fitted = [r for r in stage2 if r.fits and r.batch == batch]
    report = {
        "gpu": gpu["name"],
        "total_gb": round(gpu["total_gb"], 2),
        "free_gb": round(gpu["free_gb"], 2),
        "family": family,
        "task": task,
        "imgsz": imgsz,
        "requested_imgsz": requested_imgsz,
        "nc": nc,
        "max_objects": max_objects,
        "amp": amp,
        "fraction": fraction,
        "scale": scale,
        "scale_name": SCALE_NAMES.get(scale or "", ""),
        "model": model_spec(family, scale, task, ".pt") if scale else None,
        "batch": batch,
        "peak_gb": round(fitted[0].peak_gb, 2) if fitted and fitted[0].peak_gb else None,
        "stage1": [asdict(r) for r in stage1],
        "stage2": [asdict(r) for r in stage2],
    }

    if save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2))

    if verbose:
        print("\n=== Summary ===")
        if scale is None:
            print(
                f"Nothing in {'/'.join(scales)} trains on this card at imgsz={imgsz}. "
                f"Drop imgsz (e.g. {imgsz // 2}), or free VRAM and re-run."
            )
        elif batch is None:
            print(
                f"{model_spec(family, scale, task, '')} builds but no candidate batch fit — try a smaller "
                f"imgsz or the next scale down."
            )
        else:
            print(f"  model : {model_spec(family, scale, task, '.pt')}  ({SCALE_NAMES[scale]})")
            print(f"  batch : {batch}")
            print(f"  imgsz : {imgsz}")
            if report["peak_gb"]:
                print(f"  peak  : {report['peak_gb']:.2f} GB of {gpu['total_gb']:.1f} GB")
            if batch < 8 and scale != scales[-1]:
                nxt = scales[scales.index(scale) + 1]
                print(
                    f"\nNote: batch {batch} is small — gradients get noisy and epochs get "
                    f"slow. The next scale down usually trains several times faster at a "
                    f"healthier batch; compare with:\n"
                    f"    python -m utils.yolo_probe --task {task} "
                    f"--scales {','.join(scales[scales.index(nxt):])} "
                    f"--imgsz {imgsz} --nc {nc}"
                )
            print("\nPaste these into models/baseline9.py (MODEL / BATCH).")
        if save:
            print(f"\nReport written to {REPORT_PATH}")

    return report


def load_probe_report(path: Path = REPORT_PATH) -> dict | None:
    """
    Read back the last :func:`probe_yolo_capacity` report, if there is one.

    Returns ``None`` when no report exists or it is unreadable, so callers
    can fall back to their own defaults.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe which YOLO scale and batch size fit this GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family", default="yolo26", help="model family prefix, e.g. yolo26 / yolov8")
    parser.add_argument("--imgsz", type=int, default=640, help="square training resolution")
    parser.add_argument("--nc", type=int, default=9, help="number of classes in your dataset")
    parser.add_argument("--max-objects", type=int, default=12, help="objects per image to simulate")
    parser.add_argument("--ref-batch", type=int, default=4, help="batch used to compare model scales")
    parser.add_argument(
        "--scales", default=",".join(SCALE_ORDER), help="scale ladder to walk, largest first"
    )
    parser.add_argument(
        "--batches", default="1,2,4,8,16,32,64", help="batch ladder for stage 2"
    )
    parser.add_argument(
        "--task",
        choices=("detect", "classify"),
        default="detect",
        help="which head to probe — must match what you will train",
    )
    parser.add_argument("--fraction", type=float, default=0.85, help="share of free VRAM allowed")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--no-amp", action="store_true", help="probe in full precision")
    parser.add_argument("--no-refine", action="store_true", help="skip the batch binary search")
    parser.add_argument("--pretrained", action="store_true", help="download real .pt weights")
    parser.add_argument("--no-save", action="store_true", help=f"do not write {REPORT_PATH.name}")
    args = parser.parse_args()
    if not [b for b in args.batches.split(",") if b.strip() and int(b) >= 1]:
        parser.error("--batches needs at least one batch size >= 1")
    return args


if __name__ == "__main__":
    args = _parse_args()
    probe_yolo_capacity(
        family=args.family,
        imgsz=args.imgsz,
        nc=args.nc,
        max_objects=args.max_objects,
        ref_batch=args.ref_batch,
        scales=tuple(s.strip() for s in args.scales.split(",") if s.strip()),
        candidates=tuple(int(b) for b in args.batches.split(",") if b.strip()),
        task=args.task,
        amp=not args.no_amp,
        pretrained=args.pretrained,
        fraction=args.fraction,
        refine=not args.no_refine,
        device=args.device,
        save=not args.no_save,
    )
