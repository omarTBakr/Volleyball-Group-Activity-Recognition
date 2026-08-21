"""
Shared training driver for the baseline models.

Every baseline used to hand-copy the same epoch loop — train → validate →
TensorBoard scalars → JSON log → best-F1 checkpoint → early stopping — once
per stage (twice or three times for the staged/probe-finetune baselines).
``Trainer`` owns that loop exactly once. A baseline builds its data, model,
criterion, and optimizer per stage, then calls :meth:`Trainer.run_stage` for
each stage and :meth:`Trainer.run_test` at the end.

Composition, not configuration
------------------------------
There is no ``two_stages`` flag: a two-stage baseline simply calls
``run_stage`` twice, a probe→finetune baseline three times. One ``Trainer``
instance spans a whole run and carries the shared state across those calls:

* ``global_epoch`` — a monotonic counter across every stage (so TensorBoard
  and the JSON log read as one continuous timeline).
* ``metrics_history`` — a single ``logs/<baseline>/<run_id>.json``.
* best-F1 tracking **keyed by checkpoint filename** — stages that write the
  same checkpoint (e.g. B1's probe + finetune, or B6/B7's Stage-B probe +
  finetune) share one best, so a later phase only overwrites the checkpoint
  when it genuinely beats the earlier one; stages that write different
  checkpoints (e.g. Stage A vs Stage B) track independent bests.

Metric contract
---------------
The per-epoch helpers in :mod:`utils.utility` return
``(loss, accuracy, macro_f1, confusion_matrix)``; ``Trainer`` selects the best
checkpoint on validation macro-F1 (the metric every baseline optimized).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.path_config import LOGS_DIR
from utils.utility import (
    BatchUnpack,
    get_device,
    log_experiment_summary,
    save_model,
    test_one_epoch,
    train_one_epoch,
    validate_one_epoch,
)


#: JSON metric key → TensorBoard scalar tag. :meth:`Trainer.log_epoch` plots
#: every key it finds here and stores the whole record in the JSON log, so a
#: baseline that has no value for one (B9 reports no train-side accuracy, the
#: torch baselines report no top-5) simply omits it.
_TB_TAGS: dict[str, str] = {
    "train_loss": "Loss/train",
    "val_loss": "Loss/val",
    "train_acc": "Acc/train",
    "val_acc": "Acc/val",
    "val_top5": "Acc5/val",
    "train_f1": "F1/train",
    "val_f1": "F1/val",
    "learning_rate": "LR",
}


class Trainer:
    """Runs one training stage per :meth:`run_stage` call; owns cross-stage state."""

    def __init__(
        self,
        baseline: str,
        run_id: str,
        *,
        device: torch.device | None = None,
        writer: SummaryWriter | None = None,
        log_root: Path | None = None,
    ) -> None:
        self.baseline = baseline
        self.run_id = run_id
        self.device = device or get_device()

        # ``log_root`` overrides the repo's LOGS_DIR — for runs whose output has
        # to live off the repo's own disk (B9 writes to the internal SSD).
        self.log_dir = (log_root or LOGS_DIR) / baseline
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.log_dir / f"{run_id}.json"
        self.writer = writer or SummaryWriter(
            log_dir=self.log_dir / "tensorboard" / run_id,
        )

        self.global_epoch = 0
        self.metrics_history: list[dict] = []
        self._test: dict | None = None
        self._best: dict[str, float] = {}  # checkpoint_name → best val macro-F1

    # ── internal helpers ──────────────────────────────────────────────────

    def _flush_json(self) -> None:
        payload: dict = {"epochs": self.metrics_history}
        if self._test is not None:
            payload["test"] = self._test
        with self.json_path.open("w") as f:
            json.dump(payload, f, indent=4)

    @staticmethod
    def _current_lr(optimizer: optim.Optimizer, scheduler) -> float:
        if scheduler is not None:
            return float(scheduler.get_last_lr()[0])
        return float(optimizer.param_groups[0]["lr"])

    # ── public API ────────────────────────────────────────────────────────

    def log_epoch(self, record: dict, *, tb_prefix: str = "") -> None:
        """
        Record one epoch: TensorBoard scalars plus a row in the JSON history.

        This is the whole logging half of :meth:`run_stage`, split out so a
        baseline that does **not** run our epoch loop can still log the same
        way — B9 hands its training to Ultralytics and calls this from that
        trainer's ``on_fit_epoch_end`` callback.

        Parameters
        ----------
        record : dict
            One epoch's metrics; ``"epoch"`` is the step every scalar is
            plotted against. Keys listed in :data:`_TB_TAGS` also become
            scalars, the rest are JSON-only. Stored verbatim.
        tb_prefix : str, optional
            Namespace for the scalars (e.g. ``"StageA"``).

        """
        prefix = f"{tb_prefix}/" if tb_prefix else ""
        step = record["epoch"]
        for key, tag in _TB_TAGS.items():
            if record.get(key) is not None:
                self.writer.add_scalar(f"{prefix}{tag}", float(record[key]), step)
        self.metrics_history.append(record)
        self._flush_json()

    def record_test(
        self,
        *,
        test_loss: float,
        test_acc: float,
        test_f1: float,
        hparam_dict: dict,
        best_val_f1: float,
        extra: dict | None = None,
    ) -> None:
        """
        Store the final metrics in the JSON log and write the TensorBoard
        summary card. :meth:`run_test` calls this once it has evaluated the
        test set; baselines that produce those numbers some other way (B9's
        Ultralytics validator) call it directly. ``extra`` adds JSON-only
        fields, e.g. which split was evaluated.
        """
        self._test = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            **(extra or {}),
        }
        self._flush_json()

        log_experiment_summary(
            writer=self.writer,
            run_id=self.run_id,
            hparam_dict=hparam_dict,
            test_f1=test_f1,
            test_acc=test_acc,
            test_loss=test_loss,
            best_val_f1=best_val_f1,
        )

    def run_stage(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_classes: int,
        num_epochs: int,
        checkpoint_name: str,
        class_to_idx: dict[str, int],
        stage: str = "",
        batch_unpack: BatchUnpack | None = None,
        scheduler=None,
        accum_steps: int = 1,
        patience: int = 0,
        save_ref: nn.Module | None = None,
        tb_prefix: str | None = None,
        desc: str | None = None,
        select_metric: str = "f1",
    ) -> float:
        """Train ``model`` for one stage; return the best validation score for
        ``checkpoint_name`` (shared across stages writing the same file).

        Parameters mirror the per-baseline loops. ``save_ref`` is the module
        actually checkpointed — pass the unwrapped module when ``model`` is a
        ``DataParallel`` wrapper, so checkpoints stay free of ``module.``
        prefixes. ``tb_prefix`` namespaces the TensorBoard scalars (e.g.
        ``"StageA"``); it defaults to ``stage`` or the empty string.
        ``select_metric`` picks the metric the best checkpoint is chosen on —
        ``"f1"`` (macro-F1, default) or ``"acc"`` (accuracy).
        """
        if select_metric not in ("f1", "acc"):
            raise ValueError(f"select_metric must be 'f1' or 'acc', got '{select_metric}'.")
        save_ref = save_ref if save_ref is not None else model
        tag = (tb_prefix if tb_prefix is not None else stage) or ""
        label = f" ({stage})" if stage else ""
        metric_label = select_metric.upper()

        best = self._best.setdefault(checkpoint_name, float("-inf"))
        epochs_without_improvement = 0

        print(f"\n{'=' * 60}")
        print(f"  {self.baseline} — stage{label or ' train'}: "
              f"{num_epochs} epochs, {num_classes} classes")
        print(f"{'=' * 60}")

        for epoch in range(num_epochs):
            self.global_epoch += 1
            print(f"\n--- {self.baseline}{label} · Epoch {epoch + 1}/{num_epochs} "
                  f"(global {self.global_epoch}) ---")

            train_loss, train_acc, train_f1, _ = train_one_epoch(
                model, train_loader, criterion, optimizer, self.device,
                batch_unpack=batch_unpack, num_classes=num_classes,
                accumulate_grad_batches=accum_steps,
                desc=desc or f"Train[{self.baseline}{label}]",
            )
            val_loss, val_acc, val_f1, _ = validate_one_epoch(
                model, val_loader, criterion, self.device,
                batch_unpack=batch_unpack, num_classes=num_classes,
                desc=desc or f"Val[{self.baseline}{label}]",
            )

            if scheduler is not None:
                scheduler.step()
            lr = self._current_lr(optimizer, scheduler)

            print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            self.log_epoch({
                "epoch": self.global_epoch,
                "stage": stage,
                "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "learning_rate": lr,
            }, tb_prefix=tag)

            selected = val_acc if select_metric == "acc" else val_f1
            if selected > best:
                best = selected
                self._best[checkpoint_name] = best
                epochs_without_improvement = 0
                save_model(checkpoint_name, self.global_epoch, save_ref,
                           optimizer, val_loss, class_to_idx=class_to_idx)
                print(f"  ✓ New best saved to '{checkpoint_name}' (val {metric_label}: {best:.4f})")
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    print(f"  ⏹ Early stopping{label} — no improvement for {patience} epochs.")
                    break

        return self._best[checkpoint_name]

    def run_test(
        self,
        *,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        num_classes: int,
        hparam_dict: dict,
        best_val_f1: float,
        batch_unpack: BatchUnpack | None = None,
        desc: str | None = None,
    ) -> tuple[float, float, float]:
        """Evaluate ``model`` on the test set, record it in the JSON log, and
        write the TensorBoard experiment summary. Returns ``(loss, acc, f1)``.
        """
        print(f"\n--- {self.baseline} — testing best model ---")
        test_loss, test_acc, test_f1, _ = test_one_epoch(
            model, test_loader, criterion, self.device,
            batch_unpack=batch_unpack, num_classes=num_classes,
            desc=desc or f"Test[{self.baseline}]",
        )
        print(f"Final Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

        self.record_test(
            test_loss=test_loss,
            test_acc=test_acc,
            test_f1=test_f1,
            hparam_dict=hparam_dict,
            best_val_f1=best_val_f1,
        )
        return test_loss, test_acc, test_f1

    def close(self) -> None:
        self.writer.close()

    # Allocate the next run id for a baseline, matching the historical
    # "count existing json logs + 1" scheme so runs stay sequentially named.
    @staticmethod
    def next_run_id(baseline: str, log_root: Path | None = None) -> str:
        log_dir = (log_root or LOGS_DIR) / baseline
        log_dir.mkdir(parents=True, exist_ok=True)
        return f"run{len(list(log_dir.glob('*.json'))) + 1}"
