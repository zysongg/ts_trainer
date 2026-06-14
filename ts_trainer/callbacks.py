"""Custom callbacks for ts_trainer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks.progress import ProgressBar
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. GradientMonitor
# ---------------------------------------------------------------------------


class GradientMonitor(Callback):
    """Monitor gradient and parameter norms during training.

    Logs ``grad_norm`` and optionally ``param_norm`` to the logger.

    Args:
        log_every_n_steps: Log interval (default 50).
        log_grad_norm: Whether to log gradient norm.
        log_param_norm: Whether to log parameter norm.
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_grad_norm: bool = True,
        log_param_norm: bool = False,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_grad_norm = log_grad_norm
        self.log_param_norm = log_param_norm

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        if self.log_grad_norm:
            grad_norm = _compute_grad_norm(pl_module)
            pl_module.log("grad_norm", grad_norm, on_step=True, on_epoch=False)

        if self.log_param_norm:
            param_norm = _compute_param_norm(pl_module)
            pl_module.log("param_norm", param_norm, on_step=True, on_epoch=False)


def _compute_grad_norm(module: LightningModule) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _compute_param_norm(module: LightningModule) -> float:
    total = 0.0
    for p in module.parameters():
        total += p.data.norm(2).item() ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# 2. TrainingTimer
# ---------------------------------------------------------------------------


class TrainingTimer(Callback):
    """Track per-epoch and total training time.

    Attributes:
        epoch_times: List of per-epoch durations in seconds.
        total_time: Total training duration in seconds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.epoch_times: list[float] = []
        self.total_time: float = 0.0
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._train_start = time.time()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        pl_module.log("epoch_time", elapsed, on_step=False, on_epoch=True, prog_bar=False)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.total_time = time.time() - self._train_start

    @property
    def avg_epoch_time(self) -> float:
        """Average epoch duration in seconds."""
        if not self.epoch_times:
            return 0.0
        return sum(self.epoch_times) / len(self.epoch_times)


# ---------------------------------------------------------------------------
# 3. PredictionWriter
# ---------------------------------------------------------------------------


class PredictionWriter(Callback):
    """Save test/predict predictions as .npz files.

    Args:
        output_dir: Directory to save predictions.
    """

    def __init__(self, output_dir: str = "./predictions") -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect(outputs, batch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save("test")

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect(outputs, batch)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save("predict")

    def _collect(self, outputs: Any, batch: Any) -> None:
        if isinstance(outputs, dict):
            if "preds" in outputs:
                self._predictions.append(_to_numpy(outputs["preds"]))
            if "y" in outputs:
                self._targets.append(_to_numpy(outputs["y"]))
        elif isinstance(outputs, Tensor):
            self._predictions.append(_to_numpy(outputs))

    def _save(self, prefix: str) -> None:
        if not self._predictions:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, np.ndarray] = {
            "predictions": np.concatenate(self._predictions, axis=0),
        }
        if self._targets:
            data["targets"] = np.concatenate(self._targets, axis=0)
        np.savez(self.output_dir / f"{prefix}_results.npz", **data)
        self._predictions.clear()
        self._targets.clear()


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 4. SlimProgressBar
# ---------------------------------------------------------------------------


class SlimProgressBar(ProgressBar):
    """Minimal progress bar for Slurm/pipeline environments.

    Prints one line per epoch instead of tqdm bars::

        [Epoch 1/100] train_loss=0.2345 val_loss=0.1890 lr=1e-3 (2m30s)

    Args:
        log_every_n_steps: Not used (kept for API compat).
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0

    def disable(self) -> None:
        """Override to keep our print output even when PL disables progress bars."""
        pass

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._train_start = time.time()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._print_epoch(trainer, pl_module)

    def _print_epoch(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        elapsed = time.time() - self._epoch_start

        metrics = trainer.callback_metrics
        parts = [f"[Epoch {epoch}/{max_epochs}]"]

        for key in ("train_loss", "val_loss", "val/loss"):
            if key in metrics:
                val = metrics[key]
                parts.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val.item():.4f}")

        # Learning rate
        if pl_module.optimizers():
            opt = pl_module.optimizers()[0]
            lr = opt.param_groups[0]["lr"]
            parts.append(f"lr={lr:.1e}")

        parts.append(f"({_format_time(elapsed)})")
        print(" ".join(parts), flush=True)


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


__all__ = [
    "GradientMonitor",
    "TrainingTimer",
    "PredictionWriter",
    "SlimProgressBar",
]
