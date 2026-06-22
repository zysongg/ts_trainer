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

    def __init__(self, output_dir: str = "./predictions", write_files: bool = True) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.write_files = write_files
        self._inputs: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._samples: list[np.ndarray] = []
        self.saved_outputs: dict[str, dict[str, np.ndarray]] = {}

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
            inputs = outputs.get("inputs")
            if inputs is None and isinstance(batch, dict):
                inputs = batch.get("x")
            if inputs is not None:
                self._inputs.append(_to_numpy(inputs))

            samples = outputs.get("samples")
            if samples is not None:
                self._samples.append(_to_numpy(samples))

            preds = outputs.get("preds", outputs.get("pred"))
            if preds is not None:
                self._predictions.append(_to_numpy(preds))
            elif samples is not None:
                self._predictions.append(np.median(_to_numpy(samples), axis=1))

            targets = outputs.get("targets", outputs.get("y"))
            if targets is not None:
                self._targets.append(_to_numpy(targets))
        elif isinstance(outputs, Tensor):
            prediction = _to_numpy(outputs)
            if prediction.ndim > 0:
                self._predictions.append(prediction)
        elif isinstance(outputs, np.ndarray):
            if outputs.ndim > 0:
                self._predictions.append(outputs)

        if not isinstance(outputs, dict) and isinstance(batch, dict) and "x" in batch:
            self._inputs.append(_to_numpy(batch["x"]))

    def _save(self, prefix: str) -> None:
        if not self._predictions and not self._samples:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, np.ndarray] = {}
        if self._inputs:
            data["inputs"] = np.concatenate(self._inputs, axis=0)
        if self._predictions:
            data["predictions"] = np.concatenate(self._predictions, axis=0)
        elif self._samples:
            data["predictions"] = np.median(np.concatenate(self._samples, axis=0), axis=1)
        if self._targets:
            data["targets"] = np.concatenate(self._targets, axis=0)
        if self._samples:
            data["samples"] = np.concatenate(self._samples, axis=0)
        if self.write_files:
            np.savez(self.output_dir / f"{prefix}_results.npz", **data)
        self.saved_outputs[prefix] = data
        self._inputs.clear()
        self._predictions.clear()
        self._targets.clear()
        self._samples.clear()


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
        """Do nothing here - wait for validation to complete."""
        pass

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Print epoch summary after validation (when all metrics are aggregated)."""
        if not trainer.sanity_checking:
            # Only print on rank 0 in DDP
            if trainer.is_global_zero:
                self._print_epoch(trainer, pl_module)

    def _print_epoch(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        elapsed = time.time() - self._epoch_start

        metrics = trainer.callback_metrics
        parts = [f"[Epoch {epoch}/{max_epochs}]"]

        # Prefer epoch-level metrics (computed in on_*_epoch_end)
        if "train_epoch_loss" in metrics:
            val = metrics["train_epoch_loss"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            parts.append(f"train_loss={val:.4f}")
        elif "train_loss" in metrics:
            val = metrics["train_loss"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            parts.append(f"train_loss={val:.4f}")

        if "val_epoch_loss" in metrics:
            val = metrics["val_epoch_loss"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            parts.append(f"val_loss={val:.4f}")
        elif "val_loss" in metrics:
            val = metrics["val_loss"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            parts.append(f"val_loss={val:.4f}")

        if "val_epoch_mse" in metrics:
            val = metrics["val_epoch_mse"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            parts.append(f"val_mse={val:.4f}")
        elif "val_mse" in metrics:
            val = metrics["val_mse"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            parts.append(f"val_mse={val:.4f}")

        # Learning rate
        try:
            opt = pl_module.optimizers()
            if opt is not None:
                lr = opt.param_groups[0]["lr"]
                parts.append(f"lr={lr:.1e}")
        except Exception:
            pass

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
