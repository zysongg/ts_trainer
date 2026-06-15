"""Lightning module wrapper for probabilistic forecasting models."""

import inspect
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import lightning as pl


class ProbForecastModule(pl.LightningModule):
    """Lightning wrapper for probabilistic forecasting models.

    Supports two model interfaces:
      1. ``model.train_loss(batch)`` / ``model.val_loss(batch)``
         where batch is a dict {"x", "y", "idx", "x_mark", "y_mark"}.
      2. Generic forward: wrapper calls ``model(x, **kwargs)`` and
         computes loss manually.

    Args:
        model: A ts_model probabilistic forecasting model (ProbModel)
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        num_samples: Number of samples for prediction (default 100)

    Example:
        >>> from ts_model import create_model
        >>> from ts_trainer import ProbForecastModule, Trainer
        >>>
        >>> model = create_model("TimePrism", task="forecasting", ...)
        >>> module = ProbForecastModule(model, lr=1e-3)
        >>> trainer = Trainer(max_epochs=100)
        >>> trainer.fit(module, train_dataloaders=train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        num_samples: int = 100,
        test_metrics: Sequence[str] | None = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.test_metrics = tuple(test_metrics or ("CRPS", "CRPS_sum", "MSE_median", "MAE_median"))
        self._forward_params = set(inspect.signature(self.model.forward).parameters)

        # Storage for batch outputs (to compute epoch-level averages)
        self.train_outputs: List[Dict[str, torch.Tensor]] = []
        self.val_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_outputs: List[Dict[str, torch.Tensor]] = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor, **kwargs) -> Any:
        """Forward pass with automatic kwargs filtering."""
        valid_kwargs = {k: v for k, v in kwargs.items() if k in self._forward_params}
        return self.model(x, **valid_kwargs)

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Epoch 开始时预计算 cycle samples 缓存（仅对支持的模型生效）。"""
        model = self.model
        if hasattr(model, "enable_cycle_cache") and hasattr(model, "precompute_cycle_cache"):
            if getattr(model, "_cache_enabled", False):
                trainer = self.trainer
                if trainer is not None and trainer.train_dataloader is not None:
                    model.precompute_cycle_cache(trainer.train_dataloader)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        if hasattr(self.model, "train_loss"):
            loss = self.model.train_loss(batch)
        else:
            loss = self._compute_loss(batch)

        batch_size = batch["x"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.train_outputs.append({"loss": loss})
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        """Compute epoch-level average train loss and clear cycle cache."""
        if self.train_outputs:
            avg_loss = torch.stack([o["loss"] for o in self.train_outputs]).mean()
            self.log("train_epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
            self.train_outputs.clear()
        # 清空当前 epoch 的 cycle samples 缓存释放显存，但保留缓存功能开关，
        # 这样下一轮 on_train_epoch_start 仍会重新预计算。
        model = self.model
        if hasattr(model, "clear_cycle_cache"):
            model.clear_cycle_cache()

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        if hasattr(self.model, "val_loss"):
            loss = self.model.val_loss(batch)
        elif hasattr(self.model, "train_loss"):
            loss = self.model.train_loss(batch)
        else:
            loss = self._compute_loss(batch)

        batch_size = batch["x"].size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.val_outputs.append({"loss": loss})
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        """Compute epoch-level average validation metrics."""
        if not self.val_outputs:
            return
        avg_loss = torch.stack([o["loss"] for o in self.val_outputs]).mean()
        self.log("val_epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.val_outputs.clear()

    # ------------------------------------------------------------------
    #  Test
    # ------------------------------------------------------------------

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step — collect samples for probabilistic metrics."""
        samples = self._sample(batch)
        targets = self._align_target_to_prediction(batch["y"], samples)
        preds = torch.median(samples, dim=1)[0]
        result = {
            "inputs": batch["x"],
            "samples": samples,
            "preds": preds,
            "targets": targets,
            # Backward-compatible aliases.
            "y": targets,
        }
        self.test_outputs.append({
            "inputs": batch["x"].detach().cpu(),
            "samples": samples.detach().cpu(),
            "preds": preds.detach().cpu(),
            "targets": targets.detach().cpu(),
        })
        return result

    def on_test_epoch_end(self) -> None:
        """Compute probabilistic metrics at test end."""
        if not self.test_outputs:
            return

        all_samples = torch.cat([o["samples"] for o in self.test_outputs], dim=0)
        all_targets = torch.cat([o["targets"] for o in self.test_outputs], dim=0)

        try:
            from ts_metric import MetricCalculator

            calc = MetricCalculator(
                task="prediction",
                mode="probabilistic",
                metrics=list(self.test_metrics),
            )
            metric_results = calc.compute(all_targets, all_samples)
            print("\nTest Results (probabilistic forecast):")
            for name, value in metric_results.items():
                scalar = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                self.log(f"test_{name}", scalar, prog_bar=True)
                print(f"  {name}: {float(scalar):.4f}")
        except ImportError:
            print("Warning: ts_metric not installed, skipping probabilistic test metrics.")
        except Exception as e:
            print(f"Warning: Failed to compute probabilistic test metrics with ts_metric: {e}")

        self.test_outputs.clear()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generic loss computation via forward pass."""
        x = batch["x"]
        kwargs = self._build_kwargs(batch)
        pred = self._to_median_forecast(self.forward(x, **kwargs))
        target = self._align_target_to_prediction(batch["y"], pred)
        return nn.functional.mse_loss(pred, target)

    def _predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single forward prediction."""
        x = batch["x"]
        kwargs = self._build_kwargs(batch)
        return self._to_median_forecast(self.forward(x, **kwargs))

    def _sample(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample from a probabilistic model or repeat deterministic forward."""
        if hasattr(self.model, "sample"):
            with torch.no_grad():
                output = self.model.sample(batch, num_samples=self.num_samples)
            samples = self._extract_samples(output)
        else:
            samples_list = []
            for _ in range(self.num_samples):
                pred = self._predict(batch)
                samples_list.append(pred.unsqueeze(1))
            samples = torch.cat(samples_list, dim=1)
        if samples.dim() == 3:
            samples = samples.unsqueeze(1)
        if samples.dim() != 4:
            raise ValueError(f"Expected probabilistic samples with shape (B, S, C, H), got {tuple(samples.shape)}")
        return samples

    def _extract_samples(self, output: Any) -> torch.Tensor:
        """Extract a samples tensor from common model output forms."""
        if isinstance(output, dict):
            if "samples" in output:
                return output["samples"]
            if "preds" in output:
                return output["preds"]
            if "pred" in output:
                return output["pred"]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _to_median_forecast(self, output: Any) -> torch.Tensor:
        """Extract a median forecast with shape (B, C, H)."""
        if isinstance(output, dict):
            for key in ("preds", "pred", "forecast", "samples"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() == 4:
            output = torch.median(output, dim=1)[0]
        if output.dim() != 3:
            raise ValueError(f"Expected prediction with shape (B, C, H), got {tuple(output.shape)}")
        return output

    def _align_target_to_prediction(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Align target horizon with a prediction or sample tensor."""
        horizon = prediction.shape[-1]
        if target.shape[-1] != horizon:
            target = target[..., -horizon:]
        return target

    def _build_kwargs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Build kwargs dict from batch."""
        kwargs = {}
        for key in ("idx", "x_mark", "y_mark", "y"):
            val = batch.get(key)
            if val is not None:
                kwargs[key] = val
        return kwargs

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Any:
        """Prediction step."""
        samples = self._sample(batch)
        preds = torch.median(samples, dim=1)[0]
        result = {
            "inputs": batch["x"],
            "samples": samples,
            "preds": preds,
        }
        if "y" in batch:
            targets = self._align_target_to_prediction(batch["y"], samples)
            result["targets"] = targets
            result["y"] = targets
        return result

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        from ts_trainer.trainer import SchedulerFactory

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        max_epochs = getattr(self, '_max_epochs', 100)
        return SchedulerFactory.create(
            optimizer,
            scheduler_type="cosine",
            max_epochs=max_epochs,
        )


__all__ = ["ProbForecastModule"]
