"""Lightning module wrapper for point forecasting models."""

import inspect
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

from ts_model.layers import RevIN


class PointForecastModule(pl.LightningModule):
    """Lightning wrapper for point forecasting models.

    Args:
        model: A ts_model forecasting model (PointModel)
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        loss_fn: Loss function (default: MSE loss)
        use_norm: Whether to use RevIN normalization (default: False)
        norm_affine: Whether RevIN has learnable affine parameters (default: False)

    Example:
        >>> from ts_model import create_model
        >>> from ts_trainer import PointForecastModule, Trainer
        >>>
        >>> model = create_model("DLinear", task="forecasting", ...)
        >>> module = PointForecastModule(model, lr=1e-3, use_norm=True)
        >>> trainer = Trainer(max_epochs=100)
        >>> trainer.fit(module, train_dataloaders=train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_fn: Optional[Callable] = None,
        use_norm: bool = False,
        norm_affine: bool = False,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn or F.mse_loss
        self.use_norm = use_norm
        self._forward_params = set(inspect.signature(self.model.forward).parameters)

        # Setup RevIN normalization
        if use_norm:
            self.revin = self._setup_normalization(norm_affine)
        else:
            self.revin = None

        # Storage for batch outputs (to compute epoch-level averages)
        self.train_outputs: List[Dict[str, torch.Tensor]] = []
        self.val_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_outputs: List[Dict[str, torch.Tensor]] = []

        # Save hyperparameters except model
        self.save_hyperparameters(ignore=["model", "loss_fn", "revin"])

    def _setup_normalization(self, affine: bool = False) -> RevIN:
        """Setup RevIN normalization layer.

        Args:
            affine: Whether to use learnable affine parameters

        Returns:
            RevIN instance
        """
        return RevIN(
            num_features=self.model.num_features,
            eps=1e-5,
            affine=affine,
            feature_dim=1,  # (B, C, L) format
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with automatic kwargs filtering."""
        valid_kwargs = self._filter_forward_kwargs(kwargs)
        return self.model(x, **valid_kwargs)

    def _filter_forward_kwargs(self, kwargs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Keep only kwargs accepted by the wrapped model."""
        return {k: v for k, v in kwargs.items() if k in self._forward_params}

    def _build_kwargs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build optional model kwargs from a forecast batch."""
        kwargs = {}
        for key in ("idx", "x_mark", "y_mark"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value
        return kwargs

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared logic for train/val/test steps.

        Returns:
            Dict with 'pred', 'y', and 'loss' tensors.
        """
        x = batch["x"]
        y = batch["y"]

        # Apply RevIN normalization
        if self.revin is not None:
            x = self.revin(x, mode="norm")

        # Forward - pass time marks if model accepts them
        pred = self.forward(x, **self._build_kwargs(batch))

        # Apply RevIN denormalization
        if self.revin is not None:
            pred = self.revin(pred, mode="denorm")

        # Handle different y shapes
        # y shape: (B, C, label_len + pred_len) or (B, C, pred_len)
        if pred.shape != y.shape:
            # Take only the prediction length from y
            y = y[:, :, -pred.shape[2]:]

        loss = self.loss_fn(pred, y)

        return {"pred": pred, "y": y, "loss": loss}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        output = self._shared_step(batch, "train")
        batch_size = batch["x"].size(0)
        self.log("train_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.train_outputs.append({"loss": output["loss"]})
        return output

    def on_train_epoch_end(self) -> None:
        """Compute epoch-level average train loss."""
        if not self.train_outputs:
            return
        avg_loss = torch.stack([o["loss"] for o in self.train_outputs]).mean()
        self.log("train_epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.train_outputs.clear()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        output = self._shared_step(batch, "val")
        batch_size = batch["x"].size(0)
        mse = F.mse_loss(output["pred"], output["y"])
        self.log("val_loss", output["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.val_outputs.append({"loss": output["loss"], "mse": mse})
        return output

    def on_validation_epoch_end(self) -> None:
        """Compute epoch-level average validation metrics."""
        if not self.val_outputs:
            return
        avg_loss = torch.stack([o["loss"] for o in self.val_outputs]).mean()
        avg_mse = torch.stack([o["mse"] for o in self.val_outputs]).mean()
        self.log("val_epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_epoch_mse", avg_mse, prog_bar=True, sync_dist=True)
        self.val_outputs.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        output = self._shared_step(batch, "test")
        self.log("test_loss", output["loss"])
        result = {
            "inputs": batch["x"],
            "preds": output["pred"],
            "targets": output["y"],
            "loss": output["loss"],
            # Backward-compatible aliases.
            "pred": output["pred"],
            "y": output["y"],
        }
        self.test_outputs.append(result)
        return result

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        x = batch["x"]

        # Apply RevIN normalization
        if self.revin is not None:
            x = self.revin(x, mode="norm")

        pred = self.forward(x, **self._build_kwargs(batch))

        # Apply RevIN denormalization
        if self.revin is not None:
            pred = self.revin(pred, mode="denorm")

        result = {"inputs": batch["x"], "preds": pred, "pred": pred}
        if "y" in batch:
            y = batch["y"]
            if pred.shape != y.shape:
                y = y[:, :, -pred.shape[2]:]
            result["targets"] = y
            result["y"] = y
        return result

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        from ts_trainer.trainer import SchedulerFactory
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Default to cosine scheduler
        max_epochs = getattr(self, '_max_epochs', 100)
        return SchedulerFactory.create(
            optimizer,
            scheduler_type="cosine",
            max_epochs=max_epochs,
        )


__all__ = ["PointForecastModule"]
