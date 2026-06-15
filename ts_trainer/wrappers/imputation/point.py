"""Lightning module wrapper for point imputation models."""

import inspect
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl


class PointImputationModule(pl.LightningModule):
    """Lightning wrapper for point imputation models.

    Args:
        model: A ts_model imputation model (ImputationModel)
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        loss_fn: Loss function (default: MSE loss)

    Example:
        >>> from ts_model import create_model
        >>> from ts_trainer import PointImputationModule, Trainer
        >>>
        >>> model = create_model("SAITS", task="imputation", ...)
        >>> module = PointImputationModule(model, lr=1e-3)
        >>> trainer = Trainer(max_epochs=100)
        >>> trainer.fit(module, train_dataloaders=train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn or F.mse_loss

        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass."""
        sig = inspect.signature(self.model.forward)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.model(x, **valid_kwargs)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared logic for train/val/test steps."""
        x = batch["x"]
        mask = batch.get("mask")
        idx = batch.get("idx")

        # Forward
        pred = self.forward(x, mask=mask, idx=idx)

        # Compute loss on masked positions
        if mask is not None:
            loss = self.loss_fn(pred * mask, x * mask)
        else:
            loss = self.loss_fn(pred, x)

        # Log
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"))
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        x = batch["x"]
        mask = batch.get("mask")
        idx = batch.get("idx")

        pred = self.forward(x, mask=mask, idx=idx)

        if mask is not None:
            loss = self.loss_fn(pred * mask, x * mask)
        else:
            loss = self.loss_fn(pred, x)

        self.log("test_loss", loss)
        return {"pred": pred, "x": x, "mask": mask}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        x = batch["x"]
        mask = batch.get("mask")
        idx = batch.get("idx")
        return self.forward(x, mask=mask, idx=idx)

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


__all__ = ["PointImputationModule"]
