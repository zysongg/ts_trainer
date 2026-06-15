"""Lightning module wrapper for probabilistic imputation models."""

import inspect
from typing import Any, Dict

import torch
import torch.nn as nn
import lightning as pl


class ProbImputationModule(pl.LightningModule):
    """Lightning wrapper for probabilistic imputation models.

    Args:
        model: A ts_model probabilistic imputation model
        lr: Learning rate
        weight_decay: Weight decay for optimizer

    Example:
        >>> from ts_model import create_model
        >>> from ts_trainer import ProbImputationModule, Trainer
        >>>
        >>> model = create_model("CSDI", task="imputation", ...)
        >>> module = ProbImputationModule(model, lr=1e-3)
        >>> trainer = Trainer(max_epochs=100)
        >>> trainer.fit(module, train_dataloaders=train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor, **kwargs) -> Any:
        """Forward pass."""
        sig = inspect.signature(self.model.forward)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.model(x, **valid_kwargs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - expects model to return loss directly."""
        x = batch["x"]
        mask = batch.get("mask")
        idx = batch.get("idx")

        # ProbModel should return loss directly
        loss = self.model.training_step(x, mask=mask, idx=idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        x = batch["x"]
        mask = batch.get("mask")
        idx = batch.get("idx")

        loss = self.model.validation_step(x, mask=mask, idx=idx)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Test step."""
        x = batch["x"]
        mask = batch.get("mask")
        idx = batch.get("idx")

        result = self.model.test_step(x, mask=mask, idx=idx)
        if isinstance(result, dict):
            self.log("test_loss", result.get("loss", 0.0))
            return result
        else:
            self.log("test_loss", result)
            return {"loss": result}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Any:
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


__all__ = ["ProbImputationModule"]
