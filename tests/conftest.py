"""Shared test fixtures for ts_trainer."""

import sys
import os

import lightning as pl
import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class SimpleModel(pl.LightningModule):
    """Minimal LightningModule for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 1, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(input_dim, output_dim)
        self.lr = lr
        self.stage_lr = lr

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def _make_dataloader(n_samples: int = 200, input_dim: int = 10, batch_size: int = 32):
    """Create a simple random dataloader."""
    x = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def train_loader():
    return _make_dataloader(n_samples=200)


@pytest.fixture
def val_loader():
    return _make_dataloader(n_samples=64, batch_size=16)


@pytest.fixture
def test_loader():
    return _make_dataloader(n_samples=64, batch_size=16)
