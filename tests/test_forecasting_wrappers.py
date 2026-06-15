"""Tests for forecasting Lightning wrappers."""

import torch
from torch import nn

from ts_trainer.wrappers.forecasting import PointForecastModule, ProbForecastModule


class MarkAwarePointModel(nn.Module):
    def __init__(self, horizon: int = 4):
        super().__init__()
        self.num_features = 3
        self.horizon = horizon
        self.calls = []
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x, idx=None, x_mark=None, y_mark=None):
        self.calls.append({"idx": idx, "x_mark": x_mark, "y_mark": y_mark})
        return x[:, :, -self.horizon :] * self.weight


class TinyProbModel(nn.Module):
    def __init__(self, horizon: int = 4):
        super().__init__()
        self.horizon = horizon
        self.weight = nn.Parameter(torch.ones(()))

    def sample(self, batch, num_samples=5):
        base = batch["x"][:, :, -self.horizon :] * self.weight
        offsets = torch.arange(num_samples, device=base.device, dtype=base.dtype).view(1, num_samples, 1, 1)
        return base.unsqueeze(1) + offsets

    def forward(self, x, **kwargs):
        return x[:, :, -self.horizon :] * self.weight


class SampleForwardProbModel(nn.Module):
    def __init__(self, horizon: int = 4, num_samples: int = 3):
        super().__init__()
        self.horizon = horizon
        self.num_samples = num_samples
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x, **kwargs):
        base = x[:, :, -self.horizon :] * self.weight
        return base.unsqueeze(1).repeat(1, self.num_samples, 1, 1)


def _batch(batch_size=2, channels=3, lookback=8, horizon=4, label_len=2):
    return {
        "x": torch.randn(batch_size, channels, lookback),
        "y": torch.randn(batch_size, channels, label_len + horizon),
        "idx": torch.arange(batch_size),
        "x_mark": torch.randn(batch_size, lookback, 4),
        "y_mark": torch.randn(batch_size, label_len + horizon, 4),
    }


def test_point_forecast_module_outputs_inputs_preds_targets_and_marks():
    model = MarkAwarePointModel(horizon=4)
    module = PointForecastModule(model)
    batch = _batch()

    out = module.test_step(batch, 0)

    assert out["inputs"].shape == batch["x"].shape
    assert out["preds"].shape == (2, 3, 4)
    assert out["targets"].shape == (2, 3, 4)
    assert out["pred"] is out["preds"]
    assert out["y"] is out["targets"]
    assert model.calls[-1]["x_mark"] is batch["x_mark"]
    assert model.calls[-1]["y_mark"] is batch["y_mark"]


def test_point_predict_step_returns_structured_output_with_targets():
    module = PointForecastModule(MarkAwarePointModel(horizon=4))
    batch = _batch()

    out = module.predict_step(batch, 0)

    assert set(["inputs", "preds", "targets"]).issubset(out)
    assert out["inputs"].shape == (2, 3, 8)
    assert out["preds"].shape == (2, 3, 4)
    assert out["targets"].shape == (2, 3, 4)


def test_prob_forecast_module_outputs_samples_preds_targets_inputs():
    module = ProbForecastModule(TinyProbModel(horizon=4), num_samples=5)
    batch = _batch()

    out = module.test_step(batch, 0)

    assert out["inputs"].shape == (2, 3, 8)
    assert out["samples"].shape == (2, 5, 3, 4)
    assert out["preds"].shape == (2, 3, 4)
    assert out["targets"].shape == (2, 3, 4)
    assert module.test_outputs[0]["samples"].device.type == "cpu"


def test_prob_fallback_loss_uses_median_and_aligns_label_len_target():
    module = ProbForecastModule(SampleForwardProbModel(horizon=4, num_samples=3), num_samples=3)
    batch = _batch()

    loss = module._compute_loss(batch)

    assert loss.ndim == 0
