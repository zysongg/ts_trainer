"""Tests for ts_trainer.callbacks."""

import time

import pytest
import torch

from ts_trainer.callbacks import (
    GradientMonitor,
    PredictionWriter,
    SlimProgressBar,
    TrainingTimer,
)


class TestGradientMonitor:
    def test_init(self):
        gm = GradientMonitor(log_every_n_steps=10, log_grad_norm=True, log_param_norm=True)
        assert gm.log_every_n_steps == 10
        assert gm.log_grad_norm is True
        assert gm.log_param_norm is True

    def test_compute_grad_norm(self):
        from ts_trainer.callbacks import _compute_grad_norm

        model = torch.nn.Linear(10, 1)
        # Simulate a backward pass
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        norm = _compute_grad_norm(model)
        assert isinstance(norm, float)
        assert norm > 0

    def test_compute_param_norm(self):
        from ts_trainer.callbacks import _compute_param_norm

        model = torch.nn.Linear(10, 1)
        norm = _compute_param_norm(model)
        assert isinstance(norm, float)
        assert norm > 0


class TestTrainingTimer:
    def test_init(self):
        timer = TrainingTimer()
        assert timer.epoch_times == []
        assert timer.total_time == 0.0

    def test_avg_epoch_time_empty(self):
        timer = TrainingTimer()
        assert timer.avg_epoch_time == 0.0

    def test_avg_epoch_time(self):
        timer = TrainingTimer()
        timer.epoch_times = [1.0, 2.0, 3.0]
        assert timer.avg_epoch_time == 2.0

    def test_epoch_tracking(self):
        timer = TrainingTimer()
        timer._train_start = time.time()
        timer._epoch_start = time.time()
        time.sleep(0.01)
        timer.epoch_times.append(time.time() - timer._epoch_start)
        assert len(timer.epoch_times) == 1
        assert timer.epoch_times[0] >= 0


class TestPredictionWriter:
    def test_init(self, tmp_path):
        writer = PredictionWriter(output_dir=str(tmp_path / "preds"))
        assert writer._inputs == []
        assert writer._predictions == []
        assert writer._targets == []
        assert writer._samples == []

    def test_save(self, tmp_path):
        import numpy as np

        writer = PredictionWriter(output_dir=str(tmp_path / "preds"))
        writer._predictions = [np.random.randn(10, 5)]
        writer._targets = [np.random.randn(10, 5)]
        writer._save("test")
        assert (tmp_path / "preds" / "test_results.npz").exists()

        data = np.load(tmp_path / "preds" / "test_results.npz")
        assert "predictions" in data
        assert "targets" in data
        assert data["predictions"].shape == (10, 5)

    def test_save_no_predictions(self, tmp_path):
        writer = PredictionWriter(output_dir=str(tmp_path / "preds"))
        writer._save("test")  # Should not create file
        assert not (tmp_path / "preds" / "test_results.npz").exists()

    def test_save_predictions_only(self, tmp_path):
        import numpy as np

        writer = PredictionWriter(output_dir=str(tmp_path / "preds"))
        writer._predictions = [np.random.randn(10, 5)]
        writer._save("predict")
        assert (tmp_path / "preds" / "predict_results.npz").exists()

        data = np.load(tmp_path / "preds" / "predict_results.npz")
        assert "predictions" in data
        assert "targets" not in data

    def test_collect_structured_probabilistic_outputs(self, tmp_path):
        import numpy as np

        writer = PredictionWriter(output_dir=str(tmp_path / "preds"))
        outputs = {
            "inputs": torch.randn(2, 3, 8),
            "samples": torch.randn(2, 5, 3, 4),
            "preds": torch.randn(2, 3, 4),
            "targets": torch.randn(2, 3, 4),
        }
        writer._collect(outputs, batch={})
        writer._save("predict")

        data = np.load(tmp_path / "preds" / "predict_results.npz")
        assert data["inputs"].shape == (2, 3, 8)
        assert data["samples"].shape == (2, 5, 3, 4)
        assert data["predictions"].shape == (2, 3, 4)
        assert data["targets"].shape == (2, 3, 4)

    def test_collect_legacy_pred_y_outputs(self, tmp_path):
        import numpy as np

        writer = PredictionWriter(output_dir=str(tmp_path / "preds"))
        writer._collect(
            {"pred": torch.randn(2, 3, 4), "y": torch.randn(2, 3, 4)},
            batch={"x": torch.randn(2, 3, 8)},
        )
        writer._save("test")

        data = np.load(tmp_path / "preds" / "test_results.npz")
        assert data["inputs"].shape == (2, 3, 8)
        assert data["predictions"].shape == (2, 3, 4)
        assert data["targets"].shape == (2, 3, 4)


class TestSlimProgressBar:
    def test_init(self):
        bar = SlimProgressBar(log_every_n_steps=10)
        assert bar._train_start == 0.0

    def test_format_time_seconds(self):
        from ts_trainer.callbacks import _format_time

        assert _format_time(5.0) == "5.0s"
        assert _format_time(59.9) == "59.9s"

    def test_format_time_minutes(self):
        from ts_trainer.callbacks import _format_time

        assert _format_time(61.0) == "1m01s"
        assert _format_time(150.0) == "2m30s"

    def test_disable_noop(self):
        bar = SlimProgressBar()
        bar.disable()  # Should not raise
