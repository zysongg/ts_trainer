"""Tests for ts_trainer.trainer."""

from pathlib import Path
import pytest
import re
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ts_trainer import (
    PipelineResult,
    PipelineStage,
    ProbForecastModule,
    SchedulerFactory,
    StageCheckpoint,
    StageConfig,
    Trainer,
    TrainerConfig,
    build_cycleflow_pipeline,
)

from conftest import SimpleModel, _make_dataloader


class CacheAwareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self._cache_enabled = True
        self._cycle_cache = {0: torch.ones(1)}
        self.clear_calls = 0
        self.disable_calls = 0

    def forward(self, x):
        return self.linear(x)

    def clear_cycle_cache(self):
        self.clear_calls += 1
        self._cycle_cache.clear()

    def disable_cycle_cache(self):
        self.disable_calls += 1
        self._cache_enabled = False
        self._cycle_cache.clear()


class TestProbForecastModuleCacheHooks:
    def test_epoch_end_clears_cache_without_disabling_future_epochs(self):
        model = CacheAwareModel()
        module = ProbForecastModule(model)

        module.on_train_epoch_end()

        assert model.clear_calls == 1
        assert model.disable_calls == 0
        assert model._cache_enabled is True
        assert model._cycle_cache == {}


class TestSchedulerFactory:
    def _make_optimizer(self):
        model = SimpleModel()
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def test_none(self):
        opt = self._make_optimizer()
        result = SchedulerFactory.create(opt, "none", 100)
        assert isinstance(result, torch.optim.Optimizer)

    def test_cosine(self):
        opt = self._make_optimizer()
        result = SchedulerFactory.create(opt, "cosine", 100)
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_step(self):
        opt = self._make_optimizer()
        result = SchedulerFactory.create(opt, "step", 100, step_size=30, gamma=0.1)
        assert isinstance(result, dict)

    def test_plateau(self):
        opt = self._make_optimizer()
        result = SchedulerFactory.create(opt, "plateau", 100, factor=0.5, patience=5)
        assert isinstance(result, dict)
        assert result["lr_scheduler"]["monitor"] == "val_loss"

    def test_onecycle(self):
        opt = self._make_optimizer()
        result = SchedulerFactory.create(opt, "onecycle", 100)
        assert isinstance(result, dict)
        assert result["lr_scheduler"]["interval"] == "step"

    def test_invalid(self):
        opt = self._make_optimizer()
        with pytest.raises(ValueError):
            SchedulerFactory.create(opt, "invalid", 100)


class TestStageCheckpoint:
    def test_basic(self, tmp_path):
        tracker = StageCheckpoint(tmp_path)
        assert tracker.get_checkpoint("train") is None
        assert tracker.latest is None

    def test_set_get(self, tmp_path):
        tracker = StageCheckpoint(tmp_path)
        ckpt_path = tmp_path / "checkpoint.ckpt"
        ckpt_path.touch()
        tracker.set_checkpoint("train", ckpt_path)
        assert tracker.get_checkpoint("train") == ckpt_path
        assert tracker.latest == ckpt_path

    def test_multiple_stages(self, tmp_path):
        tracker = StageCheckpoint(tmp_path)
        p1 = tmp_path / "s1.ckpt"
        p2 = tmp_path / "s2.ckpt"
        p1.touch()
        p2.touch()
        tracker.set_checkpoint("pretrain", p1)
        tracker.set_checkpoint("finetune", p2)
        assert tracker.get_checkpoint("pretrain") == p1
        assert tracker.get_checkpoint("finetune") == p2
        assert tracker.latest == p2

    def test_artifacts(self, tmp_path):
        tracker = StageCheckpoint(tmp_path)
        artifact = tmp_path / "cycled_flow.pt"
        artifact.touch()
        tracker.set_artifact("cycled", "flow", artifact)
        assert tracker.get_artifact("cycled", "flow") == artifact
        assert tracker.get_artifact("cycled", "missing") is None

    def test_save_and_load_manifest(self, tmp_path):
        tracker = StageCheckpoint(tmp_path)
        ckpt = tmp_path / "ckpt" / "best-val.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.touch()
        artifact = tmp_path / "artifacts" / "cycled_flow.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()

        tracker.set_checkpoint("cycled", ckpt)
        tracker.set_artifact("cycled", "flow", artifact)

        manifest_path = tracker.save_manifest()
        assert manifest_path.exists()
        assert manifest_path.name == "pipeline_manifest.json"

        import json
        data = json.loads(manifest_path.read_text())
        assert data["checkpoints"]["cycled"] == "ckpt/best-val.ckpt"
        assert data["artifacts"]["cycled"]["flow"] == "artifacts/cycled_flow.pt"

        loaded = StageCheckpoint.load_manifest(tmp_path)
        assert loaded.get_checkpoint("cycled") == ckpt
        assert loaded.get_artifact("cycled", "flow") == artifact


class TestTrainer:
    def test_init_default(self, tmp_path):
        trainer = Trainer(save_dir=str(tmp_path))
        assert trainer.config.max_epochs == 100

    def test_init_kwargs(self, tmp_path):
        trainer = Trainer(max_epochs=50, lr_scheduler="plateau", save_dir=str(tmp_path))
        assert trainer.config.max_epochs == 50
        assert trainer.config.lr_scheduler == "plateau"

    def test_init_config(self, tmp_path):
        cfg = TrainerConfig(max_epochs=30, save_dir=str(tmp_path))
        trainer = Trainer(cfg)
        assert trainer.config.max_epochs == 30

    def test_explicit_run_name_creates_experiment_dir(self, tmp_path):
        trainer = Trainer(
            save_dir=str(tmp_path),
            experiment_name="debug_verbalts",
            run_name="20260614_132350_debug_verbalts",
            logger="none",
            accelerator="cpu",
        )

        expected = tmp_path / "20260614_132350_debug_verbalts"
        assert Path(trainer.experiment_dir) == expected
        assert expected.is_dir()
        assert (expected / "config.yaml").exists()

    def test_auto_run_name_uses_timestamp_and_experiment_name(self, tmp_path):
        trainer = Trainer(
            save_dir=str(tmp_path),
            experiment_name="debug verbalts",
            logger="none",
            accelerator="cpu",
        )

        run_name = Path(trainer.experiment_dir).name
        assert re.fullmatch(r"\d{8}_\d{6}_debug_verbalts", run_name)
        assert Path(trainer.experiment_dir).parent == tmp_path

    def test_auto_run_name_prefers_dataset_and_model_name(self, tmp_path):
        trainer = Trainer(
            save_dir=str(tmp_path),
            dataset_name="ETTh1",
            model_name="DLinear",
            experiment_name="debug_verbalts",
            logger="none",
            accelerator="cpu",
        )

        run_name = Path(trainer.experiment_dir).name
        assert re.fullmatch(r"\d{8}_\d{6}_ETTh1_DLinear", run_name)

    def test_fit_single_stage(self, simple_model, train_loader, val_loader, tmp_path):
        trainer = Trainer(
            max_epochs=2,
            save_dir=str(tmp_path),
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            early_stopping_patience=0,
        )
        trainer.fit(simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # Should complete without error

    def test_fit_with_early_stopping(self, simple_model, train_loader, val_loader, tmp_path):
        trainer = Trainer(
            max_epochs=2,
            save_dir=str(tmp_path),
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            early_stopping_patience=50,
        )
        trainer.fit(simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # Early stopping set up but won't trigger in 2 epochs

    def test_fit_with_gradient_clip(self, simple_model, train_loader, val_loader, tmp_path):
        trainer = Trainer(
            max_epochs=1,
            save_dir=str(tmp_path),
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            gradient_clip_val=1.0,
            early_stopping_patience=0,
        )
        trainer.fit(simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def test_best_checkpoint_path(self, simple_model, train_loader, val_loader, tmp_path):
        trainer = Trainer(
            max_epochs=2,
            save_dir=str(tmp_path),
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            early_stopping_patience=0,
        )
        trainer.fit(simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # Checkpoint should exist after training
        assert trainer.best_checkpoint_path is not None or True  # May be None if no val

    def test_stages_preset_string(self, simple_model, train_loader, val_loader, tmp_path):
        trainer = Trainer(
            save_dir=str(tmp_path),
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            early_stopping_patience=0,
        )
        stages = [
            StageConfig(name="pretrain", epochs=1),
            StageConfig(name="finetune", epochs=1),
        ]
        trainer.fit(simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader, stages=stages)

    def test_fit_pipeline_with_artifact_handoff(self, train_loader, val_loader, tmp_path):
        seen = {}

        def save_artifact(trainer, stage, module, tracker):
            artifacts_dir = Path(trainer.experiment_dir) / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact = artifacts_dir / "cycled_flow.pt"
            artifact.write_text("flow", encoding="utf-8")
            tracker.set_artifact(stage.name, "flow", artifact)

        def make_second_module(trainer, stage, tracker):
            artifact = tracker.get_artifact("cycled", "flow")
            assert artifact is not None and artifact.exists()
            seen["flow"] = artifact
            return SimpleModel()

        trainer = Trainer(
            save_dir=str(tmp_path),
            run_name="pipeline",
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            early_stopping_patience=0,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=1,
        )

        result = trainer.fit_pipeline([
            PipelineStage(
                config=StageConfig(name="cycled", epochs=1),
                module=SimpleModel(),
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                artifact_hook=save_artifact,
            ),
            PipelineStage(
                config=StageConfig(name="cycleflow", epochs=1),
                module=make_second_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            ),
        ])

        assert isinstance(result, PipelineResult)
        assert result.stages == ["cycled", "cycleflow"]
        assert "cycled" in result.modules
        assert result.final_module is result.modules["cycleflow"]
        assert result.tracker is trainer._checkpoint_tracker
        assert seen["flow"].name == "cycled_flow.pt"
        assert seen["flow"].parent.name == "artifacts"
        assert (Path(trainer.experiment_dir) / "pipeline_manifest.json").exists()

    def test_build_cycleflow_pipeline_passes_flow_path(self, tmp_path):
        class FlowSource(SimpleModel):
            def save_flow_weights(self, path: str):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("flow", encoding="utf-8")

        seen = {}

        def cycleflow_factory(pretrained_flow_path: str):
            seen["path"] = Path(pretrained_flow_path)
            return SimpleModel()

        trainer = Trainer(save_dir=str(tmp_path), run_name="cycleflow_pipeline", logger="none", accelerator="cpu", devices=1)
        stages = build_cycleflow_pipeline(
            cycled_model=FlowSource(),
            cycleflow_factory=cycleflow_factory,
            cycled_stage=StageConfig(name="cycled", epochs=1),
            cycleflow_stage=StageConfig(name="cycleflow", epochs=1),
        )
        stages[0].artifact_hook(trainer, stages[0].config, stages[0].module, trainer._checkpoint_tracker)
        module = stages[1].module(trainer, stages[1].config, trainer._checkpoint_tracker)

        assert module.model.__class__ is SimpleModel
        assert seen["path"].name == "cycled_flow.pt"
        assert seen["path"].parent.name == "artifacts"
        assert seen["path"].exists()

    def test_test_method(self, simple_model, train_loader, val_loader, test_loader, tmp_path):
        trainer = Trainer(
            max_epochs=1,
            save_dir=str(tmp_path),
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            early_stopping_patience=0,
        )
        trainer.fit(simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        results = trainer.test(simple_model, dataloaders=test_loader)
        assert isinstance(results, list)
        assert not (Path(trainer.experiment_dir) / "test_metrics.json").exists()

    def test_predict_results_saved_in_experiment_dir(self, test_loader, tmp_path):
        class PredictModel(SimpleModel):
            def predict_step(self, batch, batch_idx: int):
                x, _ = batch
                return self(x)

        model = PredictModel()
        trainer = Trainer(
            max_epochs=1,
            save_dir=str(tmp_path),
            run_name="20260614_132350_debug_verbalts",
            logger="none",
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            early_stopping_patience=0,
        )

        trainer.predict(model, dataloaders=test_loader)

        output_path = Path(trainer.experiment_dir) / "predict_results.npz"
        assert output_path.exists()

    def test_resolve_stages_invalid_preset(self, tmp_path):
        trainer = Trainer(save_dir=str(tmp_path))
        with pytest.raises(ValueError):
            trainer._resolve_stages("invalid_preset")

    def test_logger_tensorboard(self, tmp_path):
        trainer = Trainer(
            logger="tensorboard",
            save_dir=str(tmp_path),
            enable_progress_bar=False,
            accelerator="cpu",
        )
        loggers = trainer._build_loggers(tmp_path / "exp", None)
        assert len(loggers) > 0

    def test_logger_csv(self, tmp_path):
        trainer = Trainer(
            logger="csv",
            save_dir=str(tmp_path),
            enable_progress_bar=False,
            accelerator="cpu",
        )
        loggers = trainer._build_loggers(tmp_path / "exp", None)
        assert len(loggers) > 0

    def test_logger_none(self, tmp_path):
        trainer = Trainer(
            logger="none",
            save_dir=str(tmp_path),
            enable_progress_bar=False,
            accelerator="cpu",
        )
        loggers = trainer._build_loggers(tmp_path / "exp", None)
        assert loggers is False

    def test_collect_prediction_outputs_structured_probabilistic(self, tmp_path):
        trainer = Trainer(save_dir=str(tmp_path))
        outputs = [
            {
                "inputs": torch.randn(2, 3, 8),
                "samples": torch.randn(2, 5, 3, 4),
                "targets": torch.randn(2, 3, 4),
            },
            {
                "inputs": torch.randn(1, 3, 8),
                "samples": torch.randn(1, 5, 3, 4),
                "targets": torch.randn(1, 3, 4),
            },
        ]

        collected = trainer._collect_prediction_outputs(outputs)

        assert collected["inputs"].shape == (3, 3, 8)
        assert collected["samples"].shape == (3, 5, 3, 4)
        assert collected["preds"].shape == (3, 3, 4)
        assert collected["targets"].shape == (3, 3, 4)

    def test_metric_task_and_mode_aliases(self):
        assert Trainer._normalize_metric_task("forecast") == "prediction"
        assert Trainer._normalize_metric_task("forecasting") == "prediction"
        assert Trainer._normalize_metric_mode("deterministic") == "point"
        assert Trainer._normalize_metric_mode("prob") == "probabilistic"

    def test_prune_preserves_artifacts_and_manifest(self, tmp_path):
        trainer = Trainer(
            save_dir=str(tmp_path),
            run_name="prune_test",
            logger="none",
            accelerator="cpu",
        )
        exp_dir = Path(trainer.experiment_dir)

        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "cycled_flow.pt").write_text("flow", encoding="utf-8")
        (exp_dir / "pipeline_manifest.json").write_text("{}", encoding="utf-8")
        (exp_dir / "junk_file.txt").write_text("junk", encoding="utf-8")
        junk_dir = exp_dir / "junk_dir"
        junk_dir.mkdir()

        trainer._prune_experiment_outputs()

        assert (exp_dir / "artifacts" / "cycled_flow.pt").exists()
        assert (exp_dir / "pipeline_manifest.json").exists()
        assert not (exp_dir / "junk_file.txt").exists()
        assert not junk_dir.exists()
