"""Tests for ts_trainer.config."""

import tempfile
from pathlib import Path

import pytest

from ts_trainer.config import (
    STAGE_PRESETS,
    StageConfig,
    TrainerConfig,
)


class TestStageConfig:
    def test_defaults(self):
        stage = StageConfig(name="train")
        assert stage.name == "train"
        assert stage.epochs == 100
        assert stage.lr == 1e-3
        assert stage.freeze_modules == []
        assert stage.extra_kwargs == {}

    def test_custom(self):
        stage = StageConfig(name="pretrain", epochs=50, lr=5e-4, freeze_modules=["encoder"])
        assert stage.epochs == 50
        assert stage.freeze_modules == ["encoder"]

    def test_invalid_epochs(self):
        with pytest.raises(Exception):
            StageConfig(name="bad", epochs=0)

    def test_invalid_lr(self):
        with pytest.raises(Exception):
            StageConfig(name="bad", lr=-1)

    def test_extra_forbidden(self):
        with pytest.raises(Exception):
            StageConfig(name="bad", unknown_field=True)


class TestStagePresets:
    def test_single(self):
        assert "single" in STAGE_PRESETS
        stages = STAGE_PRESETS["single"]
        assert len(stages) == 1
        assert stages[0]["name"] == "train"

    def test_two_stage(self):
        assert "two_stage" in STAGE_PRESETS
        stages = STAGE_PRESETS["two_stage"]
        assert len(stages) == 2
        assert stages[0]["name"] == "pretrain"
        assert stages[1]["load_from_stage"] == "pretrain"

    def test_pretrain_freeze(self):
        assert "pretrain_freeze" in STAGE_PRESETS
        stages = STAGE_PRESETS["pretrain_freeze"]
        assert "freeze_modules" in stages[1]


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.max_epochs == 100
        assert cfg.lr_scheduler == "cosine"
        assert cfg.logger == "tensorboard"
        assert cfg.save_dir == "./experiments"
        assert cfg.dataset_name is None
        assert cfg.model_name is None
        assert cfg.run_name is None
        assert cfg.stages is not None
        assert len(cfg.stages) == 1
        assert cfg.stages[0].name == "train"

    def test_kwargs(self):
        cfg = TrainerConfig(max_epochs=50, lr_scheduler="plateau", logger="csv")
        assert cfg.max_epochs == 50
        assert cfg.lr_scheduler == "plateau"
        assert cfg.logger == "csv"

    def test_invalid_scheduler(self):
        with pytest.raises(Exception):
            TrainerConfig(lr_scheduler="invalid")

    def test_invalid_logger(self):
        with pytest.raises(Exception):
            TrainerConfig(logger="invalid")

    def test_negative_epochs(self):
        with pytest.raises(Exception):
            TrainerConfig(max_epochs=-1)

    def test_yaml_roundtrip(self, tmp_path):
        cfg = TrainerConfig(
            max_epochs=50,
            lr_scheduler="step",
            experiment_name="test_exp",
        )
        yaml_path = tmp_path / "config.yaml"
        cfg.to_yaml(yaml_path)

        loaded = TrainerConfig.from_yaml(yaml_path)
        assert loaded.max_epochs == 50
        assert loaded.lr_scheduler == "step"
        assert loaded.experiment_name == "test_exp"

    def test_from_yaml_not_found(self):
        with pytest.raises(FileNotFoundError):
            TrainerConfig.from_yaml("/nonexistent/config.yaml")

    def test_resolve_stages_none(self):
        cfg = TrainerConfig(max_epochs=200)
        assert len(cfg.stages) == 1
        assert cfg.stages[0].epochs == 200

    def test_resolve_stages_explicit(self):
        stages = [
            StageConfig(name="pre", epochs=30),
            StageConfig(name="main", epochs=70, load_from_stage="pre"),
        ]
        cfg = TrainerConfig(stages=stages)
        assert len(cfg.stages) == 2
        assert cfg.stages[1].load_from_stage == "pre"
