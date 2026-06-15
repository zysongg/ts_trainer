"""Helpers for two-stage forecasting training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import StageConfig
from .trainer import PipelineStage, StageCheckpoint, Trainer
from .wrappers import ProbForecastModule


def save_cycleflow_flow_artifact(
    trainer: Trainer,
    stage: StageConfig,
    module,
    tracker: StageCheckpoint,
    artifact_name: str = "flow",
    filename: str = "cycled_flow.pt",
) -> Path:
    """Save a CycleD flow artifact from a probabilistic wrapper module."""
    model = getattr(module, "model", module)
    if not hasattr(model, "save_flow_weights"):
        raise AttributeError("CycleD stage model must implement save_flow_weights(path)")
    artifacts_dir = Path(trainer.experiment_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / filename
    model.save_flow_weights(str(path))
    tracker.set_artifact(stage.name, artifact_name, path)
    return path


def build_cycleflow_pipeline(
    *,
    cycled_model,
    cycleflow_factory,
    train_dataloaders: Any = None,
    val_dataloaders: Any = None,
    datamodule: Any = None,
    cycled_stage: StageConfig | dict[str, Any] | None = None,
    cycleflow_stage: StageConfig | dict[str, Any] | None = None,
    cycled_lr: float = 1e-3,
    cycleflow_lr: float = 1e-4,
    weight_decay: float = 0.0,
    train_num_samples: int = 10,
) -> list[PipelineStage]:
    """Build CycleD -> CycleFlow pipeline stages.

    Args:
        cycled_model: A ``CycleDPretrain`` model instance.
        cycleflow_factory: Callable receiving ``pretrained_flow_path`` and
            returning a ``CycleFlow`` model instance.
        train_dataloaders: Shared train loader.
        val_dataloaders: Shared validation loader.
        datamodule: Optional shared datamodule.
        cycled_stage: Stage config for CycleD pretraining.
        cycleflow_stage: Stage config for CycleFlow downstream training.
        cycled_lr: Learning rate for CycleD.
        cycleflow_lr: Learning rate for CycleFlow.
        weight_decay: Optimizer weight decay for both wrappers.
        train_num_samples: Sample count used by the CycleFlow training wrapper.
    """
    cycled_stage = _stage(cycled_stage, name="cycled", epochs=20, lr=cycled_lr)
    cycleflow_stage = _stage(cycleflow_stage, name="cycleflow", epochs=20, lr=cycleflow_lr)

    def make_cycleflow_module(trainer: Trainer, stage: StageConfig, tracker: StageCheckpoint):
        flow_path = tracker.get_artifact(cycled_stage.name, "flow")
        if flow_path is None:
            raise RuntimeError(f"Missing CycleD flow artifact from stage '{cycled_stage.name}'")
        model = cycleflow_factory(pretrained_flow_path=str(flow_path))
        return ProbForecastModule(model, lr=stage.lr, weight_decay=weight_decay, num_samples=train_num_samples)

    return [
        PipelineStage(
            config=cycled_stage,
            module=ProbForecastModule(cycled_model, lr=cycled_stage.lr, weight_decay=weight_decay, num_samples=1),
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            artifact_hook=save_cycleflow_flow_artifact,
        ),
        PipelineStage(
            config=cycleflow_stage,
            module=make_cycleflow_module,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
        ),
    ]


def _stage(config: StageConfig | dict[str, Any] | None, *, name: str, epochs: int, lr: float) -> StageConfig:
    if config is None:
        return StageConfig(name=name, epochs=epochs, lr=lr)
    if isinstance(config, StageConfig):
        return config
    return StageConfig(**config)


__all__ = ["build_cycleflow_pipeline", "save_cycleflow_flow_artifact"]
