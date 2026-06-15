"""Pydantic v2 configuration models for ts_trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SchedulerType = Literal["cosine", "step", "plateau", "onecycle", "none"]
LoggerType = Literal["tensorboard", "wandb", "csv", "none"]


# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------

class StageConfig(BaseModel):
    """Single training stage configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Stage name (e.g., 'pretrain', 'finetune')")
    epochs: int = Field(100, ge=1, description="Epochs for this stage")
    lr: float = Field(1e-3, gt=0, description="Learning rate for this stage")
    early_stopping_patience: int = Field(10, ge=0, description="Early stopping patience (0 to disable)")
    load_from_stage: str | None = Field(
        None, description="Load weights from this stage's checkpoint"
    )
    freeze_modules: list[str] = Field(
        default_factory=list,
        description="Module names to freeze (e.g., ['encoder', 'text_encoder'])",
    )
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs passed to model (e.g., use_condition=False)",
    )


# ---------------------------------------------------------------------------
# Stage presets
# ---------------------------------------------------------------------------

STAGE_PRESETS: dict[str, list[dict[str, Any]]] = {
    "single": [
        {"name": "train", "epochs": 100},
    ],
    "two_stage": [
        {"name": "pretrain", "epochs": 50, "extra_kwargs": {"use_condition": False}},
        {"name": "finetune", "epochs": 100, "load_from_stage": "pretrain"},
    ],
    "pretrain_freeze": [
        {"name": "pretrain", "epochs": 50, "extra_kwargs": {"use_condition": False}},
        {
            "name": "finetune",
            "epochs": 100,
            "load_from_stage": "pretrain",
            "freeze_modules": ["encoder"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

class TrainerConfig(BaseModel):
    """Training configuration with multi-stage support.

    Example:
        >>> cfg = TrainerConfig(max_epochs=100, lr_scheduler="cosine")
        >>> cfg = TrainerConfig.from_yaml("config.yaml")
    """

    model_config = ConfigDict(extra="forbid")

    # -- Basic training parameters --
    max_epochs: int = Field(100, ge=1, description="Maximum training epochs")
    min_epochs: int = Field(1, ge=0, description="Minimum training epochs")
    accelerator: str = Field("auto", description="Accelerator ('cpu', 'gpu', 'auto')")
    devices: int | list[int] | str = Field("auto", description="Devices to use")
    precision: int | str = Field(32, description="Training precision (16, 32, 'bf16', etc.)")
    gradient_clip_val: float | None = Field(None, ge=0, description="Gradient clipping value (None to disable)")
    accumulate_grad_batches: int = Field(1, ge=1, description="Gradient accumulation steps")

    # -- Validation & logging --
    check_val_every_n_epoch: int = Field(1, ge=1, description="Validation check interval")
    log_every_n_steps: int = Field(50, ge=1, description="Log every N steps")
    enable_progress_bar: bool = Field(False, description="Show progress bar")
    enable_model_summary: bool = Field(True, description="Print model summary")
    limit_train_batches: int | float = Field(1.0, ge=0, description="Limit training batches")
    limit_val_batches: int | float = Field(1.0, ge=0, description="Limit validation batches")
    limit_test_batches: int | float = Field(1.0, ge=0, description="Limit test batches")
    num_sanity_val_steps: int = Field(2, ge=0, description="Sanity validation steps")

    # -- Early stopping --
    early_stopping_patience: int = Field(10, ge=0, description="Early stopping patience (0 to disable)")

    # -- LR scheduler --
    lr_scheduler: SchedulerType = Field("cosine", description="LR scheduler type")
    lr_scheduler_params: dict[str, Any] = Field(
        default_factory=dict, description="Extra scheduler params"
    )

    # -- Checkpoint --
    checkpoint_monitor: str = Field("val_loss", description="Metric to monitor for checkpointing")
    checkpoint_mode: Literal["min", "max"] = Field("min", description="Checkpoint mode")
    save_top_k: int = Field(1, ge=0, description="Save top-k checkpoints (0 to disable)")
    save_last: bool = Field(True, description="Save last checkpoint")

    # -- Paths --
    save_dir: str = Field("./experiments", description="Root directory for timestamped experiment runs")
    experiment_name: str = Field("ts_experiment", description="Experiment name suffix")
    dataset_name: str | None = Field(None, description="Dataset name used in automatic run folder names")
    model_name: str | None = Field(None, description="Model name used in automatic run folder names")
    run_name: str | None = Field(
        None,
        description="Run folder name. If None, uses YYYYMMDD_HHMMSS_<dataset_name>_<model_name> when available.",
    )

    # -- Logger --
    logger: LoggerType = Field("tensorboard", description="Logger type")
    logger_params: dict[str, Any] = Field(default_factory=dict, description="Extra logger params")

    # -- Callbacks --
    gradient_monitor: bool = Field(False, description="Enable gradient monitoring")
    gradient_monitor_every_n_steps: int = Field(50, ge=1, description="Gradient monitor interval")

    # -- Multi-stage --
    stages: list[StageConfig] | None = Field(
        None, description="Multi-stage config (None = single stage using max_epochs)"
    )

    # -- Profiling --
    profiling: list[str] | None = Field(None, description="Profiling options")

    @model_validator(mode="after")
    def _resolve_stages(self) -> TrainerConfig:
        """Auto-create single stage if stages is None."""
        if self.stages is None:
            self.stages = [StageConfig(
                name="train",
                epochs=self.max_epochs,
                early_stopping_patience=self.early_stopping_patience,
            )]
        return self

    # -- YAML serialization --

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainerConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="json", exclude_none=True),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )


__all__ = [
    "StageConfig",
    "STAGE_PRESETS",
    "TrainerConfig",
    "SchedulerType",
    "LoggerType",
]
