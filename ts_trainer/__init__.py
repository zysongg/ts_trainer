"""ts_trainer: Standardized training framework for time series tasks."""

from .callbacks import (
    GradientMonitor,
    PredictionWriter,
    SlimProgressBar,
    TrainingTimer,
)
from .config import (
    LoggerType,
    SchedulerType,
    StageConfig,
    STAGE_PRESETS,
    TrainerConfig,
)
from .trainer import SchedulerFactory, StageCheckpoint, Trainer

__version__ = "0.1.0"

__all__ = [
    "Trainer",
    "TrainerConfig",
    "StageConfig",
    "STAGE_PRESETS",
    "StageCheckpoint",
    "SchedulerFactory",
    "GradientMonitor",
    "TrainingTimer",
    "PredictionWriter",
    "SlimProgressBar",
    "SchedulerType",
    "LoggerType",
]
