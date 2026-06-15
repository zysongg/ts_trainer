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
from .trainer import PipelineResult, PipelineStage, SchedulerFactory, StageCheckpoint, Trainer
from .two_stage import build_cycleflow_pipeline, save_cycleflow_flow_artifact
from .wrappers import (
    PointForecastModule,
    ProbForecastModule,
    PointImputationModule,
    ProbImputationModule,
)

__version__ = "0.1.0"

__all__ = [
    "Trainer",
    "TrainerConfig",
    "StageConfig",
    "STAGE_PRESETS",
    "StageCheckpoint",
    "PipelineStage",
    "PipelineResult",
    "SchedulerFactory",
    "build_cycleflow_pipeline",
    "save_cycleflow_flow_artifact",
    "GradientMonitor",
    "TrainingTimer",
    "PredictionWriter",
    "SlimProgressBar",
    "SchedulerType",
    "LoggerType",
    # Forecasting
    "PointForecastModule",
    "ProbForecastModule",
    # Imputation
    "PointImputationModule",
    "ProbImputationModule",
]
