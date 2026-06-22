"""Core Trainer with multi-stage support."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from .callbacks import GradientMonitor, PredictionWriter, SlimProgressBar, TrainingTimer
from .config import STAGE_PRESETS, SchedulerType, StageConfig, TrainerConfig
from .wrappers import (
    PointForecastModule,
    ProbForecastModule,
    PointImputationModule,
    ProbImputationModule,
)


DEFAULT_EVAL_METRICS: dict[tuple[str, str], list[str]] = {
    ("prediction", "point"): ["MSE", "MAE", "RMSE", "NRMSE"],
    ("prediction", "probabilistic"): ["CRPS", "CRPS_sum", "MSE_median", "MAE_median"],
    ("imputation", "point"): ["MSE", "MAE", "RMSE", "MRE"],
    ("imputation", "probabilistic"): ["CRPS", "PICP", "QICE", "IntervalWidth"],
    ("generation", "default"): ["MDD", "ACD", "DS", "PS"],
    ("anomaly", "default"): ["F1", "PA_F1", "AUC_ROC", "AUC_PR"],
    ("classification", "default"): ["Accuracy", "Precision", "Recall", "F1"],
}


ModuleFactory = Callable[["Trainer", StageConfig, "StageCheckpoint"], pl.LightningModule]
ArtifactHook = Callable[["Trainer", StageConfig, pl.LightningModule, "StageCheckpoint"], None]


@dataclass
class PipelineStage:
    """A trainable stage that may use a different LightningModule."""

    config: StageConfig | dict[str, Any]
    module: pl.LightningModule | ModuleFactory
    train_dataloaders: Any = None
    val_dataloaders: Any = None
    datamodule: pl.LightningDataModule | None = None
    ckpt_path: str | None = None
    artifact_hook: ArtifactHook | None = None


# ---------------------------------------------------------------------------
# StageCheckpoint - cross-stage checkpoint tracker
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Structured result returned by :meth:`Trainer.fit_pipeline`.

    Attributes:
        stages: Ordered list of stage names that were executed.
        modules: Mapping from stage name to the trained LightningModule.
        final_module: The LightningModule from the last pipeline stage.
        tracker: The :class:`StageCheckpoint` containing best checkpoints and artifacts.
    """

    stages: list[str]
    modules: dict[str, pl.LightningModule]
    final_module: pl.LightningModule
    tracker: StageCheckpoint


class StageCheckpoint:
    """Track best checkpoints across training stages.

    Args:
        output_dir: Directory to store checkpoint metadata.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._checkpoints: dict[str, Path] = {}
        self._artifacts: dict[str, dict[str, Path]] = {}

    def get_checkpoint(self, stage_name: str) -> Path | None:
        """Get the best checkpoint path for a stage."""
        return self._checkpoints.get(stage_name)

    def set_checkpoint(self, stage_name: str, path: Path | str) -> None:
        """Record the best checkpoint for a stage."""
        self._checkpoints[stage_name] = Path(path) if path else None

    def get_artifact(self, stage_name: str, artifact_name: str) -> Path | None:
        """Get a named artifact path produced by a stage."""
        return self._artifacts.get(stage_name, {}).get(artifact_name)

    def set_artifact(self, stage_name: str, artifact_name: str, path: Path | str) -> None:
        """Record a named artifact path produced by a stage."""
        self._artifacts.setdefault(stage_name, {})[artifact_name] = Path(path)

    @property
    def latest(self) -> Path | None:
        """Get the most recently recorded checkpoint."""
        if not self._checkpoints:
            return None
        return list(self._checkpoints.values())[-1]

    def save_manifest(self, path: Path | str | None = None) -> Path:
        """Persist checkpoints and artifacts to ``pipeline_manifest.json``.

        Paths are stored relative to ``self.output_dir`` when possible so the
        experiment directory can be moved freely.
        """
        manifest_path = Path(path) if path else self.output_dir / "pipeline_manifest.json"

        def _rel(p: Path | None) -> str | None:
            if p is None:
                return None
            try:
                return str(p.relative_to(self.output_dir))
            except ValueError:
                return str(p)

        data: dict[str, Any] = {
            "checkpoints": {k: _rel(v) for k, v in self._checkpoints.items()},
            "artifacts": {
                stage: {name: _rel(p) for name, p in arts.items()}
                for stage, arts in self._artifacts.items()
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return manifest_path

    @classmethod
    def load_manifest(cls, output_dir: Path | str, path: Path | str | None = None) -> "StageCheckpoint":
        """Reconstruct a :class:`StageCheckpoint` from a previously saved manifest."""
        output_dir = Path(output_dir)
        manifest_path = Path(path) if path else output_dir / "pipeline_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracker = cls(output_dir)
        for stage_name, rel in data.get("checkpoints", {}).items():
            if rel is not None:
                tracker.set_checkpoint(stage_name, output_dir / rel)
        for stage_name, arts in data.get("artifacts", {}).items():
            for name, rel in arts.items():
                if rel is not None:
                    tracker.set_artifact(stage_name, name, output_dir / rel)
        return tracker


# ---------------------------------------------------------------------------
# SchedulerFactory
# ---------------------------------------------------------------------------


class SchedulerFactory:
    """LR scheduler factory for use in ``configure_optimizers()``.

    Example::

        class MyModel(pl.LightningModule):
            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
                return SchedulerFactory.create(optimizer, "cosine", self.trainer.max_epochs)
    """

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer,
        scheduler_type: SchedulerType,
        max_epochs: int,
        monitor: str = "val_loss",
        **kwargs: Any,
    ) -> dict | torch.optim.Optimizer:
        """Create a scheduler configuration dict compatible with PL's ``configure_optimizers()``.

        Returns:
            Either a dict ``{"optimizer": ..., "lr_scheduler": ...}`` or just the optimizer
            if ``scheduler_type == "none"``.
        """
        if scheduler_type == "none":
            return optimizer

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max_epochs, **kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if scheduler_type == "step":
            step_size = kwargs.pop("step_size", max(1, max_epochs // 3))
            gamma = kwargs.pop("gamma", 0.5)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if scheduler_type == "plateau":
            factor = kwargs.pop("factor", 0.3)
            patience = kwargs.pop("patience", 3)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                },
            }

        if scheduler_type == "onecycle":
            max_lr = kwargs.pop("max_lr", optimizer.param_groups[0]["lr"])
            scheduler = OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=max_epochs, **kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Unified training entry point with multi-stage support.

    Args:
        config: A :class:`TrainerConfig` instance.
        **kwargs: Override any config field.

    Example::

        trainer = Trainer(max_epochs=100, lr_scheduler="cosine")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = TrainerConfig(**kwargs)
        else:
            # Apply kwargs overrides
            data = config.model_dump()
            data.update(kwargs)
            config = TrainerConfig(**data)
        self.config = config

        # Internal state
        self._best_checkpoint_path: str | None = None
        self._last_pl_trainer: pl.Trainer | None = None
        self._experiment_dir = self._build_experiment_dir(config)
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_tracker = StageCheckpoint(self._experiment_dir)
        config.to_yaml(self._experiment_dir / "config.yaml")

    # -- Public API ----------------------------------------------------------

    def fit(
        self,
        model: pl.LightningModule,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = None,
        stages: str | list[StageConfig | dict] | None = None,
    ) -> None:
        """Train the model.

        Args:
            model: A LightningModule.
            train_dataloaders: Training data.
            val_dataloaders: Validation data.
            datamodule: Optional LightningDataModule.
            ckpt_path: Resume from checkpoint.
            stages: Stage config. Can be:
                - ``None``: use ``self.config.stages``
                - ``"single"`` / ``"two_stage"``: use preset
                - list of StageConfig or dicts
        """
        stage_list = self._resolve_stages(stages)

        if len(stage_list) == 1:
            self._run_single_stage(
                model, stage_list[0],
                train_dataloaders, val_dataloaders, datamodule, ckpt_path,
            )
        else:
            self._run_multi_stage(
                model, stage_list,
                train_dataloaders, val_dataloaders, datamodule,
            )

    def fit_pipeline(self, stages: Sequence[PipelineStage | dict[str, Any]]) -> PipelineResult:
        """Train a pipeline whose stages may use different LightningModules.

        This is intended for workflows like CycleD -> CycleFlow where the
        first stage produces an artifact consumed by the second stage, rather
        than a checkpoint that can be loaded into the same module class.

        Returns:
            A :class:`PipelineResult` with stage names, trained modules, the
            final module, and the checkpoint tracker.
        """
        tracker = self._checkpoint_tracker
        executed_stages: list[str] = []
        stage_modules: dict[str, pl.LightningModule] = {}
        last_module: pl.LightningModule | None = None

        for i, spec in enumerate(stages):
            stage_spec = self._normalize_pipeline_stage(spec)
            stage = self._coerce_stage_config(stage_spec.config)
            print(f"\n{'='*60}")
            print(f"  Pipeline stage {i+1}/{len(stages)}: {stage.name} ({stage.epochs} epochs)")
            print(f"{'='*60}\n")

            module = self._build_pipeline_module(stage_spec.module, stage, tracker)
            if hasattr(module, "set_stage"):
                module.set_stage(stage.name)
            if hasattr(module, "stage_lr"):
                module.stage_lr = stage.lr

            self._run_single_stage(
                module,
                stage,
                stage_spec.train_dataloaders,
                stage_spec.val_dataloaders,
                stage_spec.datamodule,
                stage_spec.ckpt_path,
            )

            if stage_spec.artifact_hook is not None:
                stage_spec.artifact_hook(self, stage, module, tracker)

            executed_stages.append(stage.name)
            stage_modules[stage.name] = module
            last_module = module

        tracker.save_manifest()

        assert last_module is not None, "Pipeline must have at least one stage"
        return PipelineResult(
            stages=executed_stages,
            modules=stage_modules,
            final_module=last_module,
            tracker=tracker,
        )

    def test(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule | None = None,
        dataloaders: Any = None,
        ckpt_path: str | None = None,
        save_outputs: bool = True,
    ) -> list[dict[str, float]]:
        """Test the model."""
        pl_trainer = self._create_pl_trainer(stage=None, test_mode=True, save_outputs=save_outputs)
        results = pl_trainer.test(
            model, datamodule=datamodule, dataloaders=dataloaders, ckpt_path=ckpt_path,
        )
        self._last_pl_trainer = pl_trainer
        return results

    def predict(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule | None = None,
        dataloaders: Any = None,
        ckpt_path: str | None = None,
    ) -> Any:
        """Run prediction."""
        pl_trainer = self._create_pl_trainer(stage=None, test_mode=True)
        return pl_trainer.predict(
            model, datamodule=datamodule, dataloaders=dataloaders, ckpt_path=ckpt_path,
        )

    @property
    def best_checkpoint_path(self) -> str | None:
        """Path to the best checkpoint from the last training run."""
        return self._best_checkpoint_path

    @property
    def log_dir(self) -> str | None:
        """Log directory of the last training run."""
        if self._last_pl_trainer and self._last_pl_trainer.loggers:
            return self._last_pl_trainer.loggers[0].log_dir
        return None

    @property
    def experiment_dir(self) -> str:
        """Root directory for this trainer run."""
        return str(self._experiment_dir)

    # -- Static Factory Methods ----------------------------------------------

    @staticmethod
    def create_module(
        model: torch.nn.Module,
        task: str = "forecast",
        mode: str = "point",
        **kwargs: Any,
    ) -> pl.LightningModule:
        """Create a LightningModule wrapper for the given model.

        Args:
            model: A ts_model model instance.
            task: Task type ("forecast", "imputation", "generation").
            mode: Model mode ("point", "prob" for forecasting).
            **kwargs: Additional arguments passed to the wrapper.

        Returns:
            A LightningModule wrapper.

        Example::

            model = create_model("DLinear", task="forecasting", ...)
            module = Trainer.create_module(model, task="forecast")
        """
        if task == "forecast":
            if mode == "point":
                return PointForecastModule(model, **kwargs)
            elif mode == "prob":
                return ProbForecastModule(model, **kwargs)
            else:
                raise ValueError(f"Unknown forecast mode: {mode}. Use 'point' or 'prob'.")
        elif task == "imputation":
            if mode == "point":
                return PointImputationModule(model, **kwargs)
            elif mode == "prob":
                return ProbImputationModule(model, **kwargs)
            else:
                raise ValueError(f"Unknown imputation mode: {mode}. Use 'point' or 'prob'.")
        else:
            raise ValueError(f"Unsupported task: {task}. Use 'forecast' or 'imputation'.")

    def evaluate(
        self,
        model: pl.LightningModule,
        dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = None,
        metrics: list[str] | None = None,
        task: str = "prediction",
        mode: str = "deterministic",
        prune_outputs: bool = True,
        return_outputs: bool = False,
        save_metrics: bool = True,
        save_outputs: bool = True,
    ) -> dict[str, float]:
        """End-to-end evaluation: test + collect predictions + compute metrics.

        Args:
            model: A LightningModule.
            dataloaders: Test dataloader(s).
            datamodule: Optional LightningDataModule.
            ckpt_path: Checkpoint path to load.
            metrics: List of metric names (e.g., ["MSE", "MAE"]).
                If None, uses the default 4 metrics for the task/mode.
            task: Task type for metrics ("prediction", "imputation", etc.).
            mode: Mode for metrics ("deterministic", "probabilistic").
            prune_outputs: If True, prune experiment directory to keep only
                standard artifacts. Set False when using ts_pipeline which
                manages its own directory structure.
            return_outputs: If True, return dict with both metrics and collected
                outputs (inputs, targets, predictions, samples). If False,
                return only metrics dict.
            save_metrics: If True, save metrics in the trainer experiment root.
            save_outputs: If True, save collected test outputs in the trainer
                experiment root.

        Returns:
            Dictionary with test_loss and requested metrics. If return_outputs=True,
            also includes inputs, targets, predictions, and samples tensors.

        Example::

            trainer = Trainer(max_epochs=100)
            trainer.fit(module, train_dataloaders=train_loader)
            results = trainer.evaluate(module, test_loader, metrics=["MSE", "MAE"])
            # {"test_loss": 0.042, "MSE": 0.042, "MAE": 0.159}
        """
        # 1. Run test
        test_results = self.test(
            model,
            dataloaders=dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            save_outputs=save_outputs,
        )
        results = dict(test_results[0]) if test_results else {}

        metric_task = self._normalize_metric_task(task)
        metric_mode = self._normalize_metric_mode(mode)
        selected_metrics = metrics or DEFAULT_EVAL_METRICS.get((metric_task, metric_mode))
        if not selected_metrics:
            if save_metrics:
                self._save_metrics(results, prefix="evaluate")
            if prune_outputs:
                self._prune_experiment_outputs()
            return self._maybe_return_outputs(results, {}, return_outputs)

        # 2. Reuse outputs collected during test_step. Evaluate intentionally
        # does not run predict(), because most inference workflows only use test.
        collected = self._collect_test_outputs_from_last_trainer()
        targets = collected.get("targets")
        preds = collected.get("preds")
        samples = collected.get("samples")
        if targets is None:
            print("Warning: Prediction outputs do not include targets, skipping metric computation.")
            if save_metrics:
                self._save_metrics(results, prefix="evaluate")
            if prune_outputs:
                self._prune_experiment_outputs()
            return self._maybe_return_outputs(results, collected, return_outputs)

        # 3. Compute metrics on wrapper outputs. Forecast point metrics use
        # preds, probabilistic metrics use samples.
        try:
            from ts_metric import MetricCalculator

            calc = MetricCalculator(task=metric_task, mode=metric_mode, metrics=selected_metrics)
            if metric_mode == "probabilistic":
                if samples is None:
                    print("Warning: Prediction outputs do not include samples, skipping probabilistic metrics.")
                    if save_metrics:
                        self._save_metrics(results, prefix="evaluate")
                    if prune_outputs:
                        self._prune_experiment_outputs()
                    return self._maybe_return_outputs(results, collected, return_outputs)
                metric_results = calc.compute(targets, samples)
            else:
                metric_input = preds if preds is not None else samples
                if metric_input is None:
                    print("Warning: Prediction outputs do not include predictions, skipping metric computation.")
                    if save_metrics:
                        self._save_metrics(results, prefix="evaluate")
                    if prune_outputs:
                        self._prune_experiment_outputs()
                    return self._maybe_return_outputs(results, collected, return_outputs)
                metric_results = calc.compute(targets, metric_input)
            results.update({k: v.item() if hasattr(v, 'item') else float(v) for k, v in metric_results.items()})
        except ImportError:
            print("Warning: ts_metric not installed, skipping metric computation.")
        except Exception as e:
            print(f"Warning: Failed to compute metrics: {e}")

        if save_metrics:
            self._save_metrics(results, prefix="evaluate")
        if prune_outputs:
            self._prune_experiment_outputs()
        return self._maybe_return_outputs(results, collected, return_outputs)

    def _maybe_return_outputs(
        self,
        results: dict[str, Any],
        collected: dict[str, Any],
        return_outputs: bool,
    ) -> dict[str, Any]:
        """Optionally merge collected outputs into results dict."""
        if not return_outputs:
            return results
        output = dict(results)
        for key in ("inputs", "targets", "preds", "samples"):
            if key in collected and collected[key] is not None:
                output[key] = collected[key]
        return output

    def _collect_predictions(
        self,
        model: pl.LightningModule,
        dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Collect predictions from Lightning predict outputs."""
        pred_outputs = self.predict(
            model,
            datamodule=datamodule,
            dataloaders=dataloaders,
            ckpt_path=ckpt_path,
        )
        return self._collect_prediction_outputs(pred_outputs)

    def _collect_test_outputs_from_last_trainer(self) -> dict[str, torch.Tensor]:
        """Collect saved test outputs from the PredictionWriter callback."""
        if not self._last_pl_trainer:
            return {}
        for callback in self._last_pl_trainer.callbacks:
            if isinstance(callback, PredictionWriter):
                outputs = callback.saved_outputs.get("test", {})
                collected: dict[str, torch.Tensor] = {}
                if "inputs" in outputs:
                    collected["inputs"] = self._to_cpu_tensor(outputs["inputs"])
                if "predictions" in outputs:
                    collected["preds"] = self._to_cpu_tensor(outputs["predictions"])
                if "targets" in outputs:
                    collected["targets"] = self._to_cpu_tensor(outputs["targets"])
                if "samples" in outputs:
                    collected["samples"] = self._to_cpu_tensor(outputs["samples"])
                return collected
        return {}

    def _collect_prediction_outputs(self, outputs: Any) -> dict[str, torch.Tensor]:
        """Normalize nested Lightning predict outputs into concatenated tensors."""
        buckets: dict[str, list[torch.Tensor]] = {
            "inputs": [],
            "preds": [],
            "targets": [],
            "samples": [],
            "scores": [],
        }

        for item in self._flatten_prediction_outputs(outputs):
            if isinstance(item, dict):
                if "inputs" in item and item["inputs"] is not None:
                    buckets["inputs"].append(self._to_cpu_tensor(item["inputs"]))
                preds = item.get("preds", item.get("pred"))
                if preds is not None:
                    buckets["preds"].append(self._to_cpu_tensor(preds))
                targets = item.get("targets", item.get("y"))
                if targets is not None:
                    buckets["targets"].append(self._to_cpu_tensor(targets))
                if "samples" in item and item["samples"] is not None:
                    samples = self._to_cpu_tensor(item["samples"])
                    buckets["samples"].append(samples)
                    if preds is None and samples.ndim == 4:
                        buckets["preds"].append(torch.median(samples, dim=1)[0])
                scores = item.get("scores")
                if scores is not None:
                    buckets["scores"].append(self._to_cpu_tensor(scores))
            elif isinstance(item, torch.Tensor):
                buckets["preds"].append(self._to_cpu_tensor(item))

        return {key: torch.cat(values, dim=0) for key, values in buckets.items() if values}

    def _flatten_prediction_outputs(self, outputs: Any):
        """Yield batch-level outputs from Lightning predict's nested structures."""
        if isinstance(outputs, dict) or isinstance(outputs, torch.Tensor):
            yield outputs
            return
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                yield from self._flatten_prediction_outputs(item)
            return
        yield outputs

    @staticmethod
    def _to_cpu_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value)

    @staticmethod
    def _normalize_metric_task(task: str) -> str:
        aliases = {
            "forecast": "prediction",
            "forecasting": "prediction",
            "predict": "prediction",
        }
        return aliases.get(task.lower(), task.lower())

    @staticmethod
    def _normalize_metric_mode(mode: str) -> str:
        aliases = {
            "deterministic": "point",
            "point": "point",
            "prob": "probabilistic",
            "probability": "probabilistic",
            "probabilistic": "probabilistic",
        }
        return aliases.get(mode.lower(), mode.lower())

    @staticmethod
    def _build_experiment_dir(config: TrainerConfig) -> Path:
        run_name = config.run_name
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = [
                Trainer._sanitize_path_name(part)
                for part in (config.dataset_name, config.model_name)
                if part
            ]
            suffix = "_".join(part for part in name_parts if part)
            if not suffix:
                suffix = Trainer._sanitize_path_name(config.experiment_name)
            run_name = f"{timestamp}_{suffix}" if suffix else timestamp
        return Path(config.save_dir) / run_name

    @staticmethod
    def _sanitize_path_name(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
        return cleaned.strip("_")

    def _save_metrics(self, metrics: dict[str, Any], prefix: str) -> None:
        """Save scalar metrics in the experiment root as JSON."""
        if not metrics:
            return
        scalar_metrics = {key: self._as_metric_scalar(value) for key, value in metrics.items()}
        json_path = self._experiment_dir / f"{prefix}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scalar_metrics, f, indent=2, sort_keys=True)

    @staticmethod
    def _as_metric_scalar(value: Any) -> float | int | str | bool | None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return str(value.tolist())
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)

    def _prune_experiment_outputs(self) -> None:
        """Keep the final experiment folder focused on the public artifacts."""
        allowed_names = {
            "ckpt",
            "artifacts",
            "evaluate_metrics.json",
            "test_results.npz",
            "tensorboard",
            "config.yaml",
            "pipeline_manifest.json",
        }
        if not self._experiment_dir.exists():
            return
        for path in self._experiment_dir.iterdir():
            if path.name in allowed_names:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        self._prune_checkpoint_dir()

    def _prune_checkpoint_dir(self) -> None:
        ckpt_dir = self._experiment_dir / "ckpt"
        if not ckpt_dir.exists():
            return
        allowed = {"last.ckpt", "best-val.ckpt"}
        for path in ckpt_dir.iterdir():
            if path.is_file() and path.name not in allowed:
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    # -- Internal ------------------------------------------------------------

    def _resolve_stages(
        self,
        stages: str | list[StageConfig | dict] | None,
    ) -> list[StageConfig]:
        if stages is None:
            return self.config.stages or [StageConfig(name="train", epochs=self.config.max_epochs)]

        if isinstance(stages, str):
            if stages not in STAGE_PRESETS:
                raise ValueError(f"Unknown preset: {stages}. Available: {list(STAGE_PRESETS.keys())}")
            return [StageConfig(**s) for s in STAGE_PRESETS[stages]]

        result = []
        for s in stages:
            if isinstance(s, dict):
                result.append(StageConfig(**s))
            else:
                result.append(s)
        return result

    def _normalize_pipeline_stage(self, spec: PipelineStage | dict[str, Any]) -> PipelineStage:
        if isinstance(spec, PipelineStage):
            return spec
        return PipelineStage(**spec)

    def _coerce_stage_config(self, config: StageConfig | dict[str, Any]) -> StageConfig:
        if isinstance(config, StageConfig):
            return config
        return StageConfig(**config)

    def _build_pipeline_module(
        self,
        module_or_factory: pl.LightningModule | ModuleFactory,
        stage: StageConfig,
        tracker: StageCheckpoint,
    ) -> pl.LightningModule:
        if isinstance(module_or_factory, pl.LightningModule):
            return module_or_factory
        module = module_or_factory(self, stage, tracker)
        if not isinstance(module, pl.LightningModule):
            raise TypeError("PipelineStage module factory must return a LightningModule")
        return module

    def _run_single_stage(
        self,
        model: pl.LightningModule,
        stage: StageConfig,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        # Set max_epochs on model for scheduler configuration
        model._max_epochs = stage.epochs if stage else self.config.max_epochs

        pl_trainer = self._create_pl_trainer(stage=stage)
        pl_trainer.fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        self._last_pl_trainer = pl_trainer

        # Record best checkpoint
        if pl_trainer.checkpoint_callback:
            best = pl_trainer.checkpoint_callback.best_model_path
            if best:
                self._best_checkpoint_path = best
                self._checkpoint_tracker.set_checkpoint(stage.name, best)

    def _run_multi_stage(
        self,
        model: pl.LightningModule,
        stages: list[StageConfig],
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
    ) -> None:
        tracker = self._checkpoint_tracker

        for i, stage in enumerate(stages):
            print(f"\n{'='*60}")
            print(f"  Stage {i+1}/{len(stages)}: {stage.name} ({stage.epochs} epochs)")
            print(f"{'='*60}\n")

            # Load checkpoint from previous stage
            ckpt_path = None
            if stage.load_from_stage:
                ckpt_path_obj = tracker.get_checkpoint(stage.load_from_stage)
                if ckpt_path_obj and ckpt_path_obj.exists():
                    ckpt_path = str(ckpt_path_obj)
                    print(f"  Loading checkpoint from stage '{stage.load_from_stage}': {ckpt_path}")

            # Notify model of current stage
            if hasattr(model, "set_stage"):
                model.set_stage(stage.name)

            # Handle module freezing
            frozen_params = []
            if stage.freeze_modules:
                frozen_params = self._freeze_modules(model, stage.freeze_modules)

            # Pass stage lr to model if supported
            if hasattr(model, "stage_lr"):
                model.stage_lr = stage.lr

            self._run_single_stage(
                model, stage,
                train_dataloaders, val_dataloaders, datamodule,
                ckpt_path,
            )

            # Unfreeze
            for param in frozen_params:
                param.requires_grad = True

    def _freeze_modules(
        self, model: pl.LightningModule, module_names: list[str]
    ) -> list[torch.nn.Parameter]:
        """Freeze named modules and return list of frozen parameters."""
        frozen = []
        for name, module in model.named_modules():
            if name in module_names or any(name.startswith(f"{n}.") for n in module_names):
                for param in module.parameters():
                    param.requires_grad = False
                    frozen.append(param)
                print(f"  Frozen module: {name}")
        return frozen

    def _create_pl_trainer(
        self,
        stage: StageConfig | None = None,
        test_mode: bool = False,
        save_outputs: bool = True,
    ) -> pl.Trainer:
        cfg = self.config
        max_epochs = stage.epochs if stage else cfg.max_epochs

        # -- Callbacks --
        callbacks: list[Callback] = [TrainingTimer()]

        # Early stopping
        patience = stage.early_stopping_patience if stage else cfg.early_stopping_patience
        if patience > 0:
            callbacks.append(EarlyStopping(
                monitor=cfg.checkpoint_monitor,
                mode=cfg.checkpoint_mode,
                patience=patience,
                verbose=True,
            ))

        stage_dir = self._experiment_dir

        # Checkpoint
        if not test_mode:
            ckpt_dir = self._experiment_dir / "ckpt"
            if stage and stage.name != "train":
                ckpt_dir = ckpt_dir / stage.name
            callbacks.append(ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor=cfg.checkpoint_monitor,
                mode=cfg.checkpoint_mode,
                save_top_k=cfg.save_top_k,
                save_last=cfg.save_last,
                filename="best-val",
            ))

        # LR monitor (only when logger is available)
        if cfg.logger != "none" and not test_mode:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Gradient monitor
        if cfg.gradient_monitor:
            callbacks.append(GradientMonitor(
                log_every_n_steps=cfg.gradient_monitor_every_n_steps,
            ))

        # Add SlimProgressBar by default for epoch-level output
        callbacks.append(SlimProgressBar())

        # Collect structured test/predict outputs. Pipeline callers can keep the
        # in-memory collection while disabling root-level files.
        callbacks.append(PredictionWriter(output_dir=str(self._experiment_dir), write_files=save_outputs))

        # -- Logger --
        loggers = self._build_loggers(stage_dir, stage) if not test_mode else False

        # -- Gradient clipping --
        grad_clip = cfg.gradient_clip_val
        if grad_clip is not None:
            # Disable for manual optimization (GAN etc.)
            # PL handles this automatically but we pass it explicitly
            pass

        return pl.Trainer(
            max_epochs=max_epochs,
            min_epochs=cfg.min_epochs,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
            gradient_clip_val=grad_clip,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            check_val_every_n_epoch=cfg.check_val_every_n_epoch,
            log_every_n_steps=cfg.log_every_n_steps,
            enable_progress_bar=True,  # Always True since we use SlimProgressBar
            enable_model_summary=cfg.enable_model_summary,
            limit_train_batches=cfg.limit_train_batches,
            limit_val_batches=cfg.limit_val_batches,
            limit_test_batches=cfg.limit_test_batches,
            num_sanity_val_steps=cfg.num_sanity_val_steps,
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=str(stage_dir),
        )

    def _build_loggers(
        self,
        stage_dir: Path,
        stage: StageConfig | None,
    ) -> list[Any]:
        cfg = self.config
        loggers: list[Any] = []

        if cfg.logger == "tensorboard":
            loggers.append(TensorBoardLogger(
                save_dir=str(stage_dir),
                name="",
                version="tensorboard",
                **cfg.logger_params,
            ))
        elif cfg.logger == "csv":
            loggers.append(CSVLogger(
                save_dir=str(stage_dir),
                name="csv_logs",
                version=stage.name if stage else None,
                **cfg.logger_params,
            ))
        elif cfg.logger == "wandb":
            try:
                from lightning.pytorch.loggers import WandbLogger
                loggers.append(WandbLogger(
                    project=cfg.experiment_name,
                    name=stage.name if stage else None,
                    save_dir=str(stage_dir),
                    **cfg.logger_params,
                ))
            except ImportError:
                print("Warning: wandb not installed, falling back to TensorBoard")
                loggers.append(TensorBoardLogger(save_dir=str(stage_dir), name="tensorboard"))
        # "none" -> empty list

        if not loggers:
            loggers = False  # type: ignore[assignment]

        return loggers


__all__ = ["Trainer", "PipelineResult", "StageCheckpoint", "SchedulerFactory"]
