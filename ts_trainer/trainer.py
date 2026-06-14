"""Core Trainer with multi-stage support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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

from .callbacks import GradientMonitor, SlimProgressBar, TrainingTimer
from .config import STAGE_PRESETS, SchedulerType, StageConfig, TrainerConfig


# ---------------------------------------------------------------------------
# StageCheckpoint - cross-stage checkpoint tracker
# ---------------------------------------------------------------------------


class StageCheckpoint:
    """Track best checkpoints across training stages.

    Args:
        output_dir: Directory to store checkpoint metadata.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._checkpoints: dict[str, Path] = {}

    def get_checkpoint(self, stage_name: str) -> Path | None:
        """Get the best checkpoint path for a stage."""
        return self._checkpoints.get(stage_name)

    def set_checkpoint(self, stage_name: str, path: Path | str) -> None:
        """Record the best checkpoint for a stage."""
        self._checkpoints[stage_name] = Path(path) if path else None

    @property
    def latest(self) -> Path | None:
        """Get the most recently recorded checkpoint."""
        if not self._checkpoints:
            return None
        return list(self._checkpoints.values())[-1]


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
        self._checkpoint_tracker = StageCheckpoint(Path(config.save_dir))

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

    def test(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule | None = None,
        dataloaders: Any = None,
        ckpt_path: str | None = None,
    ) -> list[dict[str, float]]:
        """Test the model."""
        pl_trainer = self._create_pl_trainer(stage=None, test_mode=True)
        return pl_trainer.test(
            model, datamodule=datamodule, dataloaders=dataloaders, ckpt_path=ckpt_path,
        )

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

    def _run_single_stage(
        self,
        model: pl.LightningModule,
        stage: StageConfig,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
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

        # Checkpoint
        stage_dir = Path(cfg.save_dir) / cfg.experiment_name
        if stage:
            stage_dir = stage_dir / stage.name
        callbacks.append(ModelCheckpoint(
            dirpath=str(stage_dir / "checkpoints"),
            monitor=cfg.checkpoint_monitor,
            mode=cfg.checkpoint_mode,
            save_top_k=cfg.save_top_k,
            save_last=cfg.save_last,
            filename=f"{stage.name if stage else 'train'}-{{epoch:02d}}",
        ))

        # LR monitor (only when logger is available)
        if cfg.logger != "none":
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Gradient monitor
        if cfg.gradient_monitor:
            callbacks.append(GradientMonitor(
                log_every_n_steps=cfg.gradient_monitor_every_n_steps,
            ))

        # SlimProgressBar is opt-in: users add it manually if they want minimal output.
        # When enable_progress_bar=True (default), PL uses its built-in progress bar.
        # When enable_progress_bar=False, PL disables all progress bars.

        # -- Logger --
        loggers = self._build_loggers(stage_dir, stage)

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
            enable_progress_bar=cfg.enable_progress_bar,
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
                name="tensorboard",
                version=stage.name if stage else None,
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


__all__ = ["Trainer", "StageCheckpoint", "SchedulerFactory"]
