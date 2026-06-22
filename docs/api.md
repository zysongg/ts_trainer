# ts_trainer API

`ts_trainer` owns training execution. It wraps already-created models in
Lightning modules, runs training/evaluation, saves checkpoints, and supports
multi-stage handoff. It does not own model hyperparameter defaults or
experiment YAML presets.

## Public Entry Points

```python
from ts_trainer import (
    Trainer,
    TrainerConfig,
    StageConfig,
    PipelineStage,
    StageCheckpoint,
    PointForecastModule,
    ProbForecastModule,
    build_cycleflow_pipeline,
)
```

## TrainerConfig

`TrainerConfig` is the source of truth for training behavior:

```python
from ts_trainer import Trainer, TrainerConfig

config = TrainerConfig(
    max_epochs=20,
    accelerator="gpu",
    devices=1,
    early_stopping_patience=5,
    lr_scheduler="cosine",
    save_dir="output",
    run_name="my_run",
)

trainer = Trainer(config)
```

`ts_pipeline.TrainConfig` maps user-facing experiment fields into
`TrainerConfig`/`Trainer` kwargs. Do not duplicate training behavior in
`ts_model`.

## Forecasting Wrappers

Point forecasting:

```python
module = Trainer.create_module(
    model,
    task="forecast",
    mode="point",
    lr=1e-3,
    weight_decay=1e-4,
)
```

For point models, `PointForecastModule` first checks for model-owned
`train_loss(batch)` / `val_loss(batch)`. If those methods are absent, it uses
the wrapper `loss_fn`, defaulting to MSE.

Probabilistic forecasting:

```python
module = Trainer.create_module(
    model,
    task="forecast",
    mode="prob",
    lr=1e-4,
    weight_decay=0.0,
    num_samples=100,
)
```

Wrappers expect batches with:

```python
{
    "x": Tensor,        # (B, C, lookback)
    "y": Tensor,        # (B, C, horizon) or label+horizon
    "idx": Tensor,      # optional
    "x_mark": Tensor,   # optional
    "y_mark": Tensor,   # optional
}
```

## Evaluation

```python
results = trainer.evaluate(
    module,
    dataloaders=test_loader,
    ckpt_path="output/run/ckpt/best-val.ckpt",
    metrics=["MSE", "MAE"],
    task="prediction",
    mode="point",
    return_outputs=True,
    prune_outputs=False,
    save_metrics=False,
    save_outputs=False,
)
```

`return_outputs=True` returns metrics plus collected tensors:

```text
inputs, targets, preds, samples
```

Use `prune_outputs=False`, `save_metrics=False`, and `save_outputs=False` when
`ts_pipeline` owns the run directory layout.

## Common Tasks

| Task | API |
|---|---|
| Train one model | `Trainer.fit(module, train_loader, val_loader)` |
| Wrap a point forecaster | `Trainer.create_module(..., task="forecast", mode="point")` |
| Wrap a probabilistic forecaster | `Trainer.create_module(..., task="forecast", mode="prob")` |
| Evaluate metrics and collect tensors | `Trainer.evaluate(..., return_outputs=True)` |
| Run two-stage training | `Trainer.fit_pipeline(stages)` |

`ts_trainer` expects the model object to already exist. It should not import
pipeline YAML presets or duplicate `ts_model.ModelSpec` defaults.

## Multi-Stage Training

Use `PipelineStage` when stages use different Lightning modules or need
artifact handoff:

```python
from ts_trainer import Trainer, PipelineStage, StageConfig

stages = [
    PipelineStage(
        config=StageConfig(name="pretrain", epochs=10, lr=1e-3),
        module=pretrain_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    ),
    PipelineStage(
        config=StageConfig(name="forecast", epochs=20, lr=1e-4),
        module=lambda trainer, stage, tracker: build_final_module(tracker),
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    ),
]

result = Trainer(save_dir="output", run_name="two_stage").fit_pipeline(stages)
```

`result.tracker` records stage checkpoints and artifacts.

## Boundaries

| Need | Put it in |
|---|---|
| Model architecture/default params | `ts_model` |
| Training loop/checkpoints/loggers/wrappers | `ts_trainer` |
| YAML experiment, plots, compare, profile | `ts_pipeline` |
