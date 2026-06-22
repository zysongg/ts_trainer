# ts_trainer

标准化训练框架，基于 PyTorch Lightning，支持多阶段训练和可插拔日志系统。

## 安装

```bash
cd TSLib-tool/ts_trainer
pip install -e .
```

## 功能特性

- 基于 PyTorch Lightning 的训练封装
- 多阶段训练（预训练→微调）
- Pydantic v2 类型安全配置
- 可插拔日志系统（TensorBoard / WandB / CSV）
- 4 个实用回调：梯度监控、训练计时、预测保存、精简进度条
- LR 调度器工厂（cosine / step / plateau / onecycle）
- YAML 配置导入/导出

## 快速开始

### 基本用法

```python
from ts_trainer import Trainer

trainer = Trainer(
    max_epochs=100,
    early_stopping_patience=10,
    gradient_clip_val=1.0,
    logger="tensorboard",
    save_dir="./logs",
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model, dataloaders=test_loader)
```

### 多阶段训练

```python
# 使用预设
trainer.fit(model, ..., stages="two_stage")

# 自定义阶段
from ts_trainer import StageConfig

stages = [
    StageConfig(name="warmup", epochs=10, lr=1e-4),
    StageConfig(name="main", epochs=100, lr=1e-3, load_from_stage="warmup"),
]
trainer.fit(model, ..., stages=stages)
```

### 配置驱动

```python
from ts_trainer import TrainerConfig

config = TrainerConfig.from_yaml("config.yaml")
trainer = Trainer(config)
trainer.fit(model, datamodule=dm)
```

### 模型中使用 SchedulerFactory

```python
import lightning as pl
from ts_trainer import SchedulerFactory

class MyModel(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return SchedulerFactory.create(optimizer, "cosine", self.trainer.max_epochs)
    
    def set_stage(self, stage_name: str):
        """多阶段时由 Trainer 调用（可选）"""
        self._current_stage = stage_name
```

## 配置参数

### TrainerConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_epochs` | int | 100 | 最大训练轮数 |
| `accelerator` | str | "auto" | 加速器 |
| `precision` | int/str | 32 | 训练精度 |
| `gradient_clip_val` | float | None | 梯度裁剪值 |
| `early_stopping_patience` | int | 10 | 早停耐心值 |
| `lr_scheduler` | str | "cosine" | LR 调度器 |
| `checkpoint_monitor` | str | "val_loss" | 监控指标 |
| `save_dir` | str | "./logs" | 输出目录 |
| `logger` | str | "tensorboard" | 日志系统 |
| `gradient_monitor` | bool | False | 梯度监控 |

### LR 调度器

| 类型 | 说明 |
|------|------|
| `"cosine"` | CosineAnnealingLR |
| `"step"` | StepLR |
| `"plateau"` | ReduceLROnPlateau |
| `"onecycle"` | OneCycleLR |
| `"none"` | 不使用调度器 |

### 日志系统

| 类型 | 说明 |
|------|------|
| `"tensorboard"` | TensorBoard |
| `"wandb"` | Weights & Biases |
| `"csv"` | CSV 文件 |
| `"none"` | 不记录日志 |

### 阶段预设

| 预设名 | 说明 |
|--------|------|
| `"single"` | 单阶段训练 |
| `"two_stage"` | 预训练 + 微调 |
| `"pretrain_freeze"` | 预训练 + 冻结 encoder 微调 |

## 回调

| 回调 | 功能 |
|------|------|
| `GradientMonitor` | 每 N 步记录梯度范数 |
| `TrainingTimer` | 跟踪训练时间 |
| `PredictionWriter` | 保存预测结果为 .npz |
| `SlimProgressBar` | Slurm 精简进度条 |

## License

MIT

## API 文档

完整 API 边界和示例见 [docs/api.md](docs/api.md)。
