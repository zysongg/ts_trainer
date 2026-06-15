# Forecasting Wrapper Refactor Design

## 目标

统一 `ts_trainer` 中预测任务的 point 和 probabilistic wrapper 输出接口，让训练、测试、预测保存、`Trainer.evaluate()` 和 `ts_metric` 指标计算走同一套数据契约。

本设计只覆盖本轮指定范围：

- Point: 1 / 3 / 4
- Probabilistic: 1 / 2 / 3 / 4 / 5
- Trainer: 1 / 2 / 3

暂不处理 point 的严格 target shape 校验和 RevIN `num_features` fallback，避免本轮改动范围扩散。

## 当前问题

### PointForecastModule

当前 `PointForecastModule.test_step()` 返回：

```python
{"pred": pred, "y": y}
```

但 `PredictionWriter` 主要识别：

```python
outputs["preds"]
outputs["y"]
```

这会导致 point 测试输出和保存逻辑不一致。

另外，`PointForecastModule.forward()` 每次调用都会执行 `inspect.signature(self.model.forward)`，训练 step 内频繁反射；`_shared_step()` 和 `predict_step()` 也重复构造 `idx/x_mark/y_mark` kwargs。

### ProbForecastModule

当前概率预测 wrapper 可以采样，但输出和指标链路不完整：

- `test_step()` 返回 `samples/targets`，`PredictionWriter` 不保存 `samples`。
- `on_test_epoch_end()` 只计算 sample median 的 MSE/MAE，没有 CRPS。
- `test_outputs` 将所有 batch 的 samples 留在 GPU/内存，再一次性 `torch.cat`。
- `batch["y"]` 如果是 `(B, C, label_len + H)`，没有裁剪到 forecast horizon。
- fallback `_compute_loss()` 直接对 `pred/y` 做 MSE，无法稳健处理 `label_len`、dict/tuple 输出或 sample 维度。

### Trainer.evaluate()

当前 `_collect_predictions()` 手写：

```python
pred = model.forward(x, idx=idx)
```

问题是它绕过了 Lightning wrapper 的 `predict_step()`，因此也绕过：

- 概率模型的 `sample()`
- `x_mark/y_mark`
- wrapper 内部 target 对齐逻辑
- point/prob 的统一输出字段

结果是 `evaluate(..., mode="probabilistic")` 名义上存在，但拿不到概率 samples，无法可靠计算 CRPS 等概率指标。

## 统一输出契约

所有 forecasting wrapper 对外输出统一为 dict。

### 尺度约定

推理输出、指标计算和绘图默认使用 dataset 标准化后的尺度，不做 dataset/scaler 的 inverse transform 回 csv/raw 原始尺度。

具体约定：

- `inputs` 保存 dataloader 传入 wrapper 的 `batch["x"]`，也就是数据管线标准化后的回望窗口。
- `targets` 保存 dataloader 传入 wrapper 的 `batch["y"]` 裁剪后的预测段，同样保持标准化尺度。
- `preds` 和 `samples` 如果经过 wrapper 内部 RevIN normalization，必须执行 RevIN denorm，回到进入 RevIN 前的 dataloader 标准化尺度。
- dataset/scaler 的 inverse transform 设计为可选能力，但默认关闭；本轮默认不还原 csv/raw 原始尺度。
- `ts_metric` 指标只在上述标准化尺度上计算。
- 绘图直接使用保存的 `inputs/predictions/targets/samples`，不进行反标准化。

### Point 输出

`test_step()` 和可选的结构化 `predict_step()` 输出：

```python
{
    "inputs": Tensor[B, C, L],
    "preds": Tensor[B, C, H],
    "targets": Tensor[B, C, H],
    "loss": Tensor,
}
```

其中：

- `inputs` 是回望窗口，即模型输入的历史序列，用于后续绘图。
- `preds` 是确定性预测。
- `targets` 是与 `preds` horizon 对齐后的真实值。
- `loss` 用于 test log 和调试，不进入 prediction file 的核心字段。

为兼容旧代码，`PredictionWriter` 可以继续接受旧键：

- `pred`
- `y`

但 wrapper 新代码应以 `preds/targets` 为主。

### Probabilistic 输出

`test_step()` 和 `predict_step()` 输出：

```python
{
    "inputs": Tensor[B, C, L],
    "samples": Tensor[B, S, C, H],
    "preds": Tensor[B, C, H],
    "targets": Tensor[B, C, H],
}
```

其中：

- `inputs` 是回望窗口，即模型输入的历史序列，用于和预测段一起绘图。
- `samples` 是概率样本。
- `preds` 是 `samples.median(dim=1).values`，用于 point metric。
- `targets` 是裁剪到 horizon 的真实值。

这样 `PredictionWriter` 可以同时保存：

```text
inputs
samples
predictions
targets
```

其中 `.npz` 里的 `predictions` 对应统一输出契约中的 `preds`。

## PointForecastModule 设计

### 1. 修正 test 输出字段

修改 `test_step()`：

```python
return {
    "inputs": batch["x"],
    "preds": output["pred"],
    "targets": output["y"],
    "loss": output["loss"],
}
```

日志仍保留：

```python
self.log("test_loss", output["loss"], ...)
```

如果希望保持兼容，可以短期额外返回旧字段：

```python
"pred": output["pred"],
"y": output["y"],
```

但文档和新代码都以 `preds/targets` 为准。

### 2. 缓存 forward signature

在 `__init__()` 中缓存模型 forward 可接收参数：

```python
self._forward_params = set(inspect.signature(self.model.forward).parameters)
```

新增 helper：

```python
def _filter_forward_kwargs(self, kwargs):
    return {k: v for k, v in kwargs.items() if k in self._forward_params}
```

`forward()` 改为使用缓存，避免每个 batch 反射。

### 3. 抽取 kwargs 构造

新增：

```python
def _build_kwargs(self, batch):
    kwargs = {}
    for key in ("idx", "x_mark", "y_mark"):
        value = batch.get(key)
        if value is not None:
            kwargs[key] = value
    return kwargs
```

`_shared_step()` 和 `predict_step()` 共用它。

本轮不扩展 point 的 target shape 校验，只保留现有裁剪行为，后续可单独收紧。

## ProbForecastModule 设计

### 1. PredictionWriter 支持 samples

`PredictionWriter._collect()` 扩展识别：

```python
samples -> self._samples
preds / pred -> self._predictions
targets / y -> self._targets
```

`_save()` 保存：

```python
{
    "predictions": concat(preds),
    "targets": concat(targets),
    "samples": concat(samples),
}
```

如果只有 `samples` 没有 `preds`，writer 可以自动用 median 生成 `predictions`。

### 2. Prob test 接入 ts_metric

`ProbForecastModule` 增加可配置测试指标：

```python
test_metrics: tuple[str, ...] = ("CRPS", "CRPS_sum", "MSE_median", "MAE_median")
```

测试阶段：

- CRPS 用 `ts_metric` 计算。
- `MSE_median/MAE_median` 使用 sample median，即 `preds`。
- 如果 `ts_metric` 不可用，降级为只计算 PyTorch MSE/MAE，并打印 warning。

优先尝试调用当前项目中已验证过的 API：

```python
import ts_metric as tm

tm.prediction.crps(targets, samples)
tm.prediction.mse(targets, preds)
tm.prediction.mae(targets, preds)
```

如果 API 不匹配，再 fallback 到 `MetricCalculator`。

### 3. 避免 GPU 上堆完整 samples

`test_step()` 生成 samples 后立即 detach 到 CPU 再存入 `self.test_outputs`：

```python
self.test_outputs.append({
    "inputs": batch["x"].detach().cpu(),
    "samples": samples.detach().cpu(),
    "preds": preds.detach().cpu(),
    "targets": targets.detach().cpu(),
})
```

`on_test_epoch_end()` 只在 CPU 上 concat。

这不是 streaming metric，但能避免 GPU 显存被测试输出长期占住。后续如果数据更大，再做 batch-wise streaming。

### 4. 统一 target horizon 裁剪

新增 helper：

```python
def _align_target_to_prediction(self, target, prediction):
    horizon = prediction.shape[-1]
    if target.shape[-1] != horizon:
        target = target[..., -horizon:]
    return target
```

概率 samples 是 `(B, S, C, H)`，target 是 `(B, C, label_len + H)`。

调用时用：

```python
targets = self._align_target_to_prediction(batch["y"], samples)
```

helper 需要识别 samples 比 target 多一个 sample 维度，因此 horizon 一律取 `prediction.shape[-1]`。

### 5. 改善 fallback loss 和输出标准化

新增输出标准化 helper：

```python
def _to_median_forecast(self, output):
    if isinstance(output, dict):
        output = output.get("preds") or output.get("pred") or output.get("samples")
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.dim() == 4:
        output = output.median(dim=1).values
    return output
```

`_compute_loss()` 使用：

```python
pred = self._to_median_forecast(self.forward(x, **kwargs))
target = self._align_target_to_prediction(batch["y"], pred)
return nn.functional.mse_loss(pred, target)
```

`_predict()` 同样走标准化路径，保证 repeated-forward fallback 能产出 `(B, C, H)`。

## Trainer.evaluate 设计

### 1. 使用 predict_step 收集输出

废弃 `_collect_predictions()` 中手写 forward 的路径。

改为：

```python
pred_outputs = self.predict(
    model,
    datamodule=datamodule,
    dataloaders=dataloaders,
    ckpt_path=ckpt_path,
)
```

然后用一个标准化函数解析 Lightning predict 返回：

```python
def _collect_prediction_outputs(outputs):
    ...
    return preds_or_samples, targets
```

如果 `predict_step()` 输出不含 targets，则从 batch 取 targets 会比较麻烦，因为 Lightning predict 默认不把 batch 返回。因此 forecasting wrapper 的 `predict_step()` 建议在 batch 有 `y` 时返回 `targets`。

### 2. 保留 mark 传递

因为 `predict_step()` 由 wrapper 自己处理 batch，它天然能传递：

- `idx`
- `x_mark`
- `y_mark`
- `y`

`Trainer.evaluate()` 不再手工拆 batch，也就不会漏掉 mark。

### 3. 概率指标链路

`evaluate(..., mode="probabilistic")` 的数据规则：

- 如果 outputs 里有 `inputs`，保留到推理结果中，但不参与指标计算。
- 如果 outputs 里有 `samples`，用 samples 作为概率预测。
- 如果同时有 `preds`，MSE/MAE 用 `preds`。
- 如果没有 `preds`，从 samples median 得到。
- `targets` 必须存在，否则只返回 `test()` 的结果并给出 warning。

指标计算顺序：

1. 先执行 `self.test(...)`，保留 Lightning 测试日志。
2. 如果没有传 `metrics`，直接返回 test results。
3. 执行 `self.predict(...)` 收集标准化 outputs。
4. 用 `ts_metric` 计算请求指标。

对 deterministic：

```python
calc.compute(targets, preds)
```

对 probabilistic：

```python
CRPS -> samples
MSE/MAE -> preds median
```

## 推理指标默认集

推理阶段所有指标统一通过 `ts_metric` 计算。`ts_trainer` 的 `forecast` 任务在 `ts_metric` 中对应 `prediction`。

每个 task/mode 默认计算 4 个指标：

| Trainer task | ts_metric task | mode | 默认 4 指标 | 说明 |
| --- | --- | --- | --- | --- |
| forecast | prediction | point | `MSE`, `MAE`, `RMSE`, `NRMSE` | 覆盖平方误差、绝对误差、原量纲误差和归一化误差 |
| forecast | prediction | probabilistic | `CRPS`, `CRPS_sum`, `MSE_median`, `MAE_median` | `CRPS/CRPS_sum` 对齐 K2VAE/DeepAR 风格概率预测评估，median 指标兼顾点预测质量 |
| imputation | imputation | point | `MSE`, `MAE`, `RMSE`, `MRE` | `MRE` 是插补任务更有用的相对误差，优先于不稳定的 `MAPE` |
| imputation | imputation | probabilistic | `CRPS`, `PICP`, `QICE`, `IntervalWidth` | 同时评价概率质量、覆盖率、校准误差和区间宽度 |
| generation | generation | default | `MDD`, `ACD`, `DS`, `PS` | 覆盖边际分布、时序自相关、可区分性和下游预测可用性 |
| anomaly | anomaly | default | `F1`, `PA_F1`, `AUC_ROC`, `AUC_PR` | 兼顾阈值指标、时序异常段 point-adjust 指标和阈值无关指标 |
| classification | classification | default | `Accuracy`, `Precision`, `Recall`, `F1` | 多分类通用默认集；`AUC_ROC` 仅适合二分类或有明确 score 时作为额外指标 |

实现细节：

- point forecast 使用 `MetricCalculator(task="prediction", mode="point", metrics=[...])`。
- probabilistic forecast 使用 `MetricCalculator(task="prediction", mode="probabilistic", metrics=[...])`，输入为 `targets` 和 `samples`。
- 如果需要同时记录预测中位数的通用点指标，不再手写 PyTorch MSE/MAE，而是使用 `MSE_median` 和 `MAE_median`。
- anomaly 的 `F1/PA_F1` 更适合二值 `preds`，`AUC_ROC/AUC_PR` 更适合连续 `scores`。如果模型同时输出二值预测和异常分数，可以分别调用 `ts_metric.anomaly` 的函数式 API；如果只有一个输出，则按当前输出解释。
- classification 的 `AUC_ROC` 不进入默认 4 项，避免多分类场景下默认失败；二分类评估可以通过用户自定义 metrics 显式开启。

## PredictionWriter 设计

虽然用户本轮范围没有单独列 callback，但 prob 1 需要 writer 支持 samples，因此必须配套改。

内部状态：

```python
self._predictions: list[np.ndarray] = []
self._targets: list[np.ndarray] = []
self._samples: list[np.ndarray] = []
```

收集规则：

```text
inputs: outputs["inputs"] or batch["x"]
predictions: outputs["preds"] or outputs["pred"]
targets: outputs["targets"] or outputs["y"]
samples: outputs["samples"]
```

保存规则：

- 有 predictions 时保存 `predictions`。
- 有 inputs 时保存 `inputs`。
- 有 targets 时保存 `targets`。
- 有 samples 时保存 `samples`。
- 如果没有 predictions 但有 samples，则保存 sample median 到 `predictions`。

`inputs` 必须保存 wrapper 归一化前的回望窗口，也就是 dataloader 给出的标准化 `batch["x"]`。如果 wrapper 内部使用 RevIN，`predictions/samples` 必须 RevIN denorm 回到同一尺度，绘图使用进入 RevIN 前的标准化历史值和 RevIN denorm 后的预测值；默认不做数据集 inverse transform 回原始尺度。

## 测试计划

### 单元测试

新增或扩展 callback 测试：

- `PredictionWriter` 能保存 `preds/targets`。
- `PredictionWriter` 能保存 `samples/preds/targets`。
- 旧字段 `pred/y` 仍能保存。

新增 forecasting wrapper 测试：

- point/prob `test_step()` 和 `predict_step()` 输出包含 `inputs`，shape 为 `(B, C, L)`。
- point `test_step()` 返回 `preds/targets/loss`。
- point `predict_step()` 使用缓存 kwargs，并能传 `idx/x_mark/y_mark`。
- prob `test_step()` 返回 `samples/preds/targets`。
- prob 对 `y=(B,C,label_len+H)` 正确裁剪到 `(B,C,H)`。
- prob fallback forward 输出为 `(B,S,C,H)` 时，loss 使用 median 和裁剪后的 target。

新增 Trainer evaluate 测试：

- deterministic evaluate 使用 wrapper `predict_step()` 输出，而不是手写 forward。
- probabilistic evaluate 能从 `samples` 计算 CRPS，并从 median 计算 MSE/MAE。
- batch 中带 `x_mark/y_mark` 时，由 wrapper 接收并传给底层模型。

### 集成验证

在 `TSLib-tool` 根目录执行：

```bash
pytest ts_trainer/tests -q
```

如果要额外验证 ETTh1 概率模型：

```bash
python tests_integration/train_infer_nsdiff.py --epochs 0 --max-test-batches 1
```

## 实现顺序

1. 修改 `PredictionWriter`，先让保存层能识别新旧字段和 samples。
2. 修改 `PointForecastModule` 输出字段、signature 缓存和 `_build_kwargs()`。
3. 修改 `ProbForecastModule` 的 target 裁剪、输出标准化、CPU 收集和 `ts_metric` 指标。
4. 修改 `Trainer.evaluate()` / `_collect_predictions()`，改用 `predict()` 输出。
5. 增加测试，先覆盖接口契约，再跑现有测试。

## 风险和兼容

- 改 `predict_step()` 返回 dict 后，原来直接期望 tensor 的用户代码可能需要取 `outputs[i]["preds"]`。为降低风险，可以只让 `test_step()` 返回 dict，`predict_step()` 也返回 dict 但文档明确新契约。
- `ts_metric` API 版本可能不同，需要写 fallback。
- 概率 samples 很大，CPU concat 仍可能占内存。当前方案先解决 GPU 堆积问题，大数据 streaming 可后续做。
- `Trainer.evaluate()` 使用 `predict()` 会多跑一遍预测，这是现有 evaluate 的行为延续：先 test，再 collect predictions。后续可考虑从 test outputs 直接复用。
