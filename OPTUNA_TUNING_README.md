# Optuna 超参数调优指南

本目录包含用于优化 TRIXLatentMechanism 模型超参数的 Optuna 脚本。

## 文件说明

1. **analyze_results.py** - 结果分析脚本（推荐首先运行）
   - 对比基准数据和当前运行结果
   - 计算各项指标的提升
   - 生成详细对比报告
   - **无需额外依赖**

2. **smart_optuna_tuning.py** - 智能参数优化建议（推荐）
   - 基于分析结果自动识别问题数据集
   - 为每个问题数据集生成针对性参数建议
   - 自动生成优化后的配置文件
   - **无需额外依赖**

3. **optuna_hyperparameter_tuning.py** - 完整的 Optuna 调参脚本
   - 自动运行训练/评估
   - 优化关键超参数
   - 相对于基准数据最大化提升
   - **需要安装 optuna: `pip install optuna`**

4. **quick_optuna_tuning.py** - 快速调参脚本
   - 基于已有 Optuna 结果生成优化配置
   - 适用于已经运行过调参的情况
   - **需要安装 optuna**

## 使用方法

### 1. 分析当前结果

首先运行分析脚本，了解当前性能：

```bash
python analyze_results.py
```

这将生成 `comparison_analysis.json` 文件，包含详细的对比结果。

### 2. 运行 Optuna 调参

运行完整的调参脚本（需要较长时间）：

```bash
python optuna_hyperparameter_tuning.py
```

**注意**: 这个脚本会：
- 自动运行多个试验（默认50个）
- 每个试验会在多个数据集上评估
- 预计需要数小时到数天时间

**建议**: 如果时间有限，可以先修改脚本中的 `n_trials` 参数，减少试验数量。

### 3. 查看调参结果

调参完成后，运行快速脚本查看结果：

```bash
python quick_optuna_tuning.py
```

这将：
- 显示最佳参数
- 生成优化后的配置文件（保存在 `config_optimized/` 目录）

## 优化的超参数

脚本会优化以下超参数：

| 参数 | 范围 | 说明 |
|------|------|------|
| `mechanism_beta` | [1e-5, 1e-2] | KL散度权重，控制正则化强度 |
| `mechanism_z_dim` | [16, 32, 64] | 潜在机制维度 |
| `trix_feature_dim` | [32, 64] | 特征维度 |
| `trix_num_layer` | [2, 3, 4] | 模型层数 |
| `optimizer_lr` | [1e-5, 1e-3] | 学习率 |
| `train_batch_size` | [16, 32, 64] | 批次大小 |
| `task_adversarial_temperature` | [0.5, 1.5] | 对抗训练温度 |

## 输出文件

- `optuna_logs/trix_hyperparameter_optimization.db` - Optuna study 数据库
- `optuna_logs/best_params.json` - 最佳参数（JSON格式）
- `config_optimized/` - 优化后的配置文件
- `optuna_tuning.log` - 调参日志
- `comparison_analysis.json` - 结果对比分析

## 自定义调参

### 修改数据集列表

在 `optuna_hyperparameter_tuning.py` 中修改 `datasets` 变量：

```python
datasets = [
    ("FB15k237Inductive", "v1", True),
    ("CoDExSmall", None, False),
    # 添加更多数据集...
]
```

### 修改参数范围

在 `objective` 函数中修改参数建议：

```python
# 例如，限制 beta 的范围
beta = trial.suggest_float('mechanism_beta', 1e-4, 1e-2, log=True)
```

### 修改试验数量

```python
study.optimize(
    lambda trial: objective(trial, baseline_metrics, checkpoint, datasets),
    n_trials=20,  # 减少试验数量
    timeout=3600 * 12,  # 12小时超时
)
```

## 性能优化建议

1. **并行运行**: 如果有多个GPU，可以修改脚本支持并行试验
2. **早停机制**: 使用 Pruner 提前终止表现差的试验
3. **减少数据集**: 在调参阶段只使用关键数据集，节省时间
4. **增量调参**: 先在小范围调参，找到大致方向后再扩大范围

## 注意事项

1. 确保检查点文件存在：`/T20030104/ynj/TRIX/ckpts/rel_5.pth`
2. 确保基准日志文件存在：`inference_rel.log`
3. 调参过程会创建大量临时配置文件，注意磁盘空间
4. 建议在后台运行长时间调参任务

## 示例输出

```
Trial 10 开始
处理数据集: FB15k237Inductive (version: v1, inductive: True)
数据集 FB15k237Inductive 提升: 0.0234
  基准 MRR: 0.6895, 当前 MRR: 0.7129
Trial 10 完成，平均提升: 0.0156

最佳试验: 25
最佳提升: 0.0234
最佳超参数:
  mechanism_beta: 0.0005
  mechanism_z_dim: 32
  trix_feature_dim: 64
  ...
```

## 故障排除

### 问题：评估超时
- 解决方案：增加超时时间或减少数据集数量

### 问题：内存不足
- 解决方案：减小 batch_size 或 feature_dim

### 问题：找不到检查点
- 解决方案：检查 `checkpoint` 变量指向正确的路径

## 联系与支持

如有问题，请检查日志文件 `optuna_tuning.log` 获取详细错误信息。

