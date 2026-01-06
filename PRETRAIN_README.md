# 预训练版本使用说明

## 概述

`TRIXLatentMechanismPretrain` 是保持 TRIX 原始 query 初始化方式的预训练版本，与 `TRIXLatentMechanism`（query-free 版本）的区别：

| 特性 | TRIXLatentMechanismPretrain | TRIXLatentMechanism |
|------|----------------------------|---------------------|
| Query 初始化 | 使用 `r_index[:,0]`（与原始 TRIX 一致） | 使用 learnable `query_token`（query-free） |
| 关系表示初始化 | `torch.ones()`（与原始 TRIX 一致） | `relation_token`（learnable） |
| 适用场景 | 预训练 | 微调/推理 |
| Latent Mechanism z | ✅ 支持 | ✅ 支持 |
| KL 正则 | ✅ 支持 | ✅ 支持 |

## 配置文件

配置文件：`config/pretrain_relation_mech_pretrain.yaml`

关键配置：
```yaml
model:
  class: TRIXLatentMechanismPretrain  # 使用预训练版本
  mechanism:
    z_dim: 32
    beta: 1.0e-3
    deterministic_eval: true
```

## 运行预训练

```bash
python src/pretrain_relation.py \
    -c ./config/pretrain_relation_mech_pretrain.yaml \
    --gpus [0]
```

## 预训练后的使用

预训练完成后，可以：

1. **直接用于推理**：使用预训练好的 checkpoint
2. **微调到 query-free 版本**：将 checkpoint 加载到 `TRIXLatentMechanism` 进行微调

### 加载预训练权重示例

```python
# 加载预训练权重到 query-free 版本
pretrain_state = torch.load("pretrain_checkpoint.pth")
model = TRIXLatentMechanism(...)

# 只加载兼容的权重（relation_model, entity_model, mechanism 部分）
model.load_state_dict(pretrain_state["model"], strict=False)
```

## 注意事项

1. **预训练时**：使用 `TRIXLatentMechanismPretrain`，保持 TRIX 原始初始化方式
2. **微调/推理时**：使用 `TRIXLatentMechanism`（query-free），避免信息泄漏
3. **权重兼容性**：两个版本的 `relation_model`、`entity_model` 和 `mechanism` 部分权重可以共享



