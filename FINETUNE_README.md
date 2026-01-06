# 针对三个数据集的继续训练说明

## 目标
在以下三个表现较差的数据集上继续训练模型，提升它们的结果：
1. **Metafam(None)** - 所有指标都大幅下降
2. **NELLInductive(v1)** - 所有指标都下降  
3. **WikiTopicsMT3(infra)** - 所有指标都下降

## 训练策略
- **基础checkpoint**: `model_epoch_5.pth`
- **训练轮数**: 5轮
- **学习率**: 1.0e-4 (原始lr的1/5，避免破坏其他数据集表现)
- **训练顺序**: 依次在三个数据集上训练，每个数据集训练完成后保存checkpoint供下一个使用

## 使用方法

### 方法1: 使用Python脚本（推荐）

```bash
cd /T20030104/ynj/TRIX
python train_three_datasets.py
```

### 方法2: 使用Shell脚本

```bash
cd /T20030104/ynj/TRIX
chmod +x train_three_datasets.sh
./train_three_datasets.sh
```

### 方法3: 手动逐个训练

```bash
# 1. Metafam
python ./src/run_relation.py \
    -c ./config/run_relation_inductive_mech_finetune.yaml \
    --dataset Metafam \
    --version null \
    --ckpt /T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/model_epoch_5.pth \
    --gpus [0] \
    --epochs 5 \
    --bpe null \
    --lr 1.0e-4

# 2. NELLInductive(v1) - 使用上一步的checkpoint
python ./src/run_relation.py \
    -c ./config/run_relation_inductive_mech_finetune.yaml \
    --dataset NELLInductive \
    --version v1 \
    --ckpt <上一步生成的checkpoint路径> \
    --gpus [0] \
    --epochs 5 \
    --bpe null \
    --lr 1.0e-4

# 3. WikiTopicsMT3(infra) - 使用上一步的checkpoint
python ./src/run_relation.py \
    -c ./config/run_relation_inductive_mech_finetune.yaml \
    --dataset WikiTopicsMT3 \
    --version infra \
    --ckpt <上一步生成的checkpoint路径> \
    --gpus [0] \
    --epochs 5 \
    --bpe null \
    --lr 1.0e-4
```

## 输出文件

训练完成后，最终checkpoint将保存在：
```
/T20030104/ynj/TRIX/output_rel/finetune_three_datasets/final_checkpoint.pth
```

中间checkpoint：
- `checkpoint_after_metafam.pth` - Metafam训练后
- `checkpoint_after_nell.pth` - NELLInductive(v1)训练后
- `final_checkpoint.pth` - WikiTopicsMT3(infra)训练后（最终）

## 验证结果

训练完成后，可以使用最终checkpoint在三个数据集上评估：

```bash
# 评估Metafam
python ./src/run_relation.py \
    -c ./config/run_relation_inductive_mech.yaml \
    --dataset Metafam \
    --version null \
    --ckpt /T20030104/ynj/TRIX/output_rel/finetune_three_datasets/final_checkpoint.pth \
    --gpus [0] \
    --epochs 0 \
    --bpe null

# 评估NELLInductive(v1)
python ./src/run_relation.py \
    -c ./config/run_relation_inductive_mech.yaml \
    --dataset NELLInductive \
    --version v1 \
    --ckpt /T20030104/ynj/TRIX/output_rel/finetune_three_datasets/final_checkpoint.pth \
    --gpus [0] \
    --epochs 0 \
    --bpe null

# 评估WikiTopicsMT3(infra)
python ./src/run_relation.py \
    -c ./config/run_relation_inductive_mech.yaml \
    --dataset WikiTopicsMT3 \
    --version infra \
    --ckpt /T20030104/ynj/TRIX/output_rel/finetune_three_datasets/final_checkpoint.pth \
    --gpus [0] \
    --epochs 0 \
    --bpe null
```

## 注意事项

1. **学习率设置**: 使用较小的学习率(1.0e-4)是为了避免过度训练导致其他数据集表现下降
2. **训练顺序**: 按照Metafam → NELLInductive(v1) → WikiTopicsMT3(infra)的顺序训练
3. **Checkpoint管理**: 每个数据集训练完成后会自动保存checkpoint供下一个使用
4. **工作目录**: 训练过程中的checkpoint会保存在自动生成的时间戳目录中，脚本会自动找到最新的checkpoint

## 预期效果

期望在这三个数据集上看到：
- **Metafam**: MRR和Hits@1的提升（当前MRR: 0.250, Hits@1: 0.005）
- **NELLInductive(v1)**: 所有指标的提升（当前MRR: 0.522, Hits@1: 0.269）
- **WikiTopicsMT3(infra)**: 所有指标的提升（当前MRR: 0.908, Hits@1: 0.826）

同时保持其他数据集的表现基本不变。



