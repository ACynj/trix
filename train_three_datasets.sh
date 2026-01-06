#!/bin/bash
# 针对三个表现较差的数据集进行继续训练
# Metafam, NELLInductive(v1), WikiTopicsMT3(infra)

# 设置基础路径
BASE_CKPT="/T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/model_epoch_5.pth"
CONFIG_FILE="./config/run_relation_inductive_mech_finetune.yaml"
OUTPUT_DIR="/T20030104/ynj/TRIX/output_rel/finetune_three_datasets"
GPUS="[0]"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 训练参数：使用较小的学习率和较少的epochs，避免破坏其他数据集的表现
EPOCHS=5
LR=1.0e-4  # 原始lr是5.0e-4，这里使用1/5的学习率
BPE="null"

echo "=========================================="
echo "开始针对三个数据集进行继续训练"
echo "基础checkpoint: ${BASE_CKPT}"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LR}"
echo "=========================================="

# 1. Metafam
echo ""
echo ">>> 训练 Metafam 数据集"
echo "=========================================="
python ./src/run_relation.py \
    -c ${CONFIG_FILE} \
    --dataset Metafam \
    --version null \
    --ckpt ${BASE_CKPT} \
    --gpus ${GPUS} \
    --epochs ${EPOCHS} \
    --bpe ${BPE} \
    --lr ${LR} \

# 保存checkpoint (checkpoint会保存在自动生成的工作目录中)
# 需要从工作目录中找到最新的checkpoint
LATEST_METAFAM_DIR=$(ls -td ${OUTPUT_DIR}/TRIXLatentMechanism/Metafam/* 2>/dev/null | head -1)
if [ -n "$LATEST_METAFAM_DIR" ] && [ -f "${LATEST_METAFAM_DIR}/model_epoch_${EPOCHS}.pth" ]; then
    cp ${LATEST_METAFAM_DIR}/model_epoch_${EPOCHS}.pth ${OUTPUT_DIR}/checkpoint_after_metafam.pth
    echo "✓ Metafam训练完成，checkpoint已保存"
else
    echo "✗ Metafam训练失败"
    exit 1
fi

# 2. NELLInductive(v1)
echo ""
echo ">>> 训练 NELLInductive(v1) 数据集"
echo "=========================================="
python ./src/run_relation.py \
    -c ${CONFIG_FILE} \
    --dataset NELLInductive \
    --version v1 \
    --ckpt ${OUTPUT_DIR}/checkpoint_after_metafam.pth \
    --gpus ${GPUS} \
    --epochs ${EPOCHS} \
    --bpe ${BPE} \
    --lr ${LR} \

# 保存checkpoint
LATEST_NELL_DIR=$(ls -td ${OUTPUT_DIR}/TRIXLatentMechanism/NELLInductive/* 2>/dev/null | head -1)
if [ -n "$LATEST_NELL_DIR" ] && [ -f "${LATEST_NELL_DIR}/model_epoch_${EPOCHS}.pth" ]; then
    cp ${LATEST_NELL_DIR}/model_epoch_${EPOCHS}.pth ${OUTPUT_DIR}/checkpoint_after_nell.pth
    echo "✓ NELLInductive(v1)训练完成，checkpoint已保存"
else
    echo "✗ NELLInductive(v1)训练失败"
    exit 1
fi

# 3. WikiTopicsMT3(infra)
echo ""
echo ">>> 训练 WikiTopicsMT3(infra) 数据集"
echo "=========================================="
python ./src/run_relation.py \
    -c ${CONFIG_FILE} \
    --dataset WikiTopicsMT3 \
    --version infra \
    --ckpt ${OUTPUT_DIR}/checkpoint_after_nell.pth \
    --gpus ${GPUS} \
    --epochs ${EPOCHS} \
    --bpe ${BPE} \
    --lr ${LR} \

# 保存最终checkpoint
LATEST_WIKI_DIR=$(ls -td ${OUTPUT_DIR}/TRIXLatentMechanism/WikiTopicsMT3/* 2>/dev/null | head -1)
if [ -n "$LATEST_WIKI_DIR" ] && [ -f "${LATEST_WIKI_DIR}/model_epoch_${EPOCHS}.pth" ]; then
    cp ${LATEST_WIKI_DIR}/model_epoch_${EPOCHS}.pth ${OUTPUT_DIR}/final_checkpoint.pth
    echo "✓ WikiTopicsMT3(infra)训练完成，最终checkpoint已保存"
    echo ""
    echo "=========================================="
    echo "所有三个数据集训练完成！"
    echo "最终checkpoint: ${OUTPUT_DIR}/final_checkpoint.pth"
    echo "=========================================="
else
    echo "✗ WikiTopicsMT3(infra)训练失败"
    exit 1
fi

