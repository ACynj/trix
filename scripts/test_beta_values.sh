#!/bin/bash
# 快速测试不同beta值对FB15k237 v1的影响
# 用于找到最优beta，提升普适性

CKPT="/T20030104/ynj/TRIX/ckpts/rel_5.pth"
CONFIG="./config/run_relation_inductive_mech_adaptive.yaml"
DATASET="FB15k237Inductive"
VERSION="v1"
GPUS="[0]"

echo "=========================================="
echo "测试不同beta值对 ${DATASET}(${VERSION}) 的影响"
echo "=========================================="

# 测试不同的beta值
BETAS=(1.0e-4 5.0e-4 1.0e-3 2.0e-3)

for beta in "${BETAS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "测试 beta = ${beta}"
    echo "----------------------------------------"
    
    python ./src/run_relation.py \
        -c ${CONFIG} \
        --dataset ${DATASET} \
        --version ${VERSION} \
        --ckpt ${CKPT} \
        --gpus ${GPUS} \
        --epochs 0 \
        --bpe null \
        --beta ${beta} \
        2>&1 | grep -E "(mrr|hits@1|hits@3|hits@10)"
    
    echo "完成 beta = ${beta}"
done

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "请对比结果，选择最优beta值"
echo "=========================================="

