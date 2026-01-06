#!/bin/bash
# 快速启动三个数据集的微调训练

cd /T20030104/ynj/TRIX

echo "=========================================="
echo "开始针对三个数据集进行微调训练"
echo "=========================================="
echo ""
echo "目标数据集:"
echo "  1. Metafam(None)"
echo "  2. NELLInductive(v1)"
echo "  3. WikiTopicsMT3(infra)"
echo ""
echo "训练参数:"
echo "  - 基础checkpoint: model_epoch_5.pth"
echo "  - 训练轮数: 5"
echo "  - 学习率: 1.0e-4 (原始lr的1/5)"
echo ""
echo "=========================================="
echo ""

# 运行Python脚本
python train_three_datasets.py

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "最终checkpoint位置:"
echo "  /T20030104/ynj/TRIX/output_rel/finetune_three_datasets/final_checkpoint.pth"
echo ""
echo "可以使用以下命令评估结果:"
echo "  python ./src/run_relation.py -c ./config/run_relation_inductive_mech.yaml \\"
echo "    --dataset Metafam --version null \\"
echo "    --ckpt /T20030104/ynj/TRIX/output_rel/finetune_three_datasets/final_checkpoint.pth \\"
echo "    --gpus [0] --epochs 0 --bpe null"
echo ""



