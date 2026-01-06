#!/bin/bash
# 快速修复脚本：在表现差的数据集上测试不同beta值

CKPT="/T20030104/ynj/TRIX/ckpts/rel_5.pth"
CONFIG_BASE="./config/run_relation_inductive_mech.yaml"
GPUS="[0]"

echo "=========================================="
echo "快速修复：测试不同Beta值"
echo "=========================================="

# 创建临时配置文件
TEMP_CONFIG="/tmp/temp_config_mech.yaml"
cp $CONFIG_BASE $TEMP_CONFIG

# 测试数据集和beta值
declare -A TEST_CASES=(
    ["FB15k237Inductive v1"]="1.0e-4 5.0e-4 1.0e-3"
    ["ILPC2022 large"]="5.0e-5 1.0e-4 5.0e-4"
    ["FB15k237Inductive v4"]="1.0e-4 5.0e-4"
)

for dataset_info in "${!TEST_CASES[@]}"; do
    IFS=' ' read -r dataset version <<< "$dataset_info"
    betas="${TEST_CASES[$dataset_info]}"
    
    echo ""
    echo "----------------------------------------"
    echo "测试数据集: $dataset $version"
    echo "----------------------------------------"
    
    for beta in $betas; do
        echo ""
        echo ">>> 测试 beta = $beta"
        
        # 修改配置文件中的beta值
        sed -i "s/beta: .*/beta: $beta/" $TEMP_CONFIG
        
        # 构建命令
        if [ -n "$version" ]; then
            CMD="python ./src/run_relation.py -c $TEMP_CONFIG --dataset $dataset --version $version --ckpt $CKPT --gpus $GPUS --epochs 0 --bpe null"
        else
            CMD="python ./src/run_relation.py -c $TEMP_CONFIG --dataset $dataset --ckpt $CKPT --gpus $GPUS --epochs 0 --bpe null"
        fi
        
        # 运行并提取关键指标
        $CMD 2>&1 | tee /tmp/temp_output.log
        
        # 提取MRR和Hits@10
        MRR=$(grep -oP 'mrr: \K[0-9.]+' /tmp/temp_output.log | tail -1)
        H10=$(grep -oP 'hits@10: \K[0-9.]+' /tmp/temp_output.log | tail -1)
        
        echo "结果: MRR=$MRR, Hits@10=$H10"
    done
    
    # 恢复原始配置
    cp $CONFIG_BASE $TEMP_CONFIG
done

rm -f $TEMP_CONFIG /tmp/temp_output.log

echo ""
echo "=========================================="
echo "测试完成！请对比结果选择最优beta"
echo "=========================================="



