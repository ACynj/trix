#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对三个表现较差的数据集进行继续训练的Python脚本
Metafam, NELLInductive(v1), WikiTopicsMT3(infra)
"""

import subprocess
import os
import sys
from pathlib import Path

# 配置参数
BASE_CKPT = "/T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/model_epoch_5.pth"
CONFIG_FILE = "./config/run_relation_inductive_mech_finetune.yaml"
OUTPUT_DIR = "/T20030104/ynj/TRIX/output_rel/finetune_three_datasets"
GPUS = "[0]"
EPOCHS = 5
LR = 1.0e-4  # 原始lr是5.0e-4，使用1/5的学习率以保持其他数据集表现
BPE = "null"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 三个数据集配置
datasets = [
    {
        "name": "Metafam",
        "dataset": "Metafam",
        "version": "null",
        "checkpoint_in": BASE_CKPT,
        "checkpoint_out": f"{OUTPUT_DIR}/checkpoint_after_metafam.pth",
        "output_dir": f"{OUTPUT_DIR}/Metafam"
    },
    {
        "name": "NELLInductive(v1)",
        "dataset": "NELLInductive",
        "version": "v1",
        "checkpoint_in": f"{OUTPUT_DIR}/checkpoint_after_metafam.pth",
        "checkpoint_out": f"{OUTPUT_DIR}/checkpoint_after_nell.pth",
        "output_dir": f"{OUTPUT_DIR}/NELLInductive_v1"
    },
    {
        "name": "WikiTopicsMT3(infra)",
        "dataset": "WikiTopicsMT3",
        "version": "infra",
        "checkpoint_in": f"{OUTPUT_DIR}/checkpoint_after_nell.pth",
        "checkpoint_out": f"{OUTPUT_DIR}/final_checkpoint.pth",
        "output_dir": f"{OUTPUT_DIR}/WikiTopicsMT3_infra"
    }
]

def run_training(dataset_config):
    """运行单个数据集的训练"""
    print(f"\n{'='*60}")
    print(f">>> 训练 {dataset_config['name']} 数据集")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        "python", "./src/run_relation.py",
        "-c", CONFIG_FILE,
        "--dataset", dataset_config["dataset"],
        "--version", dataset_config["version"],
        "--ckpt", dataset_config["checkpoint_in"],
        "--gpus", GPUS,
        "--epochs", str(EPOCHS),
        "--bpe", BPE,
        "--lr", str(LR)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    # 执行训练
    result = subprocess.run(cmd, cwd="/T20030104/ynj/TRIX")
    
    if result.returncode != 0:
        print(f"✗ {dataset_config['name']} 训练失败 (退出码: {result.returncode})")
        return False
    
    # 检查checkpoint是否生成（checkpoint会保存在自动生成的工作目录中）
    # 工作目录格式: output_dir/TRIXLatentMechanism/DatasetClass/YYYY-MM-DD-HH-MM-SS/
    import glob
    import shutil
    from pathlib import Path
    
    # 查找最新的工作目录
    # 工作目录在output_rel下自动创建: output_rel/TRIXLatentMechanism/DatasetClass/YYYY-MM-DD-HH-MM-SS/
    base_output = "/T20030104/ynj/TRIX/output_rel"
    pattern = f"{base_output}/TRIXLatentMechanism/{dataset_config['dataset']}/*"
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    if dirs:
        latest_dir = dirs[0]
        checkpoint_path = f"{latest_dir}/model_epoch_{EPOCHS}.pth"
        if os.path.exists(checkpoint_path):
            # 复制checkpoint
            shutil.copy(checkpoint_path, dataset_config["checkpoint_out"])
            print(f"✓ {dataset_config['name']} 训练完成")
            print(f"  Checkpoint已保存: {dataset_config['checkpoint_out']}")
            print(f"  工作目录: {latest_dir}")
            return True
        else:
            print(f"✗ {dataset_config['name']} 训练失败: checkpoint未找到在 {checkpoint_path}")
            return False
    else:
        print(f"✗ {dataset_config['name']} 训练失败: 工作目录未找到")
        return False

def main():
    print("="*60)
    print("针对三个表现较差的数据集进行继续训练")
    print("="*60)
    print(f"基础checkpoint: {BASE_CKPT}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LR} (原始lr的1/5)")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)
    
    # 检查基础checkpoint是否存在
    if not os.path.exists(BASE_CKPT):
        print(f"✗ 错误: 基础checkpoint不存在: {BASE_CKPT}")
        sys.exit(1)
    
    # 依次训练三个数据集
    for i, dataset_config in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] 开始训练 {dataset_config['name']}")
        
        if not run_training(dataset_config):
            print(f"\n✗ 训练失败，停止执行")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("所有三个数据集训练完成！")
    print("="*60)
    print(f"最终checkpoint: {OUTPUT_DIR}/final_checkpoint.pth")
    print("="*60)

if __name__ == "__main__":
    main()

