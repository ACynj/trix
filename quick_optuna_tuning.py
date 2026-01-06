#!/usr/bin/env python3
"""
快速 Optuna 调参脚本 - 基于已有评估结果进行超参数优化
适用于已经运行过评估的情况
"""

import os
import sys
import re
import json
import yaml
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从日志中提取的参数和结果
def extract_trial_params_from_logs(log_files: List[str]) -> List[Dict]:
    """从日志文件中提取试验参数和结果"""
    trials = []
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            continue
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 提取配置块
        config_blocks = re.findall(r"\{'checkpoint':.*?'num_epoch': \d+\}", content, re.DOTALL)
        
        for block in config_blocks:
            try:
                # 解析配置
                config_str = block.replace("'", '"')
                config = eval(block)  # 简单解析，实际应该用更安全的方法
                
                # 提取关键参数
                params = {}
                if 'model' in config:
                    model = config['model']
                    if 'mechanism' in model:
                        mech = model['mechanism']
                        params['mechanism_beta'] = mech.get('beta', 0.001)
                        params['mechanism_z_dim'] = mech.get('z_dim', 32)
                    
                    if 'trix' in model:
                        trix = model['trix']
                        params['trix_feature_dim'] = trix.get('feature_dim', 32)
                        params['trix_num_layer'] = trix.get('num_layer', 3)
                
                if 'optimizer' in config:
                    params['optimizer_lr'] = config['optimizer'].get('lr', 0.0005)
                
                if 'train' in config:
                    params['train_batch_size'] = config['train'].get('batch_size', 32)
                
                if 'task' in config:
                    params['task_adversarial_temperature'] = config['task'].get('adversarial_temperature', 1)
                
                # 提取结果（需要找到对应的结果行）
                # 这里简化处理，实际需要更精确的匹配
                
                trials.append({
                    'params': params,
                    'value': None  # 需要从结果中提取
                })
            except:
                continue
    
    return trials


def create_optimized_config(best_params: Dict, base_config: str, output_config: str,
                           dataset: str, version: str = None, is_inductive: bool = True):
    """根据最佳参数创建优化后的配置文件"""
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 应用最佳参数
    if 'mechanism_beta' in best_params:
        config['model']['mechanism']['beta'] = best_params['mechanism_beta']
    
    if 'mechanism_z_dim' in best_params:
        config['model']['mechanism']['z_dim'] = int(best_params['mechanism_z_dim'])
    
    if 'trix_feature_dim' in best_params:
        feature_dim = int(best_params['trix_feature_dim'])
        config['model']['trix']['feature_dim'] = feature_dim
        # 更新相关维度
        config['model']['relation_model']['input_dim'] = feature_dim
        config['model']['relation_model']['hidden_dims'] = [feature_dim, feature_dim]
        config['model']['entity_model']['input_dim'] = feature_dim
        config['model']['entity_model']['hidden_dims'] = [feature_dim, feature_dim]
    
    if 'trix_num_layer' in best_params:
        config['model']['trix']['num_layer'] = int(best_params['trix_num_layer'])
    
    if 'optimizer_lr' in best_params:
        config['optimizer']['lr'] = best_params['optimizer_lr']
    
    if 'train_batch_size' in best_params:
        config['train']['batch_size'] = int(best_params['train_batch_size'])
    
    if 'task_adversarial_temperature' in best_params:
        config['task']['adversarial_temperature'] = best_params['task_adversarial_temperature']
    
    # 设置数据集
    config['dataset']['class'] = dataset
    if version and is_inductive:
        config['dataset']['version'] = version
    
    # 保存
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config


def main():
    """主函数 - 基于已有结果进行参数优化建议"""
    logger.info("快速 Optuna 调参 - 基于已有结果")
    
    # 检查是否有 Optuna 结果
    optuna_db = "optuna_logs/trix_hyperparameter_optimization.db"
    if os.path.exists(optuna_db):
        logger.info(f"加载已有 Optuna study: {optuna_db}")
        study = optuna.load_study(
            study_name="trix_hyperparameter_optimization",
            storage=f"sqlite:///{optuna_db}"
        )
        
        logger.info(f"找到 {len(study.trials)} 个试验")
        logger.info(f"最佳试验: {study.best_trial.number}")
        logger.info(f"最佳值: {study.best_value:.4f}")
        logger.info("\n最佳参数:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        # 生成优化后的配置文件
        logger.info("\n生成优化后的配置文件...")
        os.makedirs("config_optimized", exist_ok=True)
        
        # 为关键数据集生成配置
        datasets = [
            ("FB15k237Inductive", "v1", True, "run_relation_inductive_mech.yaml"),
            ("CoDExSmall", None, False, "run_relation_transductive_mech.yaml"),
            ("FB15k237_10", None, False, "run_relation_transductive_mech.yaml"),
        ]
        
        for dataset, version, is_inductive, base_config_name in datasets:
            base_config = f"config/{base_config_name}"
            if os.path.exists(base_config):
                output_config = f"config_optimized/{dataset}_{version or 'none'}_optimized.yaml"
                create_optimized_config(
                    study.best_params,
                    base_config,
                    output_config,
                    dataset,
                    version,
                    is_inductive
                )
                logger.info(f"  生成: {output_config}")
        
        # 保存最佳参数
        best_params_file = "config_optimized/best_params.yaml"
        with open(best_params_file, 'w') as f:
            yaml.dump({
                'best_trial': study.best_trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params
            }, f, default_flow_style=False)
        
        logger.info(f"\n最佳参数已保存到: {best_params_file}")
        
    else:
        logger.warning("未找到 Optuna study，请先运行 optuna_hyperparameter_tuning.py")
        logger.info("\n建议的参数范围:")
        logger.info("  mechanism_beta: [1e-5, 1e-2] (log scale)")
        logger.info("  mechanism_z_dim: [16, 32, 64]")
        logger.info("  trix_feature_dim: [32, 64]")
        logger.info("  trix_num_layer: [2, 3, 4]")
        logger.info("  optimizer_lr: [1e-5, 1e-3] (log scale)")
        logger.info("  train_batch_size: [16, 32, 64]")
        logger.info("  task_adversarial_temperature: [0.5, 1.5]")


if __name__ == "__main__":
    main()



