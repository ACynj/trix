#!/usr/bin/env python3
"""
智能 Optuna 调参脚本
基于已有结果分析，针对表现不佳的数据集进行重点优化
"""

import os
import sys
import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from analyze_results import parse_log_metrics, calculate_improvement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASELINE_LOG = "inference_rel.log"
CURRENT_LOGS = [
    "run_commands_20251225_041027.log",
    "run_commands_20251225_065140.log",
    "run_commands_20251225_133504.log"
]


def identify_problematic_datasets(baseline_metrics: Dict, current_metrics: Dict, 
                                  threshold: float = -0.02) -> List[Tuple[str, float]]:
    """识别表现下降的数据集"""
    problematic = []
    
    for dataset in baseline_metrics:
        if dataset not in current_metrics:
            continue
        
        baseline = baseline_metrics[dataset]
        current = current_metrics[dataset]
        
        if 'mrr' in baseline and 'mrr' in current:
            mrr_imp = (current['mrr'] - baseline['mrr']) / baseline['mrr'] if baseline['mrr'] > 0 else 0
            if mrr_imp < threshold:
                problematic.append((dataset, mrr_imp))
    
    # 按下降程度排序
    problematic.sort(key=lambda x: x[1])
    return problematic


def suggest_parameter_ranges(dataset_type: str, current_performance: float) -> Dict:
    """根据数据集类型和当前性能建议参数范围"""
    ranges = {}
    
    # 对于表现差的数据集，使用更激进的参数
    if current_performance < 0.6:
        # 低性能数据集：增加模型容量，降低正则化
        ranges['mechanism_beta'] = (1e-4, 5e-3)  # 降低KL权重
        ranges['trix_feature_dim'] = [64]  # 使用更大维度
        ranges['trix_num_layer'] = [3, 4]  # 更多层
        ranges['optimizer_lr'] = (1e-4, 1e-3)  # 更高学习率
    elif current_performance < 0.75:
        # 中等性能：平衡
        ranges['mechanism_beta'] = (5e-4, 2e-3)
        ranges['trix_feature_dim'] = [32, 64]
        ranges['trix_num_layer'] = [2, 3, 4]
        ranges['optimizer_lr'] = (5e-5, 5e-4)
    else:
        # 高性能：保守调整
        ranges['mechanism_beta'] = (1e-3, 5e-3)
        ranges['trix_feature_dim'] = [32, 64]
        ranges['trix_num_layer'] = [2, 3]
        ranges['optimizer_lr'] = (1e-5, 1e-4)
    
    # 通用参数
    ranges['mechanism_z_dim'] = [16, 32, 64]
    ranges['train_batch_size'] = [16, 32, 64]
    ranges['task_adversarial_temperature'] = (0.5, 1.5)
    
    return ranges


def create_optimized_config(params: Dict, base_config: str, output_config: str,
                           dataset: str, version: str = None, is_inductive: bool = True):
    """创建优化后的配置文件"""
    with open(base_config, 'r') as f:
        content = f.read()
        # 处理模板变量
        content = content.replace('{{ dataset }}', dataset)
        if version and is_inductive:
            content = content.replace('{{ version }}', version)
        else:
            content = re.sub(r'version:\s*\{\{\s*version\s*\}\}\s*\n', '', content)
        content = content.replace('{{ gpus }}', '[0]')
        content = content.replace('{{ epochs }}', '0')
        content = content.replace('{{ bpe }}', 'null')
        content = content.replace('{{ ckpt }}', '/T20030104/ynj/TRIX/ckpts/rel_5.pth')
    
    config = yaml.safe_load(content)
    
    # 应用参数
    if 'mechanism_beta' in params:
        config['model']['mechanism']['beta'] = params['mechanism_beta']
    if 'mechanism_z_dim' in params:
        config['model']['mechanism']['z_dim'] = int(params['mechanism_z_dim'])
    if 'trix_feature_dim' in params:
        feature_dim = int(params['trix_feature_dim'])
        config['model']['trix']['feature_dim'] = feature_dim
        config['model']['relation_model']['input_dim'] = feature_dim
        config['model']['relation_model']['hidden_dims'] = [feature_dim, feature_dim]
        config['model']['entity_model']['input_dim'] = feature_dim
        config['model']['entity_model']['hidden_dims'] = [feature_dim, feature_dim]
    if 'trix_num_layer' in params:
        config['model']['trix']['num_layer'] = int(params['trix_num_layer'])
    if 'optimizer_lr' in params:
        config['optimizer']['lr'] = params['optimizer_lr']
    if 'train_batch_size' in params:
        config['train']['batch_size'] = int(params['train_batch_size'])
    if 'task_adversarial_temperature' in params:
        config['task']['adversarial_temperature'] = params['task_adversarial_temperature']
    
    # 设置数据集
    config['dataset']['class'] = dataset
    if version and is_inductive:
        config['dataset']['version'] = version
    elif 'version' in config['dataset']:
        del config['dataset']['version']
    
    # 保存
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config


def parse_dataset_info(dataset_key: str) -> Tuple[str, str, bool]:
    """解析数据集信息"""
    parts = dataset_key.split('_')
    if len(parts) >= 2 and parts[-1] in ['v1', 'v2', 'v3', 'v4']:
        version = parts[-1]
        dataset = '_'.join(parts[:-1])
        is_inductive = True
    else:
        version = None
        dataset = dataset_key
        # 判断是否是归纳数据集
        inductive_datasets = ['FB15k237Inductive', 'WN18RRInductive', 'NELLInductive']
        is_inductive = any(d in dataset for d in inductive_datasets)
    
    return dataset, version, is_inductive


def generate_parameter_suggestions():
    """基于分析结果生成参数建议"""
    logger.info("加载基准和当前指标...")
    
    # 加载数据
    baseline_metrics = parse_log_metrics(BASELINE_LOG)
    all_current_metrics = {}
    for log_file in CURRENT_LOGS:
        if os.path.exists(log_file):
            metrics = parse_log_metrics(log_file)
            all_current_metrics.update(metrics)
    
    # 识别问题数据集
    problematic = identify_problematic_datasets(baseline_metrics, all_current_metrics, threshold=-0.02)
    
    logger.info(f"\n发现 {len(problematic)} 个表现下降的数据集:")
    for dataset, imp in problematic[:10]:
        logger.info(f"  {dataset}: {imp:.2%}")
    
    # 为每个问题数据集生成参数建议
    suggestions = {}
    
    for dataset_key, imp in problematic[:5]:  # 只处理前5个最差的
        dataset, version, is_inductive = parse_dataset_info(dataset_key)
        baseline = baseline_metrics.get(dataset_key, {})
        current = all_current_metrics.get(dataset_key, {})
        
        current_mrr = current.get('mrr', 0.5)
        ranges = suggest_parameter_ranges(dataset, current_mrr)
        
        suggestions[dataset_key] = {
            'dataset': dataset,
            'version': version,
            'is_inductive': is_inductive,
            'current_mrr': current_mrr,
            'baseline_mrr': baseline.get('mrr', 0),
            'improvement': imp,
            'suggested_ranges': ranges,
            'recommended_params': {
                'mechanism_beta': ranges['mechanism_beta'][0] if isinstance(ranges['mechanism_beta'], tuple) else 1e-3,
                'mechanism_z_dim': ranges['mechanism_z_dim'][1] if isinstance(ranges['mechanism_z_dim'], list) else 32,
                'trix_feature_dim': ranges['trix_feature_dim'][-1] if isinstance(ranges['trix_feature_dim'], list) else 64,
                'trix_num_layer': ranges['trix_num_layer'][-1] if isinstance(ranges['trix_num_layer'], list) else 3,
                'optimizer_lr': ranges['optimizer_lr'][1] if isinstance(ranges['optimizer_lr'], tuple) else 5e-4,
                'train_batch_size': 32,
                'task_adversarial_temperature': 1.0,
            }
        }
    
    # 保存建议
    output_file = "parameter_suggestions.json"
    with open(output_file, 'w') as f:
        json.dump(suggestions, f, indent=2)
    
    logger.info(f"\n参数建议已保存到: {output_file}")
    
    # 生成优化后的配置文件
    logger.info("\n生成优化后的配置文件...")
    os.makedirs("config_suggested", exist_ok=True)
    
    for dataset_key, suggestion in suggestions.items():
        dataset = suggestion['dataset']
        version = suggestion['version']
        is_inductive = suggestion['is_inductive']
        params = suggestion['recommended_params']
        
        # 确定基础配置
        if is_inductive:
            base_config = "config/run_relation_inductive_mech.yaml"
        else:
            base_config = "config/run_relation_transductive_mech.yaml"
        
        output_config = f"config_suggested/{dataset}_{version or 'none'}_suggested.yaml"
        
        if os.path.exists(base_config):
            create_optimized_config(params, base_config, output_config, dataset, version, is_inductive)
            logger.info(f"  生成: {output_config}")
    
    return suggestions


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("智能参数优化建议")
    logger.info("="*60)
    
    suggestions = generate_parameter_suggestions()
    
    logger.info("\n" + "="*60)
    logger.info("推荐参数总结:")
    logger.info("="*60)
    
    for dataset_key, suggestion in suggestions.items():
        logger.info(f"\n数据集: {dataset_key}")
        logger.info(f"  当前 MRR: {suggestion['current_mrr']:.4f}")
        logger.info(f"  基准 MRR: {suggestion['baseline_mrr']:.4f}")
        logger.info(f"  下降: {suggestion['improvement']:.2%}")
        logger.info("  推荐参数:")
        for key, value in suggestion['recommended_params'].items():
            logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    main()

