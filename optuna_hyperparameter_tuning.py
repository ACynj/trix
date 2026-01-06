#!/usr/bin/env python3
"""
Optuna 超参数调优脚本
目标：优化 TRIXLatentMechanism 模型的超参数，使其相对于基准有最大提升
"""

import os
import sys
import re
import json
import subprocess
import yaml
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optuna_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 基准数据文件
BASELINE_LOG = "inference_rel.log"
CURRENT_LOGS = [
    "run_commands_20251225_041027.log",
    "run_commands_20251225_065140.log",
    "run_commands_20251225_133504.log"
]

# 数据集配置（从日志中提取的关键数据集）
KEY_DATASETS = [
    ("FB15k237Inductive", "v1"),
    ("FB15k237Inductive", "v2"),
    ("CoDExSmall", None),
    ("CoDExLarge", None),
    ("FB15k237_10", None),
    ("FB15k237_20", None),
]


def parse_log_metrics(log_file: str) -> Dict[str, Dict[str, float]]:
    """从日志文件中解析评估指标"""
    metrics = {}
    current_dataset = None
    current_version = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # 检测数据集名称
        dataset_match = re.search(r'(\w+)(?:\((\w+)\))?\s+dataset', line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            current_version = dataset_match.group(2) if dataset_match.group(2) else None
            continue
        
        # 解析指标
        if 'mrr:' in line or 'hits@' in line:
            if current_dataset:
                key = f"{current_dataset}_{current_version}" if current_version else current_dataset
                if key not in metrics:
                    metrics[key] = {}
                
                # 解析 MRR
                mrr_match = re.search(r'mrr:\s+([\d.]+)', line)
                if mrr_match:
                    metrics[key]['mrr'] = float(mrr_match.group(1))
                
                # 解析 Hits@1, Hits@3, Hits@10
                for k in [1, 3, 10]:
                    hits_match = re.search(rf'hits@{k}:\s+([\d.]+)', line)
                    if hits_match:
                        metrics[key][f'hits@{k}'] = float(hits_match.group(1))
    
    return metrics


def load_baseline_metrics() -> Dict[str, Dict[str, float]]:
    """加载基准指标"""
    logger.info(f"加载基准数据: {BASELINE_LOG}")
    baseline = parse_log_metrics(BASELINE_LOG)
    logger.info(f"找到 {len(baseline)} 个数据集的基准指标")
    return baseline


def load_current_metrics() -> Dict[str, Dict[str, float]]:
    """加载当前运行指标"""
    all_metrics = {}
    for log_file in CURRENT_LOGS:
        if os.path.exists(log_file):
            logger.info(f"解析日志文件: {log_file}")
            metrics = parse_log_metrics(log_file)
            # 合并指标（取最新值）
            for key, values in metrics.items():
                all_metrics[key] = values
    logger.info(f"找到 {len(all_metrics)} 个数据集的当前指标")
    return all_metrics


def calculate_improvement(baseline: Dict[str, float], current: Dict[str, float]) -> float:
    """计算相对于基准的提升（使用加权平均）"""
    if not baseline or not current:
        return 0.0
    
    improvements = []
    weights = []
    
    # MRR 权重最高
    if 'mrr' in baseline and 'mrr' in current:
        imp = (current['mrr'] - baseline['mrr']) / baseline['mrr'] if baseline['mrr'] > 0 else 0
        improvements.append(imp)
        weights.append(3.0)
    
    # Hits@1 权重较高
    if 'hits@1' in baseline and 'hits@1' in current:
        imp = (current['hits@1'] - baseline['hits@1']) / baseline['hits@1'] if baseline['hits@1'] > 0 else 0
        improvements.append(imp)
        weights.append(2.0)
    
    # Hits@3 和 Hits@10
    for metric in ['hits@3', 'hits@10']:
        if metric in baseline and metric in current:
            imp = (current[metric] - baseline[metric]) / baseline[metric] if baseline[metric] > 0 else 0
            improvements.append(imp)
            weights.append(1.0)
    
    if not improvements:
        return 0.0
    
    # 加权平均
    weighted_sum = sum(imp * w for imp, w in zip(improvements, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def create_config_file(trial: optuna.Trial, base_config: str, output_config: str, 
                       dataset: str, version: str = None, is_inductive: bool = True):
    """根据 trial 参数创建配置文件"""
    with open(base_config, 'r') as f:
        content = f.read()
        # 处理模板变量
        content = content.replace('{{ dataset }}', dataset)
        if version and is_inductive:
            content = content.replace('{{ version }}', version)
        else:
            # 移除 version 字段
            content = re.sub(r'version:\s*\{\{\s*version\s*\}\}\s*\n', '', content)
        content = content.replace('{{ gpus }}', '[0]')
        content = content.replace('{{ epochs }}', '0')
        content = content.replace('{{ bpe }}', 'null')
        content = content.replace('{{ ckpt }}', '/T20030104/ynj/TRIX/ckpts/rel_5.pth')
    
    config = yaml.safe_load(content)
    
    # 优化超参数
    # 1. Mechanism beta (KL散度权重)
    beta = trial.suggest_float('mechanism_beta', 1e-5, 1e-2, log=True)
    config['model']['mechanism']['beta'] = beta
    
    # 2. z_dim (潜在机制维度)
    z_dim = trial.suggest_int('mechanism_z_dim', 16, 64, step=16)
    config['model']['mechanism']['z_dim'] = z_dim
    
    # 3. feature_dim (特征维度)
    feature_dim = trial.suggest_categorical('trix_feature_dim', [32, 64])
    config['model']['trix']['feature_dim'] = feature_dim
    
    # 4. num_layer (层数)
    num_layer = trial.suggest_int('trix_num_layer', 2, 4)
    config['model']['trix']['num_layer'] = num_layer
    
    # 5. 学习率
    lr = trial.suggest_float('optimizer_lr', 1e-5, 1e-3, log=True)
    config['optimizer']['lr'] = lr
    
    # 6. batch_size
    batch_size = trial.suggest_categorical('train_batch_size', [16, 32, 64])
    config['train']['batch_size'] = batch_size
    
    # 7. adversarial_temperature
    adv_temp = trial.suggest_float('task_adversarial_temperature', 0.5, 1.5)
    config['task']['adversarial_temperature'] = adv_temp
    
    # 8. 更新 hidden_dims 以匹配 feature_dim
    config['model']['relation_model']['input_dim'] = feature_dim
    config['model']['relation_model']['hidden_dims'] = [feature_dim, feature_dim]
    config['model']['entity_model']['input_dim'] = feature_dim
    config['model']['entity_model']['hidden_dims'] = [feature_dim, feature_dim]
    
    # 设置数据集
    config['dataset']['class'] = dataset
    if version and is_inductive:
        config['dataset']['version'] = version
    elif 'version' in config['dataset']:
        del config['dataset']['version']
    
    # 保存配置文件
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config


def run_evaluation(config_file: str, checkpoint: str, dataset: str, version: str = None, 
                   is_inductive: bool = True) -> Dict[str, float]:
    """运行评估并返回指标"""
    if is_inductive:
        cmd = [
            'python', './src/run_relation.py',
            '-c', config_file,
            '--dataset', dataset,
            '--version', version if version else 'null',
            '--ckpt', checkpoint,
            '--gpus', '[0]',
            '--epochs', '0',
            '--bpe', 'null'
        ]
    else:
        cmd = [
            'python', './src/run_relation.py',
            '-c', config_file,
            '--dataset', dataset,
            '--ckpt', checkpoint,
            '--gpus', '[0]',
            '--epochs', '0',
            '--bpe', 'null'
        ]
    
    logger.info(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            logger.error(f"评估失败: {result.stderr}")
            return {}
        
        # 解析输出
        output = result.stdout + result.stderr
        metrics = {}
        
        mrr_match = re.search(r'mrr:\s+([\d.]+)', output)
        if mrr_match:
            metrics['mrr'] = float(mrr_match.group(1))
        
        for k in [1, 3, 10]:
            hits_match = re.search(rf'hits@{k}:\s+([\d.]+)', output)
            if hits_match:
                metrics[f'hits@{k}'] = float(hits_match.group(1))
        
        return metrics
    
    except subprocess.TimeoutExpired:
        logger.error("评估超时")
        return {}
    except Exception as e:
        logger.error(f"评估异常: {e}")
        return {}


def objective(trial: optuna.Trial, baseline_metrics: Dict, checkpoint: str, 
              datasets: List[Tuple[str, str, bool]]) -> float:
    """Optuna 目标函数"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial.number} 开始")
    logger.info(f"{'='*60}")
    
    total_improvement = 0.0
    valid_runs = 0
    
    # 为每个数据集运行评估
    for dataset, version, is_inductive in datasets:
        logger.info(f"\n处理数据集: {dataset} (version: {version}, inductive: {is_inductive})")
        
        # 确定基础配置文件
        if is_inductive:
            base_config = "config/run_relation_inductive_mech.yaml"
        else:
            base_config = "config/run_relation_transductive_mech.yaml"
        
        # 创建临时配置文件
        config_dir = Path("tmp_optuna_configs")
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / f"trial_{trial.number}_{dataset}_{version or 'none'}.yaml"
        
        try:
            # 创建配置
            create_config_file(trial, base_config, str(config_file), dataset, version, is_inductive)
            
            # 运行评估
            metrics = run_evaluation(str(config_file), checkpoint, dataset, version, is_inductive)
            
            if not metrics:
                logger.warning(f"数据集 {dataset} 评估失败，跳过")
                continue
            
            # 计算提升
            dataset_key = f"{dataset}_{version}" if version else dataset
            baseline = baseline_metrics.get(dataset_key, {})
            
            if baseline:
                improvement = calculate_improvement(baseline, metrics)
                total_improvement += improvement
                valid_runs += 1
                
                logger.info(f"数据集 {dataset} 提升: {improvement:.4f}")
                logger.info(f"  基准 MRR: {baseline.get('mrr', 0):.4f}, 当前 MRR: {metrics.get('mrr', 0):.4f}")
            else:
                logger.warning(f"数据集 {dataset} 没有基准数据，跳过")
        
        except Exception as e:
            logger.error(f"处理数据集 {dataset} 时出错: {e}")
            continue
    
    if valid_runs == 0:
        return -100.0  # 惩罚无效试验
    
    avg_improvement = total_improvement / valid_runs
    logger.info(f"\nTrial {trial.number} 完成，平均提升: {avg_improvement:.4f}")
    
    return avg_improvement


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Optuna 超参数调优开始")
    logger.info("="*60)
    
    # 加载基准和当前指标
    baseline_metrics = load_baseline_metrics()
    current_metrics = load_current_metrics()
    
    # 打印当前状态
    logger.info("\n当前性能对比:")
    for key in sorted(set(list(baseline_metrics.keys()) + list(current_metrics.keys()))):
        baseline = baseline_metrics.get(key, {})
        current = current_metrics.get(key, {})
        if baseline and current:
            imp = calculate_improvement(baseline, current)
            logger.info(f"{key}: 提升 {imp:.4f}")
    
    # 检查检查点文件
    checkpoint = "/T20030104/ynj/TRIX/ckpts/rel_5.pth"
    if not os.path.exists(checkpoint):
        logger.error(f"检查点文件不存在: {checkpoint}")
        return
    
    # 准备数据集列表（选择关键数据集进行快速调优）
    datasets = [
        ("FB15k237Inductive", "v1", True),
        ("FB15k237Inductive", "v2", True),
        ("CoDExSmall", None, False),
        ("FB15k237_10", None, False),
    ]
    
    # 创建 Optuna study
    study_name = "trix_hyperparameter_optimization"
    storage_url = f"sqlite:///optuna_logs/{study_name}.db"
    
    # 确保目录存在
    os.makedirs("optuna_logs", exist_ok=True)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",  # 最大化提升
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    # 运行优化
    logger.info(f"\n开始 Optuna 优化，目标试验数: 50")
    logger.info(f"使用数据集: {[d[0] for d in datasets]}")
    
    study.optimize(
        lambda trial: objective(trial, baseline_metrics, checkpoint, datasets),
        n_trials=50,
        timeout=3600 * 24,  # 24小时超时
        show_progress_bar=True
    )
    
    # 打印最佳结果
    logger.info("\n" + "="*60)
    logger.info("优化完成！")
    logger.info("="*60)
    logger.info(f"最佳试验: {study.best_trial.number}")
    logger.info(f"最佳提升: {study.best_value:.4f}")
    logger.info("\n最佳超参数:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # 保存结果
    results_file = "optuna_logs/best_params.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }, f, indent=2)
    
    logger.info(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
