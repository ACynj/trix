#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出最佳模型并进行全面对比
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import subprocess

def parse_train_log(log_file: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    """解析训练日志，提取每个epoch的验证集指标"""
    epoch_metrics = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_epoch = None
    current_section = None
    dataset_idx = 0
    datasets = ['FB15k237', 'WN18RR', 'CoDExMedium']
    
    for i, line in enumerate(lines):
        # 检测epoch结束
        if 'Epoch' in line and 'end' in line:
            epoch_match = re.search(r'Epoch (\d+) end', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                dataset_idx = 0
                if current_epoch not in epoch_metrics:
                    epoch_metrics[current_epoch] = {}
        
        # 检测评估开始
        if 'Evaluate on valid' in line:
            current_section = 'valid'
            dataset_idx = 0
        
        # 解析指标
        if current_epoch is not None and current_section == 'valid':
            metrics = {}
            if 'mr:' in line:
                metrics['mr'] = float(re.search(r'mr:\s*([\d.]+)', line).group(1))
            if 'mrr:' in line:
                metrics['mrr'] = float(re.search(r'mrr:\s*([\d.]+)', line).group(1))
            if 'hits@1:' in line:
                metrics['hits@1'] = float(re.search(r'hits@1:\s*([\d.]+)', line).group(1))
            if 'hits@3:' in line:
                metrics['hits@3'] = float(re.search(r'hits@3:\s*([\d.]+)', line).group(1))
            if 'hits@10:' in line:
                metrics['hits@10'] = float(re.search(r'hits@10:\s*([\d.]+)', line).group(1))
            
            # 如果这一行有指标，说明是一个数据集的结果
            if metrics:
                if dataset_idx < len(datasets):
                    dataset_name = datasets[dataset_idx]
                    epoch_metrics[current_epoch][dataset_name] = metrics
                    dataset_idx += 1
                
                # 如果已经有三个数据集的结果，重置
                if dataset_idx >= 3:
                    current_section = None
    
    return epoch_metrics

def parse_eval_log(log_file: str) -> Dict[str, Dict[str, float]]:
    """解析评估日志，提取测试结果"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到所有数据集和对应的指标
    # 模式：数据集名称 dataset 后跟着 Evaluate on test，然后是指标
    pattern = r'(\w+(?:Inductive|Ingram)?(?:\([^)]+\))?)\s+dataset.*?Evaluate on test.*?(?=\n\d{2}:\d{2}:\d{2}\s+(\w+.*?dataset|执行命令|\Z))'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        dataset_name = match.group(1)
        eval_section = match.group(0)
        
        metrics = {}
        if 'mr:' in eval_section:
            mr_match = re.search(r'mr:\s*([\d.]+)', eval_section)
            if mr_match:
                metrics['mr'] = float(mr_match.group(1))
        if 'mrr:' in eval_section:
            mrr_match = re.search(r'mrr:\s*([\d.]+)', eval_section)
            if mrr_match:
                metrics['mrr'] = float(mrr_match.group(1))
        if 'hits@1:' in eval_section:
            h1_match = re.search(r'hits@1:\s*([\d.]+)', eval_section)
            if h1_match:
                metrics['hits@1'] = float(h1_match.group(1))
        if 'hits@3:' in eval_section:
            h3_match = re.search(r'hits@3:\s*([\d.]+)', eval_section)
            if h3_match:
                metrics['hits@3'] = float(h3_match.group(1))
        if 'hits@10:' in eval_section:
            h10_match = re.search(r'hits@10:\s*([\d.]+)', eval_section)
            if h10_match:
                metrics['hits@10'] = float(h10_match.group(1))
        
        if metrics:
            results[dataset_name] = metrics
    
    return results

def categorize_datasets(datasets: List[str]) -> Dict[str, List[str]]:
    """根据数据集类型分类"""
    categories = {
        'transductive': [],
        'inductive': [],
        'other': []
    }
    
    transductive_keywords = ['CoDEx', 'NELL995', 'DBpedia', 'ConceptNet', 'NELL23k', 
                            'YAGO', 'Hetionet', 'WDsinger', 'AristoV4', 'FB15k237_']
    inductive_keywords = ['Inductive', 'ILPC2022', 'HM', 'Ingram', 'WikiTopics', 
                         'Metafam', 'FBNELL']
    
    for dataset in datasets:
        dataset_clean = dataset.split('(')[0]  # 移除版本信息
        if any(kw in dataset for kw in transductive_keywords):
            categories['transductive'].append(dataset)
        elif any(kw in dataset for kw in inductive_keywords):
            categories['inductive'].append(dataset)
        else:
            categories['other'].append(dataset)
    
    return categories

def calculate_category_averages(results: Dict[str, Dict[str, float]], 
                                category_datasets: List[str]) -> Dict[str, float]:
    """计算某个类别的平均指标"""
    metrics = defaultdict(list)
    
    for dataset in category_datasets:
        if dataset in results:
            for metric, value in results[dataset].items():
                metrics[metric].append(value)
    
    avg_metrics = {}
    for metric, values in metrics.items():
        if values:
            avg_metrics[metric] = sum(values) / len(values)
    
    return avg_metrics

def main():
    train_log = '/T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/log.txt'
    
    print("="*80)
    print("分析训练日志，找出最佳模型")
    print("="*80)
    
    # 解析训练日志
    epoch_metrics = parse_train_log(train_log)
    print(f"\n找到 {len(epoch_metrics)} 个epoch的验证结果")
    
    # 找出最佳epoch
    best_epoch = None
    best_mrr = -1
    best_metrics = {}
    
    for epoch in sorted(epoch_metrics.keys()):
        metrics = epoch_metrics[epoch]
        mrr_values = [m['mrr'] for m in metrics.values() if 'mrr' in m]
        if mrr_values:
            avg_mrr = sum(mrr_values) / len(mrr_values)
            print(f"Epoch {epoch}: 平均MRR = {avg_mrr:.4f}")
            if avg_mrr > best_mrr:
                best_mrr = avg_mrr
                best_epoch = epoch
                best_metrics = metrics
    
    print(f"\n最佳epoch: {best_epoch} (平均MRR: {best_mrr:.4f})")
    
    # 读取基准日志
    baseline_logs = {
        'inductive': '/T20030104/ynj/TRIX/run_commands_20251225_041027.log',
        'transductive': '/T20030104/ynj/TRIX/run_commands_20251225_065140.log',
        'fb15k237': '/T20030104/ynj/TRIX/run_commands_20251225_133504.log'
    }
    
    print("\n" + "="*80)
    print("解析基准日志")
    print("="*80)
    
    baseline_results = {}
    for name, log_file in baseline_logs.items():
        if Path(log_file).exists():
            results = parse_eval_log(log_file)
            baseline_results[name] = results
            print(f"{name}: {len(results)} 个数据集")
    
    # 输出详细对比
    print("\n" + "="*80)
    print("最佳模型验证集指标 (Epoch {})".format(best_epoch))
    print("="*80)
    for dataset, metrics in best_metrics.items():
        print(f"\n{dataset}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 输出建议
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    print(f"使用模型: model_epoch_{best_epoch}.pth")
    print(f"路径: /T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/model_epoch_{best_epoch}.pth")
    print("\n注意：需要运行command_rel.md中的命令来获取测试集结果，然后与基准对比")

if __name__ == '__main__':
    main()



