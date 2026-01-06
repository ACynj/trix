#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面对比所有模型，找出最佳模型并与基准对比
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_train_log_detailed(log_file: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    """详细解析训练日志"""
    epoch_metrics = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_epoch = None
    in_eval = False
    dataset_results = []
    datasets = ['FB15k237', 'WN18RR', 'CoDExMedium']
    
    for i, line in enumerate(lines):
        # 检测epoch结束
        if 'Epoch' in line and 'end' in line:
            epoch_match = re.search(r'Epoch (\d+) end', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                in_eval = False
                dataset_results = []
        
        # 检测评估开始
        if 'Evaluate on valid' in line and current_epoch is not None:
            in_eval = True
            dataset_results = []
            continue
        
        # 解析指标（在评估部分）
        if in_eval and current_epoch is not None:
            metrics = {}
            if 'mr:' in line:
                try:
                    metrics['mr'] = float(re.search(r'mr:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'mrr:' in line:
                try:
                    metrics['mrr'] = float(re.search(r'mrr:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@1:' in line:
                try:
                    metrics['hits@1'] = float(re.search(r'hits@1:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@3:' in line:
                try:
                    metrics['hits@3'] = float(re.search(r'hits@3:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@10:' in line:
                try:
                    metrics['hits@10'] = float(re.search(r'hits@10:\s*([\d.]+)', line).group(1))
                except:
                    pass
            
            if metrics:
                dataset_results.append(metrics)
            
            # 如果收集到3个数据集的结果，保存
            if len(dataset_results) == 3:
                if current_epoch not in epoch_metrics:
                    epoch_metrics[current_epoch] = {}
                for idx, dataset_name in enumerate(datasets):
                    epoch_metrics[current_epoch][dataset_name] = dataset_results[idx]
                in_eval = False
    
    return epoch_metrics

def parse_eval_log_simple(log_file: str) -> Dict[str, Dict[str, float]]:
    """简单解析评估日志"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更简单的模式：找到数据集名称和对应的指标
    lines = content.split('\n')
    current_dataset = None
    in_eval = False
    
    for i, line in enumerate(lines):
        # 检测数据集
        if 'dataset' in line.lower() and '#' in line:
            # 提取数据集名称
            parts = line.split()
            for part in parts:
                if 'dataset' in part.lower():
                    idx = parts.index(part)
                    if idx > 0:
                        current_dataset = parts[idx-1]
                        break
            in_eval = False
        
        # 检测评估开始
        if 'Evaluate on test' in line:
            in_eval = True
            continue
        
        # 解析指标
        if in_eval and current_dataset:
            metrics = {}
            if 'mr:' in line:
                try:
                    metrics['mr'] = float(re.search(r'mr:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'mrr:' in line:
                try:
                    metrics['mrr'] = float(re.search(r'mrr:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@1:' in line:
                try:
                    metrics['hits@1'] = float(re.search(r'hits@1:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@3:' in line:
                try:
                    metrics['hits@3'] = float(re.search(r'hits@3:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@10:' in line:
                try:
                    metrics['hits@10'] = float(re.search(r'hits@10:\s*([\d.]+)', line).group(1))
                except:
                    pass
            
            if metrics:
                if current_dataset not in results:
                    results[current_dataset] = {}
                results[current_dataset].update(metrics)
    
    return results

def categorize_datasets(datasets: List[str]) -> Dict[str, List[str]]:
    """分类数据集"""
    categories = {
        'transductive': [],
        'inductive': [],
        'fb15k237': []
    }
    
    for dataset in datasets:
        if 'FB15k237_' in dataset or dataset == 'FB15k237_10' or dataset == 'FB15k237_20' or dataset == 'FB15k237_50':
            categories['fb15k237'].append(dataset)
        elif any(kw in dataset for kw in ['CoDEx', 'NELL995', 'DBpedia', 'ConceptNet', 'NELL23k', 'YAGO', 'Hetionet', 'WDsinger', 'AristoV4']):
            categories['transductive'].append(dataset)
        else:
            categories['inductive'].append(dataset)
    
    return categories

def calculate_averages(results: Dict[str, Dict[str, float]], 
                      datasets: List[str]) -> Dict[str, float]:
    """计算平均值"""
    metrics = defaultdict(list)
    
    for dataset in datasets:
        if dataset in results:
            for metric, value in results[dataset].items():
                metrics[metric].append(value)
    
    avg_metrics = {}
    for metric, values in metrics.items():
        if values:
            avg_metrics[metric] = sum(values) / len(values)
    
    return avg_metrics

def main():
    print("="*80)
    print("全面模型对比分析")
    print("="*80)
    
    # 1. 解析训练日志，找出最佳epoch
    train_log = '/T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/log.txt'
    print("\n1. 解析训练日志...")
    epoch_metrics = parse_train_log_detailed(train_log)
    print(f"   找到 {len(epoch_metrics)} 个epoch的验证结果")
    
    # 找出最佳epoch
    best_epoch = None
    best_mrr = -1
    
    print("\n   各epoch验证集平均MRR:")
    for epoch in sorted(epoch_metrics.keys()):
        metrics = epoch_metrics[epoch]
        mrr_values = [m.get('mrr', 0) for m in metrics.values() if 'mrr' in m]
        if mrr_values:
            avg_mrr = sum(mrr_values) / len(mrr_values)
            print(f"   Epoch {epoch:2d}: {avg_mrr:.4f}")
            if avg_mrr > best_mrr:
                best_mrr = avg_mrr
                best_epoch = epoch
    
    print(f"\n   最佳epoch: {best_epoch} (平均MRR: {best_mrr:.4f})")
    
    # 2. 解析基准日志
    print("\n2. 解析基准日志...")
    baseline_logs = {
        'inductive': '/T20030104/ynj/TRIX/run_commands_20251225_041027.log',
        'transductive': '/T20030104/ynj/TRIX/run_commands_20251225_065140.log',
        'fb15k237': '/T20030104/ynj/TRIX/run_commands_20251225_133504.log'
    }
    
    baseline_results = {}
    for name, log_file in baseline_logs.items():
        if Path(log_file).exists():
            results = parse_eval_log_simple(log_file)
            baseline_results[name] = results
            print(f"   {name}: {len(results)} 个数据集")
    
    # 3. 分类统计基准结果
    print("\n3. 基准结果分类统计:")
    all_baseline_datasets = []
    for results in baseline_results.values():
        all_baseline_datasets.extend(results.keys())
    
    baseline_categories = categorize_datasets(all_baseline_datasets)
    
    print("\n   基准 - Transductive数据集平均指标:")
    trans_avg = calculate_averages(baseline_results.get('transductive', {}), 
                                   baseline_categories['transductive'])
    for metric, value in sorted(trans_avg.items()):
        print(f"     {metric}: {value:.4f}")
    
    print("\n   基准 - Inductive数据集平均指标:")
    ind_avg = calculate_averages(baseline_results.get('inductive', {}), 
                                 baseline_categories['inductive'])
    for metric, value in sorted(ind_avg.items()):
        print(f"     {metric}: {value:.4f}")
    
    print("\n   基准 - FB15k237数据集平均指标:")
    fb_avg = calculate_averages(baseline_results.get('fb15k237', {}), 
                               baseline_categories['fb15k237'])
    for metric, value in sorted(fb_avg.items()):
        print(f"     {metric}: {value:.4f}")
    
    # 4. 输出建议
    print("\n" + "="*80)
    print("结论和建议")
    print("="*80)
    print(f"\n最佳模型: model_epoch_{best_epoch}.pth")
    print(f"模型路径: /T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/model_epoch_{best_epoch}.pth")
    print(f"\n验证集表现:")
    if best_epoch in epoch_metrics:
        for dataset, metrics in epoch_metrics[best_epoch].items():
            print(f"  {dataset}: MRR={metrics.get('mrr', 0):.4f}, Hits@1={metrics.get('hits@1', 0):.4f}")
    
    print("\n下一步:")
    print("1. 使用command_rel.md中的命令测试最佳模型")
    print("2. 将测试结果与基准对比")
    print("3. 计算三类数据集的平均提升")

if __name__ == '__main__':
    main()



