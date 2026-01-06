#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练日志，找出最佳模型并进行对比
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import subprocess
import sys

def parse_log_file(log_file: str) -> Dict[int, Dict[str, float]]:
    """解析训练日志，提取每个epoch的验证集指标"""
    epoch_metrics = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到所有epoch的评估结果
    # 格式：Evaluate on valid 后跟着三个数据集的结果
    epoch_pattern = r'Epoch (\d+) end'
    evaluate_pattern = r'Evaluate on valid\n(.*?)(?=\n>>>>>>>>>>>>>>>>>>>>>>>>>>>|\nEpoch|\Z)'
    
    epochs = re.findall(epoch_pattern, content)
    evaluates = re.findall(evaluate_pattern, content, re.DOTALL)
    
    current_epoch = None
    for i, epoch_str in enumerate(epochs):
        epoch = int(epoch_str)
        if i < len(evaluates):
            eval_text = evaluates[i]
            # 解析三个数据集的结果
            datasets_results = []
            lines = eval_text.strip().split('\n')
            current_dataset = []
            for line in lines:
                if line.strip() and not line.startswith('>>>>>>>>>>>>>>>>>>>>>>>>>>>'):
                    if 'mr:' in line or 'mrr:' in line or 'hits@' in line:
                        current_dataset.append(line)
                    elif current_dataset:
                        datasets_results.append(current_dataset)
                        current_dataset = []
            if current_dataset:
                datasets_results.append(current_dataset)
            
            # 解析指标
            metrics = {}
            for dataset_result in datasets_results:
                dataset_metrics = {}
                for line in dataset_result:
                    if 'mr:' in line:
                        dataset_metrics['mr'] = float(re.search(r'mr:\s*([\d.]+)', line).group(1))
                    elif 'mrr:' in line:
                        dataset_metrics['mrr'] = float(re.search(r'mrr:\s*([\d.]+)', line).group(1))
                    elif 'hits@1:' in line:
                        dataset_metrics['hits@1'] = float(re.search(r'hits@1:\s*([\d.]+)', line).group(1))
                    elif 'hits@3:' in line:
                        dataset_metrics['hits@3'] = float(re.search(r'hits@3:\s*([\d.]+)', line).group(1))
                    elif 'hits@10:' in line:
                        dataset_metrics['hits@10'] = float(re.search(r'hits@10:\s*([\d.]+)', line).group(1))
                
                if dataset_metrics:
                    # 三个数据集：FB15k237, WN18RR, CoDExMedium
                    if len(metrics) == 0:
                        metrics['FB15k237'] = dataset_metrics
                    elif len(metrics) == 1:
                        metrics['WN18RR'] = dataset_metrics
                    elif len(metrics) == 2:
                        metrics['CoDExMedium'] = dataset_metrics
            
            if metrics:
                epoch_metrics[epoch] = metrics
    
    return epoch_metrics

def parse_evaluation_log(log_file: str) -> Dict[str, Dict[str, float]]:
    """解析评估日志文件，提取测试结果"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到每个数据集的评估结果
    # 格式：数据集名称后跟着指标
    dataset_pattern = r'(\w+(?:Inductive|Ingram)?(?:\([^)]+\))?)\s+dataset'
    metric_pattern = r'(mr|mrr|hits@[0-9]+):\s*([\d.]+)'
    
    lines = content.split('\n')
    current_dataset = None
    
    for line in lines:
        # 检查是否是数据集声明
        dataset_match = re.search(dataset_pattern, line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            results[current_dataset] = {}
        
        # 检查是否是评估结果（在Evaluate on test之后）
        if current_dataset and 'Evaluate on test' in line:
            # 读取接下来的几行
            continue
        
        # 解析指标
        if current_dataset:
            metric_match = re.search(metric_pattern, line)
            if metric_match:
                metric_name = metric_match.group(1)
                metric_value = float(metric_match.group(2))
                results[current_dataset][metric_name] = metric_value
    
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
        if any(kw in dataset for kw in transductive_keywords):
            categories['transductive'].append(dataset)
        elif any(kw in dataset for kw in inductive_keywords):
            categories['inductive'].append(dataset)
        else:
            categories['other'].append(dataset)
    
    return categories

def calculate_average_metrics(results: Dict[str, Dict[str, float]], 
                             datasets: List[str]) -> Dict[str, float]:
    """计算数据集的平均指标"""
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

def compare_models(epoch_results: Dict[int, Dict[str, Dict[str, float]]],
                   baseline_results: Dict[str, Dict[str, float]]) -> Tuple[int, Dict]:
    """比较所有epoch，找出最佳模型"""
    best_epoch = None
    best_score = -1
    best_metrics = {}
    
    for epoch, metrics in epoch_results.items():
        # 计算平均MRR作为评分
        mrr_values = []
        for dataset_metrics in metrics.values():
            if 'mrr' in dataset_metrics:
                mrr_values.append(dataset_metrics['mrr'])
        
        if mrr_values:
            avg_mrr = sum(mrr_values) / len(mrr_values)
            if avg_mrr > best_score:
                best_score = avg_mrr
                best_epoch = epoch
                best_metrics = metrics
    
    return best_epoch, best_metrics

def main():
    # 训练日志路径
    train_log = '/T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/log.txt'
    
    # 解析训练日志
    print("正在解析训练日志...")
    epoch_metrics = parse_log_file(train_log)
    
    print(f"找到 {len(epoch_metrics)} 个epoch的验证结果")
    
    # 找出最佳epoch（基于验证集MRR）
    best_epoch = None
    best_mrr = -1
    
    for epoch, metrics in epoch_metrics.items():
        mrr_values = []
        for dataset_metrics in metrics.values():
            if 'mrr' in dataset_metrics:
                mrr_values.append(dataset_metrics['mrr'])
        
        if mrr_values:
            avg_mrr = sum(mrr_values) / len(mrr_values)
            print(f"Epoch {epoch}: 平均MRR = {avg_mrr:.4f}")
            if avg_mrr > best_mrr:
                best_mrr = avg_mrr
                best_epoch = epoch
    
    print(f"\n最佳epoch: {best_epoch} (平均MRR: {best_mrr:.4f})")
    
    # 读取基准日志文件
    baseline_logs = [
        '/T20030104/ynj/TRIX/run_commands_20251225_041027.log',
        '/T20030104/ynj/TRIX/run_commands_20251225_065140.log',
        '/T20030104/ynj/TRIX/run_commands_20251225_133504.log'
    ]
    
    print("\n正在解析基准日志...")
    baseline_results = {}
    for log_file in baseline_logs:
        if Path(log_file).exists():
            results = parse_evaluation_log(log_file)
            baseline_results[Path(log_file).stem] = results
            print(f"已解析 {Path(log_file).name}: {len(results)} 个数据集")
    
    # 输出结果
    print("\n" + "="*80)
    print("分析结果")
    print("="*80)
    print(f"最佳模型: model_epoch_{best_epoch}.pth")
    print(f"验证集平均MRR: {best_mrr:.4f}")
    
    if best_epoch in epoch_metrics:
        print("\n验证集详细指标:")
        for dataset, metrics in epoch_metrics[best_epoch].items():
            print(f"  {dataset}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")

if __name__ == '__main__':
    main()



