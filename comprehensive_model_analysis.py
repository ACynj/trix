#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面分析模型，找出最佳模型并与基准对比
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_train_log_complete(log_file: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    """完整解析训练日志"""
    epoch_metrics = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到所有epoch的评估结果
    # 模式：Epoch X end 后跟着 Evaluate on valid，然后三个数据集的结果
    epoch_sections = re.split(r'Epoch (\d+) end', content)
    
    for i in range(1, len(epoch_sections), 2):
        if i + 1 < len(epoch_sections):
            epoch = int(epoch_sections[i])
            section = epoch_sections[i + 1]
            
            # 找到Evaluate on valid部分
            eval_match = re.search(r'Evaluate on valid(.*?)(?=Epoch|\Z)', section, re.DOTALL)
            if eval_match:
                eval_text = eval_match.group(1)
                
                # 提取三个数据集的结果
                datasets = ['FB15k237', 'WN18RR', 'CoDExMedium']
                dataset_results = []
                
                # 找到所有指标行
                metric_lines = re.findall(r'(mr|mrr|hits@[0-9]+):\s*([\d.]+)', eval_text)
                
                # 每5个指标为一组（mr, mrr, hits@1, hits@3, hits@10）
                for j in range(0, len(metric_lines), 5):
                    if j + 4 < len(metric_lines):
                        metrics = {}
                        for k in range(5):
                            metric_name, metric_value = metric_lines[j + k]
                            if metric_name == 'mr':
                                metrics['mr'] = float(metric_value)
                            elif metric_name == 'mrr':
                                metrics['mrr'] = float(metric_value)
                            elif metric_name == 'hits@1':
                                metrics['hits@1'] = float(metric_value)
                            elif metric_name == 'hits@3':
                                metrics['hits@3'] = float(metric_value)
                            elif metric_name == 'hits@10':
                                metrics['hits@10'] = float(metric_value)
                        
                        if metrics:
                            dataset_results.append(metrics)
                
                # 保存结果
                if len(dataset_results) >= 3:
                    epoch_metrics[epoch] = {}
                    for idx, dataset_name in enumerate(datasets):
                        if idx < len(dataset_results):
                            epoch_metrics[epoch][dataset_name] = dataset_results[idx]
    
    return epoch_metrics

def parse_baseline_log(log_file: str) -> Dict[str, Dict[str, float]]:
    """解析基准日志文件"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_dataset = None
    in_eval = False
    current_metrics = {}
    
    for i, line in enumerate(lines):
        # 检测数据集名称（在dataset关键字前）
        if 'dataset' in line.lower():
            # 提取数据集名称
            parts = line.split()
            for j, part in enumerate(parts):
                if 'dataset' in part.lower() and j > 0:
                    # 数据集名称通常在dataset之前
                    dataset_name = parts[j-1]
                    # 移除可能的版本信息括号
                    if '(' in dataset_name:
                        dataset_name = dataset_name.split('(')[0]
                    current_dataset = dataset_name
                    in_eval = False
                    current_metrics = {}
                    break
        
        # 检测评估开始
        if 'Evaluate on test' in line:
            in_eval = True
            continue
        
        # 解析指标
        if in_eval and current_dataset:
            if 'mr:' in line:
                try:
                    current_metrics['mr'] = float(re.search(r'mr:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'mrr:' in line:
                try:
                    current_metrics['mrr'] = float(re.search(r'mrr:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@1:' in line:
                try:
                    current_metrics['hits@1'] = float(re.search(r'hits@1:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@3:' in line:
                try:
                    current_metrics['hits@3'] = float(re.search(r'hits@3:\s*([\d.]+)', line).group(1))
                except:
                    pass
            if 'hits@10:' in line:
                try:
                    current_metrics['hits@10'] = float(re.search(r'hits@10:\s*([\d.]+)', line).group(1))
                except:
                    pass
            
            # 如果收集到所有指标，保存结果
            if len(current_metrics) >= 5:
                results[current_dataset] = current_metrics.copy()
                in_eval = False
    
    return results

def categorize_datasets_complete(datasets: List[str]) -> Dict[str, List[str]]:
    """完整分类数据集"""
    categories = {
        'transductive': [],
        'inductive': [],
        'fb15k237': []
    }
    
    transductive_keywords = ['CoDEx', 'NELL995', 'DBpedia', 'ConceptNet', 'NELL23k', 
                            'YAGO', 'Hetionet', 'WDsinger', 'AristoV4']
    inductive_keywords = ['Inductive', 'ILPC2022', 'HM', 'Ingram', 'WikiTopics', 
                         'Metafam', 'FBNELL']
    
    for dataset in datasets:
        dataset_clean = dataset.split('(')[0].strip()
        if 'FB15k237_' in dataset or dataset_clean == 'FB15k237_10' or dataset_clean == 'FB15k237_20' or dataset_clean == 'FB15k237_50':
            categories['fb15k237'].append(dataset)
        elif any(kw in dataset for kw in transductive_keywords):
            categories['transductive'].append(dataset)
        elif any(kw in dataset for kw in inductive_keywords):
            categories['inductive'].append(dataset)
        else:
            # 默认归为inductive
            categories['inductive'].append(dataset)
    
    return categories

def calculate_category_averages(results: Dict[str, Dict[str, float]], 
                                category_datasets: List[str]) -> Dict[str, float]:
    """计算类别平均指标"""
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
    print("="*80)
    print("全面模型分析 - 找出最佳模型并与基准对比")
    print("="*80)
    
    # 1. 解析训练日志
    train_log = '/T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/log.txt'
    print("\n1. 解析训练日志...")
    
    epoch_metrics = parse_train_log_complete(train_log)
    print(f"   找到 {len(epoch_metrics)} 个epoch的验证结果")
    
    # 找出最佳epoch（基于平均MRR），排除epoch 14（已测试过）
    excluded_epochs = [14]  # 排除已测试的epoch
    best_epoch = None
    best_mrr = -1
    best_metrics = {}
    
    print("\n   各epoch验证集平均MRR (排除已测试的epoch 14):")
    epoch_mrrs = []
    for epoch in sorted(epoch_metrics.keys()):
        if epoch in excluded_epochs:
            continue  # 跳过已测试的epoch
        metrics = epoch_metrics[epoch]
        mrr_values = [m.get('mrr', 0) for m in metrics.values() if 'mrr' in m]
        if mrr_values:
            avg_mrr = sum(mrr_values) / len(mrr_values)
            epoch_mrrs.append((epoch, avg_mrr))
            print(f"   Epoch {epoch:2d}: {avg_mrr:.4f}")
            if avg_mrr > best_mrr:
                best_mrr = avg_mrr
                best_epoch = epoch
                best_metrics = metrics
    
    print(f"\n   最终选择的最佳epoch: {best_epoch} (平均MRR: {best_mrr:.4f})")
    
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
            results = parse_baseline_log(log_file)
            baseline_results[name] = results
            print(f"   {name}: {len(results)} 个数据集")
            # 显示前几个数据集名称用于验证
            if results:
                sample_datasets = list(results.keys())[:3]
                print(f"     示例数据集: {', '.join(sample_datasets)}")
    
    # 3. 分类统计基准结果
    print("\n3. 基准结果分类统计:")
    
    # 合并所有基准数据集
    all_baseline_datasets = []
    for name, results in baseline_results.items():
        all_baseline_datasets.extend(results.keys())
    
    baseline_categories = categorize_datasets_complete(all_baseline_datasets)
    
    print(f"\n   Transductive数据集 ({len(baseline_categories['transductive'])}个):")
    if baseline_categories['transductive']:
        trans_results = {}
        for name, results in baseline_results.items():
            for dataset in baseline_categories['transductive']:
                if dataset in results:
                    trans_results[dataset] = results[dataset]
        trans_avg = calculate_category_averages(trans_results, baseline_categories['transductive'])
        for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
            if metric in trans_avg:
                print(f"     {metric}: {trans_avg[metric]:.4f}")
    
    print(f"\n   Inductive数据集 ({len(baseline_categories['inductive'])}个):")
    if baseline_categories['inductive']:
        ind_results = {}
        for name, results in baseline_results.items():
            for dataset in baseline_categories['inductive']:
                if dataset in results:
                    ind_results[dataset] = results[dataset]
        ind_avg = calculate_category_averages(ind_results, baseline_categories['inductive'])
        for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
            if metric in ind_avg:
                print(f"     {metric}: {ind_avg[metric]:.4f}")
    
    print(f"\n   FB15k237数据集 ({len(baseline_categories['fb15k237'])}个):")
    if baseline_categories['fb15k237']:
        fb_results = {}
        for name, results in baseline_results.items():
            for dataset in baseline_categories['fb15k237']:
                if dataset in results:
                    fb_results[dataset] = results[dataset]
        fb_avg = calculate_category_averages(fb_results, baseline_categories['fb15k237'])
        for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
            if metric in fb_avg:
                print(f"     {metric}: {fb_avg[metric]:.4f}")
    
    # 4. 输出最佳模型信息
    print("\n" + "="*80)
    print("最佳模型信息")
    print("="*80)
    print(f"\n模型: model_epoch_{best_epoch}.pth")
    print(f"路径: /T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/model_epoch_{best_epoch}.pth")
    print(f"\n验证集表现:")
    if best_epoch in epoch_metrics:
        for dataset, metrics in epoch_metrics[best_epoch].items():
            print(f"  {dataset}:")
            for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric in metrics:
                    print(f"    {metric}: {metrics[metric]:.4f}")
    
    print("\n" + "="*80)
    print("下一步操作")
    print("="*80)
    print("\n1. 使用command_rel.md中的命令测试最佳模型")
    print("2. 将测试结果保存到日志文件")
    print("3. 运行对比脚本计算三类数据集的平均提升")
    print(f"\n建议测试命令（修改checkpoint路径为最佳模型）:")
    print(f"   python ./src/run_relation.py -c ./config/run_relation_inductive_mech.yaml \\")
    print(f"     --dataset FB15k237Inductive --version v1 \\")
    print(f"     --ckpt /T20030104/ynj/TRIX/output_rel/TRIXLatentMechanism/JointDataset/2025-12-31-17-00-51/model_epoch_{best_epoch}.pth \\")
    print(f"     --gpus [0] --epochs 0 --bpe null")

if __name__ == '__main__':
    main()

