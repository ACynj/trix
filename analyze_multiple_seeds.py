#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析多个随机种子的日志文件，计算平均值和方差，与基准文件对比
"""

import re
import numpy as np
from collections import defaultdict
from pathlib import Path

def parse_log_file(log_file):
    """解析日志文件，提取数据集和指标"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配数据集名称和指标
    # 模式1: dataset name (version) dataset 或 dataset dataset
    dataset_pattern = r'(\d{2}:\d{2}:\d{2})\s+(\w+(?:Inductive|Ingram|TopicsMT\d)?)\(?([^)]*)\)?\s+dataset'
    
    # 匹配指标行: "mr: 1.234", "mrr: 0.567", "hits@1: 0.123" 等
    metric_pattern = r'(\d{2}:\d{2}:\d{2})\s+(mr|mrr|hits@\d+):\s+([\d.]+)'
    
    # 找到所有数据集名称
    dataset_matches = list(re.finditer(dataset_pattern, content))
    metric_matches = list(re.finditer(metric_pattern, content))
    
    # 按时间戳组织数据
    current_dataset = None
    current_time = None
    dataset_metrics = {}
    
    for metric_match in metric_matches:
        time_str = metric_match.group(1)
        metric_name = metric_match.group(2)
        metric_value = float(metric_match.group(3))
        
        # 找到这个时间戳之前最近的数据集
        for i, ds_match in enumerate(dataset_matches):
            ds_time = ds_match.group(1)
            if ds_time <= time_str:
                # 检查是否在同一个命令块中（时间接近）
                if i < len(dataset_matches) - 1:
                    next_ds_time = dataset_matches[i+1].group(1)
                    if time_str < next_ds_time:
                        dataset_name = ds_match.group(2)
                        version = ds_match.group(3).strip()
                        
                        if version:
                            full_name = f"{dataset_name}({version})"
                        else:
                            full_name = dataset_name
                        
                        if full_name not in dataset_metrics:
                            dataset_metrics[full_name] = {}
                        dataset_metrics[full_name][metric_name] = metric_value
                        break
    
    return dataset_metrics

def normalize_dataset_name(name):
    """标准化数据集名称"""
    # 处理各种格式
    if '(' in name:
        return name
    # 如果没有版本号，尝试从日志中推断
    return name

def calculate_stats(values):
    """计算平均值和标准差"""
    if not values:
        return None, None
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std

def get_dataset_category(dataset_name):
    """根据数据集名称判断类别"""
    transductive = ['CoDExSmall', 'CoDExLarge', 'NELL995', 'DBpedia100k', 
                    'ConceptNet100k', 'NELL23k', 'YAGO310', 'Hetionet', 
                    'WDsinger', 'AristoV4', 'FB15k237_10', 'FB15k237_20', 'FB15k237_50']
    
    inductive_e = ['FB15k237Inductive', 'WN18RRInductive', 'NELLInductive', 
                   'ILPC2022', 'HM']
    
    if any(ds in dataset_name for ds in transductive):
        return 'Transductive'
    elif any(ds in dataset_name for ds in inductive_e):
        return 'Inductive(e)'
    else:
        return 'Inductive(e,r)'

def main():
    # 文件路径
    baseline_file = '/T20030104/ynj/TRIX/inference_rel.log'
    log_files = [
        '/T20030104/ynj/TRIX/run_commands_20260105_113925.log',
        '/T20030104/ynj/TRIX/run_commands_20260104_052013.log',
        '/T20030104/ynj/TRIX/run_commands_20260101_035905.log'
    ]
    
    print("正在解析基准文件...")
    baseline_results = parse_log_file(baseline_file)
    
    print("正在解析三个日志文件...")
    all_results = []
    for log_file in log_files:
        results = parse_log_file(log_file)
        all_results.append(results)
        print(f"  已解析: {Path(log_file).name}, 找到 {len(results)} 个数据集")
    
    # 收集所有数据集名称
    all_datasets = set(baseline_results.keys())
    for results in all_results:
        all_datasets.update(results.keys())
    
    all_datasets = sorted(all_datasets)
    
    # 计算每个数据集的统计信息
    comparison_data = []
    category_data = defaultdict(lambda: {'baseline': [], 'mean': [], 'std': []})
    
    for dataset in all_datasets:
        baseline_metrics = baseline_results.get(dataset, {})
        category = get_dataset_category(dataset)
        
        for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
            baseline_value = baseline_metrics.get(metric)
            if baseline_value is None:
                continue
            
            # 收集三次运行的值
            values = []
            for results in all_results:
                if dataset in results and metric in results[dataset]:
                    values.append(results[dataset][metric])
            
            if not values:
                continue
            
            mean_val, std_val = calculate_stats(values)
            
            diff = mean_val - baseline_value
            rel_change = (diff / baseline_value * 100) if baseline_value != 0 else 0
            
            comparison_data.append({
                'dataset': dataset,
                'metric': metric,
                'baseline': baseline_value,
                'mean': mean_val,
                'std': std_val,
                'diff': diff,
                'rel_change': rel_change,
                'category': category
            })
            
            # 按类别收集数据
            category_data[category]['baseline'].append(baseline_value)
            category_data[category]['mean'].append(mean_val)
            category_data[category]['std'].append(std_val)
    
    # 生成报告
    output_file = '/T20030104/ynj/TRIX/comparison_report_multiple_seeds.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 多随机种子结果对比分析报告\n")
        f.write(f"**基准文件**: inference_rel.log\n")
        f.write(f"**对比文件**: \n")
        for log_file in log_files:
            f.write(f"- {Path(log_file).name}\n")
        f.write("\n")
        f.write("---\n")
        f.write("## 1. 每个数据集的详细对比（平均值±标准差）\n")
        f.write("| 数据集 | 指标 | inference_rel.log | 三次运行平均值 | 标准差 | 差异 | 相对提升 |\n")
        f.write("|--------|------|-------------------|----------------|--------|------|----------|\n")
        
        for item in comparison_data:
            f.write(f"| {item['dataset']} | {item['metric']} | {item['baseline']:.4f} | "
                   f"{item['mean']:.4f} | {item['std']:.4f} | "
                   f"{item['diff']:+.4f} | {item['rel_change']:+.2f}% |\n")
        
        f.write("\n---\n")
        f.write("## 2. 按类别进行平均对比\n")
        
        for category in ['Transductive', 'Inductive(e)', 'Inductive(e,r)']:
            if category not in category_data:
                continue
            
            baseline_avg = np.mean(category_data[category]['baseline'])
            mean_avg = np.mean(category_data[category]['mean'])
            std_avg = np.mean(category_data[category]['std'])
            diff = mean_avg - baseline_avg
            rel_change = (diff / baseline_avg * 100) if baseline_avg != 0 else 0
            
            # 计算该类别下的数据集数量（每个指标算一个）
            count = len(category_data[category]['baseline'])
            
            f.write(f"### {category} ({count//4}个数据集)\n")
            f.write("| 指标 | inference_rel.log | 三次运行平均值 | 平均标准差 | 差异 | 相对提升 |\n")
            f.write("|------|-------------------|----------------|------------|------|----------|\n")
            
            # 按指标分组计算
            for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
                metric_baseline = []
                metric_mean = []
                metric_std = []
                
                for item in comparison_data:
                    if item['category'] == category and item['metric'] == metric:
                        metric_baseline.append(item['baseline'])
                        metric_mean.append(item['mean'])
                        metric_std.append(item['std'])
                
                if metric_baseline:
                    bl_avg = np.mean(metric_baseline)
                    mn_avg = np.mean(metric_mean)
                    std_avg = np.mean(metric_std)
                    diff = mn_avg - bl_avg
                    rel_change = (diff / bl_avg * 100) if bl_avg != 0 else 0
                    
                    f.write(f"| {metric} | {bl_avg:.4f} | {mn_avg:.4f} | {std_avg:.4f} | "
                           f"{diff:+.4f} | {rel_change:+.2f}% |\n")
        
        f.write("\n---\n")
        f.write("## 3. 全部数据集平均对比\n")
        f.write("| 指标 | inference_rel.log | 三次运行平均值 | 平均标准差 | 差异 | 相对提升 |\n")
        f.write("|------|-------------------|----------------|------------|------|----------|\n")
        
        for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
            metric_baseline = []
            metric_mean = []
            metric_std = []
            
            for item in comparison_data:
                if item['metric'] == metric:
                    metric_baseline.append(item['baseline'])
                    metric_mean.append(item['mean'])
                    metric_std.append(item['std'])
            
            if metric_baseline:
                bl_avg = np.mean(metric_baseline)
                mn_avg = np.mean(metric_mean)
                std_avg = np.mean(metric_std)
                diff = mn_avg - bl_avg
                rel_change = (diff / bl_avg * 100) if bl_avg != 0 else 0
                
                f.write(f"| {metric} | {bl_avg:.4f} | {mn_avg:.4f} | {std_avg:.4f} | "
                       f"{diff:+.4f} | {rel_change:+.2f}% |\n")
        
        f.write("\n---\n")
        f.write("## 4. 方差分析\n")
        f.write("### 方差最大的数据集（按标准差排序）\n")
        f.write("| 数据集 | 指标 | 平均值 | 标准差 | 变异系数(%) |\n")
        f.write("|--------|------|--------|--------|-------------|\n")
        
        # 计算变异系数 (CV = std/mean * 100)
        variance_data = []
        for item in comparison_data:
            cv = (item['std'] / item['mean'] * 100) if item['mean'] != 0 else 0
            variance_data.append({
                'dataset': item['dataset'],
                'metric': item['metric'],
                'mean': item['mean'],
                'std': item['std'],
                'cv': cv
            })
        
        variance_data.sort(key=lambda x: x['std'], reverse=True)
        for item in variance_data[:20]:  # 显示前20个
            f.write(f"| {item['dataset']} | {item['metric']} | {item['mean']:.4f} | "
                   f"{item['std']:.4f} | {item['cv']:.2f}% |\n")
    
    print(f"\n报告已生成: {output_file}")
    print(f"共分析了 {len(all_datasets)} 个数据集")

if __name__ == "__main__":
    main()

