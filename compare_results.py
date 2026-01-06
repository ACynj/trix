#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比分析两次实验结果
"""
import re
from collections import defaultdict

def parse_log_file(filepath):
    """解析日志文件，提取数据集和指标"""
    results = {}
    current_dataset = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取数据集名称
            if '执行命令' in line:
                # 提取dataset参数
                match = re.search(r'--dataset\s+(\w+)(?:\s+--version\s+(\S+))?', line)
                if match:
                    dataset = match.group(1)
                    version = match.group(2) if match.group(2) else None
                    if version and version != 'null':
                        current_dataset = f"{dataset}_{version}"
                    else:
                        current_dataset = dataset
            
            # 提取指标
            if current_dataset and ':' in line:
                for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                    pattern = rf'{metric}:\s*([\d.]+)'
                    match = re.search(pattern, line)
                    if match:
                        if current_dataset not in results:
                            results[current_dataset] = {}
                        results[current_dataset][metric] = float(match.group(1))
    
    return results

def classify_dataset(dataset_name):
    """将数据集分类为三类：Transductive, Inductive, Ingram"""
    if 'Inductive' in dataset_name or 'ILPC2022' in dataset_name or 'HM' in dataset_name:
        return 'Inductive'
    elif 'Ingram' in dataset_name or 'WikiTopics' in dataset_name or 'Metafam' in dataset_name or 'FBNELL' in dataset_name:
        return 'Ingram'
    else:
        return 'Transductive'

def calculate_averages(results):
    """计算三类数据集的平均值"""
    categories = defaultdict(lambda: defaultdict(list))
    
    for dataset, metrics in results.items():
        category = classify_dataset(dataset)
        for metric, value in metrics.items():
            categories[category][metric].append(value)
    
    averages = {}
    for category, metrics_dict in categories.items():
        averages[category] = {}
        for metric, values in metrics_dict.items():
            averages[category][metric] = sum(values) / len(values) if values else 0
    
    return averages

def compare_results(exp1_results, exp2_results):
    """对比两次实验结果"""
    # 合并所有数据集
    all_datasets = set(exp1_results.keys()) | set(exp2_results.keys())
    
    comparison = {}
    for dataset in all_datasets:
        if dataset in exp1_results and dataset in exp2_results:
            comparison[dataset] = {}
            for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric in exp1_results[dataset] and metric in exp2_results[dataset]:
                    val1 = exp1_results[dataset][metric]
                    val2 = exp2_results[dataset][metric]
                    if metric == 'mr':  # MR越小越好
                        change = val1 - val2  # 正数表示提升
                        change_pct = (change / val1 * 100) if val1 > 0 else 0
                    else:  # 其他指标越大越好
                        change = val2 - val1  # 正数表示提升
                        change_pct = (change / val1 * 100) if val1 > 0 else 0
                    comparison[dataset][metric] = {
                        'exp1': val1,
                        'exp2': val2,
                        'change': change,
                        'change_pct': change_pct
                    }
    
    return comparison

def main():
    # 解析第一次实验（三个日志文件）
    exp1_results = {}
    for log_file in [
        'run_commands_20251225_041027.log',
        'run_commands_20251225_065140.log',
        'run_commands_20251225_133504.log'
    ]:
        results = parse_log_file(log_file)
        exp1_results.update(results)
    
    # 解析第二次实验
    exp2_results = parse_log_file('run_commands_20260101_035905.log')
    
    # 计算平均值
    exp1_avg = calculate_averages(exp1_results)
    exp2_avg = calculate_averages(exp2_results)
    
    # 对比分析
    comparison = compare_results(exp1_results, exp2_results)
    
    # 打印结果
    print("=" * 80)
    print("实验结果对比分析")
    print("=" * 80)
    print("\n第一次实验（2025-12-25）")
    print("-" * 80)
    for category in ['Transductive', 'Inductive', 'Ingram']:
        if category in exp1_avg:
            print(f"\n{category} 数据集平均结果:")
            for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric in exp1_avg[category]:
                    print(f"  {metric:10s}: {exp1_avg[category][metric]:.6f}")
    
    print("\n\n第二次实验（2026-01-01）")
    print("-" * 80)
    for category in ['Transductive', 'Inductive', 'Ingram']:
        if category in exp2_avg:
            print(f"\n{category} 数据集平均结果:")
            for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric in exp2_avg[category]:
                    print(f"  {metric:10s}: {exp2_avg[category][metric]:.6f}")
    
    print("\n\n三类数据集平均对比（提升/下降）")
    print("=" * 80)
    for category in ['Transductive', 'Inductive', 'Ingram']:
        if category in exp1_avg and category in exp2_avg:
            print(f"\n{category} 数据集:")
            print(f"{'指标':<12} {'第一次':<12} {'第二次':<12} {'变化':<12} {'变化率':<12}")
            print("-" * 60)
            for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric in exp1_avg[category] and metric in exp2_avg[category]:
                    val1 = exp1_avg[category][metric]
                    val2 = exp2_avg[category][metric]
                    if metric == 'mr':
                        change = val1 - val2
                        change_pct = (change / val1 * 100) if val1 > 0 else 0
                        change_str = f"{change:+.6f}" if change >= 0 else f"{change:.6f}"
                        change_pct_str = f"{change_pct:+.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
                    else:
                        change = val2 - val1
                        change_pct = (change / val1 * 100) if val1 > 0 else 0
                        change_str = f"{change:+.6f}" if change >= 0 else f"{change:.6f}"
                        change_pct_str = f"{change_pct:+.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
                    print(f"{metric:<12} {val1:<12.6f} {val2:<12.6f} {change_str:<12} {change_pct_str:<12}")
    
    # 保存详细对比结果到文件
    with open('comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("实验结果详细对比分析\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("第一次实验（2025-12-25）平均值\n")
        f.write("-" * 80 + "\n")
        for category in ['Transductive', 'Inductive', 'Ingram']:
            if category in exp1_avg:
                f.write(f"\n{category} 数据集平均结果:\n")
                for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                    if metric in exp1_avg[category]:
                        f.write(f"  {metric:10s}: {exp1_avg[category][metric]:.6f}\n")
        
        f.write("\n\n第二次实验（2026-01-01）平均值\n")
        f.write("-" * 80 + "\n")
        for category in ['Transductive', 'Inductive', 'Ingram']:
            if category in exp2_avg:
                f.write(f"\n{category} 数据集平均结果:\n")
                for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                    if metric in exp2_avg[category]:
                        f.write(f"  {metric:10s}: {exp2_avg[category][metric]:.6f}\n")
        
        f.write("\n\n三类数据集平均对比（提升/下降）\n")
        f.write("=" * 80 + "\n")
        for category in ['Transductive', 'Inductive', 'Ingram']:
            if category in exp1_avg and category in exp2_avg:
                f.write(f"\n{category} 数据集:\n")
                f.write(f"{'指标':<12} {'第一次':<12} {'第二次':<12} {'变化':<12} {'变化率':<12}\n")
                f.write("-" * 60 + "\n")
                for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                    if metric in exp1_avg[category] and metric in exp2_avg[category]:
                        val1 = exp1_avg[category][metric]
                        val2 = exp2_avg[category][metric]
                        if metric == 'mr':
                            change = val1 - val2
                            change_pct = (change / val1 * 100) if val1 > 0 else 0
                        else:
                            change = val2 - val1
                            change_pct = (change / val1 * 100) if val1 > 0 else 0
                        f.write(f"{metric:<12} {val1:<12.6f} {val2:<12.6f} {change:+.6f} {change_pct:+.2f}%\n")
    
    print("\n\n详细对比报告已保存到 comparison_report.txt")

if __name__ == '__main__':
    main()



