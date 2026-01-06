#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Mann-Whitney U Test 对对比报告进行显著性检验分析
"""

import re
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

def parse_comparison_report(filepath: str) -> Dict[str, Dict[str, List[float]]]:
    """解析对比报告，提取数据"""
    data = {
        'mrr': {'baseline': [], 'comparison': []},
        'hits@1': {'baseline': [], 'comparison': []},
        'hits@3': {'baseline': [], 'comparison': []},
        'hits@10': {'baseline': [], 'comparison': []}
    }
    
    # 按类别分类
    categories = {
        'transductive': {
            'mrr': {'baseline': [], 'comparison': []},
            'hits@1': {'baseline': [], 'comparison': []},
            'hits@3': {'baseline': [], 'comparison': []},
            'hits@10': {'baseline': [], 'comparison': []}
        },
        'inductive_e': {
            'mrr': {'baseline': [], 'comparison': []},
            'hits@1': {'baseline': [], 'comparison': []},
            'hits@3': {'baseline': [], 'comparison': []},
            'hits@10': {'baseline': [], 'comparison': []}
        },
        'inductive_er': {
            'mrr': {'baseline': [], 'comparison': []},
            'hits@1': {'baseline': [], 'comparison': []},
            'hits@3': {'baseline': [], 'comparison': []},
            'hits@10': {'baseline': [], 'comparison': []}
        }
    }
    
    # 定义数据集分类
    transductive_datasets = [
        'CoDExSmall', 'CoDExLarge', 'NELL995', 'DBpedia100k', 'ConceptNet100k',
        'NELL23k', 'YAGO310', 'Hetionet', 'WDsinger', 'AristoV4',
        'FB15k237_10', 'FB15k237_20', 'FB15k237_50'
    ]
    
    inductive_e_datasets = [
        'FB15k237Inductive', 'WN18RRInductive', 'NELLInductive',
        'ILPC2022', 'HM'
    ]
    
    inductive_er_datasets = [
        'FBIngram', 'WKIngram', 'NLIngram',
        'WikiTopicsMT1', 'WikiTopicsMT2', 'WikiTopicsMT3', 'WikiTopicsMT4',
        'Metafam', 'FBNELL'
    ]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析每个数据集的数据
    for line in lines:
        # 匹配表格行: | 数据集 | 指标 | 值1 | 值2 | 差异 |
        match = re.match(r'^\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([+-]?[\d.]+)\s+\|', line)
        if match:
            dataset = match.group(1).strip()
            metric = match.group(2).strip()
            val1 = float(match.group(3))
            val2 = float(match.group(4))
            
            # 添加到总体数据
            if metric in data:
                data[metric]['baseline'].append(val1)
                data[metric]['comparison'].append(val2)
            
            # 按类别分类
            category = None
            if any(ds in dataset for ds in transductive_datasets):
                category = 'transductive'
            elif any(ds in dataset for ds in inductive_e_datasets):
                category = 'inductive_e'
            elif any(ds in dataset for ds in inductive_er_datasets):
                category = 'inductive_er'
            
            if category and metric in categories[category]:
                categories[category][metric]['baseline'].append(val1)
                categories[category][metric]['comparison'].append(val2)
    
    return data, categories

def mannwhitney_test(data1: List[float], data2: List[float]) -> Tuple[float, float, str]:
    """执行 Mann-Whitney U Test"""
    if len(data1) == 0 or len(data2) == 0:
        return 0.0, 1.0, "N/A"
    
    # 执行 Mann-Whitney U Test
    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    # 判断显著性
    if p_value < 0.001:
        sig_level = "***"
        sig_desc = "高度显著 (p < 0.001)"
    elif p_value < 0.01:
        sig_level = "**"
        sig_desc = "高度显著 (p < 0.01)"
    elif p_value < 0.05:
        sig_level = "*"
        sig_desc = "显著 (p < 0.05)"
    else:
        sig_level = ""
        sig_desc = "不显著 (p >= 0.05)"
    
    return statistic, p_value, sig_desc

def calculate_effect_size(data1: List[float], data2: List[float]) -> float:
    """计算效应量 (Cohen's d)"""
    if len(data1) == 0 or len(data2) == 0:
        return 0.0
    
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    std1 = np.std(data1, ddof=1)
    std2 = np.std(data2, ddof=1)
    
    # 合并标准差
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (mean2 - mean1) / pooled_std
    return cohens_d

def interpret_effect_size(d: float) -> str:
    """解释效应量"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "微小"
    elif abs_d < 0.5:
        return "小"
    elif abs_d < 0.8:
        return "中等"
    else:
        return "大"

def generate_report(data: Dict, categories: Dict) -> str:
    """生成统计检验报告"""
    report = []
    report.append("# Mann-Whitney U Test 显著性检验分析报告\n")
    report.append("**说明**: Mann-Whitney U Test 用于检验两个独立样本的分布是否存在显著差异。\n")
    report.append("**注意**: 虽然这是配对数据，但按照要求使用 Mann-Whitney U Test 进行分析。\n")
    report.append("对于配对数据，Wilcoxon signed-rank test 可能更合适。\n\n")
    report.append("---\n\n")
    
    metrics = ['mrr', 'hits@1', 'hits@3', 'hits@10']
    metric_names = {
        'mrr': 'MRR (平均倒数排名)',
        'hits@1': 'Hits@1',
        'hits@3': 'Hits@3',
        'hits@10': 'Hits@10'
    }
    
    # 1. 全部数据集分析
    report.append("## 1. 全部数据集分析 (54个数据集)\n\n")
    report.append("| 指标 | 基准均值 | 对比均值 | U统计量 | p值 | 显著性 | 效应量(Cohen's d) | 效应量解释 |\n")
    report.append("|------|----------|----------|---------|-----|--------|------------------|------------|\n")
    
    for metric in metrics:
        baseline = data[metric]['baseline']
        comparison = data[metric]['comparison']
        
        if len(baseline) > 0 and len(comparison) > 0:
            mean1 = np.mean(baseline)
            mean2 = np.mean(comparison)
            statistic, p_value, sig_desc = mannwhitney_test(baseline, comparison)
            effect_size = calculate_effect_size(baseline, comparison)
            effect_desc = interpret_effect_size(effect_size)
            
            report.append(f"| {metric_names[metric]} | {mean1:.4f} | {mean2:.4f} | {statistic:.2f} | {p_value:.4f} | {sig_desc} | {effect_size:+.3f} | {effect_desc} |\n")
    
    report.append("\n---\n\n")
    
    # 2. 按类别分析
    category_names = {
        'transductive': 'Transductive (13个数据集)',
        'inductive_e': 'Inductive(e) (18个数据集)',
        'inductive_er': 'Inductive(e,r) (23个数据集)'
    }
    
    report.append("## 2. 按类别进行显著性检验\n\n")
    
    for category_key, category_name in category_names.items():
        report.append(f"### {category_name}\n\n")
        report.append("| 指标 | 基准均值 | 对比均值 | U统计量 | p值 | 显著性 | 效应量(Cohen's d) | 效应量解释 |\n")
        report.append("|------|----------|----------|---------|-----|--------|------------------|------------|\n")
        
        for metric in metrics:
            baseline = categories[category_key][metric]['baseline']
            comparison = categories[category_key][metric]['comparison']
            
            if len(baseline) > 0 and len(comparison) > 0:
                mean1 = np.mean(baseline)
                mean2 = np.mean(comparison)
                statistic, p_value, sig_desc = mannwhitney_test(baseline, comparison)
                effect_size = calculate_effect_size(baseline, comparison)
                effect_desc = interpret_effect_size(effect_size)
                
                report.append(f"| {metric_names[metric]} | {mean1:.4f} | {mean2:.4f} | {statistic:.2f} | {p_value:.4f} | {sig_desc} | {effect_size:+.3f} | {effect_desc} |\n")
        
        report.append("\n")
    
    report.append("---\n\n")
    
    # 3. 总结
    report.append("## 3. 总结\n\n")
    report.append("### 显著性水平说明\n")
    report.append("- ***: p < 0.001 (高度显著)\n")
    report.append("- **: p < 0.01 (高度显著)\n")
    report.append("- *: p < 0.05 (显著)\n")
    report.append("- 无标记: p >= 0.05 (不显著)\n\n")
    
    report.append("### 效应量说明 (Cohen's d)\n")
    report.append("- |d| < 0.2: 微小效应\n")
    report.append("- 0.2 ≤ |d| < 0.5: 小效应\n")
    report.append("- 0.5 ≤ |d| < 0.8: 中等效应\n")
    report.append("- |d| ≥ 0.8: 大效应\n\n")
    
    report.append("### 注意事项\n")
    report.append("1. Mann-Whitney U Test 适用于独立样本的比较。\n")
    report.append("2. 对于配对数据，Wilcoxon signed-rank test 可能更合适。\n")
    report.append("3. p值小于0.05通常被认为是统计显著的。\n")
    report.append("4. 效应量用于评估差异的实际意义，而不仅仅是统计显著性。\n")
    
    return "".join(report)

if __name__ == "__main__":
    report_file = "/T20030104/ynj/TRIX/comparison_report_3.md"
    
    print("正在解析对比报告...")
    data, categories = parse_comparison_report(report_file)
    
    print("正在执行 Mann-Whitney U Test...")
    report = generate_report(data, categories)
    
    output_file = "/T20030104/ynj/TRIX/mannwhitney_analysis_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {output_file}")



