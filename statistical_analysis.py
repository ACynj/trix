#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对对比报告进行显著性检验分析
"""

import re
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

def parse_comparison_report(filepath: str) -> Dict[str, Dict[str, List[float]]]:
    """解析对比报告，提取数据"""
    data = {
        'mrr': {'baseline': [], 'comparison': [], 'diff': []},
        'hits@1': {'baseline': [], 'comparison': [], 'diff': []},
        'hits@3': {'baseline': [], 'comparison': [], 'diff': []},
        'hits@10': {'baseline': [], 'comparison': [], 'diff': []}
    }
    
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
            diff = float(match.group(5))
            
            if metric in data:
                data[metric]['baseline'].append(val1)
                data[metric]['comparison'].append(val2)
                data[metric]['diff'].append(diff)
    
    return data

def perform_statistical_tests(data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict]:
    """进行统计显著性检验"""
    results = {}
    
    for metric, values in data.items():
        baseline = np.array(values['baseline'])
        comparison = np.array(values['comparison'])
        diff = np.array(values['diff'])
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(baseline, comparison)
        
        # Wilcoxon符号秩检验（非参数检验，更稳健）
        try:
            w_stat, w_p_value = stats.wilcoxon(baseline, comparison, alternative='two-sided')
        except:
            w_stat, w_p_value = None, None
        
        # 计算效应量（Cohen's d）
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # 计算置信区间（95%）
        n = len(diff)
        se = std_diff / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, n-1)
        ci_lower = mean_diff - t_critical * se
        ci_upper = mean_diff + t_critical * se
        
        results[metric] = {
            'mean_baseline': np.mean(baseline),
            'mean_comparison': np.mean(comparison),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'n': n,
            't_statistic': t_stat,
            'p_value': p_value,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_p_value': w_p_value,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05,
            'highly_significant': p_value < 0.01,
            'very_highly_significant': p_value < 0.001
        }
    
    return results

def categorize_datasets() -> Dict[str, List[int]]:
    """数据集分类索引"""
    # Transductive: 前13个数据集 (索引0-12)
    # Inductive(e): 接下来18个数据集 (索引13-30)
    # Inductive(e,r): 最后23个数据集 (索引31-53)
    return {
        'transductive': list(range(13)),
        'inductive_e': list(range(13, 31)),
        'inductive_er': list(range(31, 54))
    }

def analyze_by_category(data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict]]:
    """按类别进行显著性检验"""
    categories = categorize_datasets()
    category_results = {}
    
    for category_name, indices in categories.items():
        category_data = {}
        for metric, values in data.items():
            category_data[metric] = {
                'baseline': [values['baseline'][i] for i in indices],
                'comparison': [values['comparison'][i] for i in indices],
                'diff': [values['diff'][i] for i in indices]
            }
        
        category_results[category_name] = perform_statistical_tests(category_data)
    
    return category_results

def generate_statistical_report(results: Dict[str, Dict], 
                                category_results: Dict[str, Dict[str, Dict]]) -> str:
    """生成统计检验报告"""
    report = []
    report.append("# 显著性检验分析报告\n")
    report.append("## 1. 全部数据集显著性检验 (54个数据集)\n\n")
    
    for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
        r = results[metric]
        report.append(f"### {metric.upper()}\n")
        report.append(f"- **基准均值**: {r['mean_baseline']:.6f}\n")
        report.append(f"- **对比均值**: {r['mean_comparison']:.6f}\n")
        report.append(f"- **平均差异**: {r['mean_diff']:+.6f}\n")
        report.append(f"- **差异标准差**: {r['std_diff']:.6f}\n")
        report.append(f"- **样本数**: {r['n']}\n")
        report.append(f"- **配对t检验**: t = {r['t_statistic']:.4f}, p = {r['p_value']:.6f}\n")
        if r['wilcoxon_p_value'] is not None:
            report.append(f"- **Wilcoxon符号秩检验**: W = {r['wilcoxon_statistic']:.4f}, p = {r['wilcoxon_p_value']:.6f}\n")
        report.append(f"- **Cohen's d (效应量)**: {r['cohens_d']:.4f}\n")
        report.append(f"- **95%置信区间**: [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]\n")
        
        # 显著性判断
        if r['very_highly_significant']:
            sig_level = "*** (p < 0.001)"
        elif r['highly_significant']:
            sig_level = "** (p < 0.01)"
        elif r['significant']:
            sig_level = "* (p < 0.05)"
        else:
            sig_level = "不显著 (p ≥ 0.05)"
        
        report.append(f"- **显著性**: {sig_level}\n\n")
    
    report.append("\n---\n")
    report.append("## 2. 按类别显著性检验\n\n")
    
    category_names = {
        'transductive': 'Transductive (13个数据集)',
        'inductive_e': 'Inductive(e) (18个数据集)',
        'inductive_er': 'Inductive(e,r) (23个数据集)'
    }
    
    for category_key, category_name in category_names.items():
        report.append(f"### {category_name}\n\n")
        cat_results = category_results[category_key]
        
        for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
            r = cat_results[metric]
            report.append(f"#### {metric.upper()}\n")
            report.append(f"- **基准均值**: {r['mean_baseline']:.6f}\n")
            report.append(f"- **对比均值**: {r['mean_comparison']:.6f}\n")
            report.append(f"- **平均差异**: {r['mean_diff']:+.6f}\n")
            report.append(f"- **配对t检验**: t = {r['t_statistic']:.4f}, p = {r['p_value']:.6f}\n")
            if r['wilcoxon_p_value'] is not None:
                report.append(f"- **Wilcoxon检验**: W = {r['wilcoxon_statistic']:.4f}, p = {r['wilcoxon_p_value']:.6f}\n")
            report.append(f"- **Cohen's d**: {r['cohens_d']:.4f}\n")
            report.append(f"- **95%置信区间**: [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]\n")
            
            if r['very_highly_significant']:
                sig_level = "*** (p < 0.001)"
            elif r['highly_significant']:
                sig_level = "** (p < 0.01)"
            elif r['significant']:
                sig_level = "* (p < 0.05)"
            else:
                sig_level = "不显著 (p ≥ 0.05)"
            
            report.append(f"- **显著性**: {sig_level}\n\n")
    
    report.append("\n---\n")
    report.append("## 3. 效应量解释\n\n")
    report.append("- **|d| < 0.2**: 微小效应\n")
    report.append("- **0.2 ≤ |d| < 0.5**: 小效应\n")
    report.append("- **0.5 ≤ |d| < 0.8**: 中等效应\n")
    report.append("- **|d| ≥ 0.8**: 大效应\n")
    
    return "".join(report)

if __name__ == "__main__":
    filepath = "/T20030104/ynj/TRIX/comparison_report.md"
    
    print("正在解析对比报告...")
    data = parse_comparison_report(filepath)
    
    print("正在进行显著性检验...")
    results = perform_statistical_tests(data)
    category_results = analyze_by_category(data)
    
    print("正在生成统计报告...")
    report = generate_statistical_report(results, category_results)
    
    output_file = "/T20030104/ynj/TRIX/statistical_analysis.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"统计检验报告已保存到: {output_file}")



