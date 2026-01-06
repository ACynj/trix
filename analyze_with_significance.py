#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析多个随机种子的日志文件，计算平均值和方差，与基准文件对比
并进行显著性检验
"""

import re
import numpy as np
from scipy import stats
from collections import defaultdict
from pathlib import Path

def parse_log_file(log_file: str):
    """解析日志文件，提取每个数据集的指标"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    current_dataset = None
    for i, line in enumerate(lines):
        # 匹配数据集名称
        dataset_match = re.search(r'(\w+(?:Inductive|Ingram|TopicsMT\d)?)(?:\(([^)]+)\))?\s+dataset', line)
        if dataset_match:
            dataset_name = dataset_match.group(1)
            version = dataset_match.group(2) if dataset_match.group(2) else None
            if version:
                current_dataset = f"{dataset_name}({version})"
            else:
                current_dataset = dataset_name
            if current_dataset not in results:
                results[current_dataset] = {}
            continue
        
        # 匹配指标
        if current_dataset:
            mr_match = re.search(r'mr:\s+([\d.]+)', line)
            mrr_match = re.search(r'mrr:\s+([\d.]+)', line)
            hits1_match = re.search(r'hits@1:\s+([\d.]+)', line)
            hits3_match = re.search(r'hits@3:\s+([\d.]+)', line)
            hits10_match = re.search(r'hits@10:\s+([\d.]+)', line)
            
            if mr_match or mrr_match or hits1_match or hits3_match or hits10_match:
                if mr_match:
                    results[current_dataset]['mr'] = float(mr_match.group(1))
                if mrr_match:
                    results[current_dataset]['mrr'] = float(mrr_match.group(1))
                if hits1_match:
                    results[current_dataset]['hits@1'] = float(hits1_match.group(1))
                if hits3_match:
                    results[current_dataset]['hits@3'] = float(hits3_match.group(1))
                if hits10_match:
                    results[current_dataset]['hits@10'] = float(hits10_match.group(1))
    
    return results

def calculate_stats(values):
    """计算平均值和标准差"""
    if not values:
        return None, None
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std

def perform_t_test(sample_values, population_mean):
    """
    执行单样本t检验
    零假设H0: 样本均值等于基准值
    备择假设H1: 样本均值不等于基准值
    """
    if len(sample_values) < 2:
        return None, None, None
    
    sample_values = np.array(sample_values)
    t_stat, p_value = stats.ttest_1samp(sample_values, population_mean)
    
    # 计算95%置信区间
    n = len(sample_values)
    mean = np.mean(sample_values)
    std = np.std(sample_values, ddof=1)
    se = std / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)  # 95%置信区间，双边
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se
    
    return t_stat, p_value, (ci_lower, ci_upper)

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

def interpret_significance(p_value):
    """解释p值的显著性"""
    if p_value is None:
        return "N/A", ""
    if p_value < 0.001:
        return "***", "p<0.001"
    elif p_value < 0.01:
        return "**", "p<0.01"
    elif p_value < 0.05:
        return "*", "p<0.05"
    elif p_value < 0.1:
        return ".", "p<0.1"
    else:
        return "", "p≥0.05"

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
    print(f"  基准文件找到 {len(baseline_results)} 个数据集")
    
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
    print(f"\n总共找到 {len(all_datasets)} 个唯一数据集")
    
    # 计算每个数据集的统计信息和显著性检验
    comparison_data = []
    category_data = defaultdict(lambda: {'baseline': defaultdict(list), 'mean': defaultdict(list), 'std': defaultdict(list), 'p_value': defaultdict(list)})
    
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
            
            if len(values) < 2:  # 至少需要2个值才能做t检验
                continue
            
            mean_val, std_val = calculate_stats(values)
            
            # 执行t检验
            t_stat, p_value, ci = perform_t_test(values, baseline_value)
            
            diff = mean_val - baseline_value
            rel_change = (diff / baseline_value * 100) if baseline_value != 0 else 0
            
            sig_symbol, sig_text = interpret_significance(p_value)
            
            comparison_data.append({
                'dataset': dataset,
                'metric': metric,
                'baseline': baseline_value,
                'mean': mean_val,
                'std': std_val,
                'diff': diff,
                'rel_change': rel_change,
                'category': category,
                't_stat': t_stat,
                'p_value': p_value,
                'ci_lower': ci[0] if ci else None,
                'ci_upper': ci[1] if ci else None,
                'sig_symbol': sig_symbol,
                'sig_text': sig_text
            })
            
            # 按类别收集数据
            category_data[category]['baseline'][metric].append(baseline_value)
            category_data[category]['mean'][metric].append(mean_val)
            category_data[category]['std'][metric].append(std_val)
            if p_value is not None:
                category_data[category]['p_value'][metric].append(p_value)
    
    print(f"\n共生成 {len(comparison_data)} 条对比数据")
    
    # 统计显著性结果
    significant_count = {'***': 0, '**': 0, '*': 0, '.': 0, '': 0}
    for item in comparison_data:
        significant_count[item['sig_symbol']] = significant_count.get(item['sig_symbol'], 0) + 1
    
    print(f"\n显著性统计:")
    print(f"  *** (p<0.001): {significant_count['***']}")
    print(f"  ** (p<0.01): {significant_count['**']}")
    print(f"  * (p<0.05): {significant_count['*']}")
    print(f"  . (p<0.1): {significant_count['.']}")
    print(f"  不显著 (p≥0.05): {significant_count['']}")
    
    # 生成报告
    output_file = '/T20030104/ynj/TRIX/comparison_report_multiple_seeds.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 多随机种子结果对比分析报告（含显著性检验）\n")
        f.write(f"**基准文件**: inference_rel.log\n")
        f.write(f"**对比文件**: \n")
        for log_file in log_files:
            f.write(f"- {Path(log_file).name}\n")
        f.write("\n")
        f.write("**显著性标记说明**: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, 无标记表示p≥0.05\n")
        f.write("\n")
        f.write("---\n")
        f.write("## 1. 每个数据集的详细对比（平均值±标准差，含显著性检验）\n")
        f.write("| 数据集 | 指标 | inference_rel.log | 三次运行平均值 | 标准差 | 差异 | 相对提升 | p值 | 显著性 | 95%置信区间 |\n")
        f.write("|--------|------|-------------------|----------------|--------|------|----------|-----|--------|-------------|\n")
        
        for item in comparison_data:
            ci_str = f"[{item['ci_lower']:.4f}, {item['ci_upper']:.4f}]" if item['ci_lower'] is not None else "N/A"
            p_str = f"{item['p_value']:.4f}" if item['p_value'] is not None else "N/A"
            f.write(f"| {item['dataset']} | {item['metric']} | {item['baseline']:.4f} | "
                   f"{item['mean']:.4f} | {item['std']:.4f} | "
                   f"{item['diff']:+.4f} | {item['rel_change']:+.2f}% | "
                   f"{p_str} | {item['sig_symbol']} {item['sig_text']} | {ci_str} |\n")
        
        f.write("\n---\n")
        f.write("## 2. 按类别进行平均对比（含显著性检验）\n")
        
        for category in ['Transductive', 'Inductive(e)', 'Inductive(e,r)']:
            if category not in category_data:
                continue
            
            f.write(f"### {category}\n")
            f.write("| 指标 | inference_rel.log | 三次运行平均值 | 平均标准差 | 差异 | 相对提升 | 平均p值 | 显著比例 |\n")
            f.write("|------|-------------------|----------------|------------|------|----------|--------|----------|\n")
            
            # 按指标分组计算
            for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
                metric_baseline = category_data[category]['baseline'][metric]
                metric_mean = category_data[category]['mean'][metric]
                metric_std = category_data[category]['std'][metric]
                metric_p_values = category_data[category]['p_value'][metric]
                
                if metric_baseline:
                    bl_avg = np.mean(metric_baseline)
                    mn_avg = np.mean(metric_mean)
                    std_avg = np.mean(metric_std)
                    diff = mn_avg - bl_avg
                    rel_change = (diff / bl_avg * 100) if bl_avg != 0 else 0
                    
                    # 计算显著比例
                    if metric_p_values:
                        sig_count = sum(1 for p in metric_p_values if p < 0.05 and not np.isnan(p))
                        sig_ratio = sig_count / len(metric_p_values) * 100
                        valid_p_values = [p for p in metric_p_values if not np.isnan(p)]
                        avg_p = np.mean(valid_p_values) if valid_p_values else None
                    else:
                        sig_ratio = 0
                        avg_p = None
                    
                    avg_p_str = f"{avg_p:.4f}" if avg_p is not None and not np.isnan(avg_p) else "N/A"
                    f.write(f"| {metric} | {bl_avg:.4f} | {mn_avg:.4f} | {std_avg:.4f} | "
                           f"{diff:+.4f} | {rel_change:+.2f}% | {avg_p_str} | {sig_ratio:.1f}% |\n")
        
        f.write("\n---\n")
        f.write("## 3. 全部数据集平均对比（含显著性检验）\n")
        f.write("| 指标 | inference_rel.log | 三次运行平均值 | 平均标准差 | 差异 | 相对提升 | 平均p值 | 显著比例 |\n")
        f.write("|------|-------------------|----------------|------------|------|----------|--------|----------|\n")
        
        for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
            metric_baseline = []
            metric_mean = []
            metric_std = []
            metric_p_values = []
            
            for item in comparison_data:
                if item['metric'] == metric:
                    metric_baseline.append(item['baseline'])
                    metric_mean.append(item['mean'])
                    metric_std.append(item['std'])
                    if item['p_value'] is not None:
                        metric_p_values.append(item['p_value'])
            
            if metric_baseline:
                bl_avg = np.mean(metric_baseline)
                mn_avg = np.mean(metric_mean)
                std_avg = np.mean(metric_std)
                diff = mn_avg - bl_avg
                rel_change = (diff / bl_avg * 100) if bl_avg != 0 else 0
                
                if metric_p_values:
                    valid_p_values = [p for p in metric_p_values if not np.isnan(p)]
                    sig_count = sum(1 for p in valid_p_values if p < 0.05)
                    sig_ratio = sig_count / len(valid_p_values) * 100 if valid_p_values else 0
                    avg_p = np.mean(valid_p_values) if valid_p_values else None
                else:
                    sig_ratio = 0
                    avg_p = None
                
                avg_p_str = f"{avg_p:.4f}" if avg_p is not None and not np.isnan(avg_p) else "N/A"
                f.write(f"| {metric} | {bl_avg:.4f} | {mn_avg:.4f} | {std_avg:.4f} | "
                       f"{diff:+.4f} | {rel_change:+.2f}% | {avg_p_str} | {sig_ratio:.1f}% |\n")
        
        f.write("\n---\n")
        f.write("## 4. 显著性检验总结\n")
        f.write("### 显著性分布统计\n")
        f.write("| 显著性水平 | 数量 | 比例 |\n")
        f.write("|------------|------|------|\n")
        total = len(comparison_data)
        for level, count in [('*** (p<0.001)', significant_count['***']),
                             ('** (p<0.01)', significant_count['**']),
                             ('* (p<0.05)', significant_count['*']),
                             ('. (p<0.1)', significant_count['.']),
                             ('不显著 (p≥0.05)', significant_count[''])]:
            ratio = (count / total * 100) if total > 0 else 0
            f.write(f"| {level} | {count} | {ratio:.1f}% |\n")
        
        f.write("\n### 显著提升的数据集（p<0.05且差异为正）\n")
        f.write("| 数据集 | 指标 | 基准值 | 平均值 | 差异 | p值 |\n")
        f.write("|--------|------|--------|--------|------|-----|\n")
        significant_improvements = [item for item in comparison_data 
                                   if item['p_value'] is not None and item['p_value'] < 0.05 
                                   and item['diff'] > 0]
        significant_improvements.sort(key=lambda x: x['p_value'])
        for item in significant_improvements[:30]:  # 显示前30个
            f.write(f"| {item['dataset']} | {item['metric']} | {item['baseline']:.4f} | "
                   f"{item['mean']:.4f} | {item['diff']:+.4f} | {item['p_value']:.4f} |\n")
        
        f.write("\n### 显著下降的数据集（p<0.05且差异为负）\n")
        f.write("| 数据集 | 指标 | 基准值 | 平均值 | 差异 | p值 |\n")
        f.write("|--------|------|--------|--------|------|-----|\n")
        significant_declines = [item for item in comparison_data 
                               if item['p_value'] is not None and item['p_value'] < 0.05 
                               and item['diff'] < 0]
        significant_declines.sort(key=lambda x: x['p_value'])
        for item in significant_declines[:30]:  # 显示前30个
            f.write(f"| {item['dataset']} | {item['metric']} | {item['baseline']:.4f} | "
                   f"{item['mean']:.4f} | {item['diff']:+.4f} | {item['p_value']:.4f} |\n")
        
        f.write("\n---\n")
        f.write("## 5. 方差分析\n")
        f.write("### 方差最大的数据集（按标准差排序，前20个）\n")
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

if __name__ == "__main__":
    main()

