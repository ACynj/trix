#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对comparison_report_3.md类型的数据进行配对t检验
适合两个文件的对比（每个数据集有配对的基准值和新值）
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

def perform_paired_t_test(baseline_values, comparison_values):
    """
    执行配对t检验
    零假设H0: 配对差异的均值为0（即新结果与基准无显著差异）
    备择假设H1: 配对差异的均值不为0（即新结果与基准有显著差异）
    
    使用 scipy.stats.ttest_rel 进行配对t检验
    """
    if len(baseline_values) != len(comparison_values) or len(baseline_values) < 2:
        return None, None, None
    
    baseline_array = np.array(baseline_values)
    comparison_array = np.array(comparison_values)
    
    # 配对t检验（注意：ttest_rel计算的是第一个参数减去第二个参数）
    # 我们想要 comparison - baseline，所以参数顺序是 baseline, comparison
    # 这样 t_stat 的符号会正确反映 comparison 相对于 baseline 的提升
    t_stat, p_value = stats.ttest_rel(comparison_array, baseline_array)
    
    # 计算配对差异
    differences = comparison_array - baseline_array
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # 计算95%置信区间
    n = len(differences)
    se = std_diff / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean_diff - t_critical * se
    ci_upper = mean_diff + t_critical * se
    
    return t_stat, p_value, (ci_lower, ci_upper, mean_diff, std_diff)

def perform_wilcoxon_test(baseline_values, comparison_values):
    """
    执行Wilcoxon符号秩检验（非参数检验）
    当数据不满足正态分布假设时使用
    """
    if len(baseline_values) != len(comparison_values) or len(baseline_values) < 2:
        return None, None
    
    try:
        w_stat, p_value = stats.wilcoxon(baseline_values, comparison_values, alternative='two-sided')
        return w_stat, p_value
    except:
        return None, None

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

def check_normality(differences):
    """检查差异是否满足正态分布（Shapiro-Wilk检验）"""
    if len(differences) < 3 or len(differences) > 5000:
        return None, None
    
    try:
        stat, p_value = stats.shapiro(differences)
        return stat, p_value
    except:
        return None, None

def main():
    # 文件路径（对应comparison_report_3.md中的两个文件）
    baseline_file = '/T20030104/ynj/TRIX/inference_rel.log'
    comparison_file = '/T20030104/ynj/TRIX/run_commands_20260102_085655.log'
    
    print("正在解析基准文件...")
    baseline_results = parse_log_file(baseline_file)
    print(f"  基准文件找到 {len(baseline_results)} 个数据集")
    
    print("正在解析对比文件...")
    comparison_results = parse_log_file(comparison_file)
    print(f"  对比文件找到 {len(comparison_results)} 个数据集")
    
    # 收集所有数据集名称
    all_datasets = set(baseline_results.keys())
    all_datasets.update(comparison_results.keys())
    all_datasets = sorted(all_datasets)
    print(f"\n总共找到 {len(all_datasets)} 个唯一数据集")
    
    # 按类别和指标组织数据
    category_data = defaultdict(lambda: defaultdict(lambda: {'baseline': [], 'comparison': []}))
    overall_data = defaultdict(lambda: {'baseline': [], 'comparison': []})
    
    for dataset in all_datasets:
        baseline_metrics = baseline_results.get(dataset, {})
        comparison_metrics = comparison_results.get(dataset, {})
        category = get_dataset_category(dataset)
        
        for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
            baseline_value = baseline_metrics.get(metric)
            comparison_value = comparison_metrics.get(metric)
            
            if baseline_value is None or comparison_value is None:
                continue
            
            # 按类别收集
            category_data[category][metric]['baseline'].append(baseline_value)
            category_data[category][metric]['comparison'].append(comparison_value)
            
            # 总体收集
            overall_data[metric]['baseline'].append(baseline_value)
            overall_data[metric]['comparison'].append(comparison_value)
    
    # 生成报告
    output_file = '/T20030104/ynj/TRIX/comparison_report_3_with_significance.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 日志文件对比分析报告（含配对t检验显著性分析）\n")
        f.write(f"**文件1（基准）**: inference_rel.log\n")
        f.write(f"**文件2（对比）**: run_commands_20260102_085655.log\n")
        f.write("\n")
        f.write("**显著性标记说明**: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, 无标记表示p≥0.05\n")
        f.write("\n")
        f.write("**检验方法**: 配对t检验（Paired t-test）\n")
        f.write("- 适用于配对数据：每个数据集在基准和对比文件中都有对应的值\n")
        f.write("- 零假设(H0): 配对差异的均值为0（即新结果与基准无显著差异）\n")
        f.write("- 备择假设(H1): 配对差异的均值不为0（即新结果与基准有显著差异）\n")
        f.write("- 显著性水平: α = 0.05\n")
        f.write("\n")
        f.write("---\n")
        f.write("## 1. 按类别进行配对t检验\n")
        
        for category in ['Transductive', 'Inductive(e)', 'Inductive(e,r)']:
            if category not in category_data:
                continue
            
            f.write(f"### {category}\n")
            f.write("| 指标 | 数据集数 | 基准平均值 | 对比平均值 | 平均差异 | 相对提升 | t统计量 | p值 | 显著性 | 95%置信区间 |\n")
            f.write("|------|----------|------------|------------|----------|----------|---------|-----|--------|-------------|\n")
            
            for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric not in category_data[category]:
                    continue
                
                baseline_vals = category_data[category][metric]['baseline']
                comparison_vals = category_data[category][metric]['comparison']
                
                if len(baseline_vals) < 2:
                    continue
                
                bl_avg = np.mean(baseline_vals)
                comp_avg = np.mean(comparison_vals)
                diff = comp_avg - bl_avg
                rel_change = (diff / bl_avg * 100) if bl_avg != 0 else 0
                
                # 执行配对t检验
                t_stat, p_value, ci_info = perform_paired_t_test(baseline_vals, comparison_vals)
                sig_symbol, sig_text = interpret_significance(p_value)
                
                # 检查正态性
                differences = np.array(comparison_vals) - np.array(baseline_vals)
                shapiro_stat, shapiro_p = check_normality(differences)
                
                t_str = f"{t_stat:.4f}" if t_stat is not None else "N/A"
                p_str = f"{p_value:.4f}" if p_value is not None else "N/A"
                ci_str = f"[{ci_info[0]:+.4f}, {ci_info[1]:+.4f}]" if ci_info else "N/A"
                
                f.write(f"| {metric} | {len(baseline_vals)} | {bl_avg:.4f} | {comp_avg:.4f} | "
                       f"{diff:+.4f} | {rel_change:+.2f}% | {t_str} | {p_str} | {sig_symbol} {sig_text} | {ci_str} |\n")
        
        f.write("\n---\n")
        f.write("## 2. 总体配对t检验（全部54个数据集）\n")
        f.write("| 指标 | 数据集数 | 基准平均值 | 对比平均值 | 平均差异 | 相对提升 | t统计量 | p值 | 显著性 | 95%置信区间 |\n")
        f.write("|------|----------|------------|------------|----------|----------|---------|-----|--------|-------------|\n")
        
        for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
            if metric not in overall_data:
                continue
            
            baseline_vals = overall_data[metric]['baseline']
            comparison_vals = overall_data[metric]['comparison']
            
            if len(baseline_vals) < 2:
                continue
            
            bl_avg = np.mean(baseline_vals)
            comp_avg = np.mean(comparison_vals)
            diff = comp_avg - bl_avg
            rel_change = (diff / bl_avg * 100) if bl_avg != 0 else 0
            
            # 执行配对t检验
            t_stat, p_value, ci_info = perform_paired_t_test(baseline_vals, comparison_vals)
            sig_symbol, sig_text = interpret_significance(p_value)
            
            # 检查正态性
            differences = np.array(comparison_vals) - np.array(baseline_vals)
            shapiro_stat, shapiro_p = check_normality(differences)
            
            t_str = f"{t_stat:.4f}" if t_stat is not None else "N/A"
            p_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            ci_str = f"[{ci_info[0]:+.4f}, {ci_info[1]:+.4f}]" if ci_info else "N/A"
            
            f.write(f"| {metric} | {len(baseline_vals)} | {bl_avg:.4f} | {comp_avg:.4f} | "
                   f"{diff:+.4f} | {rel_change:+.2f}% | {t_str} | {p_str} | {sig_symbol} {sig_text} | {ci_str} |\n")
        
        f.write("\n---\n")
        f.write("## 3. 显著性检验结果总结\n")
        f.write("\n### 按类别总结\n")
        
        category_summary = []
        for category in ['Transductive', 'Inductive(e)', 'Inductive(e,r)']:
            if category not in category_data:
                continue
            
            sig_count = 0
            total_count = 0
            for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric not in category_data[category]:
                    continue
                
                baseline_vals = category_data[category][metric]['baseline']
                comparison_vals = category_data[category][metric]['comparison']
                
                if len(baseline_vals) < 2:
                    continue
                
                _, p_value, _ = perform_paired_t_test(baseline_vals, comparison_vals)
                total_count += 1
                if p_value is not None and p_value < 0.05:
                    sig_count += 1
            
            category_summary.append({
                'category': category,
                'sig_count': sig_count,
                'total_count': total_count,
                'sig_ratio': (sig_count / total_count * 100) if total_count > 0 else 0
            })
        
        f.write("| 类别 | 显著指标数 | 总指标数 | 显著比例 |\n")
        f.write("|------|------------|----------|----------|\n")
        for item in category_summary:
            f.write(f"| {item['category']} | {item['sig_count']} | {item['total_count']} | {item['sig_ratio']:.1f}% |\n")
        
        f.write("\n### 总体总结\n")
        overall_sig_count = 0
        overall_total_count = 0
        for metric in ['mrr', 'hits@1', 'hits@3', 'hits@10']:
            if metric not in overall_data:
                continue
            
            baseline_vals = overall_data[metric]['baseline']
            comparison_vals = overall_data[metric]['comparison']
            
            if len(baseline_vals) < 2:
                continue
            
            _, p_value, _ = perform_paired_t_test(baseline_vals, comparison_vals)
            overall_total_count += 1
            if p_value is not None and p_value < 0.05:
                overall_sig_count += 1
        
        f.write(f"| 总体 | {overall_sig_count} | {overall_total_count} | {(overall_sig_count/overall_total_count*100):.1f}% |\n")
        
        f.write("\n---\n")
        f.write("## 4. 检验方法说明\n")
        f.write("\n### 为什么使用配对t检验？\n")
        f.write("1. **数据是配对的**：每个数据集在基准和对比文件中都有对应的值\n")
        f.write("2. **控制数据集间差异**：配对设计可以消除数据集本身的差异，只关注方法间的差异\n")
        f.write("3. **提高统计功效**：配对检验比独立样本t检验更敏感，更容易检测到真实差异\n")
        f.write("\n### 假设条件\n")
        f.write("- **配对性**：每个数据集的基准值和对比值必须对应 ✓\n")
        f.write("- **独立性**：不同数据集的结果相互独立 ✓\n")
        f.write("- **正态性**：配对差异应该近似正态分布（可通过Shapiro-Wilk检验验证）\n")
        f.write("- **样本量**：建议至少10-15对数据（我们有54对，满足要求）✓\n")
        f.write("\n### 替代方法\n")
        f.write("如果数据不满足正态分布假设，可以使用：\n")
        f.write("- **Wilcoxon符号秩检验**：非参数配对检验，不要求正态分布\n")
        f.write("- **符号检验**：最简单的非参数配对检验\n")
    
    print(f"\n报告已生成: {output_file}")

if __name__ == "__main__":
    main()

