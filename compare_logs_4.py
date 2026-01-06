#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 inference_rel.log 和 run_commands_20260101_035905.log
"""

from compare_logs import parse_log_file, get_dataset_categories, calculate_averages
from generate_comparison_report import format_number

def generate_markdown_report_with_relative(results1, results2, name1, name2):
    """生成包含相对提升的Markdown格式对比报告"""
    
    categories = get_dataset_categories()
    all_datasets = categories['transductive'] + categories['inductive_e'] + categories['inductive_er']
    metrics = ['mrr', 'hits@1', 'hits@3', 'hits@10']
    
    report = []
    report.append("# 日志文件对比分析报告\n")
    report.append(f"**文件1**: {name1}  \n")
    report.append(f"**文件2**: {name2}\n")
    report.append("\n---\n")
    
    # 1. 每个数据集的详细对比（包含相对提升）
    report.append("## 1. 每个数据集的详细对比\n")
    report.append("| 数据集 | 指标 | " + name1 + " | " + name2 + " | 差异 | 相对提升 |\n")
    report.append("|--------|------|" + "-" * (len(name1) + 2) + "|" + "-" * (len(name2) + 2) + "|------|----------|\n")
    
    for dataset in all_datasets:
        # 查找匹配的数据集
        matched_dataset1 = None
        matched_dataset2 = None
        
        for d in results1.keys():
            if dataset in d or d in dataset:
                matched_dataset1 = d
                break
        
        for d in results2.keys():
            if dataset in d or d in dataset:
                matched_dataset2 = d
                break
        
        if matched_dataset1 and matched_dataset2:
            for metric in metrics:
                val1 = results1[matched_dataset1].get(metric, 0.0)
                val2 = results2[matched_dataset2].get(metric, 0.0)
                diff = val2 - val1
                diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
                
                # 计算相对提升
                if val1 != 0:
                    relative_improvement = (diff / val1) * 100
                    rel_str = f"{relative_improvement:+.2f}%"
                else:
                    rel_str = "N/A"
                
                report.append(f"| {dataset} | {metric} | {format_number(val1)} | {format_number(val2)} | {diff_str} | {rel_str} |\n")
    
    report.append("\n---\n")
    
    # 2. 按类别进行平均对比
    report.append("## 2. 按类别进行平均对比\n")
    
    category_names = {
        'transductive': 'Transductive (13个数据集)',
        'inductive_e': 'Inductive(e) (18个数据集)',
        'inductive_er': 'Inductive(e,r) (23个数据集)'
    }
    
    for category_key, category_datasets in categories.items():
        category_name = category_names[category_key]
        report.append(f"### {category_name}\n")
        report.append("| 指标 | " + name1 + " | " + name2 + " | 差异 | 相对提升 |\n")
        report.append("|------|" + "-" * (len(name1) + 2) + "|" + "-" * (len(name2) + 2) + "|------|----------|\n")
        
        avg1 = calculate_averages(results1, category_datasets)
        avg2 = calculate_averages(results2, category_datasets)
        
        for metric in metrics:
            val1 = avg1.get(metric, 0.0)
            val2 = avg2.get(metric, 0.0)
            diff = val2 - val1
            rel_improve = (diff / val1 * 100) if val1 != 0 else 0.0
            report.append(f"| {metric} | {format_number(val1)} | {format_number(val2)} | {diff:+.6f} | {rel_improve:+.2f}% |\n")
        
        report.append("\n")
    
    report.append("---\n")
    
    # 3. 全部数据集平均对比
    report.append("## 3. 全部数据集平均对比 (54个数据集)\n")
    report.append("| 指标 | " + name1 + " | " + name2 + " | 差异 | 相对提升 |\n")
    report.append("|------|" + "-" * (len(name1) + 2) + "|" + "-" * (len(name2) + 2) + "|------|----------|\n")
    
    avg1_all = calculate_averages(results1, all_datasets)
    avg2_all = calculate_averages(results2, all_datasets)
    
    for metric in metrics:
        val1 = avg1_all.get(metric, 0.0)
        val2 = avg2_all.get(metric, 0.0)
        diff = val2 - val1
        rel_improve = (diff / val1 * 100) if val1 != 0 else 0.0
        report.append(f"| {metric} | {format_number(val1)} | {format_number(val2)} | {diff:+.6f} | {rel_improve:+.2f}% |\n")
    
    return "".join(report)

if __name__ == "__main__":
    log_file1 = "/T20030104/ynj/TRIX/inference_rel.log"
    log_file2 = "/T20030104/ynj/TRIX/run_commands_20260101_035905.log"
    
    print("正在解析日志文件...")
    results1 = parse_log_file(log_file1)
    results2 = parse_log_file(log_file2)
    
    print(f"文件1找到 {len(results1)} 个数据集的结果")
    print(f"文件2找到 {len(results2)} 个数据集的结果")
    
    print("正在生成Markdown报告...")
    report = generate_markdown_report_with_relative(
        results1, 
        results2,
        name1="inference_rel.log",
        name2="run_commands_20260101_035905.log"
    )
    
    output_file = "/T20030104/ynj/TRIX/comparison_report_4.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {output_file}")


