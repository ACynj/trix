#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照command_rel.md的分割符对数据集进行分类，进行详细对比分析
"""
import re
from collections import defaultdict

def parse_command_file(filepath):
    """解析command_rel.md，获取数据集分类信息"""
    transductive_datasets = []
    inductive_datasets = []
    ingram_datasets = []
    
    current_section = 'transductive'  # transductive, inductive, ingram
    
    # Ingram数据集的关键词
    ingram_keywords = ['Ingram', 'WikiTopics', 'Metafam', 'FBNELL']
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 检测分割符
            if '===' in line and len(line.strip()) > 10:
                if current_section == 'transductive':
                    current_section = 'inductive'
                continue
            
            # 提取数据集名称
            if '--dataset' in line:
                match = re.search(r'--dataset\s+(\w+)(?:\s+--version\s+(\S+))?', line)
                if match:
                    dataset = match.group(1)
                    version = match.group(2) if match.group(2) else None
                    
                    if version and version != 'null':
                        dataset_key = f"{dataset}_{version}"
                    else:
                        dataset_key = dataset
                    
                    # 根据数据集名称判断是否为Ingram
                    is_ingram = any(keyword in dataset for keyword in ingram_keywords)
                    
                    if current_section == 'transductive':
                        transductive_datasets.append(dataset_key)
                    elif current_section == 'inductive':
                        if is_ingram:
                            ingram_datasets.append(dataset_key)
                        else:
                            inductive_datasets.append(dataset_key)
    
    return {
        'transductive': transductive_datasets,
        'inductive': inductive_datasets,
        'ingram': ingram_datasets
    }

def parse_log_file(filepath):
    """解析日志文件，提取数据集和指标"""
    results = {}
    current_dataset = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取数据集名称
            if '执行命令' in line or 'python' in line and '--dataset' in line:
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

def classify_dataset_by_command(dataset_name, command_categories):
    """根据command_rel.md的分类对数据集进行分类"""
    for category, datasets in command_categories.items():
        if dataset_name in datasets:
            return category
    # 如果找不到，尝试模糊匹配
    if 'Inductive' in dataset_name or 'ILPC2022' in dataset_name or 'HM' in dataset_name:
        return 'inductive'
    elif 'Ingram' in dataset_name or 'WikiTopics' in dataset_name or 'Metafam' in dataset_name or 'FBNELL' in dataset_name:
        return 'ingram'
    else:
        return 'transductive'

def calculate_averages_by_category(results, command_categories):
    """按类别计算平均值"""
    categories = defaultdict(lambda: defaultdict(list))
    
    for dataset, metrics in results.items():
        category = classify_dataset_by_command(dataset, command_categories)
        for metric, value in metrics.items():
            categories[category][metric].append(value)
    
    averages = {}
    for category, metrics_dict in categories.items():
        averages[category] = {}
        for metric, values in metrics_dict.items():
            if values:
                averages[category][metric] = sum(values) / len(values)
            else:
                averages[category][metric] = 0
    
    return averages

def calculate_overall_average(results):
    """计算所有数据集的整体平均值"""
    overall = defaultdict(list)
    
    for dataset, metrics in results.items():
        for metric, value in metrics.items():
            overall[metric].append(value)
    
    averages = {}
    for metric, values in overall.items():
        if values:
            averages[metric] = sum(values) / len(values)
        else:
            averages[metric] = 0
    
    return averages

def compare_results(exp1_results, exp2_results):
    """对比两次实验结果"""
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
    # 解析command_rel.md获取分类信息
    command_categories = parse_command_file('command_rel.md')
    
    print("数据集分类信息：")
    print(f"  Transductive: {len(command_categories['transductive'])} 个")
    print(f"  Inductive: {len(command_categories['inductive'])} 个")
    print(f"  Ingram: {len(command_categories['ingram'])} 个")
    print(f"  总计: {sum(len(v) for v in command_categories.values())} 个\n")
    
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
    
    # 按类别计算平均值
    exp1_avg = calculate_averages_by_category(exp1_results, command_categories)
    exp2_avg = calculate_averages_by_category(exp2_results, command_categories)
    
    # 计算整体平均值
    exp1_overall = calculate_overall_average(exp1_results)
    exp2_overall = calculate_overall_average(exp2_results)
    
    # 打印结果
    print("=" * 100)
    print("实验结果对比分析（按command_rel.md分割符分类）")
    print("=" * 100)
    
    print("\n第一次实验（2025-12-25）")
    print("-" * 100)
    for category in ['transductive', 'inductive', 'ingram']:
        category_name = category.capitalize()
        if category in exp1_avg:
            print(f"\n{category_name} 数据集平均结果 ({len([d for d in exp1_results.keys() if classify_dataset_by_command(d, command_categories) == category])} 个数据集):")
            for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric in exp1_avg[category]:
                    print(f"  {metric:10s}: {exp1_avg[category][metric]:.6f}")
    
    print("\n\n第二次实验（2026-01-01）")
    print("-" * 100)
    for category in ['transductive', 'inductive', 'ingram']:
        category_name = category.capitalize()
        if category in exp2_avg:
            print(f"\n{category_name} 数据集平均结果 ({len([d for d in exp2_results.keys() if classify_dataset_by_command(d, command_categories) == category])} 个数据集):")
            for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                if metric in exp2_avg[category]:
                    print(f"  {metric:10s}: {exp2_avg[category][metric]:.6f}")
    
    print("\n\n三类数据集平均对比（提升/下降）")
    print("=" * 100)
    for category in ['transductive', 'inductive', 'ingram']:
        category_name = category.capitalize()
        if category in exp1_avg and category in exp2_avg:
            print(f"\n{category_name} 数据集:")
            print(f"{'指标':<12} {'第一次':<15} {'第二次':<15} {'变化':<15} {'变化率':<15}")
            print("-" * 72)
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
                    change_str = f"{change:+.6f}" if change >= 0 else f"{change:.6f}"
                    change_pct_str = f"{change_pct:+.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
                    print(f"{metric:<12} {val1:<15.6f} {val2:<15.6f} {change_str:<15} {change_pct_str:<15}")
    
    print("\n\n54个数据集整体平均对比")
    print("=" * 100)
    print(f"{'指标':<12} {'第一次':<15} {'第二次':<15} {'变化':<15} {'变化率':<15}")
    print("-" * 72)
    for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
        if metric in exp1_overall and metric in exp2_overall:
            val1 = exp1_overall[metric]
            val2 = exp2_overall[metric]
            if metric == 'mr':
                change = val1 - val2
                change_pct = (change / val1 * 100) if val1 > 0 else 0
            else:
                change = val2 - val1
                change_pct = (change / val1 * 100) if val1 > 0 else 0
            change_str = f"{change:+.6f}" if change >= 0 else f"{change:.6f}"
            change_pct_str = f"{change_pct:+.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
            print(f"{metric:<12} {val1:<15.6f} {val2:<15.6f} {change_str:<15} {change_pct_str:<15}")
    
    # 统计提升和下降的数据集数量
    comparison = compare_results(exp1_results, exp2_results)
    improved_datasets = defaultdict(int)
    declined_datasets = defaultdict(int)
    
    for dataset, metrics in comparison.items():
        category = classify_dataset_by_command(dataset, command_categories)
        improved_count = 0
        declined_count = 0
        
        for metric, data in metrics.items():
            if metric == 'mr':
                if data['change'] > 0:  # MR下降表示提升
                    improved_count += 1
                elif data['change'] < 0:
                    declined_count += 1
            else:
                if data['change'] > 0:  # 其他指标上升表示提升
                    improved_count += 1
                elif data['change'] < 0:
                    declined_count += 1
        
        if improved_count > declined_count:
            improved_datasets[category] += 1
        elif declined_count > improved_count:
            declined_datasets[category] += 1
    
    print("\n\n数据集提升/下降统计")
    print("=" * 100)
    for category in ['transductive', 'inductive', 'ingram']:
        category_name = category.capitalize()
        total = improved_datasets[category] + declined_datasets[category]
        print(f"{category_name}: 提升 {improved_datasets[category]} 个, 下降 {declined_datasets[category]} 个, 总计 {total} 个")
    
    # 保存详细报告
    with open('comparison_report_detailed.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("实验结果详细对比分析（按command_rel.md分割符分类）\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("数据集分类信息：\n")
        f.write(f"  Transductive: {len(command_categories['transductive'])} 个\n")
        f.write(f"  Inductive: {len(command_categories['inductive'])} 个\n")
        f.write(f"  Ingram: {len(command_categories['ingram'])} 个\n")
        f.write(f"  总计: {sum(len(v) for v in command_categories.values())} 个\n\n")
        
        f.write("第一次实验（2025-12-25）平均值\n")
        f.write("-" * 100 + "\n")
        for category in ['transductive', 'inductive', 'ingram']:
            category_name = category.capitalize()
            if category in exp1_avg:
                f.write(f"\n{category_name} 数据集平均结果:\n")
                for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                    if metric in exp1_avg[category]:
                        f.write(f"  {metric:10s}: {exp1_avg[category][metric]:.6f}\n")
        
        f.write("\n\n第二次实验（2026-01-01）平均值\n")
        f.write("-" * 100 + "\n")
        for category in ['transductive', 'inductive', 'ingram']:
            category_name = category.capitalize()
            if category in exp2_avg:
                f.write(f"\n{category_name} 数据集平均结果:\n")
                for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
                    if metric in exp2_avg[category]:
                        f.write(f"  {metric:10s}: {exp2_avg[category][metric]:.6f}\n")
        
        f.write("\n\n三类数据集平均对比（提升/下降）\n")
        f.write("=" * 100 + "\n")
        for category in ['transductive', 'inductive', 'ingram']:
            category_name = category.capitalize()
            if category in exp1_avg and category in exp2_avg:
                f.write(f"\n{category_name} 数据集:\n")
                f.write(f"{'指标':<12} {'第一次':<15} {'第二次':<15} {'变化':<15} {'变化率':<15}\n")
                f.write("-" * 72 + "\n")
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
                        f.write(f"{metric:<12} {val1:<15.6f} {val2:<15.6f} {change:+.6f} {change_pct:+.2f}%\n")
        
        f.write("\n\n54个数据集整体平均对比\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'指标':<12} {'第一次':<15} {'第二次':<15} {'变化':<15} {'变化率':<15}\n")
        f.write("-" * 72 + "\n")
        for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
            if metric in exp1_overall and metric in exp2_overall:
                val1 = exp1_overall[metric]
                val2 = exp2_overall[metric]
                if metric == 'mr':
                    change = val1 - val2
                    change_pct = (change / val1 * 100) if val1 > 0 else 0
                else:
                    change = val2 - val1
                    change_pct = (change / val1 * 100) if val1 > 0 else 0
                f.write(f"{metric:<12} {val1:<15.6f} {val2:<15.6f} {change:+.6f} {change_pct:+.2f}%\n")
    
    print("\n\n详细对比报告已保存到 comparison_report_detailed.txt")

if __name__ == '__main__':
    main()

