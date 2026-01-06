#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析日志文件，对比不同checkpoint的结果
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_log_file(log_file: str) -> Dict[str, Dict[str, float]]:
    """解析日志文件，提取每个数据集的指标"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配数据集名称和指标
    # 模式：数据集名称 dataset 或 dataset(version)
    # 然后找到对应的指标行
    lines = content.split('\n')
    
    current_dataset = None
    for i, line in enumerate(lines):
        # 匹配数据集名称
        dataset_match = re.search(r'(\w+)(?:\(([^)]+)\))?\s+dataset', line)
        if dataset_match:
            dataset_name = dataset_match.group(1)
            version = dataset_match.group(2) if dataset_match.group(2) else None
            if version:
                current_dataset = f"{dataset_name}({version})"
            else:
                current_dataset = dataset_name
            continue
        
        # 匹配指标
        if current_dataset:
            mr_match = re.search(r'mr:\s+([\d.]+)', line)
            mrr_match = re.search(r'mrr:\s+([\d.]+)', line)
            hits1_match = re.search(r'hits@1:\s+([\d.]+)', line)
            hits3_match = re.search(r'hits@3:\s+([\d.]+)', line)
            hits10_match = re.search(r'hits@10:\s+([\d.]+)', line)
            
            if mr_match or mrr_match or hits1_match:
                if current_dataset not in results:
                    results[current_dataset] = {}
                
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

def get_dataset_groups():
    """根据command_rel.md的分组定义数据集分类"""
    # 第一组：Transductive (第2-26行，分隔线前)
    group1_transductive = [
        'CoDExSmall', 'CoDExLarge', 'NELL995', 'DBpedia100k', 'ConceptNet100k',
        'NELL23k', 'YAGO310', 'Hetionet', 'WDsinger', 'AristoV4',
        'FB15k237_10', 'FB15k237_20', 'FB15k237_50'
    ]
    
    # 第二组：Inductive (分隔线后)
    group2_inductive = [
        'FB15k237Inductive(v1)', 'FB15k237Inductive(v2)', 'FB15k237Inductive(v3)', 'FB15k237Inductive(v4)',
        'WN18RRInductive(v1)', 'WN18RRInductive(v2)', 'WN18RRInductive(v3)', 'WN18RRInductive(v4)',
        'NELLInductive(v1)', 'NELLInductive(v2)', 'NELLInductive(v3)', 'NELLInductive(v4)',
        'ILPC2022(small)', 'ILPC2022(large)',
        'HM(1k)', 'HM(3k)', 'HM(5k)', 'HM(indigo)',
        'FBIngram(25)', 'FBIngram(50)', 'FBIngram(75)', 'FBIngram(100)',
        'WKIngram(25)', 'WKIngram(50)', 'WKIngram(75)', 'WKIngram(100)',
        'NLIngram(0)', 'NLIngram(25)', 'NLIngram(50)', 'NLIngram(75)', 'NLIngram(100)',
        'WikiTopicsMT1(tax)', 'WikiTopicsMT1(health)',
        'WikiTopicsMT2(org)', 'WikiTopicsMT2(sci)',
        'WikiTopicsMT3(art)', 'WikiTopicsMT3(infra)',
        'WikiTopicsMT4(sci)', 'WikiTopicsMT4(health)',
        'Metafam(None)', 'FBNELL(None)'  # 注意：日志中是 Metafam(None) 和 FBNELL(None)
    ]
    
    return {
        'Transductive': group1_transductive,
        'Inductive': group2_inductive
    }

def normalize_dataset_name(name: str) -> str:
    """标准化数据集名称以匹配日志中的格式"""
    # 处理版本号
    if 'FB15k237Inductive' in name:
        return name.replace('FB15k237Inductive', 'FB15k237Inductive')
    if 'WN18RRInductive' in name:
        return name.replace('WN18RRInductive', 'WN18RRInductive')
    if 'NELLInductive' in name:
        return name.replace('NELLInductive', 'NELLInductive')
    if 'ILPC2022' in name:
        return name.replace('ILPC2022', 'ILPC2022')
    if 'HM' in name:
        return name.replace('HM', 'HM')
    if 'FBIngram' in name:
        return name.replace('FBIngram', 'FBIngram')
    if 'WKIngram' in name:
        return name.replace('WKIngram', 'WKIngram')
    if 'NLIngram' in name:
        return name.replace('NLIngram', 'NLIngram')
    if 'WikiTopicsMT' in name:
        return name.replace('WikiTopicsMT', 'WikiTopicsMT')
    if 'Metafam' in name:
        return name.replace('Metafam(None)', 'Metafam')
    if 'FBNELL' in name:
        return name.replace('FBNELL(None)', 'FBNELL')
    return name

def match_dataset_name(target: str, available_keys: List[str]) -> str:
    """匹配数据集名称"""
    # 精确匹配
    if target in available_keys:
        return target
    
    # 处理版本号匹配 - 提取基础名称和版本号
    target_has_ver = '(' in target
    if target_has_ver:
        target_base = target.split('(')[0]
        target_ver = target.split('(')[1].rstrip(')')
    else:
        target_base = target
        target_ver = None
    
    # 尝试匹配
    for key in available_keys:
        key_has_ver = '(' in key
        if key_has_ver:
            key_base = key.split('(')[0]
            key_ver = key.split('(')[1].rstrip(')')
        else:
            key_base = key
            key_ver = None
        
        # 基础名称必须匹配
        if target_base == key_base:
            # 如果都有版本号，版本号必须匹配
            if target_has_ver and key_has_ver:
                # 处理None/null版本号
                if (target_ver == key_ver or 
                    (target_ver in ['None', 'null'] and key_ver in ['None', 'null'])):
                    return key
            # 如果都没有版本号
            elif not target_has_ver and not key_has_ver:
                return key
    
    return None

def calculate_averages(results: Dict[str, Dict[str, float]], datasets: List[str], debug=False) -> Dict[str, float]:
    """计算指定数据集的平均指标"""
    metrics = ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']
    averages = {metric: [] for metric in metrics}
    
    available_keys = list(results.keys())
    matched_count = 0
    
    if debug:
        print(f"  可用数据集数量: {len(available_keys)}")
        print(f"  前5个可用数据集: {available_keys[:5]}")
        print(f"  目标数据集数量: {len(datasets)}")
        print(f"  前5个目标数据集: {datasets[:5]}")
    
    for dataset in datasets:
        matched_key = match_dataset_name(dataset, available_keys)
        if matched_key and matched_key in results:
            result = results[matched_key]
            # 只要有mr, mrr, hits@1就可以计算平均值（hits@3和hits@10可能缺失）
            required_metrics = ['mr', 'mrr', 'hits@1']
            if all(metric in result for metric in required_metrics):
                for metric in metrics:
                    if metric in result:
                        averages[metric].append(result[metric])
                matched_count += 1
                if debug:
                    print(f"  匹配: {dataset} -> {matched_key}")
        elif debug:
            print(f"  未匹配: {dataset} (尝试匹配)")
    
    if debug:
        print(f"  总匹配数: {matched_count}/{len(datasets)}")
    
    # 计算平均值
    avg_results = {}
    for metric in metrics:
        if averages[metric]:
            avg_results[metric] = sum(averages[metric]) / len(averages[metric])
        else:
            avg_results[metric] = 0.0
    
    return avg_results

def calculate_improvement(old_val: float, new_val: float) -> float:
    """计算提升/下降幅度（百分比）"""
    if old_val == 0:
        return 0.0
    return ((new_val - old_val) / old_val) * 100

def main():
    # 解析四个日志文件
    log_files = [
        'run_commands_20251225_041027.log',  # 基线1 (rel_5.pth, Inductive)
        'run_commands_20251225_065140.log',  # 基线2 (rel_5.pth, Transductive)
        'run_commands_20251225_133504.log',  # 基线3 (rel_5.pth, FB15k237_10/20/50)
        'run_commands_20260101_151803.log'   # 新结果 (model_epoch_5.pth)
    ]
    
    print("正在解析日志文件...")
    all_results = {}
    for log_file in log_files:
        print(f"  解析 {log_file}...")
        results = parse_log_file(log_file)
        all_results[log_file] = results
        print(f"    找到 {len(results)} 个数据集的结果")
    
    # 合并基线结果
    baseline_results = {}
    for log_file in log_files[:3]:  # 前三个是基线
        for dataset, metrics in all_results[log_file].items():
            if dataset not in baseline_results:
                baseline_results[dataset] = metrics
            else:
                # 如果已存在，取平均值
                for metric, value in metrics.items():
                    if metric in baseline_results[dataset]:
                        baseline_results[dataset][metric] = (
                            baseline_results[dataset][metric] + value
                        ) / 2
                    else:
                        baseline_results[dataset][metric] = value
    
    new_results = all_results[log_files[3]]  # 最新的结果
    
    # 获取数据集分组
    groups = get_dataset_groups()
    
    # 分析每个分组
    print("\n" + "="*80)
    print("结果对比分析")
    print("="*80)
    
    metrics = ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']
    
    # 对每个分组进行分析
    for group_name, dataset_list in groups.items():
        print(f"\n【{group_name} 数据集组】")
        print("-" * 80)
        
        # 计算基线平均值
        print(f"\n正在匹配 {group_name} 组的数据集...")
        baseline_avg = calculate_averages(baseline_results, dataset_list, debug=True)
        new_avg = calculate_averages(new_results, dataset_list, debug=True)
        
        print(f"\n基线平均值 (rel_5.pth):")
        for metric in metrics:
            print(f"  {metric:12s}: {baseline_avg[metric]:.6f}")
        
        print(f"\n新结果平均值 (model_epoch_5.pth):")
        for metric in metrics:
            print(f"  {metric:12s}: {new_avg[metric]:.6f}")
        
        print(f"\n变化幅度 (%):")
        for metric in metrics:
            improvement = calculate_improvement(baseline_avg[metric], new_avg[metric])
            if metric == 'mr':  # MR越小越好，所以反向计算
                improvement = -improvement
            sign = "+" if improvement > 0 else ""
            print(f"  {metric:12s}: {sign}{improvement:.4f}%")
    
    # 计算所有数据集的总体平均
    print(f"\n\n【所有数据集总体平均】")
    print("="*80)
    
    all_datasets = groups['Transductive'] + groups['Inductive']
    baseline_total = calculate_averages(baseline_results, all_datasets)
    new_total = calculate_averages(new_results, all_datasets)
    
    print(f"\n基线平均值 (rel_5.pth):")
    for metric in metrics:
        print(f"  {metric:12s}: {baseline_total[metric]:.6f}")
    
    print(f"\n新结果平均值 (model_epoch_5.pth):")
    for metric in metrics:
        print(f"  {metric:12s}: {new_total[metric]:.6f}")
    
    print(f"\n总体变化幅度 (%):")
    for metric in metrics:
        improvement = calculate_improvement(baseline_total[metric], new_total[metric])
        if metric == 'mr':  # MR越小越好
            improvement = -improvement
        sign = "+" if improvement > 0 else ""
        print(f"  {metric:12s}: {sign}{improvement:.4f}%")
    
    # 生成详细对比表
    print(f"\n\n【详细数据集对比】")
    print("="*80)
    print(f"{'数据集':<40} {'指标':<8} {'基线':<12} {'新结果':<12} {'变化%':<10}")
    print("-" * 80)
    
    for dataset in all_datasets:
        # 找到匹配的数据集名称
        baseline_key = match_dataset_name(dataset, list(baseline_results.keys()))
        new_key = match_dataset_name(dataset, list(new_results.keys()))
        
        baseline_metrics = baseline_results.get(baseline_key) if baseline_key else None
        new_metrics = new_results.get(new_key) if new_key else None
        
        if baseline_metrics and new_metrics:
            for metric in metrics:
                old_val = baseline_metrics.get(metric, 0)
                new_val = new_metrics.get(metric, 0)
                improvement = calculate_improvement(old_val, new_val)
                if metric == 'mr':
                    improvement = -improvement
                sign = "+" if improvement > 0 else ""
                print(f"{dataset:<40} {metric:<8} {old_val:<12.6f} {new_val:<12.6f} {sign}{improvement:<9.4f}%")

if __name__ == '__main__':
    main()
