#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两个日志文件的结果
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict

def parse_log_file(log_file: str) -> Dict[str, Dict[str, float]]:
    """解析日志文件，提取每个数据集的指标"""
    results = {}
    current_dataset = None
    in_eval_section = False
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 方法1: 从"dataset"行提取完整名称（如 "FB15k237Inductive(v1) dataset"）
        dataset_line_match = re.search(r'(\S+\([^)]+\)|\S+)\s+dataset', line)
        if dataset_line_match:
            dataset_name = dataset_line_match.group(1)
            current_dataset = dataset_name
            in_eval_section = False
        
        # 方法2: 从命令中提取（作为备选）
        if not current_dataset:
            dataset_match = re.search(r'--dataset\s+(\S+)', line)
            if dataset_match:
                dataset_name = dataset_match.group(1)
                # 检查是否有version参数
                version_match = re.search(r'--version\s+(\S+)', line)
                if version_match:
                    version = version_match.group(1)
                    if version != 'null':
                        dataset_name = f"{dataset_name}({version})"
                    else:
                        dataset_name = f"{dataset_name}(None)"
                
                current_dataset = dataset_name
                in_eval_section = False
        
        # 检测评估开始
        if 'Evaluate on test' in line or 'Evaluate' in line:
            in_eval_section = True
        
        # 查找指标结果（在评估部分）
        if in_eval_section and current_dataset:
            if current_dataset not in results:
                results[current_dataset] = {}
            
            mrr_match = re.search(r'mrr:\s+([\d.]+)', line)
            hits1_match = re.search(r'hits@1:\s+([\d.]+)', line)
            hits3_match = re.search(r'hits@3:\s+([\d.]+)', line)
            hits10_match = re.search(r'hits@10:\s+([\d.]+)', line)
            
            if mrr_match:
                results[current_dataset]['mrr'] = float(mrr_match.group(1))
            if hits1_match:
                results[current_dataset]['hits@1'] = float(hits1_match.group(1))
            if hits3_match:
                results[current_dataset]['hits@3'] = float(hits3_match.group(1))
            if hits10_match:
                results[current_dataset]['hits@10'] = float(hits10_match.group(1))
        
        i += 1
    
    return results

def get_dataset_categories() -> Dict[str, List[str]]:
    """根据command_rel.md定义数据集分类"""
    # Transductive (13个数据集)
    transductive = [
        'CoDExSmall', 'CoDExLarge', 'NELL995', 'DBpedia100k', 'ConceptNet100k',
        'NELL23k', 'YAGO310', 'Hetionet', 'WDsinger', 'AristoV4',
        'FB15k237_10', 'FB15k237_20', 'FB15k237_50'
    ]
    
    # Inductive(e) (18个数据集) - 只涉及实体归纳
    inductive_e = [
        'FB15k237Inductive(v1)', 'FB15k237Inductive(v2)', 'FB15k237Inductive(v3)', 'FB15k237Inductive(v4)',
        'WN18RRInductive(v1)', 'WN18RRInductive(v2)', 'WN18RRInductive(v3)', 'WN18RRInductive(v4)',
        'NELLInductive(v1)', 'NELLInductive(v2)', 'NELLInductive(v3)', 'NELLInductive(v4)',
        'ILPC2022(small)', 'ILPC2022(large)',
        'HM(1k)', 'HM(3k)', 'HM(5k)', 'HM(indigo)'
    ]
    
    # Inductive(e,r) (23个数据集) - 涉及实体和关系的归纳
    inductive_er = [
        'FBIngram(25)', 'FBIngram(50)', 'FBIngram(75)', 'FBIngram(100)',
        'WKIngram(25)', 'WKIngram(50)', 'WKIngram(75)', 'WKIngram(100)',
        'NLIngram(0)', 'NLIngram(25)', 'NLIngram(50)', 'NLIngram(75)', 'NLIngram(100)',
        'WikiTopicsMT1(tax)', 'WikiTopicsMT1(health)',
        'WikiTopicsMT2(org)', 'WikiTopicsMT2(sci)',
        'WikiTopicsMT3(art)', 'WikiTopicsMT3(infra)',
        'WikiTopicsMT4(sci)', 'WikiTopicsMT4(health)',
        'Metafam(None)', 'FBNELL(None)'
    ]
    
    return {
        'transductive': transductive,
        'inductive_e': inductive_e,
        'inductive_er': inductive_er
    }

def normalize_dataset_name(name: str) -> str:
    """标准化数据集名称以匹配分类"""
    # 移除可能的空格和特殊字符
    name = name.strip()
    return name

def calculate_averages(results: Dict[str, Dict[str, float]], 
                      datasets: List[str]) -> Dict[str, float]:
    """计算指定数据集的平均指标"""
    metrics = ['mrr', 'hits@1', 'hits@3', 'hits@10']
    averages = {metric: [] for metric in metrics}
    
    for dataset in datasets:
        # 尝试匹配数据集名称（支持部分匹配）
        matched = False
        for result_dataset in results.keys():
            # 检查是否匹配
            if dataset in result_dataset or result_dataset in dataset:
                matched = True
                for metric in metrics:
                    if metric in results[result_dataset]:
                        averages[metric].append(results[result_dataset][metric])
                break
        
        if not matched:
            # 尝试直接匹配
            if dataset in results:
                for metric in metrics:
                    if metric in results[dataset]:
                        averages[metric].append(results[dataset][metric])
    
    # 计算平均值
    result = {}
    for metric in metrics:
        if averages[metric]:
            result[metric] = sum(averages[metric]) / len(averages[metric])
        else:
            result[metric] = 0.0
    
    return result

def compare_results(results1: Dict[str, Dict[str, float]], 
                    results2: Dict[str, Dict[str, float]],
                    name1: str = "File1",
                    name2: str = "File2"):
    """对比两个结果集"""
    
    categories = get_dataset_categories()
    all_datasets = categories['transductive'] + categories['inductive_e'] + categories['inductive_er']
    
    print("=" * 100)
    print("日志文件对比分析")
    print("=" * 100)
    print(f"\n文件1: {name1}")
    print(f"文件2: {name2}\n")
    
    # 1. 每个数据集的详细对比
    print("\n" + "=" * 100)
    print("1. 每个数据集的详细对比")
    print("=" * 100)
    print(f"{'数据集':<40} {'指标':<10} {name1:<15} {name2:<15} {'差异':<15}")
    print("-" * 100)
    
    metrics = ['mrr', 'hits@1', 'hits@3', 'hits@10']
    
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
                print(f"{dataset:<40} {metric:<10} {val1:<15.6f} {val2:<15.6f} {diff_str:<15}")
        elif matched_dataset1:
            print(f"{dataset:<40} {'N/A':<10} {'Found':<15} {'Missing':<15} {'N/A':<15}")
        elif matched_dataset2:
            print(f"{dataset:<40} {'N/A':<10} {'Missing':<15} {'Found':<15} {'N/A':<15}")
    
    # 2. 按类别进行平均对比
    print("\n" + "=" * 100)
    print("2. 按类别进行平均对比")
    print("=" * 100)
    
    for category_name, category_datasets in categories.items():
        print(f"\n{category_name.upper()} ({len(category_datasets)}个数据集):")
        print("-" * 100)
        print(f"{'指标':<10} {name1:<15} {name2:<15} {'差异':<15} {'相对提升':<15}")
        print("-" * 100)
        
        avg1 = calculate_averages(results1, category_datasets)
        avg2 = calculate_averages(results2, category_datasets)
        
        for metric in metrics:
            val1 = avg1.get(metric, 0.0)
            val2 = avg2.get(metric, 0.0)
            diff = val2 - val1
            rel_improve = (diff / val1 * 100) if val1 != 0 else 0.0
            print(f"{metric:<10} {val1:<15.6f} {val2:<15.6f} {diff:+.6f} {rel_improve:+.2f}%")
    
    # 3. 全部数据集平均对比
    print("\n" + "=" * 100)
    print("3. 全部数据集平均对比 (54个数据集)")
    print("=" * 100)
    print(f"{'指标':<10} {name1:<15} {name2:<15} {'差异':<15} {'相对提升':<15}")
    print("-" * 100)
    
    avg1_all = calculate_averages(results1, all_datasets)
    avg2_all = calculate_averages(results2, all_datasets)
    
    for metric in metrics:
        val1 = avg1_all.get(metric, 0.0)
        val2 = avg2_all.get(metric, 0.0)
        diff = val2 - val1
        rel_improve = (diff / val1 * 100) if val1 != 0 else 0.0
        print(f"{metric:<10} {val1:<15.6f} {val2:<15.6f} {diff:+.6f} {rel_improve:+.2f}%")

if __name__ == "__main__":
    log_file1 = "/T20030104/ynj/TRIX/inference_rel.log"
    log_file2 = "/T20030104/ynj/TRIX/run_commands_20260101_035905.log"
    
    print("正在解析日志文件...")
    results1 = parse_log_file(log_file1)
    results2 = parse_log_file(log_file2)
    
    print(f"文件1找到 {len(results1)} 个数据集的结果")
    print(f"文件2找到 {len(results2)} 个数据集的结果")
    
    # 打印找到的数据集
    print("\n文件1中的数据集:")
    for d in sorted(results1.keys()):
        print(f"  - {d}")
    
    print("\n文件2中的数据集:")
    for d in sorted(results2.keys()):
        print(f"  - {d}")
    
    compare_results(results1, results2, "inference_rel.log", "run_commands_20260101_035905.log")

