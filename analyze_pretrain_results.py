#!/usr/bin/env python3
"""
分析预训练模型在下游任务上的表现，优化预训练配置
"""

import re
from collections import defaultdict

def parse_log_file(log_file):
    """解析日志文件，提取数据集和MRR指标"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    current_dataset = None
    current_version = None
    
    for i, line in enumerate(lines):
        # 匹配数据集名称和版本
        dataset_match = re.search(r'(\w+)(?:\(([^)]+)\))?\s+dataset', line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            current_version = dataset_match.group(2) if dataset_match.group(2) else None
            continue
        
        # 匹配MRR
        mrr_match = re.search(r'mrr:\s+([\d.]+)', line)
        if mrr_match and current_dataset:
            mrr = float(mrr_match.group(1))
            key = f"{current_dataset}"
            if current_version:
                key += f"({current_version})"
            else:
                key += "(None)"
            
            results[key] = mrr
    
    return results

def analyze_results():
    """分析所有结果并生成优化建议"""
    
    # 解析所有日志
    inductive = parse_log_file('run_commands_20251225_041027.log')
    transductive1 = parse_log_file('run_commands_20251225_065140.log')
    transductive2 = parse_log_file('run_commands_20251225_133504.log')
    
    all_results = {**inductive, **transductive1, **transductive2}
    
    # 分类分析
    excellent = []  # MRR > 0.9
    good = []       # 0.8 <= MRR < 0.9
    fair = []       # 0.6 <= MRR < 0.8
    poor = []       # MRR < 0.6
    
    for dataset, mrr in all_results.items():
        if mrr >= 0.9:
            excellent.append((dataset, mrr))
        elif mrr >= 0.8:
            good.append((dataset, mrr))
        elif mrr >= 0.6:
            fair.append((dataset, mrr))
        else:
            poor.append((dataset, mrr))
    
    # 计算统计信息
    all_mrr = list(all_results.values())
    avg_mrr = sum(all_mrr) / len(all_mrr)
    min_mrr = min(all_mrr)
    max_mrr = max(all_mrr)
    
    print("=" * 80)
    print("预训练模型下游任务表现分析")
    print("=" * 80)
    print(f"\n总数据集数: {len(all_results)}")
    print(f"平均MRR: {avg_mrr:.4f}")
    print(f"最低MRR: {min_mrr:.4f}")
    print(f"最高MRR: {max_mrr:.4f}")
    
    print(f"\n【优秀表现 (MRR >= 0.9)】: {len(excellent)}个")
    for dataset, mrr in sorted(excellent, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {dataset:50s} MRR: {mrr:.4f}")
    
    print(f"\n【良好表现 (0.8 <= MRR < 0.9)】: {len(good)}个")
    for dataset, mrr in sorted(good, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {dataset:50s} MRR: {mrr:.4f}")
    
    print(f"\n【一般表现 (0.6 <= MRR < 0.8)】: {len(fair)}个")
    for dataset, mrr in sorted(fair, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {dataset:50s} MRR: {mrr:.4f}")
    
    print(f"\n【较差表现 (MRR < 0.6)】: {len(poor)}个")
    for dataset, mrr in sorted(poor, key=lambda x: x[1], reverse=True):
        print(f"  {dataset:50s} MRR: {mrr:.4f}")
    
    # 分析问题
    print("\n" + "=" * 80)
    print("问题分析")
    print("=" * 80)
    
    if len(poor) > 0:
        print(f"\n⚠️  有{len(poor)}个数据集表现较差（MRR < 0.6）")
        print("   可能原因：")
        print("   1. 预训练数据与这些数据集差异较大")
        print("   2. beta值过高，限制了机制z的灵活性")
        print("   3. 需要更长的预训练轮次")
    
    if avg_mrr < 0.75:
        print(f"\n⚠️  平均MRR较低（{avg_mrr:.4f}），说明预训练效果不够好")
        print("   优化建议：")
        print("   1. 降低beta值（当前1e-3可能过高）")
        print("   2. 增加预训练轮次")
        print("   3. 调整学习率")
    
    # 生成优化配置建议
    print("\n" + "=" * 80)
    print("优化配置建议")
    print("=" * 80)
    
    if avg_mrr < 0.75:
        print("\n【推荐配置1：平衡优化】")
        print("  beta: 2.5e-4  # 降低KL约束，提升灵活性")
        print("  lr: 4.0e-4    # 适中学习率")
        print("  weight_decay: 1.0e-5")
        print("  num_epoch: 25  # 增加训练轮次")
        
        print("\n【推荐配置2：激进优化】")
        print("  beta: 1.5e-4  # 更低的KL约束")
        print("  lr: 3.5e-4    # 较低学习率，配合更长训练")
        print("  weight_decay: 1.5e-5")
        print("  num_epoch: 30  # 最长训练")
    else:
        print("\n【推荐配置：精细调优】")
        print("  beta: 3.0e-4  # 轻微降低，保持正则化")
        print("  lr: 4.0e-4")
        print("  weight_decay: 1.0e-5")
        print("  num_epoch: 20  # 适度增加")
    
    return all_results, avg_mrr

if __name__ == "__main__":
    results, avg_mrr = analyze_results()



