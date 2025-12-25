#!/usr/bin/env python3
"""
对比实验结果脚本
提取MRR和Hits@10指标，生成对比报告
"""

import re
from collections import defaultdict

def parse_log_file(log_file):
    """解析日志文件，提取数据集和指标"""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配数据集名称和版本
    dataset_pattern = r'(\w+)(?:\(([^)]+)\))?\s+dataset'
    # 匹配指标
    metric_pattern = r'(mrr|hits@10):\s+([\d.]+)'
    
    lines = content.split('\n')
    current_dataset = None
    current_version = None
    
    for i, line in enumerate(lines):
        # 查找数据集名称
        dataset_match = re.search(dataset_pattern, line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            current_version = dataset_match.group(2) if dataset_match.group(2) else None
            continue
        
        # 查找指标
        if 'mrr:' in line or 'hits@10:' in line:
            metrics = {}
            for match in re.finditer(metric_pattern, line):
                metric_name = match.group(1)
                metric_value = float(match.group(2))
                metrics[metric_name] = metric_value
            
            if metrics and current_dataset:
                key = f"{current_dataset}"
                if current_version:
                    key += f"({current_version})"
                else:
                    key += "(None)"
                
                if key not in results:
                    results[key] = {}
                results[key].update(metrics)
    
    return results

def main():
    # 解析所有日志文件
    baseline = parse_log_file('inference_rel.log')
    inductive = parse_log_file('run_commands_20251225_041027.log')
    transductive1 = parse_log_file('run_commands_20251225_065140.log')
    transductive2 = parse_log_file('run_commands_20251225_133504.log')
    
    # 合并transductive结果
    transductive = {**transductive1, **transductive2}
    
    # 找出所有共同的数据集
    all_datasets = set(baseline.keys()) | set(inductive.keys()) | set(transductive.keys())
    
    # 生成对比报告
    print("=" * 100)
    print("实验结果对比报告 - MRR 和 Hits@10")
    print("=" * 100)
    print(f"\n基准方法: TRIX (inference_rel.log)")
    print(f"新方法1: Inductive (run_commands_20251225_041027.log)")
    print(f"新方法2: Transductive (run_commands_20251225_065140.log + 133504.log)")
    print("\n" + "=" * 100)
    
    # 按数据集分组
    inductive_results = []
    transductive_results = []
    
    for dataset in sorted(all_datasets):
        base_mrr = baseline.get(dataset, {}).get('mrr', None)
        base_h10 = baseline.get(dataset, {}).get('hits@10', None)
        
        ind_mrr = inductive.get(dataset, {}).get('mrr', None)
        ind_h10 = inductive.get(dataset, {}).get('hits@10', None)
        
        trans_mrr = transductive.get(dataset, {}).get('mrr', None)
        trans_h10 = transductive.get(dataset, {}).get('hits@10', None)
        
        if base_mrr is None:
            continue
        
        # Inductive对比
        if ind_mrr is not None:
            mrr_diff = ind_mrr - base_mrr
            h10_diff = ind_h10 - base_h10 if ind_h10 and base_h10 else None
            inductive_results.append({
                'dataset': dataset,
                'base_mrr': base_mrr,
                'ind_mrr': ind_mrr,
                'mrr_diff': mrr_diff,
                'base_h10': base_h10,
                'ind_h10': ind_h10,
                'h10_diff': h10_diff
            })
        
        # Transductive对比
        if trans_mrr is not None:
            mrr_diff = trans_mrr - base_mrr
            h10_diff = trans_h10 - base_h10 if trans_h10 and base_h10 else None
            transductive_results.append({
                'dataset': dataset,
                'base_mrr': base_mrr,
                'trans_mrr': trans_mrr,
                'mrr_diff': mrr_diff,
                'base_h10': base_h10,
                'trans_h10': trans_h10,
                'h10_diff': h10_diff
            })
    
    # 打印Inductive对比
    print("\n【归纳数据集 (Inductive) 对比】")
    print("-" * 100)
    print(f"{'数据集':<40} {'基准MRR':<12} {'新方法MRR':<12} {'ΔMRR':<10} {'基准H@10':<12} {'新方法H@10':<12} {'ΔH@10':<10}")
    print("-" * 100)
    
    for r in inductive_results:
        mrr_sign = "+" if r['mrr_diff'] >= 0 else ""
        h10_sign = "+" if r['h10_diff'] and r['h10_diff'] >= 0 else ""
        h10_str = f"{r['h10_diff']:+.4f}" if r['h10_diff'] is not None else "N/A"
        
        print(f"{r['dataset']:<40} {r['base_mrr']:<12.5f} {r['ind_mrr']:<12.5f} {mrr_sign}{r['mrr_diff']:+.4f}    "
              f"{r['base_h10']:<12.5f} {r['ind_h10']:<12.5f} {h10_str}")
    
    # 统计Inductive
    ind_improved_mrr = sum(1 for r in inductive_results if r['mrr_diff'] > 0)
    ind_improved_h10 = sum(1 for r in inductive_results if r['h10_diff'] and r['h10_diff'] > 0)
    ind_degraded_mrr = sum(1 for r in inductive_results if r['mrr_diff'] < 0)
    ind_degraded_h10 = sum(1 for r in inductive_results if r['h10_diff'] and r['h10_diff'] < 0)
    
    print(f"\n归纳数据集统计: 共{len(inductive_results)}个数据集")
    print(f"  MRR提升: {ind_improved_mrr}个, 下降: {ind_degraded_mrr}个")
    print(f"  Hits@10提升: {ind_improved_h10}个, 下降: {ind_degraded_h10}个")
    
    # 打印Transductive对比
    print("\n\n【转导数据集 (Transductive) 对比】")
    print("-" * 100)
    print(f"{'数据集':<40} {'基准MRR':<12} {'新方法MRR':<12} {'ΔMRR':<10} {'基准H@10':<12} {'新方法H@10':<12} {'ΔH@10':<10}")
    print("-" * 100)
    
    for r in transductive_results:
        mrr_sign = "+" if r['mrr_diff'] >= 0 else ""
        h10_sign = "+" if r['h10_diff'] and r['h10_diff'] >= 0 else ""
        h10_str = f"{r['h10_diff']:+.4f}" if r['h10_diff'] is not None else "N/A"
        
        print(f"{r['dataset']:<40} {r['base_mrr']:<12.5f} {r['trans_mrr']:<12.5f} {mrr_sign}{r['mrr_diff']:+.4f}    "
              f"{r['base_h10']:<12.5f} {r['trans_h10']:<12.5f} {h10_str}")
    
    # 统计Transductive
    trans_improved_mrr = sum(1 for r in transductive_results if r['mrr_diff'] > 0)
    trans_improved_h10 = sum(1 for r in transductive_results if r['h10_diff'] and r['h10_diff'] > 0)
    trans_degraded_mrr = sum(1 for r in transductive_results if r['mrr_diff'] < 0)
    trans_degraded_h10 = sum(1 for r in transductive_results if r['h10_diff'] and r['h10_diff'] < 0)
    
    print(f"\n转导数据集统计: 共{len(transductive_results)}个数据集")
    print(f"  MRR提升: {trans_improved_mrr}个, 下降: {trans_degraded_mrr}个")
    print(f"  Hits@10提升: {trans_improved_h10}个, 下降: {trans_degraded_h10}个")
    
    # 生成建议
    print("\n\n" + "=" * 100)
    print("【分析建议】")
    print("=" * 100)
    
    # 找出表现最好和最差的数据集
    ind_best = max(inductive_results, key=lambda x: x['mrr_diff']) if inductive_results else None
    ind_worst = min(inductive_results, key=lambda x: x['mrr_diff']) if inductive_results else None
    
    trans_best = max(transductive_results, key=lambda x: x['mrr_diff']) if transductive_results else None
    trans_worst = min(transductive_results, key=lambda x: x['mrr_diff']) if transductive_results else None
    
    print("\n1. 归纳数据集表现:")
    if ind_best:
        h10_str = f"{ind_best['h10_diff']:+.4f}" if ind_best['h10_diff'] is not None else "N/A"
        print(f"   ✓ 最佳提升: {ind_best['dataset']} (MRR: {ind_best['mrr_diff']:+.4f}, H@10: {h10_str})")
    if ind_worst:
        h10_str = f"{ind_worst['h10_diff']:+.4f}" if ind_worst['h10_diff'] is not None else "N/A"
        print(f"   ✗ 最大下降: {ind_worst['dataset']} (MRR: {ind_worst['mrr_diff']:+.4f}, H@10: {h10_str})")
    
    print("\n2. 转导数据集表现:")
    if trans_best:
        h10_str = f"{trans_best['h10_diff']:+.4f}" if trans_best['h10_diff'] is not None else "N/A"
        print(f"   ✓ 最佳提升: {trans_best['dataset']} (MRR: {trans_best['mrr_diff']:+.4f}, H@10: {h10_str})")
    if trans_worst:
        h10_str = f"{trans_worst['h10_diff']:+.4f}" if trans_worst['h10_diff'] is not None else "N/A"
        print(f"   ✗ 最大下降: {trans_worst['dataset']} (MRR: {trans_worst['mrr_diff']:+.4f}, H@10: {h10_str})")
    
    # 计算平均提升
    if inductive_results:
        avg_mrr_diff = sum(r['mrr_diff'] for r in inductive_results) / len(inductive_results)
        avg_h10_diff = sum(r['h10_diff'] for r in inductive_results if r['h10_diff'] is not None) / len([r for r in inductive_results if r['h10_diff'] is not None])
        print(f"\n3. 归纳数据集平均变化:")
        print(f"   平均MRR变化: {avg_mrr_diff:+.4f}")
        print(f"   平均Hits@10变化: {avg_h10_diff:+.4f}")
    
    if transductive_results:
        avg_mrr_diff = sum(r['mrr_diff'] for r in transductive_results) / len(transductive_results)
        avg_h10_diff = sum(r['h10_diff'] for r in transductive_results if r['h10_diff'] is not None) / len([r for r in transductive_results if r['h10_diff'] is not None])
        print(f"\n4. 转导数据集平均变化:")
        print(f"   平均MRR变化: {avg_mrr_diff:+.4f}")
        print(f"   平均Hits@10变化: {avg_h10_diff:+.4f}")
    
    print("\n5. 改进建议:")
    print("   - 对于表现下降的数据集，建议:")
    print("     * 降低beta值（从1e-3降到1e-4或5e-4）")
    print("     * 检查数据集规模，大规模数据集可能需要更小的beta")
    print("     * 考虑数据集特定的超参数调优")
    print("   - 对于表现提升的数据集，建议:")
    print("     * 保持当前配置")
    print("     * 分析提升原因，看是否可以应用到其他数据集")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()

