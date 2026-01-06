#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 inference_rel.log 和 run_commands_20260102_085655.log
"""

from compare_logs import parse_log_file, get_dataset_categories, calculate_averages
from generate_comparison_report import generate_markdown_report

if __name__ == "__main__":
    log_file1 = "/T20030104/ynj/TRIX/inference_rel.log"
    log_file2 = "/T20030104/ynj/TRIX/run_commands_20260102_085655.log"
    
    print("正在解析日志文件...")
    results1 = parse_log_file(log_file1)
    results2 = parse_log_file(log_file2)
    
    print(f"文件1找到 {len(results1)} 个数据集的结果")
    print(f"文件2找到 {len(results2)} 个数据集的结果")
    
    print("正在生成Markdown报告...")
    report = generate_markdown_report(
        results1, 
        results2,
        name1="inference_rel.log",
        name2="run_commands_20260102_085655.log"
    )
    
    output_file = "/T20030104/ynj/TRIX/comparison_report_3.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {output_file}")



