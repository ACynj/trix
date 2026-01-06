#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为对比报告添加相对提升列
"""

import re

def add_relative_improvement(input_file: str, output_file: str):
    """为报告添加相对提升列"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 匹配表格行: | 数据集 | 指标 | 值1 | 值2 | 差异 |
        match = re.match(r'^\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([+-]?[\d.]+)\s+\|', line)
        
        if match:
            dataset = match.group(1).strip()
            metric = match.group(2).strip()
            val1 = float(match.group(3))
            val2 = float(match.group(4))
            diff = float(match.group(5))
            
            # 计算相对提升
            if val1 != 0:
                relative_improvement = (diff / val1) * 100
                rel_str = f"{relative_improvement:+.2f}%"
            else:
                rel_str = "N/A"
            
            # 更新行，添加相对提升列
            new_line = f"| {dataset} | {metric} | {val1:.4f} | {val2:.4f} | {diff:+.4f} | {rel_str} |\n"
            new_lines.append(new_line)
        elif line.startswith("| 数据集 | 指标 |"):
            # 更新表头
            new_lines.append("| 数据集 | 指标 | inference_rel.log | run_commands_20260102_085655.log | 差异 | 相对提升 |\n")
        elif line.startswith("|--------|------|"):
            # 更新分隔线
            new_lines.append("|--------|------|-------------------|----------------------------------|------|----------|\n")
        else:
            new_lines.append(line)
        
        i += 1
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"报告已更新并保存到: {output_file}")

if __name__ == "__main__":
    input_file = "/T20030104/ynj/TRIX/comparison_report_3.md"
    output_file = "/T20030104/ynj/TRIX/comparison_report_3.md"
    
    add_relative_improvement(input_file, output_file)



