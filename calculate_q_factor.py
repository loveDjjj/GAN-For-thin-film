#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q值计算程序
-----------
读取Excel文件中的波长和吸收数据，计算Q值并输出相关参数

用法示例:
    python calculate_q_factor.py --file "generated_samples/best_samples_20250402_103024/best_sample_4_absorption.xlsx" --center 5.3
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算吸收光谱的Q值')
    parser.add_argument('--file', type=str, default="generated_samples\\best_samples_20250413_184029\\best_sample_3_absorption.xlsx",
                        help='Excel文件路径，包含波长和吸收数据')
    parser.add_argument('--center', type=float, default=4.26,
                        help='指定的初始中心波长 (μm)')
    parser.add_argument('--range', type=float, default=0.1,
                        help='搜索真实峰值的波长范围 (μm)，默认为±0.2μm')
    parser.add_argument('--output_dir', type=str, default='q_results',
                        help='输出结果的目录')
    parser.add_argument('--plot', action='store_true',
                        help='生成并保存图表')
    return parser.parse_args()

def load_excel_data(file_path):
    """
    加载Excel文件中的波长和吸收数据
    
    参数:
        file_path: Excel文件路径
        
    返回:
        wavelengths: 波长数组
        absorption: 吸收率数组
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 获取前两列数据（波长和吸收率）
        column_names = df.columns
        wavelengths = df[column_names[0]].values  # 第一列为波长
        absorption = df[column_names[1]].values   # 第二列为吸收率
        
        print(f"成功读取文件: {file_path}")
        print(f"波长列: {column_names[0]}, 吸收率列: {column_names[1]}")
        
        return wavelengths, absorption
    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise

def find_peak(wavelengths, absorption, initial_center, search_range=0.2):
    """
    在指定范围内寻找真实峰值
    
    参数:
        wavelengths: 波长数组
        absorption: 吸收率数组
        initial_center: 初始中心波长
        search_range: 搜索范围 (μm)
        
    返回:
        peak_wavelength: 峰值波长
        peak_absorption: 峰值吸收率
    """
    # 确定搜索范围
    min_wavelength = initial_center - search_range
    max_wavelength = initial_center + search_range
    
    # 找出范围内的波长索引
    range_indices = np.where((wavelengths >= min_wavelength) & (wavelengths <= max_wavelength))[0]
    
    if len(range_indices) == 0:
        print(f"警告: 在指定范围 ({min_wavelength}-{max_wavelength} μm) 内没有找到数据点")
        return initial_center, 0
    
    # 在范围内找出吸收率最大值的索引
    peak_idx = range_indices[np.argmax(absorption[range_indices])]
    peak_wavelength = wavelengths[peak_idx]
    peak_absorption = absorption[peak_idx]
    
    print(f"在范围 {min_wavelength:.4f}-{max_wavelength:.4f} μm 内找到峰值:")
    print(f"  峰值波长: {peak_wavelength:.4f} μm")
    print(f"  峰值吸收率: {peak_absorption:.4f}")
    
    return peak_wavelength, peak_absorption

def calculate_fwhm(wavelengths, absorption, peak_wavelength, peak_absorption):
    """
    计算半高全宽 (FWHM)
    
    参数:
        wavelengths: 波长数组
        absorption: 吸收率数组
        peak_wavelength: 峰值波长
        peak_absorption: 峰值吸收率
        
    返回:
        fwhm: 半高全宽
        left_wavelength: 左半高波长
        right_wavelength: 右半高波长
    """
    # 计算半高值
    half_max = peak_absorption / 2
    
    # 找出峰值的索引
    peak_idx = np.argmin(np.abs(wavelengths - peak_wavelength))
    
    # 向左找半高位置
    left_idx = peak_idx
    for i in range(peak_idx, 0, -1):
        if absorption[i] <= half_max:
            left_idx = i
            break
    
    # 向右找半高位置
    right_idx = peak_idx
    for i in range(peak_idx, len(wavelengths)-1):
        if absorption[i] <= half_max:
            right_idx = i
            break
    
    # 使用线性插值找到更精确的半高波长
    if left_idx < peak_idx:
        left_wavelength = wavelengths[left_idx] + (wavelengths[left_idx+1] - wavelengths[left_idx]) * \
                        (half_max - absorption[left_idx]) / (absorption[left_idx+1] - absorption[left_idx])
    else:
        left_wavelength = wavelengths[left_idx]
    
    if right_idx > peak_idx:
        right_wavelength = wavelengths[right_idx-1] + (wavelengths[right_idx] - wavelengths[right_idx-1]) * \
                        (half_max - absorption[right_idx-1]) / (absorption[right_idx] - absorption[right_idx-1])
    else:
        right_wavelength = wavelengths[right_idx]
    
    fwhm = right_wavelength - left_wavelength
    
    print(f"半高全宽 (FWHM) 计算结果:")
    print(f"  左半高波长: {left_wavelength:.4f} μm")
    print(f"  右半高波长: {right_wavelength:.4f} μm")
    print(f"  FWHM: {fwhm:.4f} μm")
    
    return fwhm, left_wavelength, right_wavelength

def calculate_q_factor(peak_wavelength, fwhm):
    """
    计算Q值
    
    参数:
        peak_wavelength: 峰值波长
        fwhm: 半高全宽
        
    返回:
        q_factor: Q值
    """
    if fwhm <= 0:
        print("警告: FWHM <= 0，无法计算Q值")
        return 0
    
    q_factor = peak_wavelength / fwhm
    print(f"Q值计算结果: {q_factor:.2f}")
    
    return q_factor

def plot_spectrum_with_fwhm(wavelengths, absorption, peak_wavelength, 
                           left_wavelength, right_wavelength, peak_absorption, 
                           q_factor, output_path):
    """
    绘制光谱图并标记FWHM
    
    参数:
        wavelengths: 波长数组
        absorption: 吸收率数组
        peak_wavelength: 峰值波长
        left_wavelength: 左半高波长
        right_wavelength: 右半高波长
        peak_absorption: 峰值吸收率
        q_factor: Q值
        output_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制全光谱
    plt.plot(wavelengths, absorption, 'b-', linewidth=2, label='Absorption Spectrum')
    
    # 标记峰值
    plt.plot(peak_wavelength, peak_absorption, 'ro', markersize=8, label='Peak')
    
    # 标记半高宽
    plt.plot([left_wavelength, right_wavelength], 
            [peak_absorption/2, peak_absorption/2], 
            'g-', linewidth=2, label='FWHM')
    
    plt.vlines([left_wavelength, right_wavelength], 0, peak_absorption/2, 
              colors='g', linestyles='dashed')
    
    # 设置标题和标签
    plt.title(f'Absorption Spectrum - Peak: {peak_wavelength:.4f} μm, Q-factor: {q_factor:.2f}')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Absorption')
    plt.grid(True)
    plt.legend()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {output_path}")
    plt.close()

def save_results_to_file(results, output_path):
    """
    将结果保存到文本文件
    
    参数:
        results: 结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("Q值计算结果\n")
        f.write("=" * 50 + "\n\n")
        
        # 写入基本信息
        f.write(f"分析文件: {results['file_path']}\n")
        f.write(f"分析时间: {results['timestamp']}\n")
        f.write(f"初始中心波长: {results['initial_center']:.4f} μm\n")
        f.write(f"搜索范围: ±{results['search_range']:.4f} μm\n\n")
        
        # 写入计算结果
        f.write("-" * 50 + "\n")
        f.write("计算结果:\n")
        f.write(f"峰值波长: {results['peak_wavelength']:.6f} μm\n")
        f.write(f"峰值吸收率: {results['peak_absorption']:.6f}\n")
        f.write(f"左半高波长: {results['left_wavelength']:.6f} μm\n")
        f.write(f"右半高波长: {results['right_wavelength']:.6f} μm\n")
        f.write(f"半高全宽 (FWHM): {results['fwhm']:.6f} μm\n")
        f.write(f"Q值: {results['q_factor']:.2f}\n")
    
    print(f"结果已保存至: {output_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取Excel数据
    wavelengths, absorption = load_excel_data(args.file)
    
    # 在指定范围内寻找峰值
    peak_wavelength, peak_absorption = find_peak(
        wavelengths, absorption, args.center, args.range)
    
    # 计算半高全宽
    fwhm, left_wavelength, right_wavelength = calculate_fwhm(
        wavelengths, absorption, peak_wavelength, peak_absorption)
    
    # 计算Q值
    q_factor = calculate_q_factor(peak_wavelength, fwhm)
    
    # 准备结果
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 从文件路径提取文件名
    file_name = os.path.basename(args.file)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    results = {
        'file_path': args.file,
        'timestamp': timestamp,
        'initial_center': args.center,
        'search_range': args.range,
        'peak_wavelength': peak_wavelength,
        'peak_absorption': peak_absorption,
        'left_wavelength': left_wavelength,
        'right_wavelength': right_wavelength,
        'fwhm': fwhm,
        'q_factor': q_factor
    }
    
    # 保存结果到文本文件
    output_txt = os.path.join(args.output_dir, f"{file_name_without_ext}_q_results.txt")
    save_results_to_file(results, output_txt)
    
    # 如果需要，绘制并保存图表
    if args.plot:
        output_img = os.path.join(args.output_dir, f"{file_name_without_ext}_q_plot.png")
        plot_spectrum_with_fwhm(
            wavelengths, absorption, peak_wavelength, 
            left_wavelength, right_wavelength, peak_absorption, 
            q_factor, output_img)
    
    print("\n计算完成！")

if __name__ == "__main__":
    main() 