#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
薄膜结构Q值优化程序
-----------------
读取薄膜结构文件，合并相同材料层，使用传输矩阵法计算光谱，
并使用遗传算法优化结构厚度以提高特定波长处的Q值。

使用方法:
    python optimize_structure.py --file "generated_samples/best_samples_20250403_172401/best_sample_1_structure.txt" 
                               --center1 4.26 --center2 6.2 --output_dir "optimized_results"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import re
import math
from utils.config_loader import load_config

# 尝试导入PyGAD库
try:
    import pygad
    PYGAD_AVAILABLE = True
except ImportError:
    PYGAD_AVAILABLE = False
    print("警告: 未安装PyGAD库。请使用以下命令安装: pip install pygad")

# 导入传输矩阵法模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.myindex import MatDatabase
from model.TMM.optical_calculator import calculate_reflection

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='优化薄膜结构以提高Q值')
    parser.add_argument('--file', type=str, 
                      default='results\spectral_gan\\run_20250407_205045\samples\data\structure_sample_1_epoch_2000.txt',
                      help='结构文件路径')
    parser.add_argument('--center1', type=float, default=4.26,
                      help='第一个目标波长 (μm)')
    parser.add_argument('--center2', type=float, default=8.5,
                      help='第二个目标波长 (μm)')
    parser.add_argument('--range', type=float, default=0.3,
                      help='搜索真实峰值的波长范围 (μm)')
    parser.add_argument('--output_dir', type=str, default='optimized_results',
                      help='输出目录')
    parser.add_argument('--population', type=int, default=1,
                      help='遗传算法种群大小')
    parser.add_argument('--generations', type=int, default=1,
                      help='遗传算法迭代次数')
    parser.add_argument('--vary_thickness', type=float, default=0.5,
                      help='厚度变化范围因子(0-1)')
    return parser.parse_args()

def read_structure_file(file_path):
    """
    读取结构文件
    
    Args:
        file_path: 结构文件路径
        
    Returns:
        thicknesses: 每层厚度列表
        materials: 每层材料列表
    """
    thicknesses = []
    materials = []
    
    # 定义正则表达式模式
    thickness_pattern = r"Layer (\d+): ([\d\.]+)"
    material_pattern = r"Layer (\d+): (\w+)"
    
    # 读取文件
    with open(file_path, 'r') as file:
        content = file.read()
        
    # 分离厚度部分和材料部分
    parts = content.split("Dominant Material for Each Layer:")
    if len(parts) != 2:
        raise ValueError("文件格式不符合预期")
    
    thickness_section = parts[0]
    material_section = parts[1]
    
    # 提取厚度信息
    thickness_matches = re.findall(thickness_pattern, thickness_section)
    for _, thickness in thickness_matches:
        thicknesses.append(float(thickness))
    
    # 提取材料信息
    material_matches = re.findall(material_pattern, material_section)
    for _, material in material_matches:
        materials.append(material)
    
    # 确保厚度和材料列表长度相同
    if len(thicknesses) != len(materials):
        raise ValueError(f"厚度列表({len(thicknesses)})和材料列表({len(materials)})长度不一致")
    
    return thicknesses, materials

def merge_identical_layers(thicknesses, materials):
    """
    合并相邻的相同材料层
    
    Args:
        thicknesses: 每层厚度列表
        materials: 每层材料列表
        
    Returns:
        merged_thicknesses: 合并后的厚度列表
        merged_materials: 合并后的材料列表
        layer_mapping: 原始层到合并层的映射
    """
    merged_thicknesses = []
    merged_materials = []
    layer_mapping = {}  # 记录原始层被合并到哪一层
    
    current_material = None
    current_thickness = 0.0
    current_layers = []
    
    for i, (thickness, material) in enumerate(zip(thicknesses, materials)):
        if material == current_material:
            # 相同材料，累加厚度
            current_thickness += thickness
            current_layers.append(i+1)  # 记录原始层索引(1-based)
        else:
            # 不同材料，保存当前结果并开始新的合并
            if current_material is not None:
                merged_thicknesses.append(current_thickness)
                merged_materials.append(current_material)
                # 记录原始层到合并层的映射
                for layer in current_layers:
                    layer_mapping[layer] = len(merged_materials)
            
            current_material = material
            current_thickness = thickness
            current_layers = [i+1]
    
    # 处理最后一层
    if current_material is not None:
        merged_thicknesses.append(current_thickness)
        merged_materials.append(current_material)
        for layer in current_layers:
            layer_mapping[layer] = len(merged_materials)
    
    return merged_thicknesses, merged_materials, layer_mapping

def setup_optical_params(center1=4.26, center2=6.2):
    """
    设置光学计算参数
    
    Args:
        center1: 第一个目标波长
        center2: 第二个目标波长
        
    Returns:
        params: 光学参数对象
    """
    class Params:
        pass
    
    params = Params()
    
    # 设置波长范围和采样点数
    params.wavelength_range = [3, 10]  # 微米
    params.samples_total = 20000  # 增加采样点数量提高精度
    
    # 创建非均匀采样的波长向量，在目标波长附近增加采样密度
    # 计算每个区域的采样点数
    range_width = params.wavelength_range[1] - params.wavelength_range[0]
    
    # 目标波长附近的精细采样范围 (±0.1μm)
    window1 = [max(params.wavelength_range[0], center1-0.1), min(params.wavelength_range[1], center1+0.1)]
    window2 = [max(params.wavelength_range[0], center2-0.1), min(params.wavelength_range[1], center2+0.1)]
    
    # 计算各区域宽度
    width_window1 = window1[1] - window1[0]
    width_window2 = window2[1] - window2[0]
    
    # 分配采样点：目标波长区域占50%，其余区域占50%
    points_window1 = int(params.samples_total * 0.25)
    points_window2 = int(params.samples_total * 0.25)
    points_rest = params.samples_total - points_window1 - points_window2
    
    # 创建三个区域的波长数组
    lambda1 = torch.linspace(window1[0], window1[1], points_window1)
    lambda2 = torch.linspace(window2[0], window2[1], points_window2)
    
    # 创建剩余区域的波长数组（排除已覆盖的区域）
    rest_ranges = []
    current = params.wavelength_range[0]
    
    if current < window1[0]:
        rest_ranges.append((current, window1[0]))
        current = window1[1]
    elif current < window2[0]:
        current = window1[1]
    
    if current < window2[0]:
        rest_ranges.append((current, window2[0]))
        current = window2[1]
    
    if current < params.wavelength_range[1]:
        rest_ranges.append((current, params.wavelength_range[1]))
    
    # 根据各区域宽度，按比例分配剩余采样点
    lambda_rest_parts = []
    if rest_ranges:
        total_rest_width = sum(r[1] - r[0] for r in rest_ranges)
        for r_start, r_end in rest_ranges:
            r_width = r_end - r_start
            r_points = int(points_rest * (r_width / total_rest_width))
            if r_points > 0:
                lambda_rest_parts.append(torch.linspace(r_start, r_end, r_points))
    
    # 合并所有波长数组并排序
    wavelengths_parts = [lambda1, lambda2] + lambda_rest_parts
    wavelengths = torch.cat(wavelengths_parts)
    wavelengths, _ = torch.sort(wavelengths)
    
    # 确保波长数组严格递增（移除重复值）
    wavelengths_unique = torch.unique(wavelengths)
    
    # 计算波数向量
    params.k = 2 * np.pi / wavelengths_unique
    
    # 保存原始波长数组
    params.wavelengths = wavelengths_unique
    
    # 其他参数
    params.theta = torch.tensor([0.0])  # 入射角度
    params.pol = 0  # 偏振态 (0=TE, 1=TM)
    params.n_top = torch.tensor([1.0])  # 顶层介质折射率
    params.n_bot = torch.tensor([1.0])  # 底层介质折射率
    
    # 设置材料
    params.materials = ['Ge_Burnett', 'YbF3']
    params.matdatabase = MatDatabase(params.materials)
    params.n_database, params.k_database = params.matdatabase.interp_wv(
        2 * np.pi/params.k, params.materials, False)
    params.M_materials = params.n_database.size(0)
    
    return params

def calculate_spectrum(thicknesses, materials, params, device='cuda'):
    """
    计算薄膜结构的吸收光谱
    
    Args:
        thicknesses: 厚度列表 (微米)
        materials: 材料列表
        params: 光学参数
        device: 计算设备
        
    Returns:
        wavelengths: 波长数组 (微米)
        absorption: 吸收率数组
    """
    # 将参数转移到设备
    params.k = params.k.to(device)
    params.theta = params.theta.to(device)
    params.n_database = params.n_database.to(device)
    params.k_database = params.k_database.to(device)
    
    # 构建厚度张量
    batch_size = 1
    N_layers = len(thicknesses)
    thicknesses_tensor = torch.tensor(thicknesses).float().view(batch_size, N_layers).to(device)
    
    # 构建材料索引
    material_indices = []
    for material in materials:
        if material == 'Ge_Burnett':
            material_indices.append(0)
        elif material == 'YbF3':
            material_indices.append(1)
        else:
            raise ValueError(f"未知材料: {material}")
    
    # 构建复折射率
    n_part = torch.zeros((batch_size, N_layers, len(params.k)), dtype=torch.complex128, device=device)
    k_part = torch.zeros((batch_size, N_layers, len(params.k)), dtype=torch.complex128, device=device)
    
    for i, idx in enumerate(material_indices):
        n_part[:, i, :] = params.n_database[idx]
        k_part[:, i, :] = params.k_database[idx]
    
    refractive_indices = n_part + 1j * k_part
    
    # 计算反射率
    reflection = calculate_reflection(thicknesses_tensor, refractive_indices, params, device)
    
    # 计算吸收率
    absorption = (1 - reflection).float().cpu().numpy()[0]
    
    # 计算波长
    wavelengths = (2 * np.pi / params.k.cpu().numpy())
    
    return wavelengths, absorption

def find_peak(wavelengths, absorption, center, search_range=0.2):
    """
    在指定范围内寻找真实峰值
    
    Args:
        wavelengths: 波长数组
        absorption: 吸收率数组
        center: 中心波长
        search_range: 搜索范围 (μm)
        
    Returns:
        peak_wavelength: 峰值波长
        peak_absorption: 峰值吸收率
    """
    # 确定搜索范围
    min_wavelength = center - search_range
    max_wavelength = center + search_range
    
    # 找出范围内的波长索引
    range_indices = np.where((wavelengths >= min_wavelength) & (wavelengths <= max_wavelength))[0]
    
    if len(range_indices) == 0:
        return center, 0
    
    # 在范围内找出吸收率最大值的索引
    peak_idx = range_indices[np.argmax(absorption[range_indices])]
    peak_wavelength = wavelengths[peak_idx]
    peak_absorption = absorption[peak_idx]
    
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
        # 使用简单的线性插值
        left_wavelength = wavelengths[left_idx] + (wavelengths[left_idx+1] - wavelengths[left_idx]) * \
                        (half_max - absorption[left_idx]) / (absorption[left_idx+1] - absorption[left_idx])
    else:
        left_wavelength = wavelengths[left_idx]
    
    if right_idx > peak_idx:
        # 使用简单的线性插值
        right_wavelength = wavelengths[right_idx-1] + (wavelengths[right_idx] - wavelengths[right_idx-1]) * \
                        (half_max - absorption[right_idx-1]) / (absorption[right_idx] - absorption[right_idx-1])
    else:
        right_wavelength = wavelengths[right_idx]
    
    fwhm = right_wavelength - left_wavelength
    
    return fwhm

def calculate_q_factor(wavelengths, absorption, center, search_range=0.2):
    """
    计算指定波长附近的Q因子
    
    参数:
        wavelengths: 波长数组
        absorption: 吸收率数组
        center: 中心波长
        search_range: 搜索范围
        
    返回:
        q_factor: Q因子
        peak_wavelength: 峰值波长
        peak_absorption: 峰值吸收率
        fwhm: 半高全宽
    """
    # 寻找峰值
    peak_wavelength, peak_absorption = find_peak(wavelengths, absorption, center, search_range)
    
    # 计算半高全宽
    fwhm = calculate_fwhm(wavelengths, absorption, peak_wavelength, peak_absorption)
    
    # 计算Q因子
    if fwhm > 0:
        q_factor = peak_wavelength / fwhm
    else:
        q_factor = 0.1  # 避免除零错误，设置一个较小的默认值
    
    return q_factor, peak_wavelength, peak_absorption, fwhm

def create_target_lorentzian(wavelengths, center1, center2, width=0.02):
    """
    创建双峰洛伦兹曲线作为目标光谱
    
    参数:
        wavelengths: 波长数组
        center1: 第一个峰的中心波长
        center2: 第二个峰的中心波长
        width: 洛伦兹曲线的带宽(FWHM)
        
    返回:
        target: 目标洛伦兹曲线
    """
    # 创建双峰洛伦兹曲线
    gamma = width  # 带宽参数
    
    # 第一个峰
    lorentzian1 = (gamma/2)**2 / ((wavelengths - center1)**2 + (gamma/2)**2)
    
    # 第二个峰
    lorentzian2 = (gamma/2)**2 / ((wavelengths - center2)**2 + (gamma/2)**2)
    
    # 合并两个峰并归一化
    target = lorentzian1 + lorentzian2
    target = target / np.max(target)  # 归一化到最大值为1
    
    return target

def calculate_weighted_mse(spectrum, target, wavelengths, center1, center2, peak_weight=10.0, window_width=0.05):
    """
    计算加权均方差，在峰值附近增加权重
    
    参数:
        spectrum: 计算得到的光谱
        target: 目标洛伦兹曲线
        wavelengths: 波长数组
        center1: 第一个峰的中心波长
        center2: 第二个峰的中心波长
        peak_weight: 峰值附近的权重系数
        window_width: 峰值附近加权窗口的宽度(μm)
        
    返回:
        weighted_mse: 加权均方差
    """
    # 创建权重数组，默认权重为1
    weights = np.ones_like(wavelengths)
    
    # 在两个峰附近增加权重
    for center in [center1, center2]:
        peak_region = (wavelengths >= center - window_width) & (wavelengths <= center + window_width)
        weights[peak_region] = peak_weight
    
    # 计算加权均方差
    squared_errors = weights * (spectrum - target)**2
    weighted_mse = np.mean(squared_errors)
    
    return weighted_mse

def optimize_structure(initial_thicknesses, materials, params, center1, center2, search_range, 
                     vary_factor=0.2, population=25, generations=20):
    """
    使用遗传算法优化结构厚度以提高Q值
    
    Args:
        initial_thicknesses: 初始厚度列表
        materials: 材料列表
        params: 光学参数
        center1: 第一个目标波长
        center2: 第二个目标波长
        search_range: 峰值搜索范围
        vary_factor: 厚度变化范围因子
        population: 种群大小
        generations: 迭代次数
        
    Returns:
        best_thicknesses: 优化后的厚度列表
        best_q1: 第一个波长的最佳Q值
        best_q2: 第二个波长的最佳Q值
        best_spectrum: (wavelengths, absorption) 最佳光谱
    """
    # 检查是否安装了PyGAD库
    if not PYGAD_AVAILABLE:
        print("\n错误: 未安装PyGAD库。请使用以下命令安装:")
        print("pip install pygad")
        print("\n继续使用原始结构...")
        # 返回原始值，不进行优化
        wavelengths, absorption = calculate_spectrum(initial_thicknesses, materials, params, 'cuda' if torch.cuda.is_available() else 'cpu')
        q1, _, _, _ = calculate_q_factor(wavelengths, absorption, center1, search_range)
        q2, _, _, _ = calculate_q_factor(wavelengths, absorption, center2, search_range)
        return initial_thicknesses, q1, q2, (wavelengths, absorption)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 记录最佳解和迭代状态
    best_fitness = [-float('inf')]  # 使用列表便于在函数内修改
    best_q1 = 0
    best_q2 = 0
    best_thicknesses = initial_thicknesses.copy()
    iteration_count = [0]  # 使用列表以便在回调函数中修改
    last_output_time = datetime.now()
    
    # 计算初始光谱作为基准
    wavelengths, absorption = calculate_spectrum(initial_thicknesses, materials, params, device)
    best_spectrum = (wavelengths, absorption)
    
    # 创建目标双峰洛伦兹曲线
    target_lorentzian = create_target_lorentzian(wavelengths, center1, center2, width=0.02)
    
    # 计算初始结构的加权均方差
    initial_mse = calculate_weighted_mse(
        absorption, target_lorentzian, wavelengths, center1, center2, peak_weight=15.0, window_width=0.05)
    
    # 转换为初始适应度（越小越好，所以取负值并加上常数）
    initial_fitness = 1.0 / (initial_mse + 1e-6)
    
    # 计算初始Q值（仅用于显示）
    q1_init, peak1_init, abs1_init, fwhm1_init = calculate_q_factor(wavelengths, absorption, center1, search_range)
    q2_init, peak2_init, abs2_init, fwhm2_init = calculate_q_factor(wavelengths, absorption, center2, search_range)
    
    # 更新最佳适应度
    best_fitness[0] = initial_fitness
    best_q1 = q1_init
    best_q2 = q2_init
    
    # 输出初始信息
    print(f"\n初始结构评估:")
    print(f"初始适应度 = {initial_fitness:.2f}")
    print(f"λ1({center1} μm): Q值 = {q1_init:.2f}, 峰值 = {peak1_init:.4f} μm, 吸收率 = {abs1_init:.4f}, FWHM = {fwhm1_init:.6f} μm")
    print(f"λ2({center2} μm): Q值 = {q2_init:.2f}, 峰值 = {peak2_init:.4f} μm, 吸收率 = {abs2_init:.4f}, FWHM = {fwhm2_init:.6f} μm")
    
    # 设置遗传算法的搜索范围
    # 确保厚度不会小于0.02微米
    gene_space = []
    for t in initial_thicknesses:
        min_t = max(0.02, t * (1 - vary_factor))
        max_t = t * (1 + vary_factor)
        gene_space.append({'low': min_t, 'high': max_t})
    
    # 定义适应度函数
    def fitness_function(ga_instance, thicknesses, solution_idx):
        nonlocal best_q1, best_q2, best_thicknesses, best_spectrum, last_output_time, target_lorentzian
        
        # 更新迭代计数
        iteration_count[0] += 1
        
        # 计算光谱
        try:
            wavelengths, absorption = calculate_spectrum(thicknesses, materials, params, device)
            
            # 计算加权均方差
            mse = calculate_weighted_mse(
                absorption, target_lorentzian, wavelengths, center1, center2, peak_weight=15.0, window_width=0.05)
            
            # 将MSE转换为适应度值（小的MSE意味着高的适应度）
            fitness = 1.0 / (mse + 1e-6)  # 防止除零
            
            # 计算Q值（仅用于显示）
            q1, peak1, abs1, fwhm1 = calculate_q_factor(wavelengths, absorption, center1, search_range)
            q2, peak2, abs2, fwhm2 = calculate_q_factor(wavelengths, absorption, center2, search_range)
            
            # 获取指定波长处的吸收率（仅用于显示）
            center1_idx = np.argmin(np.abs(wavelengths - center1))
            center2_idx = np.argmin(np.abs(wavelengths - center2))
            abs_center1 = absorption[center1_idx]
            abs_center2 = absorption[center2_idx]
            
            # 检查是否需要输出进度
            current_time = datetime.now()
            time_elapsed = (current_time - last_output_time).total_seconds()
            output_progress = (solution_idx % (population // 2) == 0) or (time_elapsed > 10.0)
            
            # 如果这是新的最佳解
            if fitness > best_fitness[0]:
                best_fitness[0] = fitness
                best_q1 = q1
                best_q2 = q2
                best_thicknesses = thicknesses.copy()
                best_spectrum = (wavelengths.copy(), absorption.copy())
                
                # 打印新的最佳解
                print(f"\n*** 种群 {ga_instance.generations_completed}, 个体 {solution_idx}: 找到新的最佳解 ***")
                print(f"最佳适应度 = {fitness:.2f} (MSE = {mse:.6f})")
                print(f"  Q值: Q1 = {q1:.2f}, Q2 = {q2:.2f}")
                print(f"  吸收率: λ1 = {abs_center1:.4f}, λ2 = {abs_center2:.4f}")
                print(f"  半高全宽: λ1 = {fwhm1:.6f} μm, λ2 = {fwhm2:.6f} μm")
                
                # 重置上次输出时间
                last_output_time = current_time
                output_progress = False
            
            # 定期打印进度
            if output_progress:
                print(f"种群 {ga_instance.generations_completed}, 个体 {solution_idx}: " +
                      f"适应度 = {fitness:.2f}, MSE = {mse:.6f}, " +
                      f"Q1 = {q1:.2f}, Q2 = {q2:.2f}")
                last_output_time = current_time
            
            return fitness
            
        except Exception as e:
            print(f"计算适应度时出错: {e}")
            return 0  # 错误情况下返回0适应度
    
    # 定义完成每代进化后的回调函数
    def on_generation(ga_instance):
        print(f"\n完成种群 {ga_instance.generations_completed}/{generations} 的训练，当前最佳适应度: {ga_instance.best_solution()[1]:.2f}")
        return
    
    # 输出优化信息
    print(f"\n开始使用遗传算法优化，种群大小: {population}, 迭代次数: {generations}")
    print(f"每个维度的变化范围: ±{vary_factor*100:.0f}% (最小厚度限制为0.02μm)")
    print(f"初始结构层数: {len(initial_thicknesses)}")
    print(f"目标波长: {center1} μm 和 {center2} μm")
    
    try:
        # 创建遗传算法实例
        ga_instance = pygad.GA(
            num_generations=generations,
            num_parents_mating=int(population / 2),
            fitness_func=fitness_function,
            sol_per_pop=population,
            num_genes=len(initial_thicknesses),
            gene_space=gene_space,
            init_range_low=0.0,
            init_range_high=1.0,
            parent_selection_type="tournament",
            keep_parents=1,
            crossover_type="uniform",
            crossover_probability=0.8,
            mutation_type="random",  # 改为random而不是adaptive
            mutation_probability=0.1,  # 使用单个浮点数值作为所有基因的变异概率
            on_generation=on_generation,
            initial_population=np.array([initial_thicknesses] + [np.random.uniform(
                [g['low'] for g in gene_space], 
                [g['high'] for g in gene_space], 
                len(initial_thicknesses)
            ) for _ in range(population - 1)]))
        
        # 运行遗传算法
        ga_instance.run()
        
        # 获取最佳解
        solution, solution_fitness, _ = ga_instance.best_solution()
        
        print("\n遗传算法已完成运行")
        print(f"迭代次数: {ga_instance.generations_completed}")
        print(f"最终适应度: {solution_fitness:.4f}")
        
        # 如果找到了更好的解，更新最佳解
        if solution_fitness > best_fitness[0]:
            best_fitness[0] = solution_fitness
            best_thicknesses = solution.tolist()
            final_wavelengths, final_absorption = calculate_spectrum(best_thicknesses, materials, params, device)
            best_spectrum = (final_wavelengths, final_absorption)
            q1_final, _, _, _ = calculate_q_factor(final_wavelengths, final_absorption, center1, search_range)
            q2_final, _, _, _ = calculate_q_factor(final_wavelengths, final_absorption, center2, search_range)
            best_q1 = q1_final
            best_q2 = q2_final
            
    except KeyboardInterrupt:
        print("\n\n优化被用户中断!")
    except Exception as e:
        print(f"\n\n优化过程中出现错误: {str(e)}")
        print("继续使用原始结构...")
    
    # 优化完成后的输出
    print("\n优化结束！")
    print(f"最佳适应度: {best_fitness[0]:.2f}")
    
    # 计算最佳解的光谱和统计数据
    wavelengths, absorption = calculate_spectrum(best_thicknesses, materials, params, device)
    
    # 计算最终的加权均方差
    final_mse = calculate_weighted_mse(
        absorption, target_lorentzian, wavelengths, center1, center2, peak_weight=15.0, window_width=0.05)
    
    # 计算最终的Q值和吸收率（仅用于显示）
    q1, peak1, abs1, fwhm1 = calculate_q_factor(wavelengths, absorption, center1, search_range)
    q2, peak2, abs2, fwhm2 = calculate_q_factor(wavelengths, absorption, center2, search_range)
    
    # 获取指定波长处的吸收率
    center1_idx = np.argmin(np.abs(wavelengths - center1))
    center2_idx = np.argmin(np.abs(wavelengths - center2))
    abs_center1 = absorption[center1_idx]
    abs_center2 = absorption[center2_idx]
    
    # 创建损失函数信息字典
    loss_info = {
        'mse': final_mse,
        'peak_weight': 15.0,
        'window_width': 0.05
    }
    
    # 打印详细结果
    print(f"\n优化后详细结果:")
    print(f"加权均方差 (MSE): {final_mse:.6f}")
    print(f"λ1({center1} μm):")
    print(f"  Q值 = {q1:.2f} (从 {q1_init:.2f} 提升 {(q1/q1_init-1)*100:.1f}%)")
    print(f"  吸收率 = {abs_center1:.4f}")
    print(f"  FWHM = {fwhm1:.6f} μm")
    
    print(f"\nλ2({center2} μm):")
    print(f"  Q值 = {q2:.2f} (从 {q2_init:.2f} 提升 {(q2/q2_init-1)*100:.1f}%)")
    print(f"  吸收率 = {abs_center2:.4f}")
    print(f"  FWHM = {fwhm2:.6f} μm")
    
    return best_thicknesses, best_q1, best_q2, best_spectrum

def plot_comparison(wavelengths1, absorption1, wavelengths2, absorption2, 
                  center1, center2, q1_orig, q2_orig, q1_opt, q2_opt, 
                  output_path, loss_value=None):
    """
    绘制原始光谱和优化后光谱的对比图
    
    Args:
        wavelengths1: 原始光谱波长
        absorption1: 原始光谱吸收率
        wavelengths2: 优化后光谱波长
        absorption2: 优化后光谱吸收率
        center1: 第一个目标波长
        center2: 第二个目标波长
        q1_orig: 原始光谱第一个波长Q值
        q2_orig: 原始光谱第二个波长Q值
        q1_opt: 优化后光谱第一个波长Q值
        q2_opt: 优化后光谱第二个波长Q值
        output_path: 输出图像路径
        loss_value: 损失函数值
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制原始光谱
    plt.plot(wavelengths1, absorption1, 'b-', linewidth=2, label='原始光谱')
    
    # 绘制优化后光谱
    plt.plot(wavelengths2, absorption2, 'r-', linewidth=2, label='优化后光谱')
    
    # 标记目标波长
    plt.axvline(x=center1, color='g', linestyle='--', label=f'目标 λ1: {center1} μm')
    plt.axvline(x=center2, color='m', linestyle='--', label=f'目标 λ2: {center2} μm')
    
    # 设置标题和标签
    if loss_value is None:
        plt.title(f'优化结果对比\nQ1: {q1_orig:.2f} → {q1_opt:.2f}, Q2: {q2_orig:.2f} → {q2_opt:.2f}', fontproperties="SimHei")
    else:
        plt.title(f'优化结果对比\nQ1: {q1_orig:.2f} → {q1_opt:.2f}, Q2: {q2_orig:.2f} → {q2_opt:.2f}\n加权MSE: {loss_value:.6f}', fontproperties="SimHei")
    
    plt.xlabel('波长 (μm)', fontproperties="SimHei")
    plt.ylabel('吸收率', fontproperties="SimHei")
    plt.grid(True)
    plt.legend(prop={"family":"SimHei"})
    plt.xlim(center1 - 1, center2 + 1)  # 设置x轴范围为目标波长附近
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存至: {output_path}")

def save_structure_to_file(thicknesses, materials, q1, q2, output_path, center1=4.26, center2=6.2, 
                         abs1=None, abs2=None, fwhm1=None, fwhm2=None, loss_info=None):
    """
    将优化后的结构保存到文本文件
    
    Args:
        thicknesses: 厚度列表
        materials: 材料列表
        q1: 第一个波长Q值
        q2: 第二个波长Q值
        output_path: 输出文件路径
        center1: 第一个目标波长
        center2: 第二个目标波长
        abs1: 第一个波长处的吸收率
        abs2: 第二个波长处的吸收率
        fwhm1: 第一个波长的半高全宽
        fwhm2: 第二个波长的半高全宽
        loss_info: 损失函数相关信息字典
    """
    with open(output_path, 'w') as f:
        f.write("Optimized Structure Information\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Q-factor at {center1} μm: {q1:.2f}\n")
        f.write(f"Q-factor at {center2} μm: {q2:.2f}\n")
        
        # 添加吸收率和FWHM信息
        if abs1 is not None and abs2 is not None:
            f.write(f"\nAbsorption at {center1} μm: {abs1:.4f}\n")
            f.write(f"Absorption at {center2} μm: {abs2:.4f}\n")
        
        if fwhm1 is not None and fwhm2 is not None:
            f.write(f"\nFWHM at {center1} μm: {fwhm1:.6f} μm\n")
            f.write(f"FWHM at {center2} μm: {fwhm2:.6f} μm\n")
        
        # 添加损失函数信息
        if loss_info is not None:
            f.write("\n" + "-" * 40 + "\n")
            f.write("Loss Function Information:\n")
            f.write(f"Weighted Mean Square Error (MSE): {loss_info['mse']:.6f}\n\n")
            
            f.write(f"MSE Parameters:\n")
            f.write(f"  Peak Weight: {loss_info['peak_weight']:.1f}\n")
            f.write(f"  Window Width: {loss_info['window_width']:.3f} μm\n")
            f.write(f"  Target Center Wavelengths: {center1:.2f} μm, {center2:.2f} μm\n")
        
        f.write("\n" + "-" * 40 + "\n\n")
        
        f.write("Layer Thickness (μm):\n")
        for i, thickness in enumerate(thicknesses):
            f.write(f"Layer {i+1}: {thickness:.6f}\n")
        
        f.write("\n" + "-" * 40 + "\n\n")
        
        f.write("Layer Materials:\n")
        for i, material in enumerate(materials):
            f.write(f"Layer {i+1}: {material}\n")
        
        f.write("\n" + "-" * 40 + "\n\n")
        
        # 计算每种材料的总厚度
        material_thickness = {}
        for material, thickness in zip(materials, thicknesses):
            if material not in material_thickness:
                material_thickness[material] = 0
            material_thickness[material] += thickness
        
        f.write("Total Thickness by Material:\n")
        for material, thickness in material_thickness.items():
            f.write(f"{material}: {thickness:.6f} μm\n")
        
        f.write("\nTotal Structure Thickness: {:.6f} μm\n".format(sum(thicknesses)))
        
    print(f"结构信息已保存至: {output_path}")

def save_spectrum_to_excel(wavelengths, absorption, output_path):
    """
    将光谱数据保存为Excel文件
    
    Args:
        wavelengths: 波长数组
        absorption: 吸收率数组
        output_path: 输出文件路径
    """
    df = pd.DataFrame()
    df['Wavelength (μm)'] = wavelengths
    df['Absorption'] = absorption
    
    df.to_excel(output_path, index=False)
    print(f"光谱数据已保存至: {output_path}")

def plot_comparison_with_target(wavelengths, absorption, target_lorentzian, center1, center2, q1, q2, mse, output_path):
    """
    绘制优化后光谱与目标洛伦兹曲线的对比图
    
    参数:
        wavelengths: 波长数组
        absorption: 优化后的吸收光谱
        target_lorentzian: 目标双峰洛伦兹曲线
        center1: 第一个目标波长
        center2: 第二个目标波长
        q1: 第一个波长的Q值
        q2: 第二个波长的Q值
        mse: 加权均方差
        output_path: 输出图像路径
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制优化后光谱
    plt.plot(wavelengths, absorption, 'r-', linewidth=2, label='优化后光谱')
    
    # 绘制目标洛伦兹曲线
    plt.plot(wavelengths, target_lorentzian, 'b--', linewidth=2, label='目标洛伦兹曲线')
    
    # 标记目标波长
    plt.axvline(x=center1, color='g', linestyle='--', label=f'目标 λ1: {center1} μm')
    plt.axvline(x=center2, color='m', linestyle='--', label=f'目标 λ2: {center2} μm')
    
    # 设置标题和标签
    plt.title(f'优化结果与目标曲线对比\nQ1: {q1:.2f}, Q2: {q2:.2f}\n加权MSE: {mse:.6f}', fontproperties="SimHei")
    plt.xlabel('波长 (μm)', fontproperties="SimHei")
    plt.ylabel('吸收率', fontproperties="SimHei")
    plt.grid(True)
    plt.legend(prop={"family":"SimHei"})
    plt.xlim(center1 - 1, center2 + 1)  # 设置x轴范围为目标波长附近
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"目标曲线对比图已保存至: {output_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取结构文件
    print(f"读取结构文件: {args.file}")
    thicknesses, materials = read_structure_file(args.file)
    print(f"读取到 {len(thicknesses)} 层结构")
    
    # 合并相同材料层
    merged_thicknesses, merged_materials, layer_mapping = merge_identical_layers(thicknesses, materials)
    print(f"合并后的层数: {len(merged_thicknesses)}")
    for i, (t, m) in enumerate(zip(merged_thicknesses, merged_materials)):
        print(f"合并层 {i+1}: {m}, 厚度 = {t:.6f} μm")
    
    # 设置光学参数，传入目标波长
    params = setup_optical_params(args.center1, args.center2)
    print(f"设置采样点数: {params.samples_total}，在 {args.center1}μm 和 {args.center2}μm 附近增加采样密度")
    
    # 计算原始光谱
    print("\n计算原始结构光谱...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wavelengths_orig, absorption_orig = calculate_spectrum(
        merged_thicknesses, merged_materials, params, device)
    
    # 计算原始Q值
    q1_orig, peak1_orig, abs1_orig, fwhm1_orig = calculate_q_factor(
        wavelengths_orig, absorption_orig, args.center1, args.range)
    q2_orig, peak2_orig, abs2_orig, fwhm2_orig = calculate_q_factor(
        wavelengths_orig, absorption_orig, args.center2, args.range)
    
    print(f"\n原始结构Q值评估:")
    print(f"波长 {args.center1} μm: Q = {q1_orig:.2f}, 峰值 = {peak1_orig:.4f} μm, 吸收率 = {abs1_orig:.4f}, FWHM = {fwhm1_orig:.6f} μm")
    print(f"波长 {args.center2} μm: Q = {q2_orig:.2f}, 峰值 = {peak2_orig:.4f} μm, 吸收率 = {abs2_orig:.4f}, FWHM = {fwhm2_orig:.6f} μm")
    
    # 优化结构
    print("\n开始优化结构...")
    
    if PYGAD_AVAILABLE:
        print("将使用遗传算法进行优化")
    else:
        print("错误: 需要安装PyGAD库才能进行优化")
        
    opt_thicknesses, q1_opt, q2_opt, opt_spectrum = optimize_structure(
        merged_thicknesses, merged_materials, params, 
        args.center1, args.center2, args.range,
        args.vary_thickness, args.population, args.generations)
    
    # 获取优化后的光谱
    wavelengths_opt, absorption_opt = opt_spectrum
    
    # 保存优化结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 计算最佳解的光谱
    wavelengths_opt, absorption_opt = opt_spectrum
    
    # 创建目标双峰洛伦兹曲线
    target_lorentzian = create_target_lorentzian(wavelengths_opt, args.center1, args.center2, width=0.02)
    
    # 计算最终的加权均方差
    final_mse = calculate_weighted_mse(
        absorption_opt, target_lorentzian, wavelengths_opt, args.center1, args.center2, peak_weight=15.0, window_width=0.05)
    
    # 计算最终的Q值和吸收率
    q1_final, peak1, abs1, fwhm1 = calculate_q_factor(wavelengths_opt, absorption_opt, args.center1, args.range)
    q2_final, peak2, abs2, fwhm2 = calculate_q_factor(wavelengths_opt, absorption_opt, args.center2, args.range)
    
    # 获取指定波长处的吸收率
    center1_idx = np.argmin(np.abs(wavelengths_opt - args.center1))
    center2_idx = np.argmin(np.abs(wavelengths_opt - args.center2))
    abs_center1 = absorption_opt[center1_idx]
    abs_center2 = absorption_opt[center2_idx]
    
    # 创建损失函数信息字典
    loss_info = {
        'mse': final_mse,
        'peak_weight': 15.0,
        'window_width': 0.05
    }
    
    # 保存结构信息 - 包含更多参数
    structure_path = os.path.join(args.output_dir, f"optimized_structure_{timestamp}.txt")
    save_structure_to_file(
        opt_thicknesses, merged_materials, q1_final, q2_final, 
        structure_path, args.center1, args.center2,
        abs_center1, abs_center2, fwhm1, fwhm2, loss_info
    )
    
    # 保存光谱数据
    spectrum_path = os.path.join(args.output_dir, f"optimized_spectrum_{timestamp}.xlsx")
    save_spectrum_to_excel(wavelengths_opt, absorption_opt, spectrum_path)
    
    # 绘制对比图
    plot_path = os.path.join(args.output_dir, f"spectrum_comparison_{timestamp}.png")
    plot_comparison(
        wavelengths_orig, absorption_orig, 
        wavelengths_opt, absorption_opt,
        args.center1, args.center2,
        q1_orig, q2_orig, q1_final, q2_final,
        plot_path, final_mse
    )
    
    # 绘制优化后光谱与目标洛伦兹曲线的对比图
    target_comparison_path = os.path.join(args.output_dir, f"target_comparison_{timestamp}.png")
    plot_comparison_with_target(
        wavelengths_opt, absorption_opt, target_lorentzian, args.center1, args.center2, q1_final, q2_final, final_mse, target_comparison_path
    )
    
    print("\n优化完成!")
    print(f"Q值改进: λ1({args.center1} μm): {q1_orig:.2f} → {q1_final:.2f}, λ2({args.center2} μm): {q2_orig:.2f} → {q2_final:.2f}")
    print(f"结果已保存至目录: {args.output_dir}")

if __name__ == "__main__":
    main() 
