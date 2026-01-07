
"""
GAN样本分析工具
---------------
该脚本加载训练好的GAN模型，生成大量样本，按波长点收集最优的Q值和吸收峰，
然后绘制散点图进行可视化分析。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import seaborn as sns
from tqdm import tqdm
import time

# 导入自定义模块
from model.net import Generator, Discriminator
from model.TMM.optical_calculator import calculate_reflection
from utils.config_loader import Params, load_config
from data.myindex import MatDatabase

# 添加PyTorch 2.6加载问题的安全全局设置
torch.serialization.add_safe_globals([Generator, Discriminator])

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Analyze GAN generated samples by wavelength')
    parser.add_argument('--model_path', type=str, default='results\spectral_gan\generator_final.pth',
                        help='Path to trained generator model (.pth file)')
    parser.add_argument('--config_path', type=str, default='config/training_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--max_samples', type=int, default=1000000,
                        help='Maximum number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for sample generation')
    parser.add_argument('--alpha', type=float, default=200,
                        help='Alpha parameter for material selection sharpness')
    parser.add_argument('--min_peak_height', type=float, default=0.6,
                        help='Minimum peak height threshold for filtering')
    parser.add_argument('--wavelength_step', type=float, default=0.01,
                        help='Wavelength step size in μm')
    parser.add_argument('--wavelength_tolerance', type=float, default=0.01,
                        help='Tolerance for wavelength matching in μm')
    parser.add_argument('--top_k_per_wavelength', type=int, default=10,
                        help='Number of top samples to keep per wavelength')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='Save results every N samples')
    parser.add_argument('--stats_interval', type=int, default=10000,
                        help='Generate statistics every N samples')
    return parser.parse_args()

def load_parameters(config_path, device):
    """???????????"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    params = Params()
    config = load_config(config_path)

    def require(path):
        current = config
        for key in path:
            if key not in current:
                raise KeyError(f"Missing config key: {'/'.join(path)}")
            current = current[key]
        return current

    structure = require(("structure",))
    params.N_layers = structure["N_layers"]
    params.pol = structure["pol"]
    params.thickness_sup = structure["thickness_sup"]
    params.thickness_bot = structure["thickness_bot"]

    materials = require(("materials",))
    params.materials = materials["materials_list"]

    optics = require(("optics",))
    params.wavelength_range = optics["wavelength_range"]
    params.samples_total = optics["samples_total"]
    params.theta = optics["theta"]
    params.n_top = optics["n_top"]
    params.n_bot = optics["n_bot"]
    params.lorentz_width = optics["lorentz_width"]
    params.metal_name = optics["metal_name"]

    generator = require(("generator",))
    params.thickness_noise_dim = generator["thickness_noise_dim"]
    params.material_noise_dim = generator["material_noise_dim"]
    params.alpha_sup = generator["alpha_sup"]
    params.alpha = generator["alpha"]

    print(f"Configuration loaded from {config_path}")

    params.k = 2 * np.pi / torch.linspace(
        params.wavelength_range[0], params.wavelength_range[1], params.samples_total
    )

    params.theta = torch.tensor([params.theta]).to(device)
    params.n_top = torch.tensor([params.n_top])
    params.n_bot = torch.tensor([params.n_bot])

    if hasattr(params, 'materials') and params.materials:
        params.matdatabase = MatDatabase(params.materials)
        params.n_database, params.k_database = params.matdatabase.interp_wv(
            2 * np.pi/params.k, params.materials, False)
        params.M_materials = params.n_database.size(0)
        print(f"Using materials: {', '.join(params.materials)}")

    return params

def load_model(model_path, params, device):
    """加载训练好的生成器模型"""
    # 初始化生成器
    generator = Generator(params).to(device)
    
    # 加载状态字典
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # 设置模型为评估模式
    generator.eval()
    return generator

def generate_samples(generator, params, num_samples, alpha, device):
    """使用训练好的生成器生成样本"""
    with torch.no_grad():
        # 生成随机噪声
        thickness_noise = torch.randn(num_samples, params.thickness_noise_dim, device=device)
        material_noise = torch.randn(num_samples, params.material_noise_dim, device=device)
        
        # 生成样本
        thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, alpha)
        
        # 计算波长和反射率
        wavelengths = 2 * np.pi / params.k.cpu()
        reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
        
        # 转换为吸收
        absorption = (1 - reflection).float()
        
    return wavelengths.cpu().numpy(), thicknesses, refractive_indices, P, absorption.cpu().numpy()

def find_peaks_and_calculate_q(wavelengths, absorption_spectrum, min_height=0.1, min_prominence=0.05):
    """
    找到吸收谱中的峰值并计算Q值
    
    Args:
        wavelengths: 波长数组
        absorption_spectrum: 吸收谱
        min_height: 峰值检测的最小高度阈值
        min_prominence: 峰值检测的最小突出度
    
    Returns:
        peaks_info: 包含峰值信息的字典列表
    """
    # 找到峰值
    peaks, properties = find_peaks(absorption_spectrum, 
                                   height=min_height, 
                                   prominence=min_prominence)
    
    peaks_info = []
    
    for i, peak_idx in enumerate(peaks):
        peak_wavelength = wavelengths[peak_idx]
        peak_height = absorption_spectrum[peak_idx]
        
        # 计算半高全宽(FWHM)
        try:
            # 找到峰值一半高度的点
            half_height = peak_height / 2
            
            # 在峰值左侧和右侧找到半高点
            left_indices = np.where(absorption_spectrum[:peak_idx] <= half_height)[0]
            right_indices = np.where(absorption_spectrum[peak_idx:] <= half_height)[0]
            
            if len(left_indices) > 0 and len(right_indices) > 0:
                left_idx = left_indices[-1]
                right_idx = peak_idx + right_indices[0]
                
                # 线性插值得到更精确的半高点
                left_wavelength = np.interp(half_height, 
                                          absorption_spectrum[left_idx:left_idx+2], 
                                          wavelengths[left_idx:left_idx+2])
                right_wavelength = np.interp(half_height, 
                                           absorption_spectrum[right_idx-1:right_idx+1][::-1], 
                                           wavelengths[right_idx-1:right_idx+1][::-1])
                
                fwhm = right_wavelength - left_wavelength
                q_factor = peak_wavelength / fwhm if fwhm > 0 else 0
            else:
                fwhm = 0
                q_factor = 0
                
        except:
            fwhm = 0
            q_factor = 0
        
        peaks_info.append({
            'center_wavelength': peak_wavelength,
            'peak_height': peak_height,
            'fwhm': fwhm,
            'q_factor': q_factor,
            'peak_index': peak_idx
        })
    
    return peaks_info

def analyze_samples_with_filtering(wavelengths, absorption_spectra, min_height=0.8, min_q_factor=200, min_prominence=0.05):
    """
    分析样本并根据条件进行筛选
    
    Args:
        wavelengths: 波长数组
        absorption_spectra: 所有样本的吸收谱
        min_height: 峰值高度的最小阈值
        min_q_factor: Q因子的最小阈值
        min_prominence: 峰值检测的最小突出度
    
    Returns:
        useful_results: 符合条件的样本分析结果
        all_results: 所有样本的分析结果
    """
    all_results = []
    useful_results = []
    
    for i in range(len(absorption_spectra)):
        # 找到峰值并计算Q值
        peaks_info = find_peaks_and_calculate_q(wavelengths, absorption_spectra[i], 
                                                min_height=0.1, min_prominence=min_prominence)
        
        # 找到最高的Q值和对应的峰值信息
        if peaks_info:
            best_peak = max(peaks_info, key=lambda x: x['q_factor'])
            max_q = best_peak['q_factor']
            center_wavelength = best_peak['center_wavelength']
            peak_height = best_peak['peak_height']
            fwhm = best_peak['fwhm']
        else:
            max_q = 0
            center_wavelength = 0
            peak_height = 0
            fwhm = 0
        
        # 计算整个谱的最大吸收值
        max_absorption = np.max(absorption_spectra[i])
        
        result = {
            'sample_index': i,
            'max_q_factor': max_q,
            'center_wavelength': center_wavelength,
            'peak_height': peak_height,
            'fwhm': fwhm,
            'max_absorption': max_absorption,
            'num_peaks': len(peaks_info)
        }
        
        all_results.append(result)
        
        # 检查是否符合筛选条件
        if peak_height >= min_height and max_q >= min_q_factor:
            useful_results.append(result)
    
    return useful_results, all_results

def create_scatter_plot(analysis_results, save_path=None):
    """
    创建散点图，显示中心波长vs最高Q因子
    
    Args:
        analysis_results: 分析结果列表
        save_path: 保存图像的路径
    """
    # 提取数据
    center_wavelengths = [result['center_wavelength'] for result in analysis_results if result['max_q_factor'] > 0]
    max_q_factors = [result['max_q_factor'] for result in analysis_results if result['max_q_factor'] > 0]
    max_absorptions = [result['max_absorption'] for result in analysis_results if result['max_q_factor'] > 0]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建散点图，使用最大吸收值作为颜色映射
    scatter = plt.scatter(center_wavelengths, max_q_factors, 
                         c=max_absorptions, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Maximum Absorption', fontsize=12)
    
    # 设置标签和标题
    plt.xlabel('Wavelength [μm]', fontsize=12)
    plt.ylabel('Highest Q Factor', fontsize=12)
    plt.title(f'Q Factor Analysis of GAN Generated Samples\n(Total samples: {len(analysis_results)}, Valid samples: {len(center_wavelengths)})', 
              fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    if center_wavelengths:
        plt.xlim(min(center_wavelengths) - 0.5, max(center_wavelengths) + 0.5)
    if max_q_factors:
        plt.ylim(0, max(max_q_factors) * 1.1)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    
    return plt.gcf()

def save_analysis_results(analysis_results, wavelengths, absorption_spectra, output_dir):
    """保存分析结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"analysis_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存分析结果到Excel
    df = pd.DataFrame(analysis_results)
    excel_path = os.path.join(save_dir, 'analysis_results.xlsx')
    df.to_excel(excel_path, index=False)
    
    # 保存统计信息
    stats_path = os.path.join(save_dir, 'statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("GAN Sample Analysis Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本统计
        total_samples = len(analysis_results)
        valid_samples = len([r for r in analysis_results if r['max_q_factor'] > 0])
        
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Valid samples (Q > 0): {valid_samples}\n")
        f.write(f"Valid sample ratio: {valid_samples/total_samples*100:.1f}%\n\n")
        
        # Q值统计
        q_values = [r['max_q_factor'] for r in analysis_results if r['max_q_factor'] > 0]
        if q_values:
            f.write("Q Factor Statistics:\n")
            f.write(f"  Maximum Q: {max(q_values):.2f}\n")
            f.write(f"  Average Q: {np.mean(q_values):.2f}\n")
            f.write(f"  Median Q: {np.median(q_values):.2f}\n")
            f.write(f"  Q Standard Deviation: {np.std(q_values):.2f}\n\n")
        
        # 中心波长统计
        wavelengths_valid = [r['center_wavelength'] for r in analysis_results if r['max_q_factor'] > 0]
        if wavelengths_valid:
            f.write("Center Wavelength Statistics:\n")
            f.write(f"  Wavelength range: {min(wavelengths_valid):.2f} - {max(wavelengths_valid):.2f} μm\n")
            f.write(f"  Average wavelength: {np.mean(wavelengths_valid):.2f} μm\n")
            f.write(f"  Median wavelength: {np.median(wavelengths_valid):.2f} μm\n")
            f.write(f"  Wavelength standard deviation: {np.std(wavelengths_valid):.2f} μm\n\n")
        
        # 峰值高度统计
        peak_heights = [r['peak_height'] for r in analysis_results if r['max_q_factor'] > 0]
        if peak_heights:
            f.write("Peak Height Statistics:\n")
            f.write(f"  Maximum peak: {max(peak_heights):.3f}\n")
            f.write(f"  Average peak: {np.mean(peak_heights):.3f}\n")
            f.write(f"  Median peak: {np.median(peak_heights):.3f}\n")
            f.write(f"  Peak standard deviation: {np.std(peak_heights):.3f}\n")
    
    # 保存散点图
    fig = create_scatter_plot(analysis_results)
    plot_path = os.path.join(save_dir, 'q_factor_analysis.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Analysis results saved to: {save_dir}")
    return save_dir

def create_wavelength_targets(wavelength_range, step_size):
    """
    创建波长目标点
    
    Args:
        wavelength_range: [min_wavelength, max_wavelength]
        step_size: 波长步长
    
    Returns:
        wavelength_targets: 目标波长列表
    """
    min_wl, max_wl = wavelength_range
    wavelengths = np.arange(min_wl, max_wl + step_size, step_size)
    return wavelengths

def get_q_threshold(wavelength):
    """
    根据波长获取Q值阈值
    
    Args:
        wavelength: 波长值
    
    Returns:
        q_threshold: Q值阈值
    """
    if 3.0 <= wavelength <= 5.0:
        return 300
    elif 5.0 < wavelength <= 8.0:
        return 250
    else:
        return 200  # 默认值

def find_best_wavelength_match(target_wavelengths, sample_wavelength, tolerance):
    """
    找到样本波长最匹配的目标波长
    
    Args:
        target_wavelengths: 目标波长列表
        sample_wavelength: 样本波长
        tolerance: 容差
    
    Returns:
        best_match: 最佳匹配的波长索引，如果没有匹配则返回None
    """
    distances = np.abs(target_wavelengths - sample_wavelength)
    min_distance_idx = np.argmin(distances)
    
    if distances[min_distance_idx] <= tolerance:
        return min_distance_idx
    else:
        return None

def update_wavelength_samples(wavelength_samples, wavelength_idx, new_sample, top_k):
    """
    更新指定波长点的样本列表
    
    Args:
        wavelength_samples: 波长样本字典
        wavelength_idx: 波长索引
        new_sample: 新样本
        top_k: 保留的样本数量
    
    Returns:
        updated: 是否更新了样本
    """
    if wavelength_idx not in wavelength_samples:
        wavelength_samples[wavelength_idx] = []
    
    # 检查是否应该添加这个样本
    current_samples = wavelength_samples[wavelength_idx]
    
    # 如果样本数量少于top_k，直接添加
    if len(current_samples) < top_k:
        current_samples.append(new_sample)
        current_samples.sort(key=lambda x: x['q_factor'], reverse=True)
        return True
    
    # 如果新样本的Q值比最差的样本好，则替换
    if new_sample['q_factor'] > current_samples[-1]['q_factor']:
        current_samples.append(new_sample)
        current_samples.sort(key=lambda x: x['q_factor'], reverse=True)
        current_samples = current_samples[:top_k]  # 保持top_k个
        wavelength_samples[wavelength_idx] = current_samples
        return True
    
    return False

def analyze_batch_samples(wavelengths, absorption_spectra, target_wavelengths, 
                         wavelength_tolerance, min_peak_height, top_k, wavelength_samples):
    """
    分析一批样本并更新波长样本
    
    Args:
        wavelengths: 波长数组
        absorption_spectra: 吸收谱数组
        target_wavelengths: 目标波长列表
        wavelength_tolerance: 波长容差
        min_peak_height: 最小峰值高度
        top_k: 每个波长保留的样本数
        wavelength_samples: 波长样本字典
    
    Returns:
        updated_count: 更新的样本数量
    """
    updated_count = 0
    
    for i in range(len(absorption_spectra)):
        # 找到峰值并计算Q值
        peaks_info = find_peaks_and_calculate_q(wavelengths, absorption_spectra[i], 
                                                min_height=0.1, min_prominence=0.05)
        
        if not peaks_info:
            continue
        
        # 找到最佳峰值
        best_peak = max(peaks_info, key=lambda x: x['q_factor'])
        
        # 检查峰值高度
        if best_peak['peak_height'] < min_peak_height:
            continue
        
        # 获取Q值阈值
        q_threshold = get_q_threshold(best_peak['center_wavelength'])
        
        # 检查Q值
        if best_peak['q_factor'] < q_threshold:
            continue
        
        # 找到匹配的波长目标
        wavelength_idx = find_best_wavelength_match(
            target_wavelengths, best_peak['center_wavelength'], wavelength_tolerance)
        
        if wavelength_idx is not None:
            # 创建样本信息
            sample_info = {
                'sample_index': i,
                'center_wavelength': best_peak['center_wavelength'],
                'peak_height': best_peak['peak_height'],
                'q_factor': best_peak['q_factor'],
                'fwhm': best_peak['fwhm'],
                'max_absorption': np.max(absorption_spectra[i]),
                'absorption_spectrum': absorption_spectra[i].copy()
            }
            
            # 更新样本
            if update_wavelength_samples(wavelength_samples, wavelength_idx, sample_info, top_k):
                updated_count += 1
    
    return updated_count

def collect_all_sample_statistics(wavelengths, absorption_spectra, target_wavelengths, 
                                 wavelength_tolerance, min_peak_height):
    """
    收集所有样本的统计信息（包括有效和无效样本）
    
    Args:
        wavelengths: 波长数组
        absorption_spectra: 吸收谱数组
        target_wavelengths: 目标波长列表
        wavelength_tolerance: 波长容差
        min_peak_height: 最小峰值高度
    
    Returns:
        wavelength_stats: 波长分布统计
        q_value_stats: Q值分布统计
        total_valid_samples: 有效样本总数
    """
    wavelength_counts = {i: 0 for i in range(len(target_wavelengths))}
    q_value_counts = {}
    total_valid_samples = 0
    
    for i in range(len(absorption_spectra)):
        # 找到峰值并计算Q值
        peaks_info = find_peaks_and_calculate_q(wavelengths, absorption_spectra[i], 
                                                min_height=0.1, min_prominence=0.05)
        
        if not peaks_info:
            continue
        
        # 找到最佳峰值
        best_peak = max(peaks_info, key=lambda x: x['q_factor'])
        
        # 检查峰值高度
        if best_peak['peak_height'] < min_peak_height:
            continue
        
        # 获取Q值阈值
        q_threshold = get_q_threshold(best_peak['center_wavelength'])
        
        # 检查Q值
        if best_peak['q_factor'] < q_threshold:
            continue
        
        # 找到匹配的波长目标
        wavelength_idx = find_best_wavelength_match(
            target_wavelengths, best_peak['center_wavelength'], wavelength_tolerance)
        
        if wavelength_idx is not None:
            # 统计波长分布
            wavelength_counts[wavelength_idx] += 1
            
            # 统计Q值分布（保留整数）
            q_value_int = int(best_peak['q_factor'])
            if q_value_int not in q_value_counts:
                q_value_counts[q_value_int] = 0
            q_value_counts[q_value_int] += 1
            
            total_valid_samples += 1
    
    return wavelength_counts, q_value_counts, total_valid_samples

def save_distribution_statistics(wavelength_counts, q_value_counts, target_wavelengths, 
                               total_generated, total_valid_samples, output_dir, 
                               stats_interval, start_time):
    """
    保存分布统计结果
    
    Args:
        wavelength_counts: 波长分布统计
        q_value_counts: Q值分布统计
        target_wavelengths: 目标波长列表
        total_generated: 总生成样本数
        total_valid_samples: 有效样本数
        output_dir: 输出目录
        stats_interval: 统计间隔
        start_time: 开始时间
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"distribution_stats_{stats_interval}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建波长分布数据
    wavelength_data = []
    for wavelength_idx, count in wavelength_counts.items():
        if count > 0:  # 只保存有样本的波长
            wavelength_data.append({
                'wavelength_index': wavelength_idx,
                'wavelength': target_wavelengths[wavelength_idx],
                'sample_count': count
            })
    
    # 创建Q值分布数据
    q_value_data = []
    for q_value, count in sorted(q_value_counts.items()):
        q_value_data.append({
            'q_value': q_value,
            'sample_count': count
        })
    
    # 保存波长分布到Excel
    if wavelength_data:
        df_wavelength = pd.DataFrame(wavelength_data)
        wavelength_excel_path = os.path.join(save_dir, 'wavelength_distribution.xlsx')
        df_wavelength.to_excel(wavelength_excel_path, index=False)
    
    # 保存Q值分布到Excel
    if q_value_data:
        df_q_value = pd.DataFrame(q_value_data)
        q_value_excel_path = os.path.join(save_dir, 'q_value_distribution.xlsx')
        df_q_value.to_excel(q_value_excel_path, index=False)
    
    # 保存统计信息
    stats_path = os.path.join(save_dir, 'distribution_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Distribution Statistics Report\n")
        f.write("=" * 50 + "\n\n")
        
        elapsed_time = time.time() - start_time
        f.write(f"Statistics generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time elapsed: {timedelta(seconds=int(elapsed_time))}\n")
        f.write(f"Statistics interval: {stats_interval:,} samples\n")
        f.write(f"Total samples generated: {total_generated:,}\n")
        f.write(f"Total valid samples: {total_valid_samples:,}\n")
        f.write(f"Valid sample ratio: {total_valid_samples/total_generated*100:.2f}%\n")
        f.write(f"Generation rate: {total_generated/elapsed_time:.1f} samples/second\n\n")
        
        # 波长分布统计
        covered_wavelengths = len([c for c in wavelength_counts.values() if c > 0])
        total_wavelengths = len(target_wavelengths)
        f.write(f"Wavelength Distribution Statistics:\n")
        f.write(f"  Total wavelength targets: {total_wavelengths}\n")
        f.write(f"  Covered wavelengths: {covered_wavelengths}\n")
        f.write(f"  Coverage ratio: {covered_wavelengths/total_wavelengths*100:.1f}%\n")
        if wavelength_data:
            max_count = max([d['sample_count'] for d in wavelength_data])
            avg_count = np.mean([d['sample_count'] for d in wavelength_data])
            f.write(f"  Maximum samples per wavelength: {max_count}\n")
            f.write(f"  Average samples per wavelength: {avg_count:.1f}\n")
        f.write(f"  Wavelength range: {target_wavelengths[0]:.2f} - {target_wavelengths[-1]:.2f} μm\n\n")
        
        # Q值分布统计
        if q_value_data:
            f.write(f"Q Value Distribution Statistics:\n")
            f.write(f"  Q value range: {min(q_value_counts.keys())} - {max(q_value_counts.keys())}\n")
            f.write(f"  Number of Q value bins: {len(q_value_counts)}\n")
            max_q_count = max(q_value_counts.values())
            avg_q_count = np.mean(list(q_value_counts.values()))
            f.write(f"  Maximum samples per Q value: {max_q_count}\n")
            f.write(f"  Average samples per Q value: {avg_q_count:.1f}\n")
            f.write(f"  Most common Q value: {max(q_value_counts, key=q_value_counts.get)} ({max_q_count} samples)\n")
    
    # 创建分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 波长分布图
    if wavelength_data:
        wavelengths_plot = [d['wavelength'] for d in wavelength_data]
        counts_plot = [d['sample_count'] for d in wavelength_data]
        ax1.bar(wavelengths_plot, counts_plot, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Wavelength [μm]', fontsize=12)
        ax1.set_ylabel('Sample Count', fontsize=12)
        ax1.set_title(f'Wavelength Distribution\n(Total: {total_valid_samples:,} samples)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Q值分布图
    if q_value_data:
        q_values_plot = [d['q_value'] for d in q_value_data]
        q_counts_plot = [d['sample_count'] for d in q_value_data]
        ax2.bar(q_values_plot, q_counts_plot, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Q Value', fontsize=12)
        ax2.set_ylabel('Sample Count', fontsize=12)
        ax2.set_title(f'Q Value Distribution\n(Total: {total_valid_samples:,} samples)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'distribution_plots.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Distribution statistics saved to: {save_dir}")
    return save_dir

def create_wavelength_analysis_plot(wavelength_samples, target_wavelengths, save_path=None):
    """
    创建波长分析图
    
    Args:
        wavelength_samples: 波长样本字典
        target_wavelengths: 目标波长列表
        save_path: 保存路径
    """
    # 收集数据
    all_wavelengths = []
    all_q_factors = []
    all_peak_heights = []
    all_max_absorptions = []
    
    for wavelength_idx, samples in wavelength_samples.items():
        target_wl = target_wavelengths[wavelength_idx]
        for sample in samples:
            all_wavelengths.append(target_wl)
            all_q_factors.append(sample['q_factor'])
            all_peak_heights.append(sample['peak_height'])
            all_max_absorptions.append(sample['max_absorption'])
    
    if not all_wavelengths:
        print("No valid samples to plot")
        return None
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Q因子 vs 波长
    scatter1 = ax1.scatter(all_wavelengths, all_q_factors, 
                          c=all_max_absorptions, cmap='viridis', 
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Wavelength [μm]', fontsize=12)
    ax1.set_ylabel('Q Factor', fontsize=12)
    ax1.set_title('Q Factor vs Wavelength', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Max Absorption')
    
    # 2. 峰值高度 vs 波长
    scatter2 = ax2.scatter(all_wavelengths, all_peak_heights, 
                          c=all_q_factors, cmap='plasma', 
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Wavelength [μm]', fontsize=12)
    ax2.set_ylabel('Peak Height', fontsize=12)
    ax2.set_title('Peak Height vs Wavelength', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Q Factor')
    
    # 3. 每个波长的样本数量
    wavelength_counts = []
    for i, wl in enumerate(target_wavelengths):
        count = len(wavelength_samples.get(i, []))
        wavelength_counts.append(count)
    
    ax3.bar(target_wavelengths, wavelength_counts, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Wavelength [μm]', fontsize=12)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Samples per Wavelength', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q因子分布直方图
    ax4.hist(all_q_factors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_xlabel('Q Factor', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Q Factor Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to: {save_path}")
    
    return fig

def save_wavelength_analysis_results(wavelength_samples, target_wavelengths, output_dir, 
                                   total_generated, start_time):
    """保存波长分析结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"wavelength_analysis_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建结果数据
    results_data = []
    for wavelength_idx, samples in wavelength_samples.items():
        target_wl = target_wavelengths[wavelength_idx]
        for i, sample in enumerate(samples):
            results_data.append({
                'wavelength_index': wavelength_idx,
                'target_wavelength': target_wl,
                'sample_rank': i + 1,
                'center_wavelength': sample['center_wavelength'],
                'peak_height': sample['peak_height'],
                'q_factor': sample['q_factor'],
                'fwhm': sample['fwhm'],
                'max_absorption': sample['max_absorption']
            })
    
    # 保存到Excel
    if results_data:
        df = pd.DataFrame(results_data)
        excel_path = os.path.join(save_dir, 'wavelength_analysis_results.xlsx')
        df.to_excel(excel_path, index=False)
    
    # 保存统计信息
    stats_path = os.path.join(save_dir, 'statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Wavelength-based GAN Sample Analysis Statistics\n")
        f.write("=" * 60 + "\n\n")
        
        elapsed_time = time.time() - start_time
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time elapsed: {timedelta(seconds=int(elapsed_time))}\n")
        f.write(f"Total samples generated: {total_generated:,}\n")
        f.write(f"Generation rate: {total_generated/elapsed_time:.1f} samples/second\n\n")
        
        # 波长覆盖统计
        covered_wavelengths = len(wavelength_samples)
        total_wavelengths = len(target_wavelengths)
        f.write(f"Wavelength Coverage:\n")
        f.write(f"  Total wavelength targets: {total_wavelengths}\n")
        f.write(f"  Covered wavelengths: {covered_wavelengths}\n")
        f.write(f"  Coverage ratio: {covered_wavelengths/total_wavelengths*100:.1f}%\n\n")
        
        # 样本统计
        total_samples = sum(len(samples) for samples in wavelength_samples.values())
        f.write(f"Sample Statistics:\n")
        f.write(f"  Total valid samples: {total_samples}\n")
        f.write(f"  Average samples per wavelength: {total_samples/covered_wavelengths:.1f}\n")
        
        # Q值统计
        all_q_factors = []
        for samples in wavelength_samples.values():
            all_q_factors.extend([s['q_factor'] for s in samples])
        
        if all_q_factors:
            f.write(f"\nQ Factor Statistics:\n")
            f.write(f"  Maximum Q: {max(all_q_factors):.2f}\n")
            f.write(f"  Average Q: {np.mean(all_q_factors):.2f}\n")
            f.write(f"  Median Q: {np.median(all_q_factors):.2f}\n")
            f.write(f"  Q Standard Deviation: {np.std(all_q_factors):.2f}\n")
    
    # 保存分析图
    fig = create_wavelength_analysis_plot(wavelength_samples, target_wavelengths)
    if fig:
        plot_path = os.path.join(save_dir, 'wavelength_analysis.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Wavelength analysis results saved to: {save_dir}")
    return save_dir

def main():
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载参数
    params = load_parameters(args.config_path, device)
    
    # 加载模型
    generator = load_model(args.model_path, params, device)
    
    # 创建波长目标点
    target_wavelengths = create_wavelength_targets(params.wavelength_range, args.wavelength_step)
    print(f"Created {len(target_wavelengths)} wavelength targets from {params.wavelength_range[0]} to {params.wavelength_range[1]} μm")
    
    # 初始化波长样本字典
    wavelength_samples = {}
    
    # 初始化分布统计变量
    all_wavelength_counts = {i: 0 for i in range(len(target_wavelengths))}
    all_q_value_counts = {}
    total_valid_samples_all = 0
    
    # 记录开始时间和进度
    start_time = time.time()
    total_generated = 0
    last_save_time = start_time
    
    print(f"\nStarting wavelength-based sample collection:")
    print(f"  - Target wavelengths: {len(target_wavelengths)}")
    print(f"  - Wavelength step: {args.wavelength_step} μm")
    print(f"  - Wavelength tolerance: {args.wavelength_tolerance} μm")
    print(f"  - Top K per wavelength: {args.top_k_per_wavelength}")
    print(f"  - Min peak height: {args.min_peak_height}")
    print(f"  - Q threshold (3-5μm): 300, (5-8μm): 250")
    print(f"  - Max samples: {args.max_samples:,}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Save interval: {args.save_interval}")
    print(f"  - Stats interval: {args.stats_interval}")
    print()
    
    # 创建进度条
    pbar = tqdm(total=args.max_samples, desc="Generating samples")
    
    while total_generated < args.max_samples:
        # 生成一批样本
        wavelengths, thicknesses, refractive_indices, P, absorption_spectra = generate_samples(
            generator, params, args.batch_size, args.alpha, device)
        
        # 分析样本并更新波长样本
        updated_count = analyze_batch_samples(
            wavelengths, absorption_spectra, target_wavelengths,
            args.wavelength_tolerance, args.min_peak_height, 
            args.top_k_per_wavelength, wavelength_samples)
        
        # 收集分布统计信息
        batch_wavelength_counts, batch_q_value_counts, batch_valid_samples = collect_all_sample_statistics(
            wavelengths, absorption_spectra, target_wavelengths,
            args.wavelength_tolerance, args.min_peak_height)
        
        # 更新总体统计
        for wavelength_idx, count in batch_wavelength_counts.items():
            all_wavelength_counts[wavelength_idx] += count
        
        for q_value, count in batch_q_value_counts.items():
            if q_value not in all_q_value_counts:
                all_q_value_counts[q_value] = 0
            all_q_value_counts[q_value] += count
        
        total_valid_samples_all += batch_valid_samples
        total_generated += args.batch_size
        pbar.update(args.batch_size)
        
        # 更新进度条描述
        covered_wavelengths = len(wavelength_samples)
        total_samples = sum(len(samples) for samples in wavelength_samples.values())
        pbar.set_description(f"Generated: {total_generated:,}, Updated: {updated_count}, "
                           f"Covered: {covered_wavelengths}/{len(target_wavelengths)} wavelengths, "
                           f"Total samples: {total_samples}, Valid: {total_valid_samples_all}")
        
        # 定期生成分布统计
        if total_generated % args.stats_interval == 0:
            print(f"\n--- Distribution Statistics Update ---")
            print(f"Generated: {total_generated:,} samples")
            print(f"Valid samples: {total_valid_samples_all:,}")
            print(f"Valid ratio: {total_valid_samples_all/total_generated*100:.2f}%")
            
            # 保存分布统计
            save_distribution_statistics(
                all_wavelength_counts, all_q_value_counts, target_wavelengths,
                total_generated, total_valid_samples_all, args.output_dir,
                args.stats_interval, start_time)
        
        # 定期保存结果
        if total_generated % args.save_interval == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = (args.max_samples - total_generated) / (total_generated / elapsed)
            
            print(f"\n--- Progress Update ---")
            print(f"Generated: {total_generated:,}/{args.max_samples:,} samples")
            print(f"Elapsed time: {timedelta(seconds=int(elapsed))}")
            print(f"Estimated remaining time: {timedelta(seconds=int(remaining))}")
            print(f"Covered wavelengths: {covered_wavelengths}/{len(target_wavelengths)}")
            print(f"Total valid samples: {total_samples}")
            print(f"Generation rate: {total_generated/elapsed:.1f} samples/second")
            
            # 保存当前结果
            save_wavelength_analysis_results(
                wavelength_samples, target_wavelengths, args.output_dir,
                total_generated, start_time)
            
            last_save_time = current_time
    
    pbar.close()
    
    # 最终保存
    print(f"\n--- Final Results ---")
    final_save_dir = save_wavelength_analysis_results(
        wavelength_samples, target_wavelengths, args.output_dir,
        total_generated, start_time)
    
    # 最终分布统计
    print(f"\n--- Final Distribution Statistics ---")
    final_stats_dir = save_distribution_statistics(
        all_wavelength_counts, all_q_value_counts, target_wavelengths,
        total_generated, total_valid_samples_all, args.output_dir,
        total_generated, start_time)
    
    # 显示最终统计
    total_time = time.time() - start_time
    covered_wavelengths = len(wavelength_samples)
    total_samples = sum(len(samples) for samples in wavelength_samples.values())
    
    print(f"\nFinal Analysis Results:")
    print(f"=" * 60)
    print(f"Total samples generated: {total_generated:,}")
    print(f"Total valid samples: {total_valid_samples_all:,}")
    print(f"Valid sample ratio: {total_valid_samples_all/total_generated*100:.2f}%")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Generation rate: {total_generated/total_time:.1f} samples/second")
    print(f"Covered wavelengths: {covered_wavelengths}/{len(target_wavelengths)} ({covered_wavelengths/len(target_wavelengths)*100:.1f}%)")
    print(f"Total valid samples: {total_samples}")
    
    if total_samples > 0:
        all_q_factors = []
        all_peak_heights = []
        for samples in wavelength_samples.values():
            all_q_factors.extend([s['q_factor'] for s in samples])
            all_peak_heights.extend([s['peak_height'] for s in samples])
        
        print(f"\nQ Factor Statistics:")
        print(f"  Range: {min(all_q_factors):.2f} - {max(all_q_factors):.2f}")
        print(f"  Average: {np.mean(all_q_factors):.2f}")
        print(f"  Median: {np.median(all_q_factors):.2f}")
        
        print(f"\nPeak Height Statistics:")
        print(f"  Range: {min(all_peak_heights):.3f} - {max(all_peak_heights):.3f}")
        print(f"  Average: {np.mean(all_peak_heights):.3f}")
        print(f"  Median: {np.median(all_peak_heights):.3f}")
    
    print(f"\nResults saved to: {final_save_dir}")
    print(f"Distribution statistics saved to: {final_stats_dir}")
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()
