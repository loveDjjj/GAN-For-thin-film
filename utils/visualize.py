"""
Visualization Module
Contains functions for plotting charts, saving results and displaying outputs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_gan_samples(wavelengths, real_samples, fake_samples, d_real, d_fake, num_samples=4):
    """Plot comparison between GAN generated samples and real samples
    
    Args:
        wavelengths: Wavelength array
        real_samples: Real sample tensor
        fake_samples: Generated fake sample tensor
        d_real: Discriminator scores for real samples
        d_fake: Discriminator scores for generated samples
        num_samples: Number of samples to plot
    
    Returns:
        fig: Figure object
    """
    # Ensure all tensors are on CPU
    if torch.is_tensor(wavelengths):
        wavelengths = wavelengths.cpu().numpy()
    if torch.is_tensor(real_samples):
        real_samples = real_samples.cpu().numpy()
    if torch.is_tensor(fake_samples):
        fake_samples = fake_samples.cpu().numpy()
    if torch.is_tensor(d_real):
        d_real = d_real.cpu()
    if torch.is_tensor(d_fake):
        d_fake = d_fake.cpu()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Plot specified number of samples
    for i in range(min(num_samples, len(real_samples))):
        # Plot real sample
        plt.subplot(num_samples, 2, 2*i+1)
        plt.plot(wavelengths, real_samples[i])
        plt.title(f'Real Sample {i+1} - D(x): {d_real[i].item():.3f}')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Absorption')
        plt.grid(True)
        
        # Plot generated sample
        plt.subplot(num_samples, 2, 2*i+2)
        plt.plot(wavelengths, fake_samples[i])
        plt.title(f'Generated Sample {i+1} - D(G(z)): {d_fake[i].item():.3f}')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Absorption')
        plt.grid(True)
    
    plt.tight_layout()
    return fig


def plot_gan_training_curves(g_losses, d_losses, d_real_scores, d_fake_scores):
    """Plot GAN training curves
    
    Args:
        g_losses: Generator loss list
        d_losses: Discriminator loss list
        d_real_scores: Discriminator scores for real samples list
        d_fake_scores: Discriminator scores for generated samples list
    
    Returns:
        fig: Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Loss curves
    plt.subplot(2, 1, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Loss')
    plt.grid(True)
    
    # Discriminator score curves
    plt.subplot(2, 1, 2)
    plt.plot(d_real_scores, label='D(Real)')
    plt.plot(d_fake_scores, label='D(Generated)')
    plt.xlabel('Iterations')
    plt.ylabel('Discriminator Score')
    plt.legend()
    plt.title('Discriminator Scores')
    plt.grid(True)
    
    plt.tight_layout()
    return fig


def save_gan_samples(wavelengths, real_samples, fake_samples, d_real, d_fake, save_path, epoch, num_samples=4):
    """Plot and save GAN sample comparison
    
    Args:
        wavelengths: Wavelength array
        real_samples: Real sample tensor
        fake_samples: Generated fake sample tensor
        d_real: Discriminator scores for real samples
        d_fake: Discriminator scores for generated samples
        save_path: Save path
        epoch: Current training epoch
        num_samples: Number of samples to plot
    """
    fig = plot_gan_samples(wavelengths, real_samples, fake_samples, d_real, d_fake, num_samples)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'epoch_{epoch}.png'), dpi=300)
    plt.close(fig)
    print(f"Sample images saved to: {os.path.join(save_path, f'epoch_{epoch}.png')}")


def save_gan_training_curves(g_losses, d_losses, d_real_scores, d_fake_scores, save_path):
    """Plot and save GAN training curves

    Args:
        g_losses: Generator loss list
        d_losses: Discriminator loss list
        d_real_scores: Discriminator scores for real samples list
        d_fake_scores: Discriminator scores for generated samples list
        save_path: Save path
    """
    fig = plot_gan_training_curves(g_losses, d_losses, d_real_scores, d_fake_scores)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


def save_alpha_entropy_curves(alpha_history, entropy_history, save_path):
    """Plot and save alpha and entropy curves during training

    Args:
        alpha_history: List of alpha values over epochs
        entropy_history: List of mean entropy values over epochs
        save_path: Save path for the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    epochs = range(1, len(alpha_history) + 1)

    # Alpha curve
    ax1 = axes[0]
    ax1.plot(epochs, alpha_history, 'b-', linewidth=2, label='α (softmax temperature)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Alpha (α)')
    ax1.set_title('Softmax Temperature (α) During Training')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add annotation explaining alpha
    ax1.annotate(f'Start: α={alpha_history[0]}', xy=(1, alpha_history[0]),
                xytext=(len(epochs)*0.1, alpha_history[0]),
                fontsize=9, color='blue')
    ax1.annotate(f'End: α={alpha_history[-1]}', xy=(len(epochs), alpha_history[-1]),
                xytext=(len(epochs)*0.8, alpha_history[-1]*1.05),
                fontsize=9, color='blue')

    # Entropy curve
    ax2 = axes[1]
    ax2.plot(epochs, entropy_history, 'r-', linewidth=2, label='Mean Entropy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Material Selection Entropy During Training (lower = more deterministic)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add reference lines for entropy
    max_entropy = np.log(2)  # For 2 materials
    ax2.axhline(y=max_entropy, color='gray', linestyle='--', alpha=0.5,
                label=f'Max Entropy (uniform) = {max_entropy:.3f}')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5,
                label='Min Entropy (deterministic) = 0')
    ax2.legend(fontsize=9)

    # Add shaded regions to indicate selection quality
    ax2.fill_between(epochs, 0, entropy_history, alpha=0.3, color='red')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Alpha-Entropy curves saved to: {save_path}")


def save_thickness_merged_layers_curves(mean_thickness_history, merged_layers_history, save_path):
    """Plot and save thickness and merged layers curves during training

    Args:
        mean_thickness_history: List of mean thickness values over epochs
        merged_layers_history: List of mean merged layers count over epochs
        save_path: Save path for the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    epochs = range(1, len(mean_thickness_history) + 1)

    # Average thickness curve
    ax1 = axes[0]
    ax1.plot(epochs, mean_thickness_history, 'g-', linewidth=2, label='Average Thickness')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Thickness (μm)')
    ax1.set_title('Average Generated Layer Thickness During Training')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add horizontal line for thickness range boundaries
    # These are based on config values: thickness_bot=0.05, thickness_sup=0.5
    ax1.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Lower Bound (0.05 μm)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Upper Bound (0.5 μm)')
    ax1.axhline(y=0.275, color='gray', linestyle=':', alpha=0.3, label='Mid Point (0.275 μm)')
    ax1.legend(fontsize=9)

    # Add annotation for start and end thickness
    if len(mean_thickness_history) > 1:
        ax1.annotate(f'Start: {mean_thickness_history[0]:.3f} μm',
                    xy=(1, mean_thickness_history[0]),
                    xytext=(len(epochs)*0.1, mean_thickness_history[0]),
                    fontsize=9, color='green')
        ax1.annotate(f'End: {mean_thickness_history[-1]:.3f} μm',
                    xy=(len(epochs), mean_thickness_history[-1]),
                    xytext=(len(epochs)*0.8, mean_thickness_history[-1]*1.05),
                    fontsize=9, color='green')

    # Merged layers curve
    ax2 = axes[1]
    ax2.plot(epochs, merged_layers_history, 'b-', linewidth=2, label='Merged Layers')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Average Number of Merged Layers During Training (adjacent same materials merged)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add horizontal lines for reference (min layers = 1, max layers = N_layers)
    ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5,
                label='Max Possible Layers (10)')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5,
                label='Min Possible Layers (1)')
    ax2.legend(fontsize=9)

    # Add shaded region to show layer reduction
    ax2.fill_between(epochs, merged_layers_history, 10, alpha=0.2, color='blue')

    # Add annotation for start and end merged layers
    if len(merged_layers_history) > 1:
        ax2.annotate(f'Start: {merged_layers_history[0]:.1f} layers',
                    xy=(1, merged_layers_history[0]),
                    xytext=(len(epochs)*0.1, merged_layers_history[0]),
                    fontsize=9, color='blue')
        ax2.annotate(f'End: {merged_layers_history[-1]:.1f} layers',
                    xy=(len(epochs), merged_layers_history[-1]),
                    xytext=(len(epochs)*0.8, merged_layers_history[-1]*1.05),
                    fontsize=9, color='blue')

        # Add text explaining what merged layers means
        ax2.text(0.02, 0.02, 'Note: Merged layers count = number of distinct material layers\n(adjacent same materials are counted as one layer)',
                transform=ax2.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Thickness-Merged Layers curves saved to: {save_path}")


def save_distribution_evolution_plots(thickness_distribution_history, merged_layers_distribution_history, save_dir):
    """保存厚度和层数分布随训练演变的可视化图

    Args:
        thickness_distribution_history: 厚度分布历史列表
        merged_layers_distribution_history: 合并层数分布历史列表
        save_dir: 保存目录
    """
    if not thickness_distribution_history or not merged_layers_distribution_history:
        print("No distribution data to visualize")
        return

    # 1. 厚度分布演变图
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Thickness Distribution Evolution During Training', fontsize=16, fontweight='bold')

    # 提取数据
    epochs = [data['epoch'] for data in thickness_distribution_history]
    thickness_means = [data['mean_thickness'] for data in thickness_distribution_history]

    # a. 平均厚度变化（已有曲线，但这里显示为参考）
    ax1 = axes1[0, 0]
    ax1.plot(epochs, thickness_means, 'g-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Thickness (μm)')
    ax1.set_title('Average Thickness Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Lower Bound (0.05 μm)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Upper Bound (0.5 μm)')
    ax1.legend(fontsize=9)

    # b. 厚度分布热力图（随时间变化）
    ax2 = axes1[0, 1]
    # 使用第一个分布作为参考的bin边缘
    bin_edges = thickness_distribution_history[0]['bin_edges']
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

    # 创建分布矩阵
    distribution_matrix = []
    for data in thickness_distribution_history:
        distribution_matrix.append(data['hist_counts'])

    im = ax2.imshow(distribution_matrix, aspect='auto', cmap='YlOrRd',
                   extent=[bin_centers[0], bin_centers[-1], epochs[-1], epochs[0]])
    ax2.set_xlabel('Thickness (μm)')
    ax2.set_ylabel('Epoch')
    ax2.set_title('Thickness Distribution Heatmap')
    plt.colorbar(im, ax=ax2, label='Count')

    # c. 起始、中间、结束三个epoch的分布对比
    ax3 = axes1[1, 0]
    if len(thickness_distribution_history) >= 3:
        start_idx = 0
        mid_idx = len(thickness_distribution_history) // 2
        end_idx = -1

        start_data = thickness_distribution_history[start_idx]
        mid_data = thickness_distribution_history[mid_idx]
        end_data = thickness_distribution_history[end_idx]

        bin_centers = [(start_data['bin_edges'][i] + start_data['bin_edges'][i+1])/2
                      for i in range(len(start_data['bin_edges'])-1)]

        ax3.bar(bin_centers, start_data['hist_counts'], alpha=0.5, width=0.02,
                label=f'Epoch {start_data["epoch"]}', color='blue')
        ax3.bar(bin_centers, mid_data['hist_counts'], alpha=0.5, width=0.02,
                label=f'Epoch {mid_data["epoch"]}', color='green')
        ax3.bar(bin_centers, end_data['hist_counts'], alpha=0.5, width=0.02,
                label=f'Epoch {end_data["epoch"]}', color='red')

        ax3.set_xlabel('Thickness (μm)')
        ax3.set_ylabel('Count')
        ax3.set_title('Thickness Distribution Comparison (Start/Mid/End)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # d. 厚度分布标准差变化
    ax4 = axes1[1, 1]
    thickness_stds = []
    for data in thickness_distribution_history:
        # 计算分布的近似标准差
        hist_counts = data['hist_counts']
        bin_edges = data['bin_edges']
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

        if np.sum(hist_counts) > 0:
            normalized_counts = hist_counts / np.sum(hist_counts)
            mean = np.sum(normalized_counts * bin_centers)
            std = np.sqrt(np.sum(normalized_counts * (bin_centers - mean)**2))
            thickness_stds.append(std)
        else:
            thickness_stds.append(0)

    ax4.plot(epochs, thickness_stds, 'b-', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Thickness Std (μm)')
    ax4.set_title('Thickness Distribution Standard Deviation')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    thickness_plot_path = os.path.join(save_dir, 'thickness_distribution_evolution.png')
    fig1.savefig(thickness_plot_path, dpi=300)
    plt.close(fig1)
    print(f"Thickness distribution evolution plot saved to: {thickness_plot_path}")

    # 2. 合并层数分布演变图
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Merged Layers Distribution Evolution During Training', fontsize=16, fontweight='bold')

    # 提取数据
    epochs_layers = [data['epoch'] for data in merged_layers_distribution_history]
    layers_means = [data['mean_merged_layers'] for data in merged_layers_distribution_history]

    # a. 平均合并层数变化
    ax1 = axes2[0, 0]
    ax1.plot(epochs_layers, layers_means, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Merged Layers')
    ax1.set_title('Average Merged Layers Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Min (1 layer)')
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Max (10 layers)')
    ax1.legend(fontsize=9)

    # b. 层数分布热力图
    ax2 = axes2[0, 1]
    max_layers = len(merged_layers_distribution_history[0]['layer_counts'])
    layer_indices = list(range(1, max_layers + 1))

    layers_matrix = []
    for data in merged_layers_distribution_history:
        layers_matrix.append(data['layer_counts'])

    im2 = ax2.imshow(layers_matrix, aspect='auto', cmap='Blues',
                    extent=[layer_indices[0]-0.5, layer_indices[-1]+0.5,
                           epochs_layers[-1], epochs_layers[0]])
    ax2.set_xlabel('Number of Merged Layers')
    ax2.set_ylabel('Epoch')
    ax2.set_title('Merged Layers Distribution Heatmap')
    ax2.set_xticks(layer_indices)
    plt.colorbar(im2, ax=ax2, label='Count')

    # c. 起始、中间、结束三个epoch的层数分布对比
    ax3 = axes2[1, 0]
    if len(merged_layers_distribution_history) >= 3:
        start_idx = 0
        mid_idx = len(merged_layers_distribution_history) // 2
        end_idx = -1

        start_data = merged_layers_distribution_history[start_idx]
        mid_data = merged_layers_distribution_history[mid_idx]
        end_data = merged_layers_distribution_history[end_idx]

        layer_indices = list(range(1, len(start_data['layer_counts']) + 1))

        width = 0.25
        ax3.bar([i - width for i in layer_indices], start_data['layer_counts'],
                width=width, alpha=0.7, label=f'Epoch {start_data["epoch"]}', color='blue')
        ax3.bar(layer_indices, mid_data['layer_counts'],
                width=width, alpha=0.7, label=f'Epoch {mid_data["epoch"]}', color='green')
        ax3.bar([i + width for i in layer_indices], end_data['layer_counts'],
                width=width, alpha=0.7, label=f'Epoch {end_data["epoch"]}', color='red')

        ax3.set_xlabel('Number of Merged Layers')
        ax3.set_ylabel('Count')
        ax3.set_title('Merged Layers Distribution Comparison (Start/Mid/End)')
        ax3.set_xticks(layer_indices)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # d. 层数分布熵（衡量分布的集中程度）
    ax4 = axes2[1, 1]
    layer_entropies = []
    for data in merged_layers_distribution_history:
        layer_counts = data['layer_counts']
        total = np.sum(layer_counts)
        if total > 0:
            probs = layer_counts / total
            # 计算熵，忽略零概率
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            layer_entropies.append(entropy)
        else:
            layer_entropies.append(0)

    ax4.plot(epochs_layers, layer_entropies, 'r-', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Entropy (bits)')
    ax4.set_title('Merged Layers Distribution Entropy (higher = more diverse)')
    ax4.grid(True, alpha=0.3)
    # 添加最大熵参考线（均匀分布）
    max_entropy = np.log(max_layers)
    ax4.axhline(y=max_entropy, color='gray', linestyle='--', alpha=0.5,
                label=f'Max entropy (uniform) = {max_entropy:.2f}')
    ax4.legend(fontsize=9)

    plt.tight_layout()
    layers_plot_path = os.path.join(save_dir, 'merged_layers_distribution_evolution.png')
    fig2.savefig(layers_plot_path, dpi=300)
    plt.close(fig2)
    print(f"Merged layers distribution evolution plot saved to: {layers_plot_path}")

    # 3. 保存分布数据到JSON文件
    import json
    distribution_data = {
        'thickness_distribution': thickness_distribution_history,
        'merged_layers_distribution': merged_layers_distribution_history,
        'config': {
            'thickness_range': [0.05, 0.5],
            'max_layers': 10,
            'num_thickness_bins': 20
        }
    }

    json_path = os.path.join(save_dir, 'distribution_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(distribution_data, f, indent=2, ensure_ascii=False)

    print(f"Distribution data saved to: {json_path}")


def analyze_inference_distribution(thicknesses, material_probs, save_dir, prefix="inference"):
    """分析推理过程中生成的样本的厚度和层数分布

    Args:
        thicknesses: 厚度张量 [num_samples, N_layers]
        material_probs: 材料选择概率张量 [num_samples, N_layers, M_materials]
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    import torch
    import json

    thicknesses_np = thicknesses.cpu().numpy() if torch.is_tensor(thicknesses) else thicknesses
    material_probs_np = material_probs.cpu().numpy() if torch.is_tensor(material_probs) else material_probs

    num_samples, N_layers = thicknesses_np.shape

    # 1. 厚度分布分析
    thickness_range = (0.05, 0.5)  # 假设的范围，可以从参数获取
    thickness_hist_counts, thickness_bin_edges = np.histogram(
        thicknesses_np.flatten(), bins=20, range=thickness_range)

    # 计算厚度统计
    thickness_mean = np.mean(thicknesses_np)
    thickness_std = np.std(thicknesses_np)
    thickness_percentiles = {
        'p10': np.percentile(thicknesses_np, 10),
        'p25': np.percentile(thicknesses_np, 25),
        'p50': np.percentile(thicknesses_np, 50),
        'p75': np.percentile(thicknesses_np, 75),
        'p90': np.percentile(thicknesses_np, 90)
    }

    # 2. 合并层数分布分析
    from train.trainer import calculate_merged_layers
    material_probs_tensor = torch.tensor(material_probs_np) if not torch.is_tensor(material_probs) else material_probs
    merged_counts = calculate_merged_layers(material_probs_tensor)
    merged_counts_np = merged_counts.cpu().numpy() if torch.is_tensor(merged_counts) else merged_counts

    max_layers = N_layers
    layer_counts = np.zeros(max_layers, dtype=int)
    for i in range(1, max_layers + 1):
        layer_counts[i-1] = np.sum(merged_counts_np == i)

    # 计算层数统计
    merged_layers_mean = np.mean(merged_counts_np)
    merged_layers_std = np.std(merged_counts_np)
    merged_layers_percentiles = {
        'p10': np.percentile(merged_counts_np, 10),
        'p25': np.percentile(merged_counts_np, 25),
        'p50': np.percentile(merged_counts_np, 50),
        'p75': np.percentile(merged_counts_np, 75),
        'p90': np.percentile(merged_counts_np, 90)
    }

    # 3. 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Distribution Analysis of Generated Samples (n={num_samples})',
                 fontsize=16, fontweight='bold')

    # a. 厚度直方图
    ax1 = axes[0, 0]
    bin_centers = [(thickness_bin_edges[i] + thickness_bin_edges[i+1])/2
                  for i in range(len(thickness_bin_edges)-1)]
    ax1.bar(bin_centers, thickness_hist_counts, width=0.02, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=thickness_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {thickness_mean:.3f} μm')
    ax1.set_xlabel('Thickness (μm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Thickness Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # b. 层数分布柱状图
    ax2 = axes[0, 1]
    layer_indices = list(range(1, max_layers + 1))
    ax2.bar(layer_indices, layer_counts, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(x=merged_layers_mean, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {merged_layers_mean:.1f} layers')
    ax2.set_xlabel('Number of Merged Layers')
    ax2.set_ylabel('Count')
    ax2.set_title('Merged Layers Distribution')
    ax2.set_xticks(layer_indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # c. 厚度vs层数散点图
    ax3 = axes[0, 2]
    # 计算每个样本的总厚度
    total_thickness = np.sum(thicknesses_np, axis=1)
    scatter = ax3.scatter(total_thickness, merged_counts_np,
                         alpha=0.6, s=30, c=thickness_mean, cmap='viridis')
    ax3.set_xlabel('Total Structure Thickness (μm)')
    ax3.set_ylabel('Number of Merged Layers')
    ax3.set_title('Total Thickness vs Merged Layers')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Average Layer Thickness (μm)')

    # d. 厚度箱线图
    ax4 = axes[1, 0]
    # 展平所有厚度值
    ax4.boxplot(thicknesses_np.flatten(), vert=False)
    ax4.set_xlabel('Thickness (μm)')
    ax4.set_title('Thickness Box Plot')
    ax4.grid(True, alpha=0.3)

    # e. 层数箱线图
    ax5 = axes[1, 1]
    ax5.boxplot(merged_counts_np, vert=False)
    ax5.set_xlabel('Number of Merged Layers')
    ax5.set_title('Merged Layers Box Plot')
    ax5.grid(True, alpha=0.3)

    # f. 累计分布函数
    ax6 = axes[1, 2]
    # 厚度CDF
    sorted_thickness = np.sort(thicknesses_np.flatten())
    thickness_cdf = np.arange(1, len(sorted_thickness)+1) / len(sorted_thickness)
    ax6.plot(sorted_thickness, thickness_cdf, 'b-', linewidth=2, label='Thickness')

    # 层数CDF
    sorted_layers = np.sort(merged_counts_np)
    layers_cdf = np.arange(1, len(sorted_layers)+1) / len(sorted_layers)
    ax6.step(sorted_layers, layers_cdf, 'r-', linewidth=2, where='post', label='Merged Layers')

    ax6.set_xlabel('Value')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('Cumulative Distribution Functions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(save_dir, f'{prefix}_distribution_analysis.png')
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Inference distribution analysis plot saved to: {plot_path}")

    # 4. 保存统计数据
    stats_data = {
        'num_samples': num_samples,
        'thickness_statistics': {
            'mean': float(thickness_mean),
            'std': float(thickness_std),
            'percentiles': {k: float(v) for k, v in thickness_percentiles.items()},
            'histogram': {
                'counts': thickness_hist_counts.tolist(),
                'bin_edges': thickness_bin_edges.tolist()
            }
        },
        'merged_layers_statistics': {
            'mean': float(merged_layers_mean),
            'std': float(merged_layers_std),
            'percentiles': {k: float(v) for k, v in merged_layers_percentiles.items()},
            'distribution': layer_counts.tolist()
        }
    }

    stats_path = os.path.join(save_dir, f'{prefix}_distribution_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    print(f"Inference distribution statistics saved to: {stats_path}")

    return stats_data