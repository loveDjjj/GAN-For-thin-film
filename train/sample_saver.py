import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.Lorentzian.lorentzian_curves import generate_lorentzian_curves
from model.TMM.optical_calculator import calculate_reflection
from utils.visualize import save_gan_samples


def calculate_entropy(probs):
    """计算概率分布的熵，衡量材料选择的确定性"""
    # 避免 log(0)
    probs_clamped = torch.clamp(probs, min=1e-10)
    entropy = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=-1)
    return entropy


def save_material_probability_analysis(P, alpha, epoch, samples_dir, materials):
    """保存材料选择概率分析图"""
    data_dir = os.path.join(samples_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # P shape: [batch_size, N_layers, M_materials]
    P_np = P.cpu().numpy()
    batch_size, n_layers, n_materials = P_np.shape

    # 计算统计量
    P_mean = P_np.mean(axis=0)  # [N_layers, M_materials]
    P_std = P_np.std(axis=0)

    # 计算熵
    entropy = calculate_entropy(P).cpu().numpy()  # [batch_size, N_layers]
    entropy_mean = entropy.mean(axis=0)  # [N_layers]

    # 创建可视化图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Material Selection Analysis (Epoch {epoch}, α={alpha})', fontsize=14)

    # 1. 材料概率热力图 (平均值)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(P_mean.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Material')
    ax1.set_yticks(range(n_materials))
    ax1.set_yticklabels(materials)
    ax1.set_xticks(range(n_layers))
    ax1.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
    ax1.set_title('Mean Material Probability per Layer')
    plt.colorbar(im1, ax=ax1, label='Probability')

    # 在热力图上添加数值标注
    for i in range(n_layers):
        for j in range(n_materials):
            text = ax1.text(i, j, f'{P_mean[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)

    # 2. 每层的材料概率柱状图
    ax2 = axes[0, 1]
    x = np.arange(n_layers)
    width = 0.35
    for mat_idx in range(n_materials):
        offset = (mat_idx - (n_materials - 1) / 2) * width
        bars = ax2.bar(x + offset, P_mean[:, mat_idx], width,
                      label=materials[mat_idx], yerr=P_std[:, mat_idx], capsize=3)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Probability')
    ax2.set_title('Material Selection Probability per Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

    # 3. 每层的熵
    ax3 = axes[1, 0]
    max_entropy = np.log(n_materials)  # 最大熵 (均匀分布)
    ax3.bar(range(n_layers), entropy_mean, color='steelblue', alpha=0.7)
    ax3.axhline(y=max_entropy, color='red', linestyle='--', label=f'Max Entropy (uniform) = {max_entropy:.3f}')
    ax3.axhline(y=0, color='green', linestyle='--', label='Min Entropy (deterministic) = 0')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Entropy')
    ax3.set_title(f'Selection Entropy per Layer (lower = more deterministic)')
    ax3.set_xticks(range(n_layers))
    ax3.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, max_entropy * 1.2)

    # 4. 主导材料统计
    ax4 = axes[1, 1]
    dominant_materials = np.argmax(P_mean, axis=1)  # 每层的主导材料
    dominant_probs = np.max(P_mean, axis=1)  # 主导材料的概率

    colors = ['#ff7f0e' if dm == 0 else '#1f77b4' for dm in dominant_materials]
    bars = ax4.bar(range(n_layers), dominant_probs, color=colors, alpha=0.8)
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Dominant Material Probability')
    ax4.set_title('Dominant Material per Layer')
    ax4.set_xticks(range(n_layers))
    ax4.set_xticklabels([f'L{i+1}\n({materials[dm]})' for i, dm in enumerate(dominant_materials)], fontsize=8)
    ax4.set_ylim(0, 1.1)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff7f0e', label=materials[0]),
                      Patch(facecolor='#1f77b4', label=materials[1] if n_materials > 1 else '')]
    ax4.legend(handles=legend_elements[:n_materials], loc='upper right')

    plt.tight_layout()

    # 保存图像
    fig_path = os.path.join(data_dir, f'material_probability_epoch_{epoch}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Material probability analysis saved to: {fig_path}")

    # 保存数值数据到CSV
    stats_data = {
        'Layer': [f'Layer_{i+1}' for i in range(n_layers)],
        'Entropy': entropy_mean,
        'Dominant_Material': [materials[dm] for dm in dominant_materials],
        'Dominant_Prob': dominant_probs,
    }
    for mat_idx, mat_name in enumerate(materials):
        stats_data[f'{mat_name}_Mean'] = P_mean[:, mat_idx]
        stats_data[f'{mat_name}_Std'] = P_std[:, mat_idx]

    stats_df = pd.DataFrame(stats_data)
    csv_path = os.path.join(data_dir, f'material_stats_epoch_{epoch}.csv')
    stats_df.to_csv(csv_path, index=False)

    return {
        'alpha': alpha,
        'epoch': epoch,
        'mean_entropy': entropy_mean.mean(),
        'P_mean': P_mean,
    }


def save_sample(generator, discriminator, params, epoch, samples_dir, device, alpha=None):
    """Save generated sample images and additional data."""
    if alpha is None:
        # 默认使用 alpha_max（训练结束时的值）
        alpha = getattr(params, 'alpha_max', getattr(params, 'alpha', 20))

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        thickness_noise = torch.randn(8, params.thickness_noise_dim, device=device)
        material_noise = torch.randn(8, params.material_noise_dim, device=device)

        thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, alpha)

        wavelengths = (2 * np.pi / params.k.to(device))
        real_samples = generate_lorentzian_curves(
            wavelengths, batch_size=8, width=params.lorentz_width, center_range=params.lorentz_center_range
        )

        reflection = calculate_reflection(thicknesses, refractive_indices, params, device)

        absorption = (1 - reflection).float()
        real_samples = real_samples.float()

        d_real = discriminator(real_samples)
        d_fake = discriminator(absorption)

        d_real_probs = torch.sigmoid(d_real)
        d_fake_probs = torch.sigmoid(d_fake)

        save_gan_samples(
            wavelengths.cpu(),
            real_samples,
            absorption,
            d_real_probs,
            d_fake_probs,
            samples_dir,
            epoch,
        )

        data_dir = os.path.join(samples_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        absorption_data = pd.DataFrame()
        absorption_data['Wavelength (μm)'] = wavelengths.cpu().numpy()

        for i in range(4):
            absorption_data[f'Sample_{i+1}_Absorption'] = absorption[i].cpu().numpy()

        excel_path = os.path.join(data_dir, f'absorption_epoch_{epoch}.xlsx')
        absorption_data.to_excel(excel_path, index=False)
        print(f"Absorption data saved to: {excel_path}")

        for i in range(4):
            structure_path = os.path.join(data_dir, f'structure_sample_{i+1}_epoch_{epoch}.txt')

            with open(structure_path, 'w') as f:
                f.write(f"Structure Information for Sample {i+1} at Epoch {epoch}\n")
                f.write("=" * 60 + "\n\n")

                f.write("Layer Thickness (μm):\n")
                thickness_values = thicknesses[i].cpu().numpy()
                for j, thickness in enumerate(thickness_values):
                    f.write(f"Layer {j+1}: {thickness:.6f}\n")

                f.write("\n" + "-" * 40 + "\n\n")

                f.write("Material Probabilities:\n")
                material_probs = P[i].cpu().numpy()

                f.write(f"{'Layer':<10}")
                for mat_idx, mat_name in enumerate(params.materials):
                    f.write(f"{mat_name:<15}")
                f.write("\n")

                for j, layer_probs in enumerate(material_probs):
                    f.write(f"{j+1:<10}")
                    for prob in layer_probs:
                        f.write(f"{prob:.6f}{'':<8}")
                    f.write("\n")

                f.write("\n" + "-" * 40 + "\n\n")
                f.write("Dominant Material for Each Layer:\n")
                for j, layer_probs in enumerate(material_probs):
                    dominant_idx = np.argmax(layer_probs)
                    dominant_prob = layer_probs[dominant_idx]
                    dominant_material = params.materials[dominant_idx]
                    f.write(
                        f"Layer {j+1}: {dominant_material} "
                        f"(Probability: {dominant_prob:.6f})\n"
                    )

            print(f"Structure information saved to: {structure_path}")

        # 保存材料概率分析（使用更大的batch来获得更稳定的统计）
        analysis_noise_t = torch.randn(64, params.thickness_noise_dim, device=device)
        analysis_noise_m = torch.randn(64, params.material_noise_dim, device=device)
        _, _, P_analysis = generator(analysis_noise_t, analysis_noise_m, alpha)
        save_material_probability_analysis(P_analysis, alpha, epoch, samples_dir, params.materials)

    generator.train()
    discriminator.train()
