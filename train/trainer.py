import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model.net import Generator, Discriminator
from model.initialize import initialize_models
from model.Lorentzian.lorentzian_curves import generate_lorentzian_curves
from model.TMM.optical_calculator import calculate_reflection
from utils.visualize import save_gan_training_curves, save_alpha_entropy_curves, save_thickness_merged_layers_curves, save_distribution_evolution_plots
from train.sample_saver import save_sample, calculate_entropy


torch.serialization.add_safe_globals([Generator, Discriminator])


def configure_numerics():
    """Stabilize numerics across GPUs (disable TF32, prefer deterministic algorithms)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def add_noise(x, noise_level=0.05):
    """Add random noise to tensor."""
    return x + noise_level * torch.randn_like(x)


def calculate_merged_layers(material_probs):
    """
    计算合并后的层数（相邻相同材料层合并为一层）

    Args:
        material_probs: 材料选择概率张量 [batch_size, N_layers, M_materials]

    Returns:
        merged_counts: 每个样本的合并层数 [batch_size]
    """
    batch_size, N_layers, _ = material_probs.shape

    # 获取每层的材料索引（概率最大的材料）
    material_indices = torch.argmax(material_probs, dim=2)  # [batch_size, N_layers]

    merged_counts = torch.zeros(batch_size, device=material_probs.device)

    for i in range(batch_size):
        count = 1  # 至少有一层
        for j in range(1, N_layers):
            if material_indices[i, j] != material_indices[i, j-1]:
                count += 1
        merged_counts[i] = count

    return merged_counts


def calculate_mean_thickness(thicknesses):
    """
    计算平均厚度

    Args:
        thicknesses: 厚度张量 [batch_size, N_layers]

    Returns:
        mean_thickness: 所有样本所有层的平均厚度（标量）
    """
    return thicknesses.mean()


def collect_thickness_distribution(thicknesses, num_bins=20, thickness_range=(0.05, 0.5)):
    """
    收集厚度分布统计

    Args:
        thicknesses: 厚度张量 [batch_size, N_layers]
        num_bins: 直方图箱数
        thickness_range: 厚度范围 (min, max)

    Returns:
        hist_counts: 直方图计数 [num_bins]
        bin_edges: 箱边界 [num_bins+1]
    """
    thicknesses_np = thicknesses.cpu().numpy().flatten()
    hist_counts, bin_edges = np.histogram(thicknesses_np, bins=num_bins, range=thickness_range)
    return hist_counts, bin_edges


def collect_merged_layers_distribution(merged_counts, max_layers=10):
    """
    收集合并层数分布统计

    Args:
        merged_counts: 合并层数张量 [batch_size]
        max_layers: 最大可能层数

    Returns:
        layer_counts: 各层数的样本计数 [max_layers]
    """
    merged_counts_np = merged_counts.cpu().numpy()
    layer_counts = np.zeros(max_layers, dtype=int)
    for i in range(1, max_layers + 1):
        layer_counts[i-1] = np.sum(merged_counts_np == i)
    return layer_counts


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Compute gradient penalty for regularizing discriminator."""
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)

    fake = torch.ones(real_samples.size(0), 1, device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_gan(config_path, output_dir, device=None, load_parameters=None, setup_directories=None):
    """Train GAN model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configure_numerics()
    print(f"Using device: {device}")

    if load_parameters is None or setup_directories is None:
        raise ValueError("load_parameters and setup_directories must be provided by train.py")

    run_dir, model_dir, samples_dir = setup_directories(output_dir)
    params = load_parameters(config_path, device)
    generator, discriminator = initialize_models(params, device)

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=params.lr_gen,
        betas=(params.beta1, params.beta2),
    )

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=params.lr_disc,
        betas=(params.beta1, params.beta2),
        weight_decay=params.weight_decay,
    )

    wavelengths = (2 * np.pi / params.k.to(device))

    g_losses = []
    d_losses = []
    gp_losses = []
    d_real_scores = []
    d_fake_scores = []
    alpha_history = []
    entropy_history = []
    mean_thickness_history = []
    merged_layers_history = []

    # 分布统计数据结构
    thickness_distribution_history = []  # 每个元素: (epoch, hist_counts, bin_edges)
    merged_layers_distribution_history = []  # 每个元素: (epoch, layer_counts)
    # 关键epoch的分布采样间隔
    distribution_save_interval = max(1, params.epochs // 10)  # 大约保存10个关键epoch的分布

    print(f"Starting training for {params.epochs} epochs...")
    print(f"Alpha schedule: {params.alpha_min} -> {params.alpha_max}")
    progress_bar = tqdm(range(params.epochs), desc="Training progress")

    for epoch in progress_bar:
        generator.train()
        discriminator.train()

        # alpha 从 alpha_min 线性增长到 alpha_max
        alpha = params.alpha_min + (params.alpha_max - params.alpha_min) * (epoch / max(params.epochs - 1, 1))

        d_loss = None
        g_loss = None
        d_real = None
        d_fake = None
        gradient_penalty = None

        for _ in range(max(1, params.d_steps)):
            thickness_noise = torch.randn(params.batch_size, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(params.batch_size, params.material_noise_dim, device=device)

            real_absorption = generate_lorentzian_curves(
                wavelengths,
                batch_size=params.batch_size,
                width=params.lorentz_width,
                center_range=params.lorentz_center_range,
            )

            d_optimizer.zero_grad()

            thicknesses, refractive_indices, P_gen = generator(thickness_noise, material_noise, alpha)

            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
            if not torch.isfinite(reflection).all():
                print("[NaNGuard] reflection non-finite; skip this discriminator step")
                continue
            fake_absorption = (1 - reflection).float()

            real_absorption = real_absorption.float()

            noisy_real = add_noise(real_absorption, params.noise_level)
            noisy_fake = add_noise(fake_absorption.detach(), params.noise_level)

            d_real = discriminator(noisy_real)
            d_fake = discriminator(noisy_fake)

            d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
            d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

            gradient_penalty = compute_gradient_penalty(
                discriminator, real_absorption, fake_absorption.detach()
            )

            d_loss = d_loss_real + d_loss_fake + params.lambda_gp * gradient_penalty

            if not torch.isfinite(d_loss):
                print("[NaNGuard] d_loss is non-finite; skip this discriminator step")
                continue

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            d_optimizer.step()

            gp_value = params.lambda_gp * gradient_penalty.item()
            gp_losses.append(gp_value)

        for _ in range(max(1, params.g_steps)):
            thickness_noise = torch.randn(params.batch_size, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(params.batch_size, params.material_noise_dim, device=device)

            g_optimizer.zero_grad()

            thicknesses, refractive_indices, P_g = generator(thickness_noise, material_noise, alpha)
            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
            if not torch.isfinite(reflection).all():
                print("[NaNGuard] reflection non-finite; skip this generator step")
                continue
            fake_absorption = (1 - reflection).float()

            noisy_fake = add_noise(fake_absorption, params.noise_level)

            d_fake = discriminator(noisy_fake)

            g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

            if not torch.isfinite(g_loss):
                print("[NaNGuard] g_loss is non-finite; skip this generator step")
                continue

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            g_optimizer.step()

        if g_loss is None:
            g_loss = torch.tensor(0.0)
        if d_loss is None:
            d_loss = torch.tensor(0.0)
        if d_real is None:
            d_real = torch.zeros(1, device=device)
        if d_fake is None:
            d_fake = torch.zeros(1, device=device)

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        d_real_scores.append(torch.sigmoid(d_real.mean()).item())
        d_fake_scores.append(torch.sigmoid(d_fake.mean()).item())

        # 跟踪 alpha 和材料选择熵
        alpha_history.append(alpha)
        with torch.no_grad():
            # 重新生成一批样本来计算跟踪指标
            thickness_noise = torch.randn(params.batch_size, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(params.batch_size, params.material_noise_dim, device=device)
            thicknesses, refractive_indices, P_current = generator(thickness_noise, material_noise, alpha)

            # 计算当前 epoch 的材料选择熵
            entropy = calculate_entropy(P_current)  # [batch_size, N_layers]
            mean_entropy = entropy.mean().item()
            entropy_history.append(mean_entropy)

            # 计算平均厚度
            mean_thickness = calculate_mean_thickness(thicknesses).item()
            mean_thickness_history.append(mean_thickness)

            # 计算合并层数
            merged_counts = calculate_merged_layers(P_current)
            mean_merged_layers = merged_counts.mean().item()
            merged_layers_history.append(mean_merged_layers)

            # 在关键epoch收集分布统计
            if epoch % distribution_save_interval == 0 or epoch == params.epochs - 1:
                # 收集厚度分布
                thickness_hist_counts, thickness_bin_edges = collect_thickness_distribution(
                    thicknesses, num_bins=20, thickness_range=(params.thickness_bot, params.thickness_sup))
                thickness_distribution_history.append({
                    'epoch': epoch,
                    'hist_counts': thickness_hist_counts.tolist(),
                    'bin_edges': thickness_bin_edges.tolist(),
                    'mean_thickness': mean_thickness
                })

                # 收集合并层数分布
                merged_layers_counts = collect_merged_layers_distribution(
                    merged_counts, max_layers=params.N_layers)
                merged_layers_distribution_history.append({
                    'epoch': epoch,
                    'layer_counts': merged_layers_counts.tolist(),
                    'mean_merged_layers': mean_merged_layers
                })

        gp_last = gp_losses[-1] if gp_losses else 0.0
        progress_bar.set_postfix({
            "G_Loss": f"{g_loss.item():.4f}",
            "D_Loss": f"{d_loss.item():.4f}",
            "GP": f"{gp_last:.4f}",
            "D(real)": f"{torch.sigmoid(d_real.mean()).item():.2f}",
            "D(fake)": f"{torch.sigmoid(d_fake.mean()).item():.2f}",
            "Thick": f"{mean_thickness:.3f}",
            "Layers": f"{mean_merged_layers:.1f}",
        })

        if (epoch + 1) % params.save_interval == 0:
            save_sample(generator, discriminator, params, epoch + 1, samples_dir, device, alpha)

            torch.save(generator.state_dict(), os.path.join(model_dir, f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f'discriminator_epoch_{epoch+1}.pth'))

    torch.save(generator.state_dict(), os.path.join(model_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, 'discriminator_final.pth'))

    save_gan_training_curves(
        g_losses,
        d_losses,
        d_real_scores,
        d_fake_scores,
        os.path.join(run_dir, 'training_metrics.png'),
    )

    # 保存 alpha 和熵的变化曲线
    save_alpha_entropy_curves(
        alpha_history,
        entropy_history,
        os.path.join(run_dir, 'alpha_entropy_curves.png'),
    )

    # 保存厚度和合并层数的变化曲线
    save_thickness_merged_layers_curves(
        mean_thickness_history,
        merged_layers_history,
        os.path.join(run_dir, 'thickness_merged_layers_curves.png'),
    )

    # 保存厚度和层数分布的演变图
    if thickness_distribution_history and merged_layers_distribution_history:
        save_distribution_evolution_plots(
            thickness_distribution_history,
            merged_layers_distribution_history,
            run_dir
        )
    else:
        print("Warning: No distribution data collected during training")

    print(f"Training complete! Results saved to: {run_dir}")
    return generator, discriminator
