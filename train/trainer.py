import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.initialize import initialize_models
from model.Lorentzian.lorentzian_curves import generate_lorentzian_curves
from model.net import Discriminator, Generator
from model.TMM.optical_calculator import calculate_reflection
from train.q_evaluator import evaluate_generator_q, save_q_evaluation_history
from train.sample_saver import calculate_entropy, save_sample
from utils.visualize import (
    save_alpha_entropy_curves,
    save_distribution_evolution_plots,
    save_gan_training_curves,
    save_thickness_merged_layers_curves,
)


torch.serialization.add_safe_globals([Generator, Discriminator])


def configure_numerics():
    """Stabilize numerics across GPUs."""
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
    """Add random noise to a tensor."""
    return x + noise_level * torch.randn_like(x)


def calculate_merged_layers(material_probs):
    """Count merged layers after combining adjacent identical materials."""
    batch_size, num_layers, _ = material_probs.shape
    material_indices = torch.argmax(material_probs, dim=2)
    merged_counts = torch.zeros(batch_size, device=material_probs.device)

    for batch_index in range(batch_size):
        count = 1
        for layer_index in range(1, num_layers):
            if material_indices[batch_index, layer_index] != material_indices[batch_index, layer_index - 1]:
                count += 1
        merged_counts[batch_index] = count

    return merged_counts


def calculate_mean_thickness(thicknesses):
    """Compute the mean layer thickness."""
    return thicknesses.mean()


def collect_thickness_distribution(thicknesses, num_bins=20, thickness_range=(0.05, 0.5)):
    """Collect histogram statistics for generated thicknesses."""
    thicknesses_np = thicknesses.detach().cpu().numpy().flatten()
    hist_counts, bin_edges = np.histogram(thicknesses_np, bins=num_bins, range=thickness_range)
    return hist_counts, bin_edges


def collect_merged_layers_distribution(merged_counts, max_layers=10):
    """Collect distribution statistics for merged layer counts."""
    merged_counts_np = merged_counts.detach().cpu().numpy()
    layer_counts = np.zeros(max_layers, dtype=int)
    for layer_count in range(1, max_layers + 1):
        layer_counts[layer_count - 1] = np.sum(merged_counts_np == layer_count)
    return layer_counts


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Compute gradient penalty for regularizing the discriminator."""
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones(real_samples.size(0), 1, device=real_samples.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


def train_gan(config_path, output_dir, device=None, load_parameters=None, setup_directories=None):
    """Train the GAN model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    wavelengths = 2 * np.pi / params.k.to(device)

    g_losses = []
    d_losses = []
    gp_losses = []
    d_real_scores = []
    d_fake_scores = []
    alpha_history = []
    entropy_history = []
    mean_thickness_history = []
    merged_layers_history = []
    q_evaluation_history = []

    thickness_distribution_history = []
    merged_layers_distribution_history = []
    q_evaluation_dir = os.path.join(run_dir, "q_evaluation")
    distribution_save_interval = max(
        1,
        int(getattr(params, "distribution_epoch_interval", max(1, params.epochs // 10))),
    )
    thickness_histogram_bins = max(1, int(getattr(params, "thickness_histogram_bins", 20)))
    heatmap_epoch_tick_step = max(
        1,
        int(getattr(params, "heatmap_epoch_tick_step", distribution_save_interval)),
    )
    q_eval_interval = max(0, int(getattr(params, "q_eval_interval", 0)))

    print(f"Starting training for {params.epochs} epochs...")
    print(f"Alpha schedule: {params.alpha_min} -> {params.alpha_max}")
    if q_eval_interval > 0:
        print(
            "Q/MSE evaluation enabled: "
            f"every {q_eval_interval} epochs, {params.q_eval_num_samples} generated samples per evaluation"
        )
    progress_bar = tqdm(range(params.epochs), desc="Training progress")

    for epoch in progress_bar:
        generator.train()
        discriminator.train()

        alpha = params.alpha_min + (params.alpha_max - params.alpha_min) * (epoch / max(params.epochs - 1, 1))

        d_loss = None
        g_loss = None
        d_real = None
        d_fake = None

        for _ in range(max(1, params.d_steps)):
            thickness_noise = torch.randn(params.batch_size, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(params.batch_size, params.material_noise_dim, device=device)

            real_absorption = generate_lorentzian_curves(
                wavelengths,
                batch_size=params.batch_size,
                width=params.lorentz_width,
                center_range=params.lorentz_center_range,
            ).float()

            d_optimizer.zero_grad()

            thicknesses, refractive_indices, _ = generator(thickness_noise, material_noise, alpha)
            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
            if not torch.isfinite(reflection).all():
                print("[NaNGuard] reflection non-finite; skip this discriminator step")
                continue

            fake_absorption = (1 - reflection).float()

            noisy_real = add_noise(real_absorption, params.noise_level)
            noisy_fake = add_noise(fake_absorption.detach(), params.noise_level)

            d_real = discriminator(noisy_real)
            d_fake = discriminator(noisy_fake)

            d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
            d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            gradient_penalty = compute_gradient_penalty(discriminator, real_absorption, fake_absorption.detach())
            d_loss = d_loss_real + d_loss_fake + params.lambda_gp * gradient_penalty

            if not torch.isfinite(d_loss):
                print("[NaNGuard] d_loss is non-finite; skip this discriminator step")
                continue

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            d_optimizer.step()

            gp_losses.append(params.lambda_gp * gradient_penalty.item())

        for _ in range(max(1, params.g_steps)):
            thickness_noise = torch.randn(params.batch_size, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(params.batch_size, params.material_noise_dim, device=device)

            g_optimizer.zero_grad()

            thicknesses, refractive_indices, _ = generator(thickness_noise, material_noise, alpha)
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
            g_loss = torch.tensor(0.0, device=device)
        if d_loss is None:
            d_loss = torch.tensor(0.0, device=device)
        if d_real is None:
            d_real = torch.zeros(1, device=device)
        if d_fake is None:
            d_fake = torch.zeros(1, device=device)

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        d_real_scores.append(torch.sigmoid(d_real.mean()).item())
        d_fake_scores.append(torch.sigmoid(d_fake.mean()).item())

        alpha_history.append(alpha)
        current_epoch = epoch + 1
        with torch.no_grad():
            thickness_noise = torch.randn(params.batch_size, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(params.batch_size, params.material_noise_dim, device=device)
            thicknesses, _, current_probabilities = generator(thickness_noise, material_noise, alpha)

            entropy = calculate_entropy(current_probabilities)
            mean_entropy = entropy.mean().item()
            entropy_history.append(mean_entropy)

            mean_thickness = calculate_mean_thickness(thicknesses).item()
            mean_thickness_history.append(mean_thickness)

            merged_counts = calculate_merged_layers(current_probabilities)
            mean_merged_layers = merged_counts.mean().item()
            merged_layers_history.append(mean_merged_layers)

            if current_epoch % distribution_save_interval == 0 or current_epoch == params.epochs:
                thickness_hist_counts, thickness_bin_edges = collect_thickness_distribution(
                    thicknesses,
                    num_bins=thickness_histogram_bins,
                    thickness_range=(params.thickness_bot, params.thickness_sup),
                )
                thickness_distribution_history.append(
                    {
                        "epoch": current_epoch,
                        "hist_counts": thickness_hist_counts.tolist(),
                        "bin_edges": thickness_bin_edges.tolist(),
                        "mean_thickness": mean_thickness,
                    }
                )

                merged_layers_counts = collect_merged_layers_distribution(
                    merged_counts,
                    max_layers=params.N_layers,
                )
                merged_layers_distribution_history.append(
                    {
                        "epoch": current_epoch,
                        "layer_counts": merged_layers_counts.tolist(),
                        "mean_merged_layers": mean_merged_layers,
                    }
                )

        latest_mean_q = q_evaluation_history[-1]["mean_q"] if q_evaluation_history else 0.0
        latest_mean_mse = q_evaluation_history[-1]["mean_mse"] if q_evaluation_history else 0.0
        if q_eval_interval > 0 and current_epoch % q_eval_interval == 0:
            q_summary = evaluate_generator_q(
                generator,
                params,
                device,
                alpha,
                epoch=current_epoch,
                save_dir=q_evaluation_dir,
            )
            q_evaluation_history.append(q_summary)
            save_q_evaluation_history(q_evaluation_history, q_evaluation_dir)
            latest_mean_q = q_summary["mean_q"]
            latest_mean_mse = q_summary["mean_mse"]
            print(
                f"[QEval] epoch={current_epoch} mean_q={q_summary['mean_q']:.4f} "
                f"mean_mse={q_summary['mean_mse']:.6f} valid_ratio={q_summary['valid_ratio'] * 100:.2f}%"
            )

        gp_last = gp_losses[-1] if gp_losses else 0.0
        progress_bar.set_postfix(
            {
                "G_Loss": f"{g_loss.item():.4f}",
                "D_Loss": f"{d_loss.item():.4f}",
                "GP": f"{gp_last:.4f}",
                "D(real)": f"{torch.sigmoid(d_real.mean()).item():.2f}",
                "D(fake)": f"{torch.sigmoid(d_fake.mean()).item():.2f}",
                "Thick": f"{mean_thickness:.3f}",
                "Layers": f"{mean_merged_layers:.1f}",
                "AvgQ": f"{latest_mean_q:.2f}" if q_evaluation_history else "N/A",
                "AvgMSE": f"{latest_mean_mse:.4e}" if q_evaluation_history else "N/A",
            }
        )

        if (epoch + 1) % params.save_interval == 0:
            save_sample(generator, discriminator, params, epoch + 1, samples_dir, device, alpha)
            torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_epoch_{epoch + 1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_epoch_{epoch + 1}.pth"))

    torch.save(generator.state_dict(), os.path.join(model_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, "discriminator_final.pth"))

    save_gan_training_curves(
        g_losses,
        d_losses,
        d_real_scores,
        d_fake_scores,
        os.path.join(run_dir, "training_metrics.png"),
    )

    save_alpha_entropy_curves(
        alpha_history,
        entropy_history,
        os.path.join(run_dir, "alpha_entropy_curves.png"),
        num_materials=len(getattr(params, "materials", [])),
    )

    save_thickness_merged_layers_curves(
        mean_thickness_history,
        merged_layers_history,
        os.path.join(run_dir, "thickness_merged_layers_curves.png"),
        thickness_range=(params.thickness_bot, params.thickness_sup),
        max_layers=params.N_layers,
    )

    if thickness_distribution_history and merged_layers_distribution_history:
        save_distribution_evolution_plots(
            thickness_distribution_history,
            merged_layers_distribution_history,
            run_dir,
            thickness_range=(params.thickness_bot, params.thickness_sup),
            max_layers=params.N_layers,
            thickness_bins=thickness_histogram_bins,
            heatmap_epoch_tick_step=heatmap_epoch_tick_step,
        )
    else:
        print("Warning: No distribution data collected during training")

    print(f"Training complete! Results saved to: {run_dir}")
    return generator, discriminator
