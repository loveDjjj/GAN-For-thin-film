import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model.net import Generator, Discriminator
from model.initialize import initialize_models
from model.Lorentzian.lorentzian_curves import generate_lorentzian_curves
from model.TMM.optical_calculator import calculate_reflection
from utils.visualize import save_gan_training_curves
from train.sample_saver import save_sample


torch.serialization.add_safe_globals([Generator, Discriminator])


def add_noise(x, noise_level=0.05):
    """Add random noise to tensor."""
    return x + noise_level * torch.randn_like(x)


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

    alpha = params.alpha

    print(f"Starting training for {params.epochs} epochs...")
    progress_bar = tqdm(range(params.epochs), desc="Training progress")

    for epoch in progress_bar:
        generator.train()
        discriminator.train()

        alpha = round(epoch / params.epochs * params.alpha_sup) + 20

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

            thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, alpha)

            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
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

            d_loss.backward()
            d_optimizer.step()

            gp_value = params.lambda_gp * gradient_penalty.item()
            gp_losses.append(gp_value)

        for _ in range(max(1, params.g_steps)):
            thickness_noise = torch.randn(params.batch_size, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(params.batch_size, params.material_noise_dim, device=device)

            g_optimizer.zero_grad()

            thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, alpha)
            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
            fake_absorption = (1 - reflection).float()

            noisy_fake = add_noise(fake_absorption, params.noise_level)

            d_fake = discriminator(noisy_fake)

            g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

            g_loss.backward()
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

        progress_bar.set_postfix({
            "G_Loss": f"{g_loss.item():.4f}",
            "D_Loss": f"{d_loss.item():.4f}",
            "GP": f"{gp_losses[-1]:.4f}",
            "D(real)": f"{torch.sigmoid(d_real.mean()).item():.2f}",
            "D(fake)": f"{torch.sigmoid(d_fake.mean()).item():.2f}",
        })

        if (epoch + 1) % params.save_interval == 0:
            save_sample(generator, discriminator, params, epoch + 1, samples_dir, device)

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

    print(f"Training complete! Results saved to: {run_dir}")
    return generator, discriminator
