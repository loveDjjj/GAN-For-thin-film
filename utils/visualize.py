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