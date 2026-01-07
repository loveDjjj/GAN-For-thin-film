import os

import torch
import numpy as np
import pandas as pd

from model.Lorentzian.lorentzian_curves import generate_lorentzian_curves
from model.TMM.optical_calculator import calculate_reflection
from utils.visualize import save_gan_samples


def save_sample(generator, discriminator, params, epoch, samples_dir, device):
    """Save generated sample images and additional data."""
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        thickness_noise = torch.randn(8, params.thickness_noise_dim, device=device)
        material_noise = torch.randn(8, params.material_noise_dim, device=device)

        thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, params.alpha)

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

    generator.train()
    discriminator.train()
