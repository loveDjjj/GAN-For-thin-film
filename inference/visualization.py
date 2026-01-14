import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np


def visualize_best_samples(wavelengths, absorption_spectra, best_indices, best_rmse, target):
    """Visualize the best samples compared to target."""
    num_samples = len(best_indices)

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        sample_idx = best_indices[i]
        rmse = best_rmse[i]

        axes[i].plot(wavelengths, absorption_spectra[sample_idx], 'b-', label='Generated')
        axes[i].plot(wavelengths, target, 'r--', label='Target Lorentzian')

        axes[i].set_title(f'Best Sample {i+1} (Index: {sample_idx}, RMSE: {rmse:.6f})')
        axes[i].set_xlabel('Wavelength (μm)')
        axes[i].set_ylabel('Absorption')
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    return fig


def save_best_results(output_dir, wavelengths, thicknesses, P, absorption_spectra,
                      best_indices, best_rmse, target, params):
    """Save best samples to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"best_samples_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    target_data = pd.DataFrame()
    target_data['Wavelength (μm)'] = wavelengths
    target_data['Target Lorentzian'] = target
    target_data.to_excel(os.path.join(save_dir, 'target_lorentzian.xlsx'), index=False)

    for i, idx in enumerate(best_indices):
        absorption_data = pd.DataFrame()
        absorption_data['Wavelength (μm)'] = wavelengths
        absorption_data['Absorption'] = absorption_spectra[idx]
        absorption_data['Target Lorentzian'] = target

        excel_path = os.path.join(save_dir, f'best_sample_{i+1}_absorption.xlsx')
        absorption_data.to_excel(excel_path, index=False)

        structure_path = os.path.join(save_dir, f'best_sample_{i+1}_structure.txt')

        with open(structure_path, 'w') as f:
            f.write(f"Structure Information for Best Sample {i+1} (Original Index: {idx})\n")
            f.write(f"RMSE with Target: {best_rmse[i]:.6f}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Layer Thickness (μm):\n")
            thickness_values = thicknesses[idx].cpu().numpy()
            for j, thickness in enumerate(thickness_values):
                f.write(f"Layer {j+1}: {thickness:.6f}\n")

            f.write("\n" + "-" * 40 + "\n\n")

            f.write("Material Probabilities:\n")
            material_probs = P[idx].cpu().numpy()

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
                dominant_idx = int(np.argmax(layer_probs))
                dominant_prob = layer_probs[dominant_idx]
                dominant_material = params.materials[dominant_idx]
                f.write(
                    f"Layer {j+1}: {dominant_material} "
                    f"(Probability: {dominant_prob:.6f})\n"
                )

    fig = visualize_best_samples(wavelengths, absorption_spectra, best_indices, best_rmse, target)
    plt.savefig(os.path.join(save_dir, 'best_samples_comparison.png'), dpi=300)
    plt.close(fig)

    print(f"Best samples saved to {save_dir}")
    return save_dir


def plot_pareto_front(weighted_rmse, total_thickness, pareto_indices):
    """Plot Pareto front for (RMSE, total_thickness)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(weighted_rmse, total_thickness, s=10, c='gray', alpha=0.6, label='Samples')

    pareto_rmse = weighted_rmse[pareto_indices]
    pareto_thick = total_thickness[pareto_indices]
    ax.scatter(pareto_rmse, pareto_thick, c='red', s=20, label='Pareto Front')

    ax.set_xlabel('Weighted RMSE')
    ax.set_ylabel('Total Thickness (μm)')
    ax.set_title('Pareto Front (RMSE vs Total Thickness)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def save_pareto_results(save_dir, weighted_rmse, total_thickness, pareto_indices):
    """Save Pareto front data and plot."""
    os.makedirs(save_dir, exist_ok=True)
    data = pd.DataFrame({
        'weighted_rmse': weighted_rmse,
        'total_thickness': total_thickness,
    })
    data['is_pareto'] = False
    data.loc[pareto_indices, 'is_pareto'] = True
    data.to_excel(os.path.join(save_dir, 'pareto_front_data.xlsx'), index=False)

    fig = plot_pareto_front(weighted_rmse, total_thickness, pareto_indices)
    fig.savefig(os.path.join(save_dir, 'pareto_front.png'), dpi=300)
    plt.close(fig)
