import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from model.TMM.optical_calculator import calculate_reflection
from utils.visualize import save_gan_samples


def calculate_entropy(probs):
    """Calculate entropy for a probability distribution."""
    probs_clamped = torch.clamp(probs, min=1e-10)
    return -torch.sum(probs_clamped * torch.log(probs_clamped), dim=-1)


def _material_colors(num_materials):
    base_colors = list(plt.get_cmap("tab20").colors)
    if num_materials <= len(base_colors):
        return base_colors[:num_materials]

    extended = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        extended.extend(list(plt.get_cmap(cmap_name).colors))
        if num_materials <= len(extended):
            return extended[:num_materials]

    dynamic_cmap = plt.get_cmap("hsv", num_materials)
    return [dynamic_cmap(index) for index in range(num_materials)]


def _build_merged_layers(thickness_values, material_probs, materials):
    material_indices = np.argmax(material_probs, axis=1)
    merged_layers = []

    current_material = int(material_indices[0])
    start_layer = 0
    merged_thickness = float(thickness_values[0])
    dominant_probs = [float(material_probs[0, current_material])]

    for layer_index in range(1, len(thickness_values)):
        material_index = int(material_indices[layer_index])
        if material_index == current_material:
            merged_thickness += float(thickness_values[layer_index])
            dominant_probs.append(float(material_probs[layer_index, material_index]))
            continue

        merged_layers.append(
            {
                "merged_layer_index": len(merged_layers) + 1,
                "material_index": current_material,
                "material": materials[current_material],
                "start_layer": start_layer + 1,
                "end_layer": layer_index,
                "original_layer_count": layer_index - start_layer,
                "merged_thickness_um": merged_thickness,
                "mean_dominant_probability": float(np.mean(dominant_probs)),
            }
        )

        current_material = material_index
        start_layer = layer_index
        merged_thickness = float(thickness_values[layer_index])
        dominant_probs = [float(material_probs[layer_index, material_index])]

    merged_layers.append(
        {
            "merged_layer_index": len(merged_layers) + 1,
            "material_index": current_material,
            "material": materials[current_material],
            "start_layer": start_layer + 1,
            "end_layer": len(thickness_values),
            "original_layer_count": len(thickness_values) - start_layer,
            "merged_thickness_um": merged_thickness,
            "mean_dominant_probability": float(np.mean(dominant_probs)),
        }
    )

    return merged_layers


def save_material_probability_analysis(P, alpha, epoch, samples_dir, materials):
    """Save material probability analysis plots and CSV files."""
    data_dir = os.path.join(samples_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    P_np = P.detach().cpu().numpy()
    _, n_layers, n_materials = P_np.shape

    layer_labels = [f"L{index + 1}" for index in range(n_layers)]
    colors = _material_colors(n_materials)

    P_mean = P_np.mean(axis=0)
    P_std = P_np.std(axis=0)

    entropy = calculate_entropy(P).detach().cpu().numpy()
    entropy_mean = entropy.mean(axis=0)

    dominant_materials = np.argmax(P_mean, axis=1)
    dominant_probs = np.max(P_mean, axis=1)

    summary_rows = []
    layer_material_rows = []
    for layer_index in range(n_layers):
        row = {
            "layer_index": layer_index + 1,
            "layer_label": layer_labels[layer_index],
            "entropy_mean": float(entropy_mean[layer_index]),
            "dominant_material": materials[int(dominant_materials[layer_index])],
            "dominant_probability": float(dominant_probs[layer_index]),
        }
        for material_index, material_name in enumerate(materials):
            mean_value = float(P_mean[layer_index, material_index])
            std_value = float(P_std[layer_index, material_index])
            row[f"{material_name}_mean"] = mean_value
            row[f"{material_name}_std"] = std_value
            layer_material_rows.append(
                {
                    "layer_index": layer_index + 1,
                    "layer_label": layer_labels[layer_index],
                    "material_index": material_index + 1,
                    "material": material_name,
                    "mean_probability": mean_value,
                    "std_probability": std_value,
                }
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(data_dir, f"material_probability_epoch_{epoch}_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    layer_material_df = pd.DataFrame(layer_material_rows)
    layer_material_csv_path = os.path.join(data_dir, f"material_probability_epoch_{epoch}_layer_materials.csv")
    layer_material_df.to_csv(layer_material_csv_path, index=False)

    figure_width = max(14, 6 + 0.45 * n_layers)
    figure_height = max(10, 8 + 0.25 * n_materials)
    fig, axes = plt.subplots(2, 2, figsize=(figure_width, figure_height))
    fig.suptitle(f"Material Selection Analysis (Epoch {epoch}, alpha={alpha:.3f})", fontsize=14)

    ax1 = axes[0, 0]
    heatmap = ax1.imshow(P_mean.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Material")
    ax1.set_yticks(range(n_materials))
    ax1.set_yticklabels(materials)
    ax1.set_xticks(range(n_layers))
    ax1.set_xticklabels(layer_labels)
    ax1.set_title("Mean Material Probability per Layer")
    plt.colorbar(heatmap, ax=ax1, label="Probability")

    if n_layers * n_materials <= 80:
        for layer_index in range(n_layers):
            for material_index in range(n_materials):
                ax1.text(
                    layer_index,
                    material_index,
                    f"{P_mean[layer_index, material_index]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    ax2 = axes[0, 1]
    x = np.arange(n_layers)
    cumulative = np.zeros(n_layers)
    for material_index, material_name in enumerate(materials):
        ax2.bar(
            x,
            P_mean[:, material_index],
            bottom=cumulative,
            color=colors[material_index],
            label=material_name,
            alpha=0.9,
        )
        cumulative += P_mean[:, material_index]
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Probability")
    ax2.set_title("Mean Material Probability Composition per Layer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    ax3 = axes[1, 0]
    max_entropy = float(np.log(max(1, n_materials)))
    ax3.bar(range(n_layers), entropy_mean, color="steelblue", alpha=0.8)
    if n_materials > 1:
        ax3.axhline(y=max_entropy, color="red", linestyle="--", label=f"Max Entropy = {max_entropy:.3f}")
    ax3.axhline(y=0, color="green", linestyle="--", label="Min Entropy = 0")
    ax3.set_xlabel("Layer Index")
    ax3.set_ylabel("Entropy")
    ax3.set_title("Selection Entropy per Layer")
    ax3.set_xticks(range(n_layers))
    ax3.set_xticklabels(layer_labels)
    ax3.legend(fontsize=8)
    entropy_upper = max(
        0.1,
        max_entropy * 1.2,
        float(np.max(entropy_mean)) * 1.1 if len(entropy_mean) else 0.0,
    )
    ax3.set_ylim(0, entropy_upper)

    ax4 = axes[1, 1]
    dominant_colors = [colors[index] for index in dominant_materials]
    ax4.bar(range(n_layers), dominant_probs, color=dominant_colors, alpha=0.85)
    ax4.set_xlabel("Layer Index")
    ax4.set_ylabel("Dominant Material Probability")
    ax4.set_title("Dominant Material Confidence per Layer")
    ax4.set_ylim(0, 1.02)
    ax4.set_xticks(range(n_layers))
    if n_layers <= 12:
        dominant_labels = [
            f"{layer_labels[layer_index]}\n({materials[int(dominant_materials[layer_index])]})"
            for layer_index in range(n_layers)
        ]
        ax4.set_xticklabels(dominant_labels, fontsize=8)
    else:
        ax4.set_xticklabels(layer_labels)
    ax4.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    legend_handles = [Patch(facecolor=colors[index], label=materials[index]) for index in range(n_materials)]
    ax4.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        ncol=1 if n_materials > 6 else min(3, n_materials),
    )

    plt.tight_layout(rect=[0, 0, 0.86, 0.97])

    fig_path = os.path.join(data_dir, f"material_probability_epoch_{epoch}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Material probability analysis saved to: {fig_path}")
    print(f"Material probability summary CSV saved to: {summary_csv_path}")
    print(f"Material probability layer CSV saved to: {layer_material_csv_path}")

    return {
        "alpha": float(alpha),
        "epoch": int(epoch),
        "mean_entropy": float(entropy_mean.mean()),
        "P_mean": P_mean,
    }


def save_sample(generator, discriminator, params, epoch, samples_dir, device, alpha=None):
    """Save generated sample images and additional data."""
    if alpha is None:
        alpha = getattr(params, "alpha_max", getattr(params, "alpha", 20))

    checkpoint_sample_count = max(1, int(getattr(params, "checkpoint_sample_count", 8)))
    sample_export_count = max(1, int(getattr(params, "sample_export_count", 4)))
    sample_export_count = min(sample_export_count, checkpoint_sample_count)
    material_analysis_batch_size = max(1, int(getattr(params, "material_analysis_batch_size", 64)))

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        thickness_noise = torch.randn(checkpoint_sample_count, params.thickness_noise_dim, device=device)
        material_noise = torch.randn(checkpoint_sample_count, params.material_noise_dim, device=device)

        thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, alpha)

        wavelengths = 2 * np.pi / params.k.to(device)
        real_samples = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            batch_size=checkpoint_sample_count,
            width=params.lorentz_width,
            center_range_1=params.lorentz_center_range_1,
            center_range_2=params.lorentz_center_range_2,
            min_peak_spacing=getattr(params, "min_peak_spacing", 0.0),
            max_peak_spacing=getattr(params, "max_peak_spacing", None),
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
            num_samples=sample_export_count,
        )

        data_dir = os.path.join(samples_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        absorption_data = pd.DataFrame()
        absorption_data["Wavelength (um)"] = wavelengths.cpu().numpy()

        for sample_index in range(sample_export_count):
            absorption_data[f"Sample_{sample_index + 1}_Absorption"] = absorption[sample_index].cpu().numpy()

        excel_path = os.path.join(data_dir, f"absorption_epoch_{epoch}.xlsx")
        absorption_data.to_excel(excel_path, index=False)
        print(f"Absorption data saved to: {excel_path}")

        for sample_index in range(sample_export_count):
            structure_path = os.path.join(data_dir, f"structure_sample_{sample_index + 1}_epoch_{epoch}.txt")
            thickness_values = thicknesses[sample_index].detach().cpu().numpy()
            material_probs = P[sample_index].detach().cpu().numpy()
            total_thickness = float(np.sum(thickness_values))
            merged_layers = _build_merged_layers(thickness_values, material_probs, params.materials)

            with open(structure_path, "w", encoding="utf-8") as file:
                file.write(f"Structure Information for Sample {sample_index + 1} at Epoch {epoch}\n")
                file.write("=" * 72 + "\n\n")

                file.write("Structure Summary:\n")
                file.write(f"Total Thickness (um): {total_thickness:.6f}\n")
                file.write(f"Original Layer Count: {len(thickness_values)}\n")
                file.write(f"Merged Layer Count: {len(merged_layers)}\n\n")

                file.write("Merged Layers After Combining Adjacent Identical Materials:\n")
                for merged_layer in merged_layers:
                    file.write(
                        f"Merged Layer {merged_layer['merged_layer_index']}: "
                        f"Material={merged_layer['material']}, "
                        f"Original Layers={merged_layer['start_layer']}-{merged_layer['end_layer']}, "
                        f"Thickness={merged_layer['merged_thickness_um']:.6f} um, "
                        f"Mean Dominant Probability={merged_layer['mean_dominant_probability']:.6f}\n"
                    )

                file.write("\n" + "-" * 48 + "\n\n")
                file.write("Original Layer Thickness (um):\n")
                for layer_index, thickness in enumerate(thickness_values, start=1):
                    file.write(f"Layer {layer_index}: {thickness:.6f}\n")

                file.write("\n" + "-" * 48 + "\n\n")
                file.write("Material Probabilities:\n")
                file.write(f"{'Layer':<10}")
                for material_name in params.materials:
                    file.write(f"{material_name:<18}")
                file.write("\n")

                for layer_index, layer_probs in enumerate(material_probs, start=1):
                    file.write(f"{layer_index:<10}")
                    for probability in layer_probs:
                        file.write(f"{probability:<18.6f}")
                    file.write("\n")

                file.write("\n" + "-" * 48 + "\n\n")
                file.write("Dominant Material for Each Layer:\n")
                for layer_index, layer_probs in enumerate(material_probs, start=1):
                    dominant_index = int(np.argmax(layer_probs))
                    dominant_probability = float(layer_probs[dominant_index])
                    dominant_material = params.materials[dominant_index]
                    file.write(
                        f"Layer {layer_index}: {dominant_material} "
                        f"(Probability: {dominant_probability:.6f})\n"
                    )

            print(f"Structure information saved to: {structure_path}")

        analysis_noise_t = torch.randn(material_analysis_batch_size, params.thickness_noise_dim, device=device)
        analysis_noise_m = torch.randn(material_analysis_batch_size, params.material_noise_dim, device=device)
        _, _, P_analysis = generator(analysis_noise_t, analysis_noise_m, alpha)
        save_material_probability_analysis(P_analysis, alpha, epoch, samples_dir, params.materials)

    generator.train()
    discriminator.train()
