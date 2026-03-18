"""Visualization helpers for training and inference."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_dataframe(dataframe, path):
    _ensure_parent(path)
    dataframe.to_csv(path, index=False)
    print(f"CSV data saved to: {path}")


def _choose_tick_positions(length, target_count=6):
    if length <= 0:
        return []
    if length <= target_count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, target_count, dtype=int).tolist()))


def _build_heatmap_epoch_ticks(epochs, tick_step):
    if not epochs:
        return [], []

    step = max(1, int(tick_step))
    positions = []
    for index, epoch in enumerate(epochs):
        if index == 0 or index == len(epochs) - 1 or epoch % step == 0:
            positions.append(index)

    positions = sorted(set(positions))
    labels = [str(epochs[index]) for index in positions]
    return positions, labels


def _select_snapshot_indices(num_items):
    if num_items <= 0:
        return []
    if num_items <= 3:
        return list(range(num_items))
    return sorted(set([0, num_items // 2, num_items - 1]))


def plot_gan_samples(wavelengths, real_samples, fake_samples, d_real, d_fake, num_samples=4):
    """Plot a comparison between generated and real samples."""
    wavelengths = _to_numpy(wavelengths)
    real_samples = _to_numpy(real_samples)
    fake_samples = _to_numpy(fake_samples)

    if torch.is_tensor(d_real):
        d_real = d_real.detach().cpu()
    if torch.is_tensor(d_fake):
        d_fake = d_fake.detach().cpu()

    fig = plt.figure(figsize=(12, 10))

    for index in range(min(num_samples, len(real_samples))):
        plt.subplot(num_samples, 2, 2 * index + 1)
        plt.plot(wavelengths, real_samples[index])
        plt.title(f"Real Sample {index + 1} - D(x): {d_real[index].item():.3f}")
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Absorption")
        plt.grid(True)

        plt.subplot(num_samples, 2, 2 * index + 2)
        plt.plot(wavelengths, fake_samples[index])
        plt.title(f"Generated Sample {index + 1} - D(G(z)): {d_fake[index].item():.3f}")
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Absorption")
        plt.grid(True)

    plt.tight_layout()
    return fig


def plot_gan_training_curves(g_losses, d_losses, d_real_scores, d_fake_scores):
    """Plot GAN training curves."""
    fig = plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Loss")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(d_real_scores, label="D(Real)")
    plt.plot(d_fake_scores, label="D(Generated)")
    plt.xlabel("Iterations")
    plt.ylabel("Discriminator Score")
    plt.legend()
    plt.title("Discriminator Scores")
    plt.grid(True)

    plt.tight_layout()
    return fig


def save_gan_samples(wavelengths, real_samples, fake_samples, d_real, d_fake, save_path, epoch, num_samples=4):
    """Save GAN sample comparison plots."""
    fig = plot_gan_samples(wavelengths, real_samples, fake_samples, d_real, d_fake, num_samples)
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, f"epoch_{epoch}.png")
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    print(f"Sample images saved to: {image_path}")


def save_gan_training_curves(g_losses, d_losses, d_real_scores, d_fake_scores, save_path):
    """Save GAN training curves."""
    fig = plot_gan_training_curves(g_losses, d_losses, d_real_scores, d_fake_scores)
    _ensure_parent(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


def save_alpha_entropy_curves(alpha_history, entropy_history, save_path, num_materials=2):
    """Save alpha and entropy curves during training."""
    if not alpha_history or not entropy_history:
        print("No alpha or entropy history to visualize")
        return

    epochs = np.arange(1, len(alpha_history) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    ax1.plot(epochs, alpha_history, "b-", linewidth=2, label="Alpha")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Alpha")
    ax1.set_title("Softmax Temperature During Training")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if len(alpha_history) > 1:
        ax1.annotate(
            f"Start: {alpha_history[0]:.3f}",
            xy=(1, alpha_history[0]),
            xytext=(max(1, len(epochs) * 0.1), alpha_history[0]),
            fontsize=9,
            color="blue",
        )
        ax1.annotate(
            f"End: {alpha_history[-1]:.3f}",
            xy=(len(epochs), alpha_history[-1]),
            xytext=(max(1, len(epochs) * 0.8), alpha_history[-1] * 1.05),
            fontsize=9,
            color="blue",
        )

    ax2 = axes[1]
    ax2.plot(epochs, entropy_history, "r-", linewidth=2, label="Mean Entropy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Entropy")
    ax2.set_title("Material Selection Entropy During Training")
    ax2.grid(True, alpha=0.3)

    max_entropy = float(np.log(max(1, int(num_materials))))
    if num_materials > 1:
        ax2.axhline(
            y=max_entropy,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Max Entropy (uniform) = {max_entropy:.3f}",
        )
    ax2.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Min Entropy = 0")
    ax2.fill_between(epochs, 0, entropy_history, alpha=0.3, color="red")
    ax2.legend(fontsize=9)

    plt.tight_layout()

    _ensure_parent(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Alpha-Entropy curves saved to: {save_path}")


def save_thickness_merged_layers_curves(
    mean_thickness_history,
    merged_layers_history,
    save_path,
    thickness_range=(0.05, 0.5),
    max_layers=10,
):
    """Save thickness and merged-layer curves during training."""
    if not mean_thickness_history or not merged_layers_history:
        print("No thickness or merged-layer history to visualize")
        return

    epochs = np.arange(1, len(mean_thickness_history) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    lower_bound, upper_bound = thickness_range

    ax1 = axes[0]
    ax1.plot(epochs, mean_thickness_history, "g-", linewidth=2, label="Average Thickness")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Thickness (um)")
    ax1.set_title("Average Generated Layer Thickness During Training")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=lower_bound, color="gray", linestyle="--", alpha=0.5, label=f"Lower Bound ({lower_bound:.3f} um)")
    ax1.axhline(y=upper_bound, color="gray", linestyle="--", alpha=0.5, label=f"Upper Bound ({upper_bound:.3f} um)")
    ax1.axhline(y=(lower_bound + upper_bound) / 2, color="gray", linestyle=":", alpha=0.3, label="Mid Point")
    ax1.legend(fontsize=9)

    if len(mean_thickness_history) > 1:
        ax1.annotate(
            f"Start: {mean_thickness_history[0]:.3f} um",
            xy=(1, mean_thickness_history[0]),
            xytext=(max(1, len(epochs) * 0.1), mean_thickness_history[0]),
            fontsize=9,
            color="green",
        )
        ax1.annotate(
            f"End: {mean_thickness_history[-1]:.3f} um",
            xy=(len(epochs), mean_thickness_history[-1]),
            xytext=(max(1, len(epochs) * 0.8), mean_thickness_history[-1] * 1.05),
            fontsize=9,
            color="green",
        )

    ax2 = axes[1]
    ax2.plot(epochs, merged_layers_history, "b-", linewidth=2, label="Merged Layers")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Number of Layers")
    ax2.set_title("Average Number of Merged Layers During Training")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=max_layers, color="gray", linestyle="--", alpha=0.5, label=f"Max Possible Layers ({max_layers})")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Min Possible Layers (1)")
    ax2.fill_between(epochs, merged_layers_history, max_layers, alpha=0.2, color="blue")
    ax2.legend(fontsize=9)

    if len(merged_layers_history) > 1:
        ax2.annotate(
            f"Start: {merged_layers_history[0]:.1f} layers",
            xy=(1, merged_layers_history[0]),
            xytext=(max(1, len(epochs) * 0.1), merged_layers_history[0]),
            fontsize=9,
            color="blue",
        )
        ax2.annotate(
            f"End: {merged_layers_history[-1]:.1f} layers",
            xy=(len(epochs), merged_layers_history[-1]),
            xytext=(max(1, len(epochs) * 0.8), merged_layers_history[-1] * 1.05),
            fontsize=9,
            color="blue",
        )
        ax2.text(
            0.02,
            0.02,
            "Merged layers count = number of distinct material blocks\n(adjacent identical materials are merged)",
            transform=ax2.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    _ensure_parent(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Thickness-Merged Layers curves saved to: {save_path}")


def save_distribution_evolution_plots(
    thickness_distribution_history,
    merged_layers_distribution_history,
    save_dir,
    thickness_range=(0.05, 0.5),
    max_layers=10,
    thickness_bins=20,
    heatmap_epoch_tick_step=10,
):
    """Save combined 2x2 plots and CSV files for thickness and merged-layer evolution."""
    if not thickness_distribution_history or not merged_layers_distribution_history:
        print("No distribution data to visualize")
        return

    os.makedirs(save_dir, exist_ok=True)

    thickness_epochs = [int(record["epoch"]) for record in thickness_distribution_history]
    thickness_means = [float(record["mean_thickness"]) for record in thickness_distribution_history]

    thickness_mean_df = pd.DataFrame(
        {
            "epoch": thickness_epochs,
            "mean_thickness_um": thickness_means,
        }
    )
    thickness_mean_csv = os.path.join(save_dir, "thickness_distribution_evolution_mean.csv")
    _save_dataframe(thickness_mean_df, thickness_mean_csv)

    first_bin_edges = np.asarray(thickness_distribution_history[0]["bin_edges"], dtype=float)
    bin_centers = (first_bin_edges[:-1] + first_bin_edges[1:]) / 2
    thickness_matrix = np.asarray(
        [record["hist_counts"] for record in thickness_distribution_history],
        dtype=float,
    )

    thickness_heatmap_rows = []
    for epoch, record in zip(thickness_epochs, thickness_distribution_history):
        bin_edges = np.asarray(record["bin_edges"], dtype=float)
        hist_counts = np.asarray(record["hist_counts"], dtype=float)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for index, count in enumerate(hist_counts):
            thickness_heatmap_rows.append(
                {
                    "epoch": epoch,
                    "bin_index": index + 1,
                    "bin_left_um": float(bin_edges[index]),
                    "bin_right_um": float(bin_edges[index + 1]),
                    "bin_center_um": float(centers[index]),
                    "count": int(count),
                }
            )
    thickness_heatmap_df = pd.DataFrame(thickness_heatmap_rows)
    thickness_heatmap_csv = os.path.join(save_dir, "thickness_distribution_evolution_heatmap.csv")
    _save_dataframe(thickness_heatmap_df, thickness_heatmap_csv)

    comparison_indices = _select_snapshot_indices(len(thickness_distribution_history))
    thickness_comparison_rows = []
    for snapshot_index in comparison_indices:
        record = thickness_distribution_history[snapshot_index]
        hist_counts = np.asarray(record["hist_counts"], dtype=float)
        for bin_index, count in enumerate(hist_counts):
            thickness_comparison_rows.append(
                {
                    "epoch": int(record["epoch"]),
                    "bin_index": bin_index + 1,
                    "bin_left_um": float(first_bin_edges[bin_index]),
                    "bin_right_um": float(first_bin_edges[bin_index + 1]),
                    "bin_center_um": float(bin_centers[bin_index]),
                    "count": int(count),
                }
            )
    thickness_comparison_csv = os.path.join(save_dir, "thickness_distribution_evolution_comparison.csv")
    _save_dataframe(pd.DataFrame(thickness_comparison_rows), thickness_comparison_csv)

    thickness_stds = []
    for record in thickness_distribution_history:
        hist_counts = np.asarray(record["hist_counts"], dtype=float)
        if hist_counts.sum() <= 0:
            thickness_stds.append(0.0)
            continue
        normalized_counts = hist_counts / hist_counts.sum()
        mean_value = np.sum(normalized_counts * bin_centers)
        variance = np.sum(normalized_counts * (bin_centers - mean_value) ** 2)
        thickness_stds.append(float(np.sqrt(max(variance, 0.0))))

    thickness_std_df = pd.DataFrame(
        {
            "epoch": thickness_epochs,
            "thickness_std_um": thickness_stds,
        }
    )
    thickness_std_csv = os.path.join(save_dir, "thickness_distribution_evolution_std.csv")
    _save_dataframe(thickness_std_df, thickness_std_csv)

    # Render the four thickness views into a single 2x2 summary figure.
    thickness_fig, thickness_axes = plt.subplots(2, 2, figsize=(18, 12))
    thickness_fig.suptitle("Thickness Distribution Evolution Overview", fontsize=16, fontweight="bold")

    ax = thickness_axes[0, 0]
    ax.plot(thickness_epochs, thickness_stds, "b-", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Thickness Std (um)")
    ax.set_title("Thickness Distribution Standard Deviation")
    ax.grid(True, alpha=0.3)

    ax = thickness_axes[0, 1]
    ax.plot(thickness_epochs, thickness_means, "g-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Thickness (um)")
    ax.set_title("Average Thickness Evolution")
    ax.grid(True, alpha=0.3)
    ax.axhline(
        y=thickness_range[0],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Lower Bound ({thickness_range[0]:.3f} um)",
    )
    ax.axhline(
        y=thickness_range[1],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Upper Bound ({thickness_range[1]:.3f} um)",
    )
    ax.legend(fontsize=9)

    ax = thickness_axes[1, 0]
    im = ax.imshow(thickness_matrix, aspect="auto", cmap="YlOrRd", origin="upper")
    ax.set_xlabel("Thickness (um)")
    ax.set_ylabel("Epoch")
    ax.set_title("Thickness Distribution Heatmap")
    x_positions = _choose_tick_positions(len(bin_centers))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{bin_centers[pos]:.3f}" for pos in x_positions])
    y_positions, y_labels = _build_heatmap_epoch_ticks(thickness_epochs, heatmap_epoch_tick_step)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    thickness_fig.colorbar(im, ax=ax, label="Count")

    ax = thickness_axes[1, 1]
    for snapshot_index in comparison_indices:
        record = thickness_distribution_history[snapshot_index]
        hist_counts = np.asarray(record["hist_counts"], dtype=float)
        ax.plot(
            bin_centers,
            hist_counts,
            linewidth=2,
            marker="o",
            label=f"Epoch {record['epoch']}",
        )
    ax.set_xlabel("Thickness (um)")
    ax.set_ylabel("Count")
    ax.set_title("Thickness Distribution Comparison")
    ax.grid(True, alpha=0.3)
    if comparison_indices:
        ax.legend()

    thickness_combined_png = os.path.join(save_dir, "thickness_distribution_evolution_combined.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    thickness_fig.savefig(thickness_combined_png, dpi=300, bbox_inches="tight")
    plt.close(thickness_fig)
    print(f"Thickness overview plot saved to: {thickness_combined_png}")

    merged_epochs = [int(record["epoch"]) for record in merged_layers_distribution_history]
    merged_means = [float(record["mean_merged_layers"]) for record in merged_layers_distribution_history]

    merged_mean_df = pd.DataFrame(
        {
            "epoch": merged_epochs,
            "mean_merged_layers": merged_means,
        }
    )
    merged_mean_csv = os.path.join(save_dir, "merged_layers_distribution_evolution_mean.csv")
    _save_dataframe(merged_mean_df, merged_mean_csv)

    merged_matrix = np.asarray(
        [record["layer_counts"] for record in merged_layers_distribution_history],
        dtype=float,
    )
    layer_indices = np.arange(1, merged_matrix.shape[1] + 1)

    merged_heatmap_rows = []
    for epoch, record in zip(merged_epochs, merged_layers_distribution_history):
        for layer_count, count in enumerate(record["layer_counts"], start=1):
            merged_heatmap_rows.append(
                {
                    "epoch": epoch,
                    "merged_layer_count": layer_count,
                    "count": int(count),
                }
            )
    merged_heatmap_df = pd.DataFrame(merged_heatmap_rows)
    merged_heatmap_csv = os.path.join(save_dir, "merged_layers_distribution_evolution_heatmap.csv")
    _save_dataframe(merged_heatmap_df, merged_heatmap_csv)

    merged_comparison_rows = []
    colors = ["tab:blue", "tab:green", "tab:red"]
    comparison_indices = _select_snapshot_indices(len(merged_layers_distribution_history))
    width = 0.25
    for plot_index, snapshot_index in enumerate(comparison_indices):
        record = merged_layers_distribution_history[snapshot_index]
        offset = (plot_index - (len(comparison_indices) - 1) / 2) * width
        for layer_count, count in enumerate(record["layer_counts"], start=1):
            merged_comparison_rows.append(
                {
                    "epoch": int(record["epoch"]),
                    "merged_layer_count": layer_count,
                    "count": int(count),
                }
            )
    merged_comparison_csv = os.path.join(save_dir, "merged_layers_distribution_evolution_comparison.csv")
    _save_dataframe(pd.DataFrame(merged_comparison_rows), merged_comparison_csv)

    merged_entropies = []
    for record in merged_layers_distribution_history:
        layer_counts = np.asarray(record["layer_counts"], dtype=float)
        total = layer_counts.sum()
        if total <= 0:
            merged_entropies.append(0.0)
            continue
        probabilities = layer_counts / total
        merged_entropies.append(float(-np.sum(probabilities * np.log(probabilities + 1e-10))))

    merged_entropy_df = pd.DataFrame(
        {
            "epoch": merged_epochs,
            "entropy": merged_entropies,
        }
    )
    merged_entropy_csv = os.path.join(save_dir, "merged_layers_distribution_evolution_entropy.csv")
    _save_dataframe(merged_entropy_df, merged_entropy_csv)

    # Render the four merged-layer views into a single 2x2 summary figure.
    merged_fig, merged_axes = plt.subplots(2, 2, figsize=(18, 12))
    merged_fig.suptitle("Merged Layers Distribution Evolution Overview", fontsize=16, fontweight="bold")

    ax = merged_axes[0, 0]
    ax.plot(merged_epochs, merged_means, "b-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Merged Layers")
    ax.set_title("Average Merged Layers Evolution")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Min (1 layer)")
    ax.axhline(y=max_layers, color="gray", linestyle="--", alpha=0.5, label=f"Max ({max_layers} layers)")
    ax.legend(fontsize=9)

    ax = merged_axes[0, 1]
    im = ax.imshow(merged_matrix, aspect="auto", cmap="Blues", origin="upper")
    ax.set_xlabel("Number of Merged Layers")
    ax.set_ylabel("Epoch")
    ax.set_title("Merged Layers Distribution Heatmap")
    ax.set_xticks(np.arange(len(layer_indices)))
    ax.set_xticklabels(layer_indices)
    y_positions, y_labels = _build_heatmap_epoch_ticks(merged_epochs, heatmap_epoch_tick_step)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    merged_fig.colorbar(im, ax=ax, label="Count")

    ax = merged_axes[1, 0]
    ax.plot(merged_epochs, merged_entropies, "r-", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.set_title("Merged Layers Distribution Entropy")
    ax.grid(True, alpha=0.3)
    max_entropy = float(np.log(max(1, int(max_layers))))
    if max_layers > 1:
        ax.axhline(
            y=max_entropy,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Max entropy = {max_entropy:.2f}",
        )
        ax.legend(fontsize=9)

    ax = merged_axes[1, 1]
    for plot_index, snapshot_index in enumerate(comparison_indices):
        record = merged_layers_distribution_history[snapshot_index]
        offset = (plot_index - (len(comparison_indices) - 1) / 2) * width
        positions = layer_indices + offset
        ax.bar(
            positions,
            record["layer_counts"],
            width=width,
            alpha=0.7,
            label=f"Epoch {record['epoch']}",
            color=colors[plot_index % len(colors)],
        )
    ax.set_xlabel("Number of Merged Layers")
    ax.set_ylabel("Count")
    ax.set_title("Merged Layers Distribution Comparison")
    ax.set_xticks(layer_indices)
    ax.grid(True, alpha=0.3)
    if comparison_indices:
        ax.legend()

    merged_combined_png = os.path.join(save_dir, "merged_layers_distribution_evolution_combined.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    merged_fig.savefig(merged_combined_png, dpi=300, bbox_inches="tight")
    plt.close(merged_fig)
    print(f"Merged-layer overview plot saved to: {merged_combined_png}")

    distribution_data = {
        "thickness_distribution": thickness_distribution_history,
        "merged_layers_distribution": merged_layers_distribution_history,
        "config": {
            "thickness_range": [float(thickness_range[0]), float(thickness_range[1])],
            "max_layers": int(max_layers),
            "num_thickness_bins": int(thickness_bins),
            "heatmap_epoch_tick_step": int(heatmap_epoch_tick_step),
        },
    }

    json_path = os.path.join(save_dir, "distribution_data.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(distribution_data, file, indent=2, ensure_ascii=False)
    print(f"Distribution data saved to: {json_path}")


def analyze_inference_distribution(
    thicknesses,
    material_probs,
    save_dir,
    prefix="inference",
    thickness_range=(0.05, 0.5),
    thickness_bins=20,
    max_layers=None,
):
    """Analyze generated thickness and merged-layer distributions during inference."""
    thicknesses_np = _to_numpy(thicknesses)
    material_probs_np = _to_numpy(material_probs)

    num_samples, num_layers = thicknesses_np.shape
    if max_layers is None:
        max_layers = num_layers

    thickness_hist_counts, thickness_bin_edges = np.histogram(
        thicknesses_np.flatten(),
        bins=thickness_bins,
        range=thickness_range,
    )

    thickness_mean = float(np.mean(thicknesses_np))
    thickness_std = float(np.std(thicknesses_np))
    thickness_percentiles = {
        "p10": float(np.percentile(thicknesses_np, 10)),
        "p25": float(np.percentile(thicknesses_np, 25)),
        "p50": float(np.percentile(thicknesses_np, 50)),
        "p75": float(np.percentile(thicknesses_np, 75)),
        "p90": float(np.percentile(thicknesses_np, 90)),
    }

    from train.trainer import calculate_merged_layers

    material_probs_tensor = torch.as_tensor(material_probs_np)
    merged_counts = calculate_merged_layers(material_probs_tensor).cpu().numpy()

    layer_counts = np.zeros(max_layers, dtype=int)
    for layer_count in range(1, max_layers + 1):
        layer_counts[layer_count - 1] = int(np.sum(merged_counts == layer_count))

    merged_layers_mean = float(np.mean(merged_counts))
    merged_layers_std = float(np.std(merged_counts))
    merged_layers_percentiles = {
        "p10": float(np.percentile(merged_counts, 10)),
        "p25": float(np.percentile(merged_counts, 25)),
        "p50": float(np.percentile(merged_counts, 50)),
        "p75": float(np.percentile(merged_counts, 75)),
        "p90": float(np.percentile(merged_counts, 90)),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Distribution Analysis of Generated Samples (n={num_samples})",
        fontsize=16,
        fontweight="bold",
    )

    bin_centers = (thickness_bin_edges[:-1] + thickness_bin_edges[1:]) / 2
    ax1 = axes[0, 0]
    bar_width = (thickness_bin_edges[1] - thickness_bin_edges[0]) * 0.9 if len(thickness_bin_edges) > 1 else 0.02
    ax1.bar(bin_centers, thickness_hist_counts, width=bar_width, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(x=thickness_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {thickness_mean:.3f} um")
    ax1.set_xlabel("Thickness (um)")
    ax1.set_ylabel("Count")
    ax1.set_title("Thickness Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    layer_indices = np.arange(1, max_layers + 1)
    ax2.bar(layer_indices, layer_counts, alpha=0.7, color="lightgreen", edgecolor="black")
    ax2.axvline(x=merged_layers_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {merged_layers_mean:.1f} layers")
    ax2.set_xlabel("Number of Merged Layers")
    ax2.set_ylabel("Count")
    ax2.set_title("Merged Layers Distribution")
    ax2.set_xticks(layer_indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    total_thickness = np.sum(thicknesses_np, axis=1)
    scatter = ax3.scatter(total_thickness, merged_counts, alpha=0.6, s=30, c=np.mean(thicknesses_np, axis=1), cmap="viridis")
    ax3.set_xlabel("Total Structure Thickness (um)")
    ax3.set_ylabel("Number of Merged Layers")
    ax3.set_title("Total Thickness vs Merged Layers")
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label="Average Layer Thickness (um)")

    ax4 = axes[1, 0]
    ax4.boxplot(thicknesses_np.flatten(), vert=False)
    ax4.set_xlabel("Thickness (um)")
    ax4.set_title("Thickness Box Plot")
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    ax5.boxplot(merged_counts, vert=False)
    ax5.set_xlabel("Number of Merged Layers")
    ax5.set_title("Merged Layers Box Plot")
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    sorted_thickness = np.sort(thicknesses_np.flatten())
    thickness_cdf = np.arange(1, len(sorted_thickness) + 1) / len(sorted_thickness)
    ax6.plot(sorted_thickness, thickness_cdf, "b-", linewidth=2, label="Thickness")

    sorted_layers = np.sort(merged_counts)
    layers_cdf = np.arange(1, len(sorted_layers) + 1) / len(sorted_layers)
    ax6.step(sorted_layers, layers_cdf, "r-", linewidth=2, where="post", label="Merged Layers")

    ax6.set_xlabel("Value")
    ax6.set_ylabel("Cumulative Probability")
    ax6.set_title("Cumulative Distribution Functions")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{prefix}_distribution_analysis.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Inference distribution analysis plot saved to: {plot_path}")

    stats_data = {
        "num_samples": int(num_samples),
        "thickness_statistics": {
            "mean": thickness_mean,
            "std": thickness_std,
            "percentiles": thickness_percentiles,
            "histogram": {
                "counts": thickness_hist_counts.tolist(),
                "bin_edges": thickness_bin_edges.tolist(),
            },
        },
        "merged_layers_statistics": {
            "mean": merged_layers_mean,
            "std": merged_layers_std,
            "percentiles": merged_layers_percentiles,
            "distribution": layer_counts.tolist(),
        },
    }

    stats_path = os.path.join(save_dir, f"{prefix}_distribution_stats.json")
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stats_data, file, indent=2, ensure_ascii=False)
    print(f"Inference distribution statistics saved to: {stats_path}")

    return stats_data
