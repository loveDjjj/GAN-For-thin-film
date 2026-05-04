"""Training-time high-quality solution collection utilities."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves


def build_high_quality_criteria(params):
    """Build the configured screening thresholds for high-quality solutions."""
    return {
        "enabled": bool(getattr(params, "high_quality_collection_enabled", False)),
        "q_min": float(getattr(params, "high_quality_q_min", 200.0)),
        "mse_max": float(getattr(params, "high_quality_mse_max", 0.0025)),
        "peak_min": float(getattr(params, "high_quality_peak_min", 0.9)),
        "dominant_material_prob_min": float(getattr(params, "high_quality_dominant_prob_min", 0.99)),
        # Reuse the same sampling schedule as q_evaluation to avoid extra generator/TMM passes.
        "interval": int(getattr(params, "q_eval_interval", 0)),
        "num_samples": int(getattr(params, "q_eval_num_samples", 0)),
    }


def _ensure_material_probabilities(material_probabilities):
    if material_probabilities.ndim == 2:
        return material_probabilities.unsqueeze(0)
    if material_probabilities.ndim != 3:
        raise ValueError("material_probabilities must have shape [batch_size, num_layers, num_materials]")
    return material_probabilities


def _generate_peak_aligned_lorentzian_curves_torch(wavelengths, peak_wavelengths, width):
    if width <= 0:
        raise ValueError("lorentzian width must be positive")

    wavelengths = wavelengths.to(device=peak_wavelengths.device, dtype=peak_wavelengths.dtype).flatten()
    gamma = torch.as_tensor(width, device=peak_wavelengths.device, dtype=peak_wavelengths.dtype)
    curves = (gamma / 2) / ((wavelengths.unsqueeze(0) - peak_wavelengths.unsqueeze(1)) ** 2 + (gamma / 2) ** 2)
    max_values = curves.amax(dim=1, keepdim=True).clamp_min(torch.finfo(curves.dtype).eps)
    return curves / max_values


def _generate_peak_aligned_double_lorentzian_curves_torch(wavelengths, peak_wavelengths_1, peak_wavelengths_2, width):
    return generate_double_lorentzian_curves(
        wavelengths=wavelengths,
        width=width,
        centers1=peak_wavelengths_1,
        centers2=peak_wavelengths_2,
    )


def build_original_layers(thickness_values, material_probs, materials):
    """Build per-layer structure records with dominant-material metadata."""
    material_indices = material_probs.argmax(dim=1)
    dominant_probabilities = material_probs.max(dim=1).values
    original_layers = []

    for layer_index in range(thickness_values.shape[0]):
        layer_probs = material_probs[layer_index]
        dominant_material_index = int(material_indices[layer_index].item())
        original_layers.append(
            {
                "layer_index": layer_index + 1,
                "thickness_um": float(thickness_values[layer_index].item()),
                "dominant_material_index": dominant_material_index,
                "dominant_material": materials[dominant_material_index],
                "dominant_probability": float(dominant_probabilities[layer_index].item()),
                "material_probabilities": {
                    material_name: float(layer_probs[material_index].item())
                    for material_index, material_name in enumerate(materials)
                },
            }
        )

    return original_layers


def build_merged_layers(thickness_values, material_probs, materials):
    """Merge adjacent layers with the same dominant material."""
    material_indices = material_probs.argmax(dim=1)
    merged_layers = []

    current_material = int(material_indices[0].item())
    start_layer = 0
    merged_thickness = float(thickness_values[0].item())
    dominant_probs = [float(material_probs[0, current_material].item())]

    for layer_index in range(1, thickness_values.shape[0]):
        material_index = int(material_indices[layer_index].item())
        if material_index == current_material:
            merged_thickness += float(thickness_values[layer_index].item())
            dominant_probs.append(float(material_probs[layer_index, material_index].item()))
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
                "mean_dominant_probability": float(sum(dominant_probs) / len(dominant_probs)),
            }
        )

        current_material = material_index
        start_layer = layer_index
        merged_thickness = float(thickness_values[layer_index].item())
        dominant_probs = [float(material_probs[layer_index, material_index].item())]

    merged_layers.append(
        {
            "merged_layer_index": len(merged_layers) + 1,
            "material_index": current_material,
            "material": materials[current_material],
            "start_layer": start_layer + 1,
            "end_layer": thickness_values.shape[0],
            "original_layer_count": thickness_values.shape[0] - start_layer,
            "merged_thickness_um": merged_thickness,
            "mean_dominant_probability": float(sum(dominant_probs) / len(dominant_probs)),
        }
    )

    return merged_layers


def _build_merged_structure_keys(merged_layers):
    """Build canonical merged-structure keys at 1nm and 10nm rounding granularities."""
    one_nm_parts = []
    ten_nm_parts = []
    for layer in merged_layers:
        material = str(layer["material"])
        thickness_um = float(layer["merged_thickness_um"])
        thickness_nm = int(round(thickness_um * 1000.0))
        thickness_10nm = int(round(thickness_um * 100.0))
        one_nm_parts.append(f"{material}:{thickness_nm}")
        ten_nm_parts.append(f"{material}:{thickness_10nm}")
    return "|".join(one_nm_parts), "|".join(ten_nm_parts)


def initialize_high_quality_collection(save_dir, criteria):
    """Prepare directories and registry files for high-quality solution collection."""
    os.makedirs(save_dir, exist_ok=True)
    summary_dir = os.path.join(save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    criteria_path = os.path.join(save_dir, "collection_criteria.json")
    with open(criteria_path, "w", encoding="utf-8") as file:
        json.dump(criteria, file, indent=2, ensure_ascii=False)

    registry_path = os.path.join(summary_dir, "high_quality_solutions.csv")
    if not os.path.exists(registry_path):
        pd.DataFrame(
            columns=[
                "sample_id",
                "epoch",
                "alpha",
                "evaluation_sample_index",
                "q1",
                "q2",
                "q_min_pair",
                "double_lorentz_mse",
                "peak_wavelength_1_um",
                "peak_wavelength_2_um",
                "peak_absorption_1",
                "peak_absorption_2",
                "fwhm_1_um",
                "fwhm_2_um",
                "total_thickness_um",
                "min_dominant_material_probability",
                "merged_layer_count",
                "merged_structure_1nm_key",
                "merged_structure_10nm_key",
                "sample_dir",
            ]
        ).to_csv(registry_path, index=False)


def collect_high_quality_solutions_batch(
    wavelengths,
    absorption_spectra,
    thicknesses,
    material_probabilities,
    q_mse_results,
    params,
    epoch,
    alpha,
    sample_offset,
    save_dir,
):
    """Save all samples in a batch that satisfy the configured high-quality criteria."""
    criteria = build_high_quality_criteria(params)
    if not criteria["enabled"]:
        return []

    initialize_high_quality_collection(save_dir, criteria)

    material_probabilities = _ensure_material_probabilities(material_probabilities)
    dominant_probs = material_probabilities.max(dim=2).values
    min_dominant_probs = dominant_probs.min(dim=1).values

    high_quality_mask = (
        (q_mse_results["q_min_pair_values"] > criteria["q_min"])
        & (q_mse_results["double_lorentz_mse_values"] < criteria["mse_max"])
        & (q_mse_results["peak_absorptions_1"] > criteria["peak_min"])
        & (q_mse_results["peak_absorptions_2"] > criteria["peak_min"])
        & (min_dominant_probs > criteria["dominant_material_prob_min"])
    )

    if not high_quality_mask.any():
        return []

    selected_indices = torch.where(high_quality_mask)[0]
    selected_peak_wavelengths_1 = q_mse_results["peak_wavelengths_1"][selected_indices]
    selected_peak_wavelengths_2 = q_mse_results["peak_wavelengths_2"][selected_indices]
    selected_targets = _generate_peak_aligned_double_lorentzian_curves_torch(
        wavelengths,
        selected_peak_wavelengths_1,
        selected_peak_wavelengths_2,
        float(getattr(params, "lorentz_width", 0.02)),
    )

    epoch_dir = os.path.join(save_dir, f"epoch_{int(epoch):04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    records = []
    for local_index, batch_index in enumerate(selected_indices.tolist()):
        global_index = int(sample_offset + batch_index)
        sample_id = f"epoch_{int(epoch):04d}_sample_{global_index:05d}"
        sample_dir = os.path.join(epoch_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        sample_absorption = absorption_spectra[batch_index].detach().cpu()
        sample_target = selected_targets[local_index].detach().cpu()
        sample_thicknesses = thicknesses[batch_index].detach().cpu()
        sample_material_probs = material_probabilities[batch_index].detach().cpu()
        sample_q1 = float(q_mse_results["q1_values"][batch_index].item())
        sample_q2 = float(q_mse_results["q2_values"][batch_index].item())
        sample_q_min_pair = float(q_mse_results["q_min_pair_values"][batch_index].item())
        sample_double_mse = float(q_mse_results["double_lorentz_mse_values"][batch_index].item())
        sample_peak_absorption_1 = float(q_mse_results["peak_absorptions_1"][batch_index].item())
        sample_peak_absorption_2 = float(q_mse_results["peak_absorptions_2"][batch_index].item())
        sample_fwhm_1 = float(q_mse_results["fwhm_1"][batch_index].item())
        sample_fwhm_2 = float(q_mse_results["fwhm_2"][batch_index].item())
        sample_peak_pos_1 = float(q_mse_results["peak_wavelengths_1"][batch_index].item())
        sample_peak_pos_2 = float(q_mse_results["peak_wavelengths_2"][batch_index].item())
        sample_min_dominant_prob = float(min_dominant_probs[batch_index].item())

        total_thickness = float(sample_thicknesses.sum().item())
        merged_layers = build_merged_layers(sample_thicknesses, sample_material_probs, params.materials)
        merged_structure_1nm_key, merged_structure_10nm_key = _build_merged_structure_keys(merged_layers)

        spectrum_df = pd.DataFrame(
            {
                "wavelength_um": wavelengths.detach().cpu().numpy(),
                "absorption": sample_absorption.numpy(),
                "peak_aligned_double_lorentzian": sample_target.numpy(),
            }
        )
        spectrum_csv_path = os.path.join(sample_dir, "spectrum.csv")
        spectrum_df.to_csv(spectrum_csv_path, index=False)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        wavelengths_np = wavelengths.detach().cpu().numpy()
        ax.plot(wavelengths_np, sample_absorption.numpy(), linewidth=2, label="Generated absorption")
        ax.plot(
            wavelengths_np,
            sample_target.numpy(),
            linewidth=1.8,
            linestyle="--",
            label="Peak-aligned Double Lorentzian",
        )
        ax.axvline(sample_peak_pos_1, color="tab:red", linestyle=":", linewidth=1.2)
        ax.axvline(sample_peak_pos_2, color="tab:purple", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Absorption")
        ax.set_ylim(-0.02, max(1.05, float(sample_absorption.max().item()) * 1.05))
        ax.set_title(
            f"High-Quality Solution | Qmin={sample_q_min_pair:.2f}, DoubleMSE={sample_double_mse:.6f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        spectrum_plot_path = os.path.join(sample_dir, "spectrum.png")
        fig.savefig(spectrum_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        original_layers = build_original_layers(sample_thicknesses, sample_material_probs, params.materials)

        structure_payload = {
            "sample_id": sample_id,
            "epoch": int(epoch),
            "alpha": float(alpha),
            "evaluation_sample_index": global_index,
            "criteria": criteria,
                "metrics": {
                    "q1": sample_q1,
                    "q2": sample_q2,
                    "q_min_pair": sample_q_min_pair,
                    "double_lorentz_mse": sample_double_mse,
                    "peak_wavelength_1_um": sample_peak_pos_1,
                    "peak_wavelength_2_um": sample_peak_pos_2,
                    "peak_absorption_1": sample_peak_absorption_1,
                    "peak_absorption_2": sample_peak_absorption_2,
                    "fwhm_1_um": sample_fwhm_1,
                    "fwhm_2_um": sample_fwhm_2,
                    "total_thickness_um": total_thickness,
                    "min_dominant_material_probability": sample_min_dominant_prob,
                    "merged_layer_count": len(merged_layers),
                    "merged_structure_1nm_key": merged_structure_1nm_key,
                    "merged_structure_10nm_key": merged_structure_10nm_key,
            },
            "materials": list(params.materials),
            "original_layers": original_layers,
            "merged_layers": merged_layers,
        }

        structure_json_path = os.path.join(sample_dir, "structure.json")
        with open(structure_json_path, "w", encoding="utf-8") as file:
            json.dump(structure_payload, file, indent=2, ensure_ascii=False)

        records.append(
            {
                "sample_id": sample_id,
                "epoch": int(epoch),
                "alpha": float(alpha),
                "evaluation_sample_index": global_index,
                "q1": sample_q1,
                "q2": sample_q2,
                "q_min_pair": sample_q_min_pair,
                "double_lorentz_mse": sample_double_mse,
                "peak_wavelength_1_um": sample_peak_pos_1,
                "peak_wavelength_2_um": sample_peak_pos_2,
                "peak_absorption_1": sample_peak_absorption_1,
                "peak_absorption_2": sample_peak_absorption_2,
                "fwhm_1_um": sample_fwhm_1,
                "fwhm_2_um": sample_fwhm_2,
                "total_thickness_um": total_thickness,
                "min_dominant_material_probability": sample_min_dominant_prob,
                "merged_layer_count": int(len(merged_layers)),
                "merged_structure_1nm_key": merged_structure_1nm_key,
                "merged_structure_10nm_key": merged_structure_10nm_key,
                "sample_dir": sample_dir,
            }
        )

    return records


def update_high_quality_collection_summary(save_dir, new_records):
    """Update cumulative CSV/JSON summaries and distribution plots for collected solutions."""
    summary_dir = os.path.join(save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    registry_path = os.path.join(summary_dir, "high_quality_solutions.csv")

    if os.path.exists(registry_path):
        existing_df = pd.read_csv(registry_path)
    else:
        existing_df = pd.DataFrame()

    if new_records:
        new_df = pd.DataFrame(new_records)
        if existing_df.empty:
            combined_df = new_df
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = existing_df

    if not combined_df.empty:
        combined_df["min_peak_absorption"] = combined_df[["peak_absorption_1", "peak_absorption_2"]].min(axis=1)
        combined_df = combined_df.drop_duplicates(subset=["sample_id"], keep="last")
        # Deduplicate by merged structure rounded to 10nm, keep best by Q first then MSE.
        if "merged_structure_10nm_key" in combined_df.columns:
            combined_df = combined_df.sort_values(
                by=["merged_structure_10nm_key", "q_min_pair", "double_lorentz_mse"],
                ascending=[True, False, True],
            )
            combined_df = combined_df.drop_duplicates(subset=["merged_structure_10nm_key"], keep="first")
        combined_df = combined_df.sort_values(
            by=["epoch", "q_min_pair", "min_peak_absorption"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    combined_df.to_csv(registry_path, index=False)

    summary_payload = {
        "total_high_quality_solutions": int(len(combined_df)),
        "best_q_min_pair": float(combined_df["q_min_pair"].max()) if not combined_df.empty else 0.0,
        "lowest_double_lorentz_mse": float(combined_df["double_lorentz_mse"].min()) if not combined_df.empty else 0.0,
        "highest_peak_absorption_1": float(combined_df["peak_absorption_1"].max()) if not combined_df.empty else 0.0,
        "highest_peak_absorption_2": float(combined_df["peak_absorption_2"].max()) if not combined_df.empty else 0.0,
        "largest_total_thickness_um": float(combined_df["total_thickness_um"].max()) if not combined_df.empty else 0.0,
        "epochs_with_hits": (
            sorted(int(epoch) for epoch in combined_df["epoch"].unique().tolist())
            if not combined_df.empty
            else []
        ),
    }

    summary_json_path = os.path.join(summary_dir, "high_quality_solution_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)

    if not combined_df.empty:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        axes[0, 0].hist(combined_df["peak_wavelength_1_um"], bins=20, color="royalblue", alpha=0.85, edgecolor="black")
        axes[0, 0].set_title("Peak 1 Wavelength Distribution")
        axes[0, 0].set_xlabel("Wavelength (um)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(combined_df["q_min_pair"], bins=20, color="darkorange", alpha=0.85, edgecolor="black")
        axes[0, 1].set_title("Q_min_pair Distribution")
        axes[0, 1].set_xlabel("Q")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].hist(combined_df["double_lorentz_mse"], bins=20, color="seagreen", alpha=0.85, edgecolor="black")
        axes[0, 2].set_title("Double-Lorentzian MSE Distribution")
        axes[0, 2].set_xlabel("MSE")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].hist(combined_df["peak_absorption_1"], bins=20, color="crimson", alpha=0.85, edgecolor="black")
        axes[1, 0].set_title("Peak 1 Absorption Distribution")
        axes[1, 0].set_xlabel("Peak Absorption")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(
            combined_df["total_thickness_um"],
            bins=20,
            color="mediumpurple",
            alpha=0.85,
            edgecolor="black",
        )
        axes[1, 1].set_title("Total Thickness Distribution")
        axes[1, 1].set_xlabel("Total Thickness (um)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)

        epoch_counts = combined_df.groupby("epoch").size()
        axes[1, 2].bar(epoch_counts.index.astype(str), epoch_counts.values, color="slategray", alpha=0.9)
        axes[1, 2].set_title("High-Quality Solution Count by Epoch")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].grid(True, axis="y", alpha=0.3)
        if len(epoch_counts) > 8:
            axes[1, 2].tick_params(axis="x", rotation=45)

        fig.suptitle(f"Collected High-Quality Solutions ({len(combined_df)} total)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        plot_path = os.path.join(summary_dir, "high_quality_solution_distributions.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "new_high_quality_count": int(len(new_records)),
        "total_high_quality_count": int(len(combined_df)),
        "registry_path": registry_path,
    }
