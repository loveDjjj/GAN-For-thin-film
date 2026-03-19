"""Training-time Q and Lorentzian-MSE evaluation helpers."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model.TMM.optical_calculator import calculate_reflection
from train.high_quality_solution_collector import (
    build_high_quality_criteria,
    collect_high_quality_solutions_batch,
    initialize_high_quality_collection,
    update_high_quality_collection_summary,
)


def _safe_delta(delta, eps):
    replacement = torch.where(delta >= 0, torch.full_like(delta, eps), torch.full_like(delta, -eps))
    return torch.where(delta.abs() < eps, replacement, delta)


def compute_q_factors_torch(wavelengths, absorption_spectra, eps=1e-12):
    """Compute per-sample Q factors in parallel on torch tensors."""
    if absorption_spectra.ndim != 2:
        raise ValueError("absorption_spectra must have shape [batch_size, num_wavelengths]")

    device = absorption_spectra.device
    wavelengths = wavelengths.to(device=device, dtype=absorption_spectra.dtype).flatten()
    absorption_spectra = absorption_spectra.to(dtype=wavelengths.dtype)

    batch_size, num_wavelengths = absorption_spectra.shape
    if wavelengths.numel() != num_wavelengths:
        raise ValueError("wavelengths length must match absorption_spectra.shape[1]")

    indices = torch.arange(num_wavelengths, device=device).unsqueeze(0).expand(batch_size, -1)

    peak_indices = torch.argmax(absorption_spectra, dim=1)
    peak_absorptions = absorption_spectra.gather(1, peak_indices.unsqueeze(1)).squeeze(1)
    peak_wavelengths = wavelengths[peak_indices]
    half_max = peak_absorptions * 0.5

    left_candidates = torch.where(
        (indices < peak_indices.unsqueeze(1)) & (absorption_spectra <= half_max.unsqueeze(1)),
        indices,
        torch.full_like(indices, -1),
    )
    left_indices = left_candidates.max(dim=1).values

    right_candidates = torch.where(
        (indices > peak_indices.unsqueeze(1)) & (absorption_spectra <= half_max.unsqueeze(1)),
        indices,
        torch.full_like(indices, num_wavelengths),
    )
    right_indices = right_candidates.min(dim=1).values

    left_wavelengths = torch.full_like(peak_wavelengths, float("nan"))
    right_wavelengths = torch.full_like(peak_wavelengths, float("nan"))
    fwhm = torch.full_like(peak_wavelengths, float("nan"))
    q_values = torch.zeros_like(peak_wavelengths)

    valid_mask = (
        (peak_absorptions > 0)
        & (left_indices >= 0)
        & (right_indices < num_wavelengths)
        & (right_indices > left_indices)
    )

    if valid_mask.any():
        valid_rows = torch.where(valid_mask)[0]
        left_idx = left_indices[valid_rows]
        right_idx = right_indices[valid_rows]
        half_max_valid = half_max[valid_rows]

        left_x0 = wavelengths[left_idx]
        left_x1 = wavelengths[left_idx + 1]
        left_y0 = absorption_spectra[valid_rows, left_idx]
        left_y1 = absorption_spectra[valid_rows, left_idx + 1]
        left_delta = _safe_delta(left_y1 - left_y0, eps)
        interpolated_left = left_x0 + (left_x1 - left_x0) * (half_max_valid - left_y0) / left_delta

        right_x0 = wavelengths[right_idx - 1]
        right_x1 = wavelengths[right_idx]
        right_y0 = absorption_spectra[valid_rows, right_idx - 1]
        right_y1 = absorption_spectra[valid_rows, right_idx]
        right_delta = _safe_delta(right_y1 - right_y0, eps)
        interpolated_right = right_x0 + (right_x1 - right_x0) * (half_max_valid - right_y0) / right_delta

        current_fwhm = interpolated_right - interpolated_left
        current_valid = current_fwhm > eps

        left_wavelengths[valid_rows] = interpolated_left
        right_wavelengths[valid_rows] = interpolated_right
        fwhm[valid_rows] = current_fwhm

        q_values_valid = torch.where(
            current_valid,
            peak_wavelengths[valid_rows] / current_fwhm.clamp_min(eps),
            torch.zeros_like(current_fwhm),
        )
        q_values[valid_rows] = q_values_valid
        valid_mask[valid_rows] = current_valid

    q_values = torch.nan_to_num(q_values, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "q_values": q_values,
        "valid_mask": valid_mask,
        "peak_indices": peak_indices,
        "peak_wavelengths": peak_wavelengths,
        "peak_absorptions": peak_absorptions,
        "left_wavelengths": left_wavelengths,
        "right_wavelengths": right_wavelengths,
        "fwhm": fwhm,
    }


def generate_peak_aligned_lorentzian_curves_torch(wavelengths, peak_wavelengths, width):
    """Build normalized Lorentzian targets centered at each sample peak wavelength."""
    if width <= 0:
        raise ValueError("lorentzian width must be positive")

    if peak_wavelengths.ndim != 1:
        raise ValueError("peak_wavelengths must have shape [batch_size]")

    wavelengths = wavelengths.to(device=peak_wavelengths.device, dtype=peak_wavelengths.dtype).flatten()
    gamma = torch.as_tensor(width, device=peak_wavelengths.device, dtype=peak_wavelengths.dtype)
    curves = (gamma / 2) / ((wavelengths.unsqueeze(0) - peak_wavelengths.unsqueeze(1)) ** 2 + (gamma / 2) ** 2)

    max_values = curves.amax(dim=1, keepdim=True).clamp_min(torch.finfo(curves.dtype).eps)
    return curves / max_values


def compute_peak_lorentzian_mse_torch(wavelengths, absorption_spectra, peak_wavelengths, width):
    """Compute per-sample MSE to a peak-aligned Lorentzian target in parallel."""
    target_curves = generate_peak_aligned_lorentzian_curves_torch(wavelengths, peak_wavelengths, width)
    mse_values = torch.mean((absorption_spectra - target_curves) ** 2, dim=1)
    mse_values = torch.nan_to_num(mse_values, nan=0.0, posinf=0.0, neginf=0.0)
    return {"mse_values": mse_values}


def compute_q_mse_metrics_torch(wavelengths, absorption_spectra, lorentz_width, eps=1e-12):
    """Compute Q factors and peak-aligned Lorentzian MSE in one batched pass."""
    q_results = compute_q_factors_torch(wavelengths, absorption_spectra, eps=eps)
    mse_results = compute_peak_lorentzian_mse_torch(
        wavelengths,
        absorption_spectra,
        q_results["peak_wavelengths"],
        lorentz_width,
    )
    return {**q_results, **mse_results}


def summarize_q_results(q_results, epoch, alpha, num_samples, lorentz_width):
    """Compute summary statistics for a batch of Q and Lorentzian-MSE results."""
    q_values = q_results["q_values"]
    mse_values = q_results["mse_values"]
    valid_mask = q_results["valid_mask"]

    return {
        "epoch": int(epoch),
        "alpha": float(alpha),
        "num_samples": int(num_samples),
        "lorentz_width": float(lorentz_width),
        "valid_count": int(valid_mask.sum().item()),
        "valid_ratio": float(valid_mask.float().mean().item()),
        "mean_q": float(q_values.mean().item()),
        "median_q": float(q_values.median().item()),
        "max_q": float(q_values.max().item()),
        "std_q": float(q_values.std(unbiased=False).item()),
        "mean_mse": float(mse_values.mean().item()),
        "median_mse": float(mse_values.median().item()),
        "min_mse": float(mse_values.min().item()),
        "max_mse": float(mse_values.max().item()),
        "std_mse": float(mse_values.std(unbiased=False).item()),
    }


def save_q_evaluation_epoch(q_results, summary, save_dir):
    """Save per-epoch Q and MSE statistics and plots."""
    os.makedirs(save_dir, exist_ok=True)

    epoch = summary["epoch"]
    q_values = q_results["q_values"].detach().cpu().numpy()
    mse_values = q_results["mse_values"].detach().cpu().numpy()
    valid_mask = q_results["valid_mask"].detach().cpu().numpy()
    peak_wavelengths = q_results["peak_wavelengths"].detach().cpu().numpy()
    peak_absorptions = q_results["peak_absorptions"].detach().cpu().numpy()
    left_wavelengths = q_results["left_wavelengths"].detach().cpu().numpy()
    right_wavelengths = q_results["right_wavelengths"].detach().cpu().numpy()
    fwhm = q_results["fwhm"].detach().cpu().numpy()

    details_df = pd.DataFrame(
        {
            "sample_index": range(len(q_values)),
            "q_value": q_values,
            "lorentz_mse": mse_values,
            "valid": valid_mask,
            "peak_wavelength_um": peak_wavelengths,
            "peak_absorption": peak_absorptions,
            "left_half_max_um": left_wavelengths,
            "right_half_max_um": right_wavelengths,
            "fwhm_um": fwhm,
        }
    )
    details_path = os.path.join(save_dir, f"q_mse_metrics_epoch_{epoch}.csv")
    details_df.to_csv(details_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(q_values, bins=30, color="steelblue", alpha=0.85, edgecolor="black")
    axes[0, 0].set_title(f"Q Distribution (Epoch {epoch})")
    axes[0, 0].set_xlabel("Q")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(peak_wavelengths, q_values, s=10, alpha=0.65, color="darkorange")
    axes[0, 1].set_title(f"Peak Wavelength vs Q (Epoch {epoch})")
    axes[0, 1].set_xlabel("Peak Wavelength (um)")
    axes[0, 1].set_ylabel("Q")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(mse_values, bins=30, color="seagreen", alpha=0.85, edgecolor="black")
    axes[1, 0].set_title(f"Lorentzian MSE Distribution (Epoch {epoch})")
    axes[1, 0].set_xlabel("MSE")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(peak_wavelengths, mse_values, s=10, alpha=0.65, color="crimson")
    axes[1, 1].set_title(f"Peak Wavelength vs Lorentzian MSE (Epoch {epoch})")
    axes[1, 1].set_xlabel("Peak Wavelength (um)")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        (
            f"Epoch {epoch} Q/MSE Evaluation | mean_q={summary['mean_q']:.2f}, "
            f"mean_mse={summary['mean_mse']:.6f}, valid={summary['valid_ratio'] * 100:.1f}%"
        ),
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(save_dir, f"q_mse_distribution_epoch_{epoch}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Q/MSE evaluation detail CSV saved to: {details_path}")
    print(f"Q/MSE evaluation distribution plot saved to: {plot_path}")


def save_q_evaluation_history(history, save_dir):
    """Save cross-epoch Q and MSE summary CSV and plots."""
    if not history:
        return

    os.makedirs(save_dir, exist_ok=True)
    history_df = pd.DataFrame(history)

    summary_path = os.path.join(save_dir, "q_mse_evaluation_summary.csv")
    history_df.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    axes[0, 0].plot(history_df["epoch"], history_df["mean_q"], marker="o", linewidth=2, label="Mean Q")
    axes[0, 0].plot(history_df["epoch"], history_df["median_q"], marker="s", linewidth=2, label="Median Q")
    axes[0, 0].plot(history_df["epoch"], history_df["max_q"], marker="^", linewidth=2, label="Max Q")
    axes[0, 0].set_ylabel("Q")
    axes[0, 0].set_title("Q Statistics During Training")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        history_df["epoch"],
        history_df["valid_ratio"] * 100.0,
        marker="o",
        linewidth=2,
        color="tab:green",
    )
    axes[0, 1].set_ylabel("Valid Ratio (%)")
    axes[0, 1].set_title("Share of Samples with Valid Half-Max Crossings")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history_df["epoch"], history_df["mean_mse"], marker="o", linewidth=2, label="Mean MSE")
    axes[1, 0].plot(history_df["epoch"], history_df["median_mse"], marker="s", linewidth=2, label="Median MSE")
    axes[1, 0].plot(history_df["epoch"], history_df["min_mse"], marker="^", linewidth=2, label="Min MSE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_title("Peak-Aligned Lorentzian MSE During Training")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(history_df["epoch"], history_df["max_mse"], marker="o", linewidth=2, label="Max MSE")
    axes[1, 1].plot(history_df["epoch"], history_df["std_mse"], marker="s", linewidth=2, label="Std MSE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_title("MSE Spread During Training")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "q_mse_evaluation_curves.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Q/MSE evaluation summary CSV saved to: {summary_path}")
    print(f"Q/MSE evaluation summary plot saved to: {plot_path}")


def evaluate_generator_q(generator, params, device, alpha, epoch, save_dir, high_quality_dir=None):
    """Generate samples in batches, compute Q and MSE in parallel, and save summary artifacts."""
    num_samples = max(1, int(getattr(params, "q_eval_num_samples", 1000)))
    batch_size = max(1, min(int(getattr(params, "batch_size", num_samples)), num_samples))
    wavelengths = (2 * torch.pi / params.k.to(device)).float()
    high_quality_criteria = build_high_quality_criteria(params)
    fixed_thickness_noise = getattr(params, "fixed_q_eval_thickness_noise", None)
    fixed_material_noise = getattr(params, "fixed_q_eval_material_noise", None)

    collected_results = []
    high_quality_records = []
    generator_was_training = generator.training
    generator.eval()

    if high_quality_criteria["enabled"] and high_quality_dir is not None:
        initialize_high_quality_collection(high_quality_dir, high_quality_criteria)

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - start)
            if fixed_thickness_noise is not None and fixed_material_noise is not None:
                thickness_noise = fixed_thickness_noise[start : start + current_batch].to(device=device)
                material_noise = fixed_material_noise[start : start + current_batch].to(device=device)
            else:
                thickness_noise = torch.randn(current_batch, params.thickness_noise_dim, device=device)
                material_noise = torch.randn(current_batch, params.material_noise_dim, device=device)

            thicknesses, refractive_indices, material_probabilities = generator(thickness_noise, material_noise, alpha)
            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
            reflection = torch.nan_to_num(reflection, nan=1.0, posinf=1.0, neginf=1.0)
            absorption = (1 - reflection).float()

            batch_results = compute_q_mse_metrics_torch(
                wavelengths,
                absorption,
                lorentz_width=float(getattr(params, "lorentz_width", 0.02)),
            )
            collected_results.append(batch_results)

            if high_quality_criteria["enabled"] and high_quality_dir is not None:
                high_quality_records.extend(
                    collect_high_quality_solutions_batch(
                        wavelengths=wavelengths,
                        absorption_spectra=absorption,
                        thicknesses=thicknesses,
                        material_probabilities=material_probabilities,
                        q_mse_results=batch_results,
                        params=params,
                        epoch=epoch,
                        alpha=alpha,
                        sample_offset=start,
                        save_dir=high_quality_dir,
                    )
                )

    if generator_was_training:
        generator.train()

    merged_results = {
        key: torch.cat([result[key].detach().cpu() for result in collected_results], dim=0)
        for key in collected_results[0]
    }
    summary = summarize_q_results(
        merged_results,
        epoch=epoch,
        alpha=alpha,
        num_samples=num_samples,
        lorentz_width=float(getattr(params, "lorentz_width", 0.02)),
    )
    collection_summary = {"new_high_quality_count": 0, "total_high_quality_count": 0}
    if high_quality_criteria["enabled"] and high_quality_dir is not None:
        collection_summary = update_high_quality_collection_summary(high_quality_dir, high_quality_records)

    summary["high_quality_count"] = int(collection_summary["new_high_quality_count"])
    summary["total_high_quality_count"] = int(collection_summary["total_high_quality_count"])
    save_q_evaluation_epoch(merged_results, summary, save_dir)
    return summary
