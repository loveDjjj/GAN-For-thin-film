"""Training-time Q and Lorentzian-MSE evaluation helpers."""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from model.TMM.optical_calculator import calculate_reflection
from train.high_quality_solution_collector import (
    build_high_quality_criteria,
    build_merged_layers,
    build_original_layers,
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


def compute_q_factors_in_window_torch(wavelengths, absorption_spectra, center, half_window, eps=1e-12):
    """Compute Q only around a target window."""
    if absorption_spectra.ndim != 2:
        raise ValueError("absorption_spectra must have shape [batch_size, num_wavelengths]")

    device = absorption_spectra.device
    wavelengths = wavelengths.to(device=device, dtype=absorption_spectra.dtype).flatten()
    absorption_spectra = absorption_spectra.to(dtype=wavelengths.dtype)

    window_mask = (wavelengths >= center - half_window) & (wavelengths <= center + half_window)
    if not window_mask.any():
        empty = torch.zeros(absorption_spectra.shape[0], device=device, dtype=absorption_spectra.dtype)
        nan = torch.full_like(empty, float("nan"))
        return {
            "q_values": empty,
            "valid_mask": torch.zeros(absorption_spectra.shape[0], dtype=torch.bool, device=device),
            "peak_wavelengths": nan,
            "peak_absorptions": empty,
            "left_wavelengths": nan,
            "right_wavelengths": nan,
            "fwhm": nan,
        }

    masked_absorption = torch.where(window_mask.unsqueeze(0), absorption_spectra, torch.full_like(absorption_spectra, -torch.inf))
    peak_indices = torch.argmax(masked_absorption, dim=1)
    peak_absorptions = absorption_spectra.gather(1, peak_indices.unsqueeze(1)).squeeze(1)
    peak_wavelengths = wavelengths[peak_indices]
    half_max = peak_absorptions * 0.5

    batch_size, num_wavelengths = absorption_spectra.shape
    indices = torch.arange(num_wavelengths, device=device).unsqueeze(0).expand(batch_size, -1)

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
        q_values[valid_rows] = torch.where(
            current_valid,
            peak_wavelengths[valid_rows] / current_fwhm.clamp_min(eps),
            torch.zeros_like(current_fwhm),
        )
        valid_mask[valid_rows] = current_valid

    q_values = torch.nan_to_num(q_values, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "q_values": q_values,
        "valid_mask": valid_mask,
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


def compute_peak_lorentzian_mse_torch(
    wavelengths,
    absorption_spectra,
    peak_wavelengths,
    width,
    mse_key="mse_values",
    rmse_key="rmse_values",
):
    """Compute per-sample MSE/RMSE to a peak-aligned Lorentzian target in parallel."""
    target_curves = generate_peak_aligned_lorentzian_curves_torch(wavelengths, peak_wavelengths, width)
    mse_values = torch.mean((absorption_spectra - target_curves) ** 2, dim=1)
    mse_values = torch.nan_to_num(mse_values, nan=0.0, posinf=0.0, neginf=0.0)
    rmse_values = torch.sqrt(mse_values.clamp_min(0.0))
    return {
        mse_key: mse_values,
        rmse_key: rmse_values,
    }


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


def compute_dual_q_metrics_torch(
    wavelengths,
    absorption_spectra,
    target_center_1,
    target_center_2,
    half_window,
    half_window_2=None,
    eps=1e-12,
):
    """Compute dual-window Q metrics and aliases for downstream consumers."""
    if half_window_2 is None:
        half_window_2 = half_window

    first = compute_q_factors_in_window_torch(
        wavelengths,
        absorption_spectra,
        center=target_center_1,
        half_window=half_window,
        eps=eps,
    )
    second = compute_q_factors_in_window_torch(
        wavelengths,
        absorption_spectra,
        center=target_center_2,
        half_window=half_window_2,
        eps=eps,
    )

    q_min_pair_values = torch.minimum(first["q_values"], second["q_values"])
    dual_valid_mask = first["valid_mask"] & second["valid_mask"]
    mean_peak_wavelengths = (first["peak_wavelengths"] + second["peak_wavelengths"]) * 0.5
    min_peak_absorptions = torch.minimum(first["peak_absorptions"], second["peak_absorptions"])
    max_fwhm = torch.maximum(first["fwhm"], second["fwhm"])

    return {
        "q1_values": first["q_values"],
        "q2_values": second["q_values"],
        "q_min_pair_values": q_min_pair_values,
        "dual_valid_mask": dual_valid_mask,
        "peak_wavelengths_1": first["peak_wavelengths"],
        "peak_wavelengths_2": second["peak_wavelengths"],
        "peak_absorptions_1": first["peak_absorptions"],
        "peak_absorptions_2": second["peak_absorptions"],
        "left_wavelengths_1": first["left_wavelengths"],
        "left_wavelengths_2": second["left_wavelengths"],
        "right_wavelengths_1": first["right_wavelengths"],
        "right_wavelengths_2": second["right_wavelengths"],
        "fwhm_1": first["fwhm"],
        "fwhm_2": second["fwhm"],
        # Compatibility aliases for existing downstream plots/history.
        "q_values": q_min_pair_values,
        "valid_mask": dual_valid_mask,
        "peak_wavelengths": mean_peak_wavelengths,
        "peak_absorptions": min_peak_absorptions,
        "left_wavelengths": first["left_wavelengths"],
        "right_wavelengths": second["right_wavelengths"],
        "fwhm": max_fwhm,
    }


def compute_double_lorentzian_mse_torch(
    wavelengths,
    absorption_spectra,
    peak_wavelengths_1,
    peak_wavelengths_2,
    width,
    mse_key="double_lorentz_mse_values",
    rmse_key="double_lorentz_rmse_values",
):
    """Compute MSE/RMSE to a peak-aligned double Lorentzian target."""
    target_curves = generate_double_lorentzian_curves(
        wavelengths=wavelengths,
        width=width,
        centers1=peak_wavelengths_1,
        centers2=peak_wavelengths_2,
    )
    mse_values = torch.mean((absorption_spectra - target_curves) ** 2, dim=1)
    mse_values = torch.nan_to_num(mse_values, nan=0.0, posinf=0.0, neginf=0.0)
    rmse_values = torch.sqrt(mse_values.clamp_min(0.0))
    return {
        mse_key: mse_values,
        rmse_key: rmse_values,
    }


def compute_dual_fom_scores_torch(q_min_pair_values, rmse_values, dual_valid_mask, q_ref, rmse_ref, weight):
    """Compute FOM from the weaker of the two Q peaks and a double-target RMSE."""
    if q_ref <= 0:
        raise ValueError("fom_q_ref must be positive")
    if rmse_ref <= 0:
        raise ValueError("fom_rmse_ref must be positive")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("fom_weight must be between 0 and 1")

    q_min_pair_values = q_min_pair_values.to(dtype=rmse_values.dtype)
    dual_valid_mask = dual_valid_mask.to(dtype=rmse_values.dtype)
    q_ref_tensor = torch.as_tensor(q_ref, device=rmse_values.device, dtype=rmse_values.dtype)
    rmse_ref_tensor = torch.as_tensor(rmse_ref, device=rmse_values.device, dtype=rmse_values.dtype)
    ln2 = torch.as_tensor(math.log(2.0), device=rmse_values.device, dtype=rmse_values.dtype)

    q_score_values = 1.0 - torch.exp(-ln2 * q_min_pair_values / q_ref_tensor)
    rmse_score_values = torch.exp(-ln2 * rmse_values / rmse_ref_tensor)
    q_score_values = q_score_values.clamp(0.0, 1.0)
    rmse_score_values = rmse_score_values.clamp(0.0, 1.0)
    fom_values = dual_valid_mask * torch.pow(q_score_values, weight) * torch.pow(rmse_score_values, 1.0 - weight)
    fom_values = fom_values.clamp(0.0, 1.0)

    return {
        "rmse_values": rmse_values,
        "q_score_values": q_score_values,
        "rmse_score_values": rmse_score_values,
        "fom_values": fom_values,
    }


def compute_fom_scores_torch(q_values, rmse_values, valid_mask, q_ref, rmse_ref, weight):
    """Compute bounded sample-level FOM scores from Q and RMSE."""
    if q_ref <= 0:
        raise ValueError("fom_q_ref must be positive")
    if rmse_ref <= 0:
        raise ValueError("fom_rmse_ref must be positive")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("fom_weight must be between 0 and 1")

    q_values = q_values.to(dtype=rmse_values.dtype)
    valid_mask = valid_mask.to(dtype=rmse_values.dtype)
    q_ref_tensor = torch.as_tensor(q_ref, device=rmse_values.device, dtype=rmse_values.dtype)
    rmse_ref_tensor = torch.as_tensor(rmse_ref, device=rmse_values.device, dtype=rmse_values.dtype)
    ln2 = torch.as_tensor(math.log(2.0), device=rmse_values.device, dtype=rmse_values.dtype)

    q_score_values = 1.0 - torch.exp(-ln2 * q_values / q_ref_tensor)
    rmse_score_values = torch.exp(-ln2 * rmse_values / rmse_ref_tensor)

    q_score_values = q_score_values.clamp(0.0, 1.0)
    rmse_score_values = rmse_score_values.clamp(0.0, 1.0)
    fom_values = valid_mask * torch.pow(q_score_values, weight) * torch.pow(rmse_score_values, 1.0 - weight)
    fom_values = fom_values.clamp(0.0, 1.0)

    return {
        "rmse_values": rmse_values,
        "q_score_values": q_score_values,
        "rmse_score_values": rmse_score_values,
        "fom_values": fom_values,
    }


def compute_material_certainty_metrics_torch(material_probabilities, dominant_prob_threshold):
    """Compute per-sample material certainty metrics from generator probabilities."""
    if material_probabilities.ndim == 2:
        material_probabilities = material_probabilities.unsqueeze(0)
    if material_probabilities.ndim != 3:
        raise ValueError("material_probabilities must have shape [batch_size, num_layers, num_materials]")

    dominant_probabilities, dominant_material_indices = material_probabilities.max(dim=2)
    fixed_layer_mask = dominant_probabilities > dominant_prob_threshold
    fixed_layer_count = fixed_layer_mask.sum(dim=1)
    fixed_layer_ratio = fixed_layer_count.float() / max(material_probabilities.shape[1], 1)
    fully_fixed_mask = fixed_layer_mask.all(dim=1)

    return {
        "dominant_material_probabilities": dominant_probabilities,
        "dominant_material_indices": dominant_material_indices,
        "fixed_layer_mask": fixed_layer_mask,
        "fixed_layer_count": fixed_layer_count,
        "fixed_layer_ratio": fixed_layer_ratio,
        "fully_fixed_mask": fully_fixed_mask,
        "min_dominant_material_probability": dominant_probabilities.min(dim=1).values,
        "mean_dominant_material_probability": dominant_probabilities.mean(dim=1),
    }


def _center_range_to_target_window(center_range):
    if center_range is None or len(center_range) != 2:
        raise ValueError("center_range must contain exactly two values")
    center_min, center_max = float(center_range[0]), float(center_range[1])
    if center_min > center_max:
        center_min, center_max = center_max, center_min
    return (center_min + center_max) * 0.5, max((center_max - center_min) * 0.5, 1e-6)


def summarize_q_results(
    q_results,
    epoch,
    alpha,
    num_samples,
    lorentz_width,
    dominant_prob_threshold,
    fom_q_ref,
    fom_lorentz_width,
    fom_rmse_ref,
    fom_weight,
):
    """Compute summary statistics for a batch of Q/MSE/certainty results."""
    q_values = q_results["q_values"]
    mse_values = q_results["mse_values"]
    rmse_values = q_results["rmse_values"]
    q1_values = q_results.get("q1_values", q_values)
    q2_values = q_results.get("q2_values", q_values)
    q_min_pair_values = q_results.get("q_min_pair_values", q_values)
    dual_valid_mask = q_results.get("dual_valid_mask", q_results["valid_mask"])
    double_lorentz_mse_values = q_results.get("double_lorentz_mse_values", mse_values)
    double_lorentz_rmse_values = q_results.get("double_lorentz_rmse_values", rmse_values)
    fom_lorentz_mse_values = q_results["fom_lorentz_mse_values"]
    fom_lorentz_rmse_values = q_results["fom_lorentz_rmse_values"]
    fom_values = q_results["fom_values"]
    valid_mask = q_results["valid_mask"]
    fixed_layer_count = q_results["fixed_layer_count"].float()
    fixed_layer_ratio = q_results["fixed_layer_ratio"]
    fully_fixed_mask = q_results["fully_fixed_mask"]
    min_dominant_probabilities = q_results["min_dominant_material_probability"]
    mean_dominant_probabilities = q_results["mean_dominant_material_probability"]
    dominant_material_probabilities = q_results["dominant_material_probabilities"]
    fixed_layer_mask = q_results["fixed_layer_mask"]

    summary = {
        "epoch": int(epoch),
        "alpha": float(alpha),
        "num_samples": int(num_samples),
        "lorentz_width": float(lorentz_width),
        "dominant_material_prob_threshold": float(dominant_prob_threshold),
        "fom_q_ref": float(fom_q_ref),
        "fom_lorentz_width": float(fom_lorentz_width),
        "fom_rmse_ref": float(fom_rmse_ref),
        "fom_weight": float(fom_weight),
        "valid_count": int(valid_mask.sum().item()),
        "valid_ratio": float(dual_valid_mask.float().mean().item()),
        "dual_valid_ratio": float(dual_valid_mask.float().mean().item()),
        "mean_q1": float(q1_values.mean().item()),
        "mean_q2": float(q2_values.mean().item()),
        "mean_q_min_pair": float(q_min_pair_values.mean().item()),
        "median_q_min_pair": float(q_min_pair_values.median().item()),
        "mean_q": float(q_values.mean().item()),
        "median_q": float(q_values.median().item()),
        "max_q": float(q_values.max().item()),
        "std_q": float(q_values.std(unbiased=False).item()),
        "mean_double_mse": float(double_lorentz_mse_values.mean().item()),
        "median_double_mse": float(double_lorentz_mse_values.median().item()),
        "mean_double_rmse": float(double_lorentz_rmse_values.mean().item()),
        "median_double_rmse": float(double_lorentz_rmse_values.median().item()),
        "mean_mse": float(mse_values.mean().item()),
        "median_mse": float(mse_values.median().item()),
        "min_mse": float(mse_values.min().item()),
        "max_mse": float(mse_values.max().item()),
        "std_mse": float(mse_values.std(unbiased=False).item()),
        "mean_rmse": float(rmse_values.mean().item()),
        "median_rmse": float(rmse_values.median().item()),
        "min_rmse": float(rmse_values.min().item()),
        "max_rmse": float(rmse_values.max().item()),
        "mean_fom_lorentz_mse": float(fom_lorentz_mse_values.mean().item()),
        "median_fom_lorentz_mse": float(fom_lorentz_mse_values.median().item()),
        "mean_fom_lorentz_rmse": float(fom_lorentz_rmse_values.mean().item()),
        "median_fom_lorentz_rmse": float(fom_lorentz_rmse_values.median().item()),
        "mean_fom": float(fom_values.mean().item()),
        "median_fom": float(fom_values.median().item()),
        "epoch_best_fom": float(fom_values.max().item()),
        "fully_fixed_count": int(fully_fixed_mask.sum().item()),
        "fully_fixed_ratio": float(fully_fixed_mask.float().mean().item()),
        "mean_fixed_layer_count": float(fixed_layer_count.mean().item()),
        "median_fixed_layer_count": float(fixed_layer_count.median().item()),
        "mean_fixed_layer_ratio": float(fixed_layer_ratio.mean().item()),
        "mean_min_dominant_material_probability": float(min_dominant_probabilities.mean().item()),
        "median_min_dominant_material_probability": float(min_dominant_probabilities.median().item()),
        "mean_mean_dominant_material_probability": float(mean_dominant_probabilities.mean().item()),
        "median_mean_dominant_material_probability": float(mean_dominant_probabilities.median().item()),
    }

    for layer_index in range(dominant_material_probabilities.shape[1]):
        layer_probabilities = dominant_material_probabilities[:, layer_index]
        layer_fixed = fixed_layer_mask[:, layer_index].float()
        summary[f"layer_{layer_index + 1}_mean_dominant_probability"] = float(layer_probabilities.mean().item())
        summary[f"layer_{layer_index + 1}_fixed_ratio"] = float(layer_fixed.mean().item())

    return summary


def _build_layer_summary_dataframe(q_results, materials):
    """Build layer-level certainty summary rows for one evaluation epoch."""
    dominant_material_probabilities = q_results["dominant_material_probabilities"].detach().cpu()
    dominant_material_indices = q_results["dominant_material_indices"].detach().cpu()
    fixed_layer_mask = q_results["fixed_layer_mask"].detach().cpu()

    rows = []
    for layer_index in range(dominant_material_probabilities.shape[1]):
        layer_probabilities = dominant_material_probabilities[:, layer_index]
        layer_fixed = fixed_layer_mask[:, layer_index]
        layer_indices = dominant_material_indices[:, layer_index]
        row = {
            "layer_index": layer_index + 1,
            "mean_dominant_probability": float(layer_probabilities.mean().item()),
            "median_dominant_probability": float(layer_probabilities.median().item()),
            "min_dominant_probability": float(layer_probabilities.min().item()),
            "max_dominant_probability": float(layer_probabilities.max().item()),
            "fixed_ratio": float(layer_fixed.float().mean().item()),
            "fixed_count": int(layer_fixed.sum().item()),
        }
        for material_index, material_name in enumerate(materials):
            dominant_mask = layer_indices == material_index
            row[f"dominant_count_{material_name}"] = int(dominant_mask.sum().item())
            row[f"dominant_ratio_{material_name}"] = float(dominant_mask.float().mean().item())
        rows.append(row)

    return pd.DataFrame(rows)


def _get_previous_global_best(save_dir, tracked_metric_name):
    """Read the previous global-best value from the history summary if available."""
    summary_path = os.path.join(save_dir, "q_mse_evaluation_summary.csv")
    if not os.path.exists(summary_path):
        return None

    history_df = pd.read_csv(summary_path)
    if history_df.empty:
        return None

    if tracked_metric_name in history_df.columns:
        return float(history_df[tracked_metric_name].max())

    fallback_columns = {
        "global_max_q": "max_q",
        "global_best_fom": "epoch_best_fom",
    }
    fallback_column = fallback_columns.get(tracked_metric_name)
    if fallback_column and fallback_column in history_df.columns:
        return float(history_df[fallback_column].max())

    return None


def _format_material_probabilities(material_probabilities):
    """Format one layer's material probabilities as compact text."""
    return ", ".join(
        f"{material_name}={probability:.6f}"
        for material_name, probability in material_probabilities.items()
    )


def _save_global_best_spectrum_plot(output_path, wavelengths, absorption, tracked_metric_name, tracked_metric_value, epoch):
    """Save the raw absorption spectrum plot for a global-best update."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(wavelengths.numpy(), absorption.numpy(), linewidth=2, color="tab:blue")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Absorption")
    ax.set_title(f"{tracked_metric_name} update at epoch {epoch} | value={tracked_metric_value:.6f}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(1.05, float(absorption.max().item()) * 1.05))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_global_best_sample_update(
    tracked_metric_name,
    tracked_metric_value,
    previous_tracked_metric_value,
    sample_index,
    summary,
    q_results,
    wavelengths,
    absorption_spectra,
    thicknesses,
    material_probabilities,
    params,
    save_dir,
):
    """Save one global-best structure txt and raw-spectrum csv/png for a refreshed best sample."""
    tracking_dir = os.path.join(save_dir, "global_best_samples")
    os.makedirs(tracking_dir, exist_ok=True)

    file_stem = f"{tracked_metric_name}_epoch_{int(summary['epoch']):04d}"
    structure_txt_path = os.path.join(tracking_dir, f"{file_stem}_structure.txt")
    spectra_csv_path = os.path.join(tracking_dir, f"{file_stem}_spectrum.csv")
    spectrum_plot_path = os.path.join(tracking_dir, f"{file_stem}_spectrum.png")
    sample_id = f"epoch_{int(summary['epoch']):04d}_sample_{int(sample_index):05d}"
    sample_thicknesses = thicknesses[sample_index].detach().cpu()
    sample_material_probs = material_probabilities[sample_index].detach().cpu()
    sample_absorption = absorption_spectra[sample_index].detach().cpu()

    original_layers = build_original_layers(sample_thicknesses, sample_material_probs, params.materials)
    merged_layers = build_merged_layers(sample_thicknesses, sample_material_probs, params.materials)
    total_thickness_um = float(sample_thicknesses.sum().item())

    wavelengths_cpu = wavelengths.detach().cpu()
    spectra_df = pd.DataFrame(
        {
            "wavelength_um": wavelengths_cpu.numpy(),
            "absorption": sample_absorption.numpy(),
        }
    )
    spectra_df.to_csv(spectra_csv_path, index=False)

    _save_global_best_spectrum_plot(
        output_path=spectrum_plot_path,
        wavelengths=wavelengths_cpu,
        absorption=sample_absorption,
        tracked_metric_name=tracked_metric_name,
        tracked_metric_value=tracked_metric_value,
        epoch=int(summary["epoch"]),
    )

    previous_value_text = (
        f"{previous_tracked_metric_value:.6f}"
        if previous_tracked_metric_value is not None
        else "N/A"
    )
    lines = [
        f"tracked_metric: {tracked_metric_name}",
        f"tracked_metric_value: {tracked_metric_value:.6f}",
        f"previous_tracked_metric_value: {previous_value_text}",
        f"epoch: {int(summary['epoch'])}",
        f"alpha: {float(summary['alpha']):.6f}",
        f"sample_id: {sample_id}",
        f"evaluation_sample_index: {int(sample_index)}",
        f"fom_lorentz_width: {float(getattr(params, 'q_eval_fom_lorentz_width', getattr(params, 'lorentz_width', 0.02))):.6f}",
        f"q_value: {float(q_results['q_values'][sample_index].item()):.6f}",
        f"q1: {float(q_results['q1_values'][sample_index].item()):.6f}",
        f"q2: {float(q_results['q2_values'][sample_index].item()):.6f}",
        f"q_min_pair: {float(q_results['q_min_pair_values'][sample_index].item()):.6f}",
        f"lorentz_mse: {float(q_results['mse_values'][sample_index].item()):.8f}",
        f"double_lorentz_mse: {float(q_results['double_lorentz_mse_values'][sample_index].item()):.8f}",
        f"lorentz_rmse: {float(q_results['rmse_values'][sample_index].item()):.8f}",
        f"double_lorentz_rmse: {float(q_results['double_lorentz_rmse_values'][sample_index].item()):.8f}",
        f"fom_lorentz_mse: {float(q_results['fom_lorentz_mse_values'][sample_index].item()):.8f}",
        f"fom_lorentz_rmse: {float(q_results['fom_lorentz_rmse_values'][sample_index].item()):.8f}",
        f"q_score: {float(q_results['q_score_values'][sample_index].item()):.8f}",
        f"rmse_score: {float(q_results['rmse_score_values'][sample_index].item()):.8f}",
        f"fom: {float(q_results['fom_values'][sample_index].item()):.8f}",
        f"peak_wavelength_um: {float(q_results['peak_wavelengths'][sample_index].item()):.6f}",
        f"peak_wavelength_1_um: {float(q_results['peak_wavelengths_1'][sample_index].item()):.6f}",
        f"peak_wavelength_2_um: {float(q_results['peak_wavelengths_2'][sample_index].item()):.6f}",
        f"peak_absorption: {float(q_results['peak_absorptions'][sample_index].item()):.6f}",
        f"peak_absorption_1: {float(q_results['peak_absorptions_1'][sample_index].item()):.6f}",
        f"peak_absorption_2: {float(q_results['peak_absorptions_2'][sample_index].item()):.6f}",
        f"fwhm_um: {float(q_results['fwhm'][sample_index].item()):.6f}",
        f"fwhm_1_um: {float(q_results['fwhm_1'][sample_index].item()):.6f}",
        f"fwhm_2_um: {float(q_results['fwhm_2'][sample_index].item()):.6f}",
        f"total_thickness_um: {total_thickness_um:.6f}",
        f"total_thickness_nm: {total_thickness_um * 1000.0:.3f}",
        "",
        "Original layers:",
    ]
    for layer_record in original_layers:
        lines.append(
            (
                f"Layer {int(layer_record['layer_index'])}: "
                f"material={layer_record['dominant_material']}, "
                f"thickness_um={float(layer_record['thickness_um']):.6f}, "
                f"thickness_nm={float(layer_record['thickness_um']) * 1000.0:.3f}, "
                f"dominant_probability={float(layer_record['dominant_probability']):.6f}"
            )
        )
        lines.append(
            f"  material_probabilities: {_format_material_probabilities(layer_record['material_probabilities'])}"
        )

    lines.append("")
    lines.append("Merged layers:")
    for merged_layer in merged_layers:
        lines.append(
            (
                f"Merged layer {int(merged_layer['merged_layer_index'])}: "
                f"material={merged_layer['material']}, "
                f"original_layers={int(merged_layer['start_layer'])}-{int(merged_layer['end_layer'])}, "
                f"merged_thickness_um={float(merged_layer['merged_thickness_um']):.6f}, "
                f"merged_thickness_nm={float(merged_layer['merged_thickness_um']) * 1000.0:.3f}, "
                f"mean_dominant_probability={float(merged_layer['mean_dominant_probability']):.6f}"
            )
        )

    with open(structure_txt_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(
        f"Tracked {tracked_metric_name} update saved: "
        f"epoch={summary['epoch']}, sample={sample_index}, value={tracked_metric_value:.6f}"
    )


def save_global_best_sample_histories(summary, q_results, wavelengths, absorption_spectra, thicknesses, material_probabilities, params, save_dir):
    """Save per-update files when global max Q or global best FOM improves."""
    tracking_dir = os.path.join(save_dir, "global_best_samples")
    os.makedirs(tracking_dir, exist_ok=True)

    metric_specs = (
        ("global_max_q", "q_values"),
        ("global_best_fom", "fom_values"),
    )
    for tracked_metric_name, tensor_key in metric_specs:
        metric_tensor = q_results[tensor_key]
        sample_index = int(torch.argmax(metric_tensor).item())
        tracked_metric_value = float(metric_tensor[sample_index].item())
        previous_value = _get_previous_global_best(save_dir, tracked_metric_name)

        if previous_value is None or tracked_metric_value > previous_value:
            _save_global_best_sample_update(
                tracked_metric_name=tracked_metric_name,
                tracked_metric_value=tracked_metric_value,
                previous_tracked_metric_value=previous_value,
                sample_index=sample_index,
                summary=summary,
                q_results=q_results,
                wavelengths=wavelengths,
                absorption_spectra=absorption_spectra,
                thicknesses=thicknesses,
                material_probabilities=material_probabilities,
                params=params,
                save_dir=save_dir,
            )


def save_q_evaluation_epoch(q_results, summary, save_dir, materials):
    """Save per-epoch Q/MSE/certainty statistics and plots."""
    os.makedirs(save_dir, exist_ok=True)

    epoch = summary["epoch"]
    q_values = q_results["q_values"].detach().cpu().numpy()
    mse_values = q_results["mse_values"].detach().cpu().numpy()
    rmse_values = q_results["rmse_values"].detach().cpu().numpy()
    fom_lorentz_mse_values = q_results["fom_lorentz_mse_values"].detach().cpu().numpy()
    fom_lorentz_rmse_values = q_results["fom_lorentz_rmse_values"].detach().cpu().numpy()
    q_score_values = q_results["q_score_values"].detach().cpu().numpy()
    rmse_score_values = q_results["rmse_score_values"].detach().cpu().numpy()
    fom_values = q_results["fom_values"].detach().cpu().numpy()
    valid_mask = q_results["valid_mask"].detach().cpu().numpy()
    peak_wavelengths = q_results["peak_wavelengths"].detach().cpu().numpy()
    peak_absorptions = q_results["peak_absorptions"].detach().cpu().numpy()
    left_wavelengths = q_results["left_wavelengths"].detach().cpu().numpy()
    right_wavelengths = q_results["right_wavelengths"].detach().cpu().numpy()
    fwhm = q_results["fwhm"].detach().cpu().numpy()
    fixed_layer_count = q_results["fixed_layer_count"].detach().cpu().numpy()
    fixed_layer_ratio = q_results["fixed_layer_ratio"].detach().cpu().numpy()
    fully_fixed_mask = q_results["fully_fixed_mask"].detach().cpu().numpy()
    min_dominant_probabilities = q_results["min_dominant_material_probability"].detach().cpu().numpy()
    mean_dominant_probabilities = q_results["mean_dominant_material_probability"].detach().cpu().numpy()
    dominant_material_probabilities = q_results["dominant_material_probabilities"].detach().cpu().numpy()
    dominant_material_indices = q_results["dominant_material_indices"].detach().cpu().numpy()
    fixed_layer_mask = q_results["fixed_layer_mask"].detach().cpu().numpy()

    details_df = pd.DataFrame(
        {
            "sample_index": range(len(q_values)),
            "q_value": q_values,
            "q1": q_results["q1_values"].detach().cpu().numpy(),
            "q2": q_results["q2_values"].detach().cpu().numpy(),
            "q_min_pair": q_results["q_min_pair_values"].detach().cpu().numpy(),
            "lorentz_mse": mse_values,
            "double_lorentz_mse": q_results["double_lorentz_mse_values"].detach().cpu().numpy(),
            "lorentz_rmse": rmse_values,
            "double_lorentz_rmse": q_results["double_lorentz_rmse_values"].detach().cpu().numpy(),
            "fom_lorentz_mse": fom_lorentz_mse_values,
            "fom_lorentz_rmse": fom_lorentz_rmse_values,
            "q_score": q_score_values,
            "rmse_score": rmse_score_values,
            "fom": fom_values,
            "valid": valid_mask,
            "dual_valid": q_results["dual_valid_mask"].detach().cpu().numpy(),
            "peak_wavelength_um": peak_wavelengths,
            "peak_wavelength_1_um": q_results["peak_wavelengths_1"].detach().cpu().numpy(),
            "peak_wavelength_2_um": q_results["peak_wavelengths_2"].detach().cpu().numpy(),
            "peak_absorption": peak_absorptions,
            "peak_absorption_1": q_results["peak_absorptions_1"].detach().cpu().numpy(),
            "peak_absorption_2": q_results["peak_absorptions_2"].detach().cpu().numpy(),
            "left_half_max_um": left_wavelengths,
            "left_half_max_1_um": q_results["left_wavelengths_1"].detach().cpu().numpy(),
            "left_half_max_2_um": q_results["left_wavelengths_2"].detach().cpu().numpy(),
            "right_half_max_um": right_wavelengths,
            "right_half_max_1_um": q_results["right_wavelengths_1"].detach().cpu().numpy(),
            "right_half_max_2_um": q_results["right_wavelengths_2"].detach().cpu().numpy(),
            "fwhm_um": fwhm,
            "fwhm_1_um": q_results["fwhm_1"].detach().cpu().numpy(),
            "fwhm_2_um": q_results["fwhm_2"].detach().cpu().numpy(),
            "fixed_layer_count": fixed_layer_count,
            "fixed_layer_ratio": fixed_layer_ratio,
            "is_fully_fixed": fully_fixed_mask,
            "min_dominant_material_probability": min_dominant_probabilities,
            "mean_dominant_material_probability": mean_dominant_probabilities,
        }
    )
    for layer_index in range(dominant_material_probabilities.shape[1]):
        layer_column_prefix = f"layer_{layer_index + 1}"
        layer_material_indices = dominant_material_indices[:, layer_index]
        details_df[f"{layer_column_prefix}_dominant_material_index"] = layer_material_indices
        details_df[f"{layer_column_prefix}_dominant_material"] = [
            (
                materials[int(material_index)]
                if 0 <= int(material_index) < len(materials)
                else f"material_{int(material_index)}"
            )
            for material_index in layer_material_indices
        ]
        details_df[f"{layer_column_prefix}_dominant_probability"] = dominant_material_probabilities[:, layer_index]
        details_df[f"{layer_column_prefix}_is_fixed"] = fixed_layer_mask[:, layer_index]

    details_path = os.path.join(save_dir, f"q_mse_metrics_epoch_{epoch}.csv")
    details_df.to_csv(details_path, index=False)

    layer_summary_df = _build_layer_summary_dataframe(q_results, materials)
    layer_summary_path = os.path.join(save_dir, f"material_certainty_layers_epoch_{epoch}.csv")
    layer_summary_df.to_csv(layer_summary_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    q1_values = q_results["q1_values"].detach().cpu().numpy()
    q2_values = q_results["q2_values"].detach().cpu().numpy()
    q_min_pair_values = q_results["q_min_pair_values"].detach().cpu().numpy()
    peak_wavelengths_1 = q_results["peak_wavelengths_1"].detach().cpu().numpy()
    peak_wavelengths_2 = q_results["peak_wavelengths_2"].detach().cpu().numpy()
    double_mse_values = q_results["double_lorentz_mse_values"].detach().cpu().numpy()

    axes[0, 0].hist(q_min_pair_values, bins=30, color="steelblue", alpha=0.85, edgecolor="black")
    axes[0, 0].set_title(f"Q_min_pair Distribution (Epoch {epoch})")
    axes[0, 0].set_xlabel("Q")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(peak_wavelengths_1, q1_values, s=10, alpha=0.65, color="darkorange", label="Peak 1 vs Q1")
    axes[0, 1].scatter(peak_wavelengths_2, q2_values, s=10, alpha=0.65, color="tab:blue", label="Peak 2 vs Q2")
    axes[0, 1].set_title(f"Target-Window Peak Wavelength vs Q (Epoch {epoch})")
    axes[0, 1].set_xlabel("Peak Wavelength (um)")
    axes[0, 1].set_ylabel("Q")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].hist(double_mse_values, bins=30, color="seagreen", alpha=0.85, edgecolor="black")
    axes[1, 0].set_title(f"Double-Lorentzian MSE Distribution (Epoch {epoch})")
    axes[1, 0].set_xlabel("MSE")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(peak_wavelengths, double_mse_values, s=10, alpha=0.65, color="crimson")
    axes[1, 1].set_title(f"Representative Peak Wavelength vs Double-Lorentzian MSE (Epoch {epoch})")
    axes[1, 1].set_xlabel("Peak Wavelength (um)")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        (
            f"Epoch {epoch} Dual-Peak Evaluation | mean_q_min_pair={summary['mean_q_min_pair']:.2f}, "
            f"mean_double_mse={summary['mean_double_mse']:.6f}, dual_valid={summary['dual_valid_ratio'] * 100:.1f}%"
        ),
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(save_dir, f"q_mse_distribution_epoch_{epoch}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    certainty_fig, certainty_axes = plt.subplots(2, 2, figsize=(14, 10))

    certainty_axes[0, 0].hist(
        min_dominant_probabilities,
        bins=30,
        color="slateblue",
        alpha=0.85,
        edgecolor="black",
    )
    certainty_axes[0, 0].axvline(
        summary["dominant_material_prob_threshold"],
        color="crimson",
        linestyle="--",
        linewidth=1.5,
    )
    certainty_axes[0, 0].set_title(f"Min Dominant Probability Distribution (Epoch {epoch})")
    certainty_axes[0, 0].set_xlabel("Min Dominant Probability")
    certainty_axes[0, 0].set_ylabel("Count")
    certainty_axes[0, 0].grid(True, alpha=0.3)

    certainty_axes[0, 1].hist(
        fixed_layer_count,
        bins=range(dominant_material_probabilities.shape[1] + 2),
        align="left",
        color="teal",
        alpha=0.85,
        edgecolor="black",
        rwidth=0.85,
    )
    certainty_axes[0, 1].set_title(f"Fixed Layer Count per Structure (Epoch {epoch})")
    certainty_axes[0, 1].set_xlabel("Fixed Layer Count")
    certainty_axes[0, 1].set_ylabel("Count")
    certainty_axes[0, 1].set_xticks(range(dominant_material_probabilities.shape[1] + 1))
    certainty_axes[0, 1].grid(True, axis="y", alpha=0.3)

    certainty_axes[1, 0].bar(
        layer_summary_df["layer_index"].astype(str),
        layer_summary_df["fixed_ratio"] * 100.0,
        color="darkorange",
        alpha=0.9,
    )
    certainty_axes[1, 0].set_title("Per-Layer Fixed Ratio")
    certainty_axes[1, 0].set_xlabel("Layer")
    certainty_axes[1, 0].set_ylabel("Fixed Ratio (%)")
    certainty_axes[1, 0].set_ylim(0, 100)
    certainty_axes[1, 0].grid(True, axis="y", alpha=0.3)

    scatter = certainty_axes[1, 1].scatter(
        min_dominant_probabilities,
        q_values,
        c=mse_values,
        cmap="viridis_r",
        s=14,
        alpha=0.7,
    )
    certainty_axes[1, 1].axvline(
        summary["dominant_material_prob_threshold"],
        color="crimson",
        linestyle="--",
        linewidth=1.5,
    )
    certainty_axes[1, 1].set_title("Min Dominant Probability vs Q")
    certainty_axes[1, 1].set_xlabel("Min Dominant Probability")
    certainty_axes[1, 1].set_ylabel("Q")
    certainty_axes[1, 1].grid(True, alpha=0.3)
    certainty_colorbar = certainty_fig.colorbar(scatter, ax=certainty_axes[1, 1])
    certainty_colorbar.set_label("Double-Lorentzian MSE")

    certainty_fig.suptitle(
        (
            f"Epoch {epoch} Material Certainty | fully_fixed={summary['fully_fixed_ratio'] * 100:.1f}%, "
            f"mean_fixed_layers={summary['mean_fixed_layer_count']:.2f}, "
            f"mean_min_prob={summary['mean_min_dominant_material_probability']:.4f}"
        ),
        fontsize=14,
    )
    certainty_fig.tight_layout(rect=[0, 0, 1, 0.96])

    certainty_plot_path = os.path.join(save_dir, f"material_certainty_epoch_{epoch}.png")
    certainty_fig.savefig(certainty_plot_path, dpi=300, bbox_inches="tight")
    plt.close(certainty_fig)

    print(f"Q/MSE evaluation detail CSV saved to: {details_path}")
    print(f"Q/MSE evaluation distribution plot saved to: {plot_path}")
    print(f"Material certainty layer CSV saved to: {layer_summary_path}")
    print(f"Material certainty plot saved to: {certainty_plot_path}")


def _save_single_metric_curve(curve_df, output_path, y_column, y_label, title, color):
    """Save a single-metric training curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve_df["epoch"], curve_df[y_column], marker="o", linewidth=2, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pick_history_column(history_df, *candidates):
    for candidate in candidates:
        if candidate in history_df.columns:
            return candidate
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def save_q_evaluation_history(history, save_dir):
    """Save cross-epoch Q/MSE/certainty summary CSV and plots."""
    if not history:
        return

    os.makedirs(save_dir, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df["global_max_q"] = history_df["max_q"].cummax()
    history_df["global_best_fom"] = history_df["epoch_best_fom"].cummax()

    summary_path = os.path.join(save_dir, "q_mse_evaluation_summary.csv")
    history_df.to_csv(summary_path, index=False)

    q_primary_mean_column = _pick_history_column(history_df, "mean_q_min_pair", "mean_q")
    q_primary_median_column = _pick_history_column(history_df, "median_q_min_pair", "median_q")
    valid_ratio_column = _pick_history_column(history_df, "dual_valid_ratio", "valid_ratio")
    mse_mean_column = _pick_history_column(history_df, "mean_double_mse", "mean_mse")
    mse_median_column = _pick_history_column(history_df, "median_double_mse", "median_mse")
    mse_min_column = _pick_history_column(history_df, "min_double_mse", "min_mse")
    mse_max_column = _pick_history_column(history_df, "max_double_mse", "max_mse")
    mse_std_column = _pick_history_column(history_df, "std_double_mse", "std_mse")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    if "mean_q1" in history_df.columns:
        axes[0, 0].plot(history_df["epoch"], history_df["mean_q1"], marker="o", linewidth=2, label="Mean Q1")
    if "mean_q2" in history_df.columns:
        axes[0, 0].plot(history_df["epoch"], history_df["mean_q2"], marker="s", linewidth=2, label="Mean Q2")
    axes[0, 0].plot(history_df["epoch"], history_df[q_primary_mean_column], marker="^", linewidth=2, label="Mean Q_min_pair")
    axes[0, 0].plot(history_df["epoch"], history_df["max_q"], marker="d", linewidth=2, label="Max Q_min_pair")
    axes[0, 0].set_ylabel("Q")
    axes[0, 0].set_title("Dual-Peak Q Statistics During Training")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        history_df["epoch"],
        history_df[valid_ratio_column] * 100.0,
        marker="o",
        linewidth=2,
        color="tab:green",
    )
    axes[0, 1].set_ylabel("Valid Ratio (%)")
    axes[0, 1].set_title("Share of Samples with Valid Dual Peaks")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history_df["epoch"], history_df[mse_mean_column], marker="o", linewidth=2, label="Mean Double MSE")
    axes[1, 0].plot(history_df["epoch"], history_df[mse_median_column], marker="s", linewidth=2, label="Median Double MSE")
    axes[1, 0].plot(history_df["epoch"], history_df[mse_min_column], marker="^", linewidth=2, label="Min Double MSE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_title("Double-Lorentzian MSE During Training")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(history_df["epoch"], history_df[mse_max_column], marker="o", linewidth=2, label="Max Double MSE")
    axes[1, 1].plot(history_df["epoch"], history_df[mse_std_column], marker="s", linewidth=2, label="Std Double MSE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_title("Double-MSE Spread During Training")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "q_mse_evaluation_curves.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    global_max_q_curve_df = history_df[["epoch", "max_q", "global_max_q"]].rename(
        columns={
            "max_q": "epoch_max_q",
        }
    )
    global_max_q_curve_path = os.path.join(save_dir, "global_max_q_curve.csv")
    global_max_q_curve_df.to_csv(global_max_q_curve_path, index=False)
    global_max_q_plot_path = os.path.join(save_dir, "global_max_q_curve.png")
    _save_single_metric_curve(
        curve_df=global_max_q_curve_df,
        output_path=global_max_q_plot_path,
        y_column="global_max_q",
        y_label="Global Max Q_min_pair",
        title="Global Max Q_min_pair During Training",
        color="tab:green",
    )

    global_best_fom_curve_df = history_df[["epoch", "epoch_best_fom", "global_best_fom"]]
    global_best_fom_curve_path = os.path.join(save_dir, "global_best_fom_curve.csv")
    global_best_fom_curve_df.to_csv(global_best_fom_curve_path, index=False)
    global_best_fom_plot_path = os.path.join(save_dir, "global_best_fom_curve.png")
    _save_single_metric_curve(
        curve_df=global_best_fom_curve_df,
        output_path=global_best_fom_plot_path,
        y_column="global_best_fom",
        y_label="Global Best FOM",
        title="Global Best FOM During Training",
        color="tab:purple",
    )

    layer_ratio_columns = [
        column
        for column in history_df.columns
        if column.startswith("layer_") and column.endswith("_fixed_ratio")
    ]
    layer_ratio_columns = sorted(layer_ratio_columns, key=lambda name: int(name.split("_")[1]))
    layer_mean_prob_columns = [
        column
        for column in history_df.columns
        if column.startswith("layer_") and column.endswith("_mean_dominant_probability")
    ]
    layer_mean_prob_columns = sorted(layer_mean_prob_columns, key=lambda name: int(name.split("_")[1]))

    layer_history_records = []
    for _, row in history_df.iterrows():
        layer_count = min(len(layer_ratio_columns), len(layer_mean_prob_columns))
        for layer_index in range(layer_count):
            layer_history_records.append(
                {
                    "epoch": int(row["epoch"]),
                    "layer_index": layer_index + 1,
                    "fixed_ratio": float(row[layer_ratio_columns[layer_index]]),
                    "mean_dominant_probability": float(row[layer_mean_prob_columns[layer_index]]),
                }
            )
    layer_history_path = os.path.join(save_dir, "material_certainty_layer_history.csv")
    pd.DataFrame(layer_history_records).to_csv(layer_history_path, index=False)

    certainty_fig, certainty_axes = plt.subplots(2, 2, figsize=(14, 10))

    certainty_axes[0, 0].plot(
        history_df["epoch"],
        history_df["fully_fixed_ratio"] * 100.0,
        marker="o",
        linewidth=2,
        label="Fully Fixed Ratio",
    )
    certainty_axes[0, 0].plot(
        history_df["epoch"],
        history_df["mean_fixed_layer_ratio"] * 100.0,
        marker="s",
        linewidth=2,
        label="Mean Fixed Layer Ratio",
    )
    certainty_axes[0, 0].set_ylabel("Ratio (%)")
    certainty_axes[0, 0].set_title("Structure Fixedness During Training")
    certainty_axes[0, 0].grid(True, alpha=0.3)
    certainty_axes[0, 0].legend()

    certainty_axes[0, 1].plot(
        history_df["epoch"],
        history_df["mean_min_dominant_material_probability"],
        marker="o",
        linewidth=2,
        label="Mean Min Dominant Prob",
    )
    certainty_axes[0, 1].plot(
        history_df["epoch"],
        history_df["median_min_dominant_material_probability"],
        marker="s",
        linewidth=2,
        label="Median Min Dominant Prob",
    )
    certainty_axes[0, 1].axhline(
        history_df["dominant_material_prob_threshold"].iloc[0],
        color="crimson",
        linestyle="--",
        linewidth=1.4,
        label="Fixed Threshold",
    )
    certainty_axes[0, 1].set_ylabel("Probability")
    certainty_axes[0, 1].set_title("Dominant Probability Floor During Training")
    certainty_axes[0, 1].grid(True, alpha=0.3)
    certainty_axes[0, 1].legend()

    certainty_axes[1, 0].plot(
        history_df["epoch"],
        history_df["mean_fixed_layer_count"],
        marker="o",
        linewidth=2,
        label="Mean Fixed Layer Count",
    )
    certainty_axes[1, 0].plot(
        history_df["epoch"],
        history_df["median_fixed_layer_count"],
        marker="s",
        linewidth=2,
        label="Median Fixed Layer Count",
    )
    certainty_axes[1, 0].set_xlabel("Epoch")
    certainty_axes[1, 0].set_ylabel("Layer Count")
    certainty_axes[1, 0].set_title("Fixed Layer Count per Structure")
    certainty_axes[1, 0].grid(True, alpha=0.3)
    certainty_axes[1, 0].legend()

    if layer_ratio_columns:
        heatmap_data = (history_df[layer_ratio_columns] * 100.0).to_numpy()
        image = certainty_axes[1, 1].imshow(
            heatmap_data,
            aspect="auto",
            cmap="YlGnBu",
            origin="lower",
        )
        certainty_axes[1, 1].set_title("Per-Layer Fixed Ratio Heatmap")
        certainty_axes[1, 1].set_xlabel("Layer")
        certainty_axes[1, 1].set_ylabel("Evaluation Index")
        certainty_axes[1, 1].set_xticks(range(len(layer_ratio_columns)))
        certainty_axes[1, 1].set_xticklabels(
            [column.split("_")[1] for column in layer_ratio_columns]
        )
        certainty_axes[1, 1].set_yticks(range(len(history_df)))
        certainty_axes[1, 1].set_yticklabels(history_df["epoch"].astype(int).tolist())
        certainty_colorbar = certainty_fig.colorbar(image, ax=certainty_axes[1, 1])
        certainty_colorbar.set_label("Fixed Ratio (%)")
    else:
        certainty_axes[1, 1].set_axis_off()

    certainty_fig.tight_layout()
    certainty_plot_path = os.path.join(save_dir, "material_certainty_curves.png")
    certainty_fig.savefig(certainty_plot_path, dpi=300, bbox_inches="tight")
    plt.close(certainty_fig)

    print(f"Q/MSE evaluation summary CSV saved to: {summary_path}")
    print(f"Q/MSE evaluation summary plot saved to: {plot_path}")
    print(f"Global max Q curve CSV saved to: {global_max_q_curve_path}")
    print(f"Global max Q curve plot saved to: {global_max_q_plot_path}")
    print(f"Global best FOM curve CSV saved to: {global_best_fom_curve_path}")
    print(f"Global best FOM curve plot saved to: {global_best_fom_plot_path}")
    print(f"Material certainty layer history CSV saved to: {layer_history_path}")
    print(f"Material certainty summary plot saved to: {certainty_plot_path}")


def evaluate_generator_q(generator, params, device, alpha, epoch, save_dir, high_quality_dir=None):
    """Generate samples in batches, compute Q/MSE/certainty in parallel, and save summary artifacts."""
    num_samples = max(1, int(getattr(params, "q_eval_num_samples", 1000)))
    batch_size = max(1, min(int(getattr(params, "batch_size", num_samples)), num_samples))
    wavelengths = (2 * torch.pi / params.k.to(device)).float()
    high_quality_criteria = build_high_quality_criteria(params)
    fixed_thickness_noise = getattr(params, "fixed_q_eval_thickness_noise", None)
    fixed_material_noise = getattr(params, "fixed_q_eval_material_noise", None)
    dominant_prob_threshold = float(getattr(params, "q_eval_dominant_prob_threshold", 0.99))
    fom_q_ref = float(getattr(params, "q_eval_fom_q_ref", 200.0))
    fom_lorentz_width = float(getattr(params, "q_eval_fom_lorentz_width", getattr(params, "lorentz_width", 0.02)))
    fom_rmse_ref = float(getattr(params, "q_eval_fom_rmse_ref", 0.05))
    fom_weight = float(getattr(params, "q_eval_fom_weight", 0.5))
    target_center_1, half_window_1 = _center_range_to_target_window(getattr(params, "lorentz_center_range_1", None))
    target_center_2, half_window_2 = _center_range_to_target_window(getattr(params, "lorentz_center_range_2", None))

    collected_results = []
    collected_absorption_spectra = []
    collected_thicknesses = []
    collected_material_probabilities = []
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

            batch_results = compute_dual_q_metrics_torch(
                wavelengths,
                absorption,
                target_center_1=target_center_1,
                target_center_2=target_center_2,
                half_window=half_window_1,
                half_window_2=half_window_2,
            )
            batch_results.update(
                compute_double_lorentzian_mse_torch(
                    wavelengths,
                    absorption,
                    batch_results["peak_wavelengths_1"],
                    batch_results["peak_wavelengths_2"],
                    width=fom_lorentz_width,
                    mse_key="fom_double_lorentz_mse_values",
                    rmse_key="fom_double_lorentz_rmse_values",
                )
            )
            batch_results.update(
                compute_double_lorentzian_mse_torch(
                    wavelengths,
                    absorption,
                    batch_results["peak_wavelengths_1"],
                    batch_results["peak_wavelengths_2"],
                    width=float(getattr(params, "lorentz_width", 0.02)),
                )
            )
            batch_results["mse_values"] = batch_results["double_lorentz_mse_values"]
            batch_results["rmse_values"] = batch_results["double_lorentz_rmse_values"]
            batch_results["fom_lorentz_mse_values"] = batch_results["fom_double_lorentz_mse_values"]
            batch_results["fom_lorentz_rmse_values"] = batch_results["fom_double_lorentz_rmse_values"]
            batch_results.update(
                compute_dual_fom_scores_torch(
                    batch_results["q_min_pair_values"],
                    batch_results["fom_lorentz_rmse_values"],
                    batch_results["dual_valid_mask"],
                    q_ref=fom_q_ref,
                    rmse_ref=fom_rmse_ref,
                    weight=fom_weight,
                )
            )
            batch_results.update(
                compute_material_certainty_metrics_torch(
                    material_probabilities,
                    dominant_prob_threshold=dominant_prob_threshold,
                )
            )
            collected_results.append(batch_results)
            collected_absorption_spectra.append(absorption.detach().cpu())
            collected_thicknesses.append(thicknesses.detach().cpu())
            collected_material_probabilities.append(material_probabilities.detach().cpu())

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
    merged_absorption_spectra = torch.cat(collected_absorption_spectra, dim=0)
    merged_thicknesses = torch.cat(collected_thicknesses, dim=0)
    merged_material_probabilities = torch.cat(collected_material_probabilities, dim=0)
    summary = summarize_q_results(
        merged_results,
        epoch=epoch,
        alpha=alpha,
        num_samples=num_samples,
        lorentz_width=float(getattr(params, "lorentz_width", 0.02)),
        dominant_prob_threshold=dominant_prob_threshold,
        fom_q_ref=fom_q_ref,
        fom_lorentz_width=fom_lorentz_width,
        fom_rmse_ref=fom_rmse_ref,
        fom_weight=fom_weight,
    )
    collection_summary = {"new_high_quality_count": 0, "total_high_quality_count": 0}
    if high_quality_criteria["enabled"] and high_quality_dir is not None:
        collection_summary = update_high_quality_collection_summary(high_quality_dir, high_quality_records)

    summary["high_quality_count"] = int(collection_summary["new_high_quality_count"])
    summary["total_high_quality_count"] = int(collection_summary["total_high_quality_count"])
    save_global_best_sample_histories(
        summary=summary,
        q_results=merged_results,
        wavelengths=wavelengths,
        absorption_spectra=merged_absorption_spectra,
        thicknesses=merged_thicknesses,
        material_probabilities=merged_material_probabilities,
        params=params,
        save_dir=save_dir,
    )
    save_q_evaluation_epoch(merged_results, summary, save_dir, materials=list(getattr(params, "materials", [])))
    return summary
