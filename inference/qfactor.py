import os

import torch


def _ensure_tensor(value, device=None, dtype=None):
    """Convert value to torch tensor, preserving tensors already on device."""
    if torch.is_tensor(value):
        if device is not None or dtype is not None:
            return value.to(device=device if device is not None else value.device, dtype=dtype or value.dtype)
        return value
    return torch.as_tensor(value, device=device, dtype=dtype)


def _safe_delta(delta, eps):
    replacement = torch.where(delta >= 0, torch.full_like(delta, eps), torch.full_like(delta, -eps))
    return torch.where(delta.abs() < eps, replacement, delta)


def compute_q_for_spectra(wavelengths, absorption_spectra, center, half_window, eps=1e-12):
    """Compute Q-factor for a batch of spectra on GPU with peak search limited to a target window."""
    absorption_spectra = _ensure_tensor(absorption_spectra, dtype=torch.float32)
    if absorption_spectra.ndim == 1:
        absorption_spectra = absorption_spectra.unsqueeze(0)
    if absorption_spectra.ndim != 2:
        raise ValueError("absorption_spectra must have shape [batch_size, num_wavelengths]")

    wavelengths = _ensure_tensor(
        wavelengths,
        device=absorption_spectra.device,
        dtype=absorption_spectra.dtype,
    ).flatten()
    if wavelengths.numel() != absorption_spectra.shape[1]:
        raise ValueError("wavelengths length must match absorption_spectra.shape[1]")

    window_mask = (wavelengths >= center - half_window) & (wavelengths <= center + half_window)
    if not window_mask.any():
        empty = torch.zeros(absorption_spectra.shape[0], device=absorption_spectra.device, dtype=absorption_spectra.dtype)
        return {
            "q_values": empty,
            "peak_wavelengths": torch.full_like(empty, float("nan")),
            "peak_absorptions": torch.zeros_like(empty),
            "left_wavelengths": torch.full_like(empty, float("nan")),
            "right_wavelengths": torch.full_like(empty, float("nan")),
            "valid_mask": torch.zeros(absorption_spectra.shape[0], dtype=torch.bool, device=absorption_spectra.device),
        }

    masked_absorption = torch.where(window_mask.unsqueeze(0), absorption_spectra, torch.full_like(absorption_spectra, -torch.inf))
    peak_indices = torch.argmax(masked_absorption, dim=1)
    peak_absorptions = absorption_spectra.gather(1, peak_indices.unsqueeze(1)).squeeze(1)
    peak_wavelengths = wavelengths[peak_indices]
    half_max = peak_absorptions * 0.5

    num_wavelengths = absorption_spectra.shape[1]
    indices = torch.arange(num_wavelengths, device=absorption_spectra.device).unsqueeze(0).expand(absorption_spectra.shape[0], -1)

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
    q_values = torch.zeros_like(peak_wavelengths)

    valid_mask = (
        window_mask.any()
        & (peak_absorptions > 0)
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

        fwhm = interpolated_right - interpolated_left
        current_valid = fwhm > eps

        left_wavelengths[valid_rows] = interpolated_left
        right_wavelengths[valid_rows] = interpolated_right
        q_values[valid_rows] = torch.where(
            current_valid,
            peak_wavelengths[valid_rows] / fwhm.clamp_min(eps),
            torch.zeros_like(fwhm),
        )
        valid_mask[valid_rows] = current_valid

    q_values = torch.nan_to_num(q_values, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "q_values": q_values,
        "peak_wavelengths": peak_wavelengths,
        "peak_absorptions": peak_absorptions,
        "left_wavelengths": left_wavelengths,
        "right_wavelengths": right_wavelengths,
        "valid_mask": valid_mask,
    }


def compute_q_for_spectrum(wavelengths, absorption, center, half_window):
    """Compute Q-factor for one spectrum using the batched torch implementation."""
    results = compute_q_for_spectra(wavelengths, absorption, center, half_window)
    peak_wavelength = results["peak_wavelengths"][0]
    peak_absorption = results["peak_absorptions"][0]
    left_wavelength = results["left_wavelengths"][0]
    right_wavelength = results["right_wavelengths"][0]
    return {
        "q": float(results["q_values"][0].item()),
        "peak_wavelength": None if torch.isnan(peak_wavelength) else float(peak_wavelength.item()),
        "peak_absorption": float(peak_absorption.item()),
        "left_wavelength": None if torch.isnan(left_wavelength) else float(left_wavelength.item()),
        "right_wavelength": None if torch.isnan(right_wavelength) else float(right_wavelength.item()),
    }


def compute_q_for_indices(wavelengths, absorption_spectra, indices, center, half_window):
    """Compute Q for a list of sample indices with batched torch kernels."""
    indices = _ensure_tensor(indices, dtype=torch.long)
    if indices.numel() == 0:
        return []

    absorption_spectra = _ensure_tensor(absorption_spectra, device=indices.device if torch.is_tensor(absorption_spectra) else None)
    selected_absorption = absorption_spectra.index_select(0, indices.to(device=absorption_spectra.device))
    results = compute_q_for_spectra(wavelengths, selected_absorption, center, half_window)

    records = []
    q_values = results["q_values"].detach().cpu()
    peak_wavelengths = results["peak_wavelengths"].detach().cpu()
    peak_absorptions = results["peak_absorptions"].detach().cpu()
    left_wavelengths = results["left_wavelengths"].detach().cpu()
    right_wavelengths = results["right_wavelengths"].detach().cpu()
    index_values = indices.detach().cpu()

    for row_index, original_index in enumerate(index_values.tolist()):
        records.append(
            {
                "index": int(original_index),
                "q": float(q_values[row_index].item()),
                "peak_wavelength": None if torch.isnan(peak_wavelengths[row_index]) else float(peak_wavelengths[row_index].item()),
                "peak_absorption": float(peak_absorptions[row_index].item()),
                "left_wavelength": None if torch.isnan(left_wavelengths[row_index]) else float(left_wavelengths[row_index].item()),
                "right_wavelength": None if torch.isnan(right_wavelengths[row_index]) else float(right_wavelengths[row_index].item()),
            }
        )
    return records


def compute_dual_q_for_spectra(wavelengths, absorption_spectra, center_1, center_2, half_window, half_window_2=None):
    """Compute dual-window Q metrics for a batch of spectra."""
    if half_window_2 is None:
        half_window_2 = half_window

    first = compute_q_for_spectra(wavelengths, absorption_spectra, center_1, half_window)
    second = compute_q_for_spectra(wavelengths, absorption_spectra, center_2, half_window_2)
    q_min_pair_values = torch.minimum(first["q_values"], second["q_values"])
    dual_valid_mask = first["valid_mask"] & second["valid_mask"]
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
    }


def compute_dual_q_for_indices(wavelengths, absorption_spectra, indices, center_1, center_2, half_window, half_window_2=None):
    """Compute dual-window Q for a list of sample indices."""
    indices = _ensure_tensor(indices, dtype=torch.long)
    if indices.numel() == 0:
        return []

    absorption_spectra = _ensure_tensor(absorption_spectra, device=indices.device if torch.is_tensor(absorption_spectra) else None)
    selected_absorption = absorption_spectra.index_select(0, indices.to(device=absorption_spectra.device))
    results = compute_dual_q_for_spectra(wavelengths, selected_absorption, center_1, center_2, half_window, half_window_2)

    records = []
    index_values = indices.detach().cpu()
    q1_values = results["q1_values"].detach().cpu()
    q2_values = results["q2_values"].detach().cpu()
    q_min_pair_values = results["q_min_pair_values"].detach().cpu()
    peak_wavelengths_1 = results["peak_wavelengths_1"].detach().cpu()
    peak_wavelengths_2 = results["peak_wavelengths_2"].detach().cpu()
    peak_absorptions_1 = results["peak_absorptions_1"].detach().cpu()
    peak_absorptions_2 = results["peak_absorptions_2"].detach().cpu()
    left_wavelengths_1 = results["left_wavelengths_1"].detach().cpu()
    left_wavelengths_2 = results["left_wavelengths_2"].detach().cpu()
    right_wavelengths_1 = results["right_wavelengths_1"].detach().cpu()
    right_wavelengths_2 = results["right_wavelengths_2"].detach().cpu()

    for row_index, original_index in enumerate(index_values.tolist()):
        records.append(
            {
                "index": int(original_index),
                "q1": float(q1_values[row_index].item()),
                "q2": float(q2_values[row_index].item()),
                "q_min_pair": float(q_min_pair_values[row_index].item()),
                "peak_wavelength_1": None if torch.isnan(peak_wavelengths_1[row_index]) else float(peak_wavelengths_1[row_index].item()),
                "peak_wavelength_2": None if torch.isnan(peak_wavelengths_2[row_index]) else float(peak_wavelengths_2[row_index].item()),
                "peak_absorption_1": float(peak_absorptions_1[row_index].item()),
                "peak_absorption_2": float(peak_absorptions_2[row_index].item()),
                "left_wavelength_1": None if torch.isnan(left_wavelengths_1[row_index]) else float(left_wavelengths_1[row_index].item()),
                "left_wavelength_2": None if torch.isnan(left_wavelengths_2[row_index]) else float(left_wavelengths_2[row_index].item()),
                "right_wavelength_1": None if torch.isnan(right_wavelengths_1[row_index]) else float(right_wavelengths_1[row_index].item()),
                "right_wavelength_2": None if torch.isnan(right_wavelengths_2[row_index]) else float(right_wavelengths_2[row_index].item()),
            }
        )
    return records


def save_q_report(path, q_results, title="Q Report"):
    """Save Q-factor results to a txt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write("=" * 60 + "\n\n")
        for result_index, result in enumerate(q_results, 1):
            file.write(f"Sample {result_index} (Original Index: {result.get('index', 'N/A')})\n")
            if "q_min_pair" in result:
                file.write(f"Q1: {result['q1']:.4f}\n")
                file.write(f"Q2: {result['q2']:.4f}\n")
                file.write(f"Q_min_pair: {result['q_min_pair']:.4f}\n")
                file.write(f"Peak Wavelength 1: {result.get('peak_wavelength_1')}\n")
                file.write(f"Peak Wavelength 2: {result.get('peak_wavelength_2')}\n")
                file.write(f"Peak Absorption 1: {result.get('peak_absorption_1')}\n")
                file.write(f"Peak Absorption 2: {result.get('peak_absorption_2')}\n")
                file.write(f"Left Half-Max 1: {result.get('left_wavelength_1')}\n")
                file.write(f"Left Half-Max 2: {result.get('left_wavelength_2')}\n")
                file.write(f"Right Half-Max 1: {result.get('right_wavelength_1')}\n")
                file.write(f"Right Half-Max 2: {result.get('right_wavelength_2')}\n")
            else:
                file.write(f"Q: {result['q']:.4f}\n")
                file.write(f"Peak Wavelength: {result.get('peak_wavelength')}\n")
                file.write(f"Peak Absorption: {result.get('peak_absorption')}\n")
                file.write(f"Left Half-Max: {result.get('left_wavelength')}\n")
                file.write(f"Right Half-Max: {result.get('right_wavelength')}\n")
            file.write("-" * 40 + "\n")
