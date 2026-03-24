import torch


def _ensure_tensor(value, device=None, dtype=None):
    """Convert inputs to tensors while preserving tensors already on device."""
    if torch.is_tensor(value):
        if device is not None or dtype is not None:
            return value.to(device=device if device is not None else value.device, dtype=dtype or value.dtype)
        return value
    return torch.as_tensor(value, device=device, dtype=dtype)


def _build_weights(wavelengths, center, region_width, weight_factor):
    wavelengths = _ensure_tensor(wavelengths, dtype=torch.float32).flatten()
    weights = torch.ones_like(wavelengths)
    central_region = (wavelengths >= center - region_width) & (wavelengths <= center + region_width)
    weights = torch.where(central_region, torch.full_like(weights, float(weight_factor)), weights)
    return wavelengths, weights


def calculate_weighted_rmse(absorption, target, wavelengths, center, region_width, weight_factor):
    """Calculate weighted RMSE on torch tensors; supports one sample or a batch."""
    absorption = _ensure_tensor(absorption, dtype=torch.float32)
    target = _ensure_tensor(target, device=absorption.device, dtype=absorption.dtype)
    wavelengths, weights = _build_weights(
        wavelengths,
        center=center,
        region_width=region_width,
        weight_factor=weight_factor,
    )
    wavelengths = wavelengths.to(device=absorption.device, dtype=absorption.dtype)
    weights = weights.to(device=absorption.device, dtype=absorption.dtype)

    if absorption.ndim == 1:
        squared_error = weights * (absorption - target) ** 2
        return torch.sqrt(torch.mean(squared_error))

    squared_error = weights.unsqueeze(0) * (absorption - target.unsqueeze(0)) ** 2
    return torch.sqrt(torch.mean(squared_error, dim=1))


def select_best_samples(absorption_spectra, wavelengths, target, center, region_width, weight_factor, num_best=4):
    """Select best samples on GPU using weighted RMSE and torch.topk."""
    rmse_values = compute_weighted_rmse_all(
        absorption_spectra,
        wavelengths,
        target,
        center,
        region_width,
        weight_factor,
    )
    num_best = min(max(int(num_best), 0), int(rmse_values.numel()))
    if num_best == 0:
        empty = torch.empty(0, dtype=torch.long, device=rmse_values.device)
        return empty, torch.empty(0, dtype=rmse_values.dtype, device=rmse_values.device)

    best_rmse, best_indices = torch.topk(rmse_values, k=num_best, largest=False, sorted=True)
    return best_indices, best_rmse


def compute_weighted_rmse_all(absorption_spectra, wavelengths, target, center, region_width, weight_factor):
    """Compute weighted RMSE for all samples on torch tensors."""
    absorption_spectra = _ensure_tensor(absorption_spectra, dtype=torch.float32)
    if absorption_spectra.ndim != 2:
        raise ValueError("absorption_spectra must have shape [num_samples, num_wavelengths]")
    return calculate_weighted_rmse(
        absorption_spectra,
        target,
        wavelengths,
        center,
        region_width,
        weight_factor,
    )


def calculate_pareto_front(weighted_rmse, total_thickness):
    """Calculate Pareto front indices on GPU for minimizing RMSE and total thickness."""
    weighted_rmse = _ensure_tensor(weighted_rmse, dtype=torch.float32).flatten()
    total_thickness = _ensure_tensor(total_thickness, device=weighted_rmse.device, dtype=weighted_rmse.dtype).flatten()
    if weighted_rmse.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=weighted_rmse.device)

    sorted_idx = torch.argsort(weighted_rmse)
    sorted_thickness = total_thickness[sorted_idx]
    running_min = torch.cummin(sorted_thickness, dim=0).values
    previous_min = torch.cat(
        [
            torch.full((1,), torch.inf, device=sorted_thickness.device, dtype=sorted_thickness.dtype),
            running_min[:-1],
        ],
        dim=0,
    )
    pareto_mask = sorted_thickness < previous_min
    return sorted_idx[pareto_mask]


def compute_total_thickness(thicknesses):
    """Sum thickness over layers for each sample on torch tensors."""
    thicknesses = _ensure_tensor(thicknesses, dtype=torch.float32)
    if thicknesses.ndim != 2:
        raise ValueError("thicknesses must have shape [num_samples, num_layers]")
    return torch.sum(thicknesses, dim=1)
