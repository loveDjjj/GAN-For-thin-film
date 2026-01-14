import numpy as np


def calculate_weighted_rmse(absorption, target, wavelengths, center, region_width, weight_factor):
    """Calculate weighted RMSE between absorption and target Lorentzian."""
    weights = np.ones_like(wavelengths)
    central_region = (wavelengths >= center - region_width) & (wavelengths <= center + region_width)
    weights[central_region] = weight_factor

    weighted_errors = weights * (absorption - target) ** 2
    rmse = np.sqrt(np.mean(weighted_errors))
    return rmse


def select_best_samples(absorption_spectra, wavelengths, target, center, region_width, weight_factor, num_best=4):
    """Select best samples based on weighted RMSE with target."""
    rmse_values = []

    for i in range(len(absorption_spectra)):
        rmse = calculate_weighted_rmse(
            absorption_spectra[i], target, wavelengths, center, region_width, weight_factor
        )
        rmse_values.append(rmse)

    rmse_values = np.array(rmse_values)
    best_indices = np.argsort(rmse_values)[:num_best]
    best_rmse = rmse_values[best_indices]

    return best_indices, best_rmse


def compute_weighted_rmse_all(absorption_spectra, wavelengths, target, center, region_width, weight_factor):
    """Compute weighted RMSE for all samples."""
    rmse_values = []
    for i in range(len(absorption_spectra)):
        rmse = calculate_weighted_rmse(
            absorption_spectra[i], target, wavelengths, center, region_width, weight_factor
        )
        rmse_values.append(rmse)
    return np.array(rmse_values)


def calculate_pareto_front(weighted_rmse, total_thickness):
    """
    Calculate Pareto front indices based on two objectives:
    weighted RMSE (minimize) and total thickness (minimize).
    """
    indices = np.arange(len(weighted_rmse))
    sorted_idx = np.argsort(weighted_rmse)
    pareto = []
    best_thickness = np.inf

    for idx in sorted_idx:
        thickness = total_thickness[idx]
        if thickness < best_thickness:
            pareto.append(idx)
            best_thickness = thickness

    pareto_indices = np.array(pareto)
    return pareto_indices


def compute_total_thickness(thicknesses):
    """Sum thickness over layers for each sample."""
    if hasattr(thicknesses, 'cpu'):
        thickness_array = thicknesses.cpu().numpy()
    else:
        thickness_array = thicknesses
    return np.sum(thickness_array, axis=1)
