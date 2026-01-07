import torch
import numpy as np

from model.TMM.optical_calculator import calculate_reflection
from inference.results import save_best_results, visualize_best_samples


def generate_samples(generator, params, num_samples, alpha, device, batch_size):
    """Generate samples using the trained generator."""
    if batch_size <= 0:
        raise ValueError("infer_batch_size must be a positive integer")

    wavelengths = 2 * np.pi / params.k.cpu()
    thicknesses_list = []
    refractive_indices_list = []
    probs_list = []
    absorption_list = []

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            current = min(batch_size, num_samples - start)
            thickness_noise = torch.randn(current, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(current, params.material_noise_dim, device=device)

            thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, alpha)
            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
            absorption = (1 - reflection).float()

            thicknesses_list.append(thicknesses.cpu())
            refractive_indices_list.append(refractive_indices.cpu())
            probs_list.append(P.cpu())
            absorption_list.append(absorption.cpu().numpy())

    thicknesses_all = torch.cat(thicknesses_list, dim=0)
    refractive_indices_all = torch.cat(refractive_indices_list, dim=0)
    probs_all = torch.cat(probs_list, dim=0)
    absorption_all = np.concatenate(absorption_list, axis=0)

    return wavelengths.cpu().numpy(), thicknesses_all, refractive_indices_all, probs_all, absorption_all


def create_target_lorentzian(wavelengths, center, width):
    """Create a target Lorentzian function with specified center and width."""
    gamma = width
    target = gamma / (2 * np.pi * ((wavelengths - center) ** 2 + (gamma / 2) ** 2))
    target = target / np.max(target)
    return target


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


def run_inference(args, load_parameters=None, load_model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if load_parameters is None or load_model is None:
        raise ValueError("load_parameters and load_model must be provided by infer.py")

    params = load_parameters(args.config_path, device)
    generator = load_model(args.model_path, params, device)

    print(f"Generating {args.num_samples} samples for screening...")
    wavelengths, thicknesses, refractive_indices, P, absorption_spectra = generate_samples(
        generator, params, args.num_samples, args.alpha, device, args.infer_batch_size
    )

    print(
        f"Creating target Lorentzian (center: {args.target_center}?m, "
        f"width: {args.target_width}?m)..."
    )
    target = create_target_lorentzian(wavelengths, args.target_center, args.target_width)

    print(f"Selecting best {args.best_samples} samples based on weighted RMSE...")
    best_indices, best_rmse = select_best_samples(
        absorption_spectra,
        wavelengths,
        target,
        args.target_center,
        args.center_region,
        args.weight_factor,
        args.best_samples,
    )

    print("Best samples selected:")
    for i, (idx, rmse) in enumerate(zip(best_indices, best_rmse)):
        print(f"  Best Sample {i+1}: Index {idx}, RMSE = {rmse:.6f}")

    save_best_results(
        args.output_dir,
        wavelengths,
        thicknesses,
        P,
        absorption_spectra,
        best_indices,
        best_rmse,
        target,
        params,
    )

    if args.visualize:
        fig = visualize_best_samples(wavelengths, absorption_spectra, best_indices, best_rmse, target)
        fig.show()

    print("Done!")
