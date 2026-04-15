import os

import torch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from model.TMM.optical_calculator import calculate_reflection
from inference import filtering
from inference import visualization
from inference import qfactor
from utils.visualize import analyze_inference_distribution


def generate_samples(generator, params, num_samples, alpha, device, batch_size):
    """Generate samples and keep tensors on GPU for downstream filtering."""
    if batch_size <= 0:
        raise ValueError("infer_batch_size must be a positive integer")

    wavelengths = (2 * torch.pi / params.k.to(device)).float()
    thicknesses_list = []
    probs_list = []
    absorption_list = []

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            current = min(batch_size, num_samples - start)
            thickness_noise = torch.randn(current, params.thickness_noise_dim, device=device)
            material_noise = torch.randn(current, params.material_noise_dim, device=device)

            thicknesses, refractive_indices, P = generator(thickness_noise, material_noise, alpha)
            if P.ndim == 2:
                P = P.unsqueeze(0)
            reflection = calculate_reflection(thicknesses, refractive_indices, params, device)
            absorption = (1 - reflection).float()

            thicknesses_list.append(thicknesses.detach())
            probs_list.append(P.detach())
            absorption_list.append(absorption.detach())

    thicknesses_all = torch.cat(thicknesses_list, dim=0)
    probs_all = torch.cat(probs_list, dim=0)
    absorption_all = torch.cat(absorption_list, dim=0)

    return wavelengths, thicknesses_all, probs_all, absorption_all


def create_double_target_lorentzian(wavelengths, center_1, center_2, width):
    """Create a normalized double Lorentzian target on the same device as wavelengths."""
    return generate_double_lorentzian_curves(
        wavelengths=wavelengths,
        width=width,
        center1=center_1,
        center2=center_2,
    )


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def run_inference(args, load_parameters=None, load_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if load_parameters is None or load_model is None:
        raise ValueError("load_parameters and load_model must be provided by infer.py")

    params = load_parameters(args.config_path, device)
    generator = load_model(args.model_path, params, device)

    print(f"Generating {args.num_samples} samples for screening...")
    wavelengths, thicknesses, P, absorption_spectra = generate_samples(
        generator, params, args.num_samples, args.alpha, device, args.infer_batch_size
    )

    print(
        f"Creating target Lorentzian (centers: {args.target_center_1}?m / {args.target_center_2}?m, "
        f"width: {args.target_width}?m)..."
    )
    target = create_double_target_lorentzian(
        wavelengths,
        args.target_center_1,
        args.target_center_2,
        args.target_width,
    )

    print(f"Selecting best {args.best_samples} samples based on weighted RMSE...")
    best_indices, best_rmse = filtering.select_best_samples(
        absorption_spectra,
        wavelengths,
        target,
        centers=(args.target_center_1, args.target_center_2),
        region_width=args.center_region,
        weight_factor=args.weight_factor,
        num_best=args.best_samples,
    )

    best_indices_cpu = best_indices.detach().cpu()
    best_rmse_cpu = best_rmse.detach().cpu()
    print("Best samples selected:")
    for result_index, (sample_index, rmse_value) in enumerate(zip(best_indices_cpu.tolist(), best_rmse_cpu.tolist()), start=1):
        print(f"  Best Sample {result_index}: Index {sample_index}, RMSE = {rmse_value:.6f}")

    wavelengths_cpu = _to_numpy(wavelengths)
    target_cpu = _to_numpy(target)
    best_absorption = _to_numpy(absorption_spectra.index_select(0, best_indices))
    best_thicknesses = thicknesses.index_select(0, best_indices).detach().cpu()
    best_probs = P.index_select(0, best_indices).detach().cpu()

    save_dir = visualization.save_best_results(
        args.output_dir,
        wavelengths_cpu,
        best_thicknesses,
        best_probs,
        best_absorption,
        list(range(best_indices.numel())),
        best_rmse_cpu.numpy(),
        target_cpu,
        params,
        original_indices=best_indices_cpu.numpy(),
    )

    print("Analyzing thickness and merged layers distribution of generated samples...")
    analyze_inference_distribution(thicknesses, P, save_dir, prefix="all_samples")

    if best_indices.numel() > 0:
        print(f"Analyzing distribution of {best_indices.numel()} best samples...")
        analyze_inference_distribution(best_thicknesses, best_probs, save_dir, prefix="best_samples")
    else:
        print("No best samples to analyze")

    weighted_rmse_all = filtering.compute_weighted_rmse_all(
        absorption_spectra,
        wavelengths,
        target,
        centers=(args.target_center_1, args.target_center_2),
        region_width=args.center_region,
        weight_factor=args.weight_factor,
    )
    total_thickness = filtering.compute_total_thickness(thicknesses)
    pareto_indices = filtering.calculate_pareto_front(weighted_rmse_all, total_thickness)
    weighted_rmse_all_cpu = _to_numpy(weighted_rmse_all)
    total_thickness_cpu = _to_numpy(total_thickness)
    pareto_indices_cpu = pareto_indices.detach().cpu().numpy()
    pareto_dir = visualization.save_pareto_results(
        save_dir,
        weighted_rmse_all_cpu,
        total_thickness_cpu,
        pareto_indices_cpu,
    )

    pareto_rmse = weighted_rmse_all.index_select(0, pareto_indices)
    pareto_absorption = _to_numpy(absorption_spectra.index_select(0, pareto_indices))
    pareto_thicknesses = thicknesses.index_select(0, pareto_indices).detach().cpu()
    pareto_probs = P.index_select(0, pareto_indices).detach().cpu()
    if pareto_indices.numel() > 0:
        visualization.save_pareto_samples(
            save_dir,
            wavelengths_cpu,
            pareto_absorption,
            pareto_thicknesses,
            pareto_probs,
            list(range(pareto_indices.numel())),
            _to_numpy(pareto_rmse),
            target_cpu,
            params,
            original_indices=pareto_indices_cpu,
        )

    if pareto_indices.numel() > 0:
        print(f"Analyzing distribution of {pareto_indices.numel()} Pareto front samples...")
        analyze_inference_distribution(pareto_thicknesses, pareto_probs, pareto_dir, prefix="pareto_samples")
    else:
        print("No Pareto front samples to analyze")

    best_q = qfactor.compute_dual_q_for_indices(
        wavelengths,
        absorption_spectra,
        best_indices,
        args.target_center_1,
        args.target_center_2,
        args.q_eval_window,
    )
    qfactor.save_q_report(os.path.join(save_dir, "best_samples_q.txt"), best_q, "Best Samples Q")

    pareto_q = qfactor.compute_dual_q_for_indices(
        wavelengths,
        absorption_spectra,
        pareto_indices,
        args.target_center_1,
        args.target_center_2,
        args.q_eval_window,
    )
    qfactor.save_q_report(os.path.join(pareto_dir, "pareto_samples_q.txt"), pareto_q, "Pareto Samples Q")

    if args.visualize:
        fig = visualization.visualize_best_samples(
            wavelengths_cpu,
            best_absorption,
            list(range(best_indices.numel())),
            best_rmse_cpu.numpy(),
            target_cpu,
            original_indices=best_indices_cpu.numpy(),
        )
        fig.show()
        pareto_fig = visualization.plot_pareto_front(weighted_rmse_all_cpu, total_thickness_cpu, pareto_indices_cpu)
        pareto_fig.show()

    print("Done!")
    return {
        "wavelengths": wavelengths.detach().cpu(),
        "thicknesses": thicknesses.detach().cpu(),
        "refractive_indices": None,
        "probs": P.detach().cpu(),
        "absorption_spectra": absorption_spectra.detach().cpu(),
        "target": target.detach().cpu(),
        "best_indices": best_indices_cpu,
        "best_rmse": best_rmse_cpu,
        "weighted_rmse_all": torch.as_tensor(weighted_rmse_all_cpu),
        "total_thickness": torch.as_tensor(total_thickness_cpu),
        "pareto_indices": torch.as_tensor(pareto_indices_cpu, dtype=torch.long),
        "best_q": best_q,
        "pareto_q": pareto_q,
    }
