import os
import torch
import numpy as np

from model.TMM.optical_calculator import calculate_reflection
from inference import filtering
from inference import visualization
from inference import qfactor
from utils.visualize import analyze_inference_distribution


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
    best_indices, best_rmse = filtering.select_best_samples(
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

    save_dir = visualization.save_best_results(
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

    # 分析所有生成样本的厚度和层数分布
    print("Analyzing thickness and merged layers distribution of generated samples...")
    analyze_inference_distribution(thicknesses, P, save_dir, prefix="all_samples")

    # 分析最佳样本的厚度和层数分布
    if best_indices.size > 0:
        print(f"Analyzing distribution of {len(best_indices)} best samples...")
        best_thicknesses = thicknesses[best_indices]
        best_P = P[best_indices]
        analyze_inference_distribution(best_thicknesses, best_P, save_dir, prefix="best_samples")
    else:
        print("No best samples to analyze")

    # 计算并保存帕累托前沿信息
    weighted_rmse_all = filtering.compute_weighted_rmse_all(
        absorption_spectra,
        wavelengths,
        target,
        args.target_center,
        args.center_region,
        args.weight_factor,
    )
    total_thickness = filtering.compute_total_thickness(thicknesses)
    pareto_indices = filtering.calculate_pareto_front(weighted_rmse_all, total_thickness)
    pareto_dir = visualization.save_pareto_results(save_dir, weighted_rmse_all, total_thickness, pareto_indices)
    pareto_rmse = weighted_rmse_all[pareto_indices]
    visualization.save_pareto_samples(
        save_dir, wavelengths, absorption_spectra, thicknesses, P, pareto_indices, pareto_rmse, target, params
    )

    # 分析帕累托样本的厚度和层数分布
    if pareto_indices.size > 0:
        print(f"Analyzing distribution of {len(pareto_indices)} Pareto front samples...")
        pareto_thicknesses = thicknesses[pareto_indices]
        pareto_P = P[pareto_indices]
        analyze_inference_distribution(pareto_thicknesses, pareto_P, pareto_dir, prefix="pareto_samples")
    else:
        print("No Pareto front samples to analyze")

    # 计算 Q 值（最佳样本与帕累托样本）
    best_q = qfactor.compute_q_for_indices(
        wavelengths, absorption_spectra, best_indices, args.target_center, args.q_eval_window
    )
    qfactor.save_q_report(os.path.join(save_dir, "best_samples_q.txt"), best_q, "Best Samples Q")

    pareto_q = qfactor.compute_q_for_indices(
        wavelengths, absorption_spectra, pareto_indices, args.target_center, args.q_eval_window
    )
    qfactor.save_q_report(os.path.join(pareto_dir, "pareto_samples_q.txt"), pareto_q, "Pareto Samples Q")

    if args.visualize:
        fig = visualization.visualize_best_samples(
            wavelengths, absorption_spectra, best_indices, best_rmse, target
        )
        fig.show()
        pareto_fig = visualization.plot_pareto_front(weighted_rmse_all, total_thickness, pareto_indices)
        pareto_fig.show()

    print("Done!")
    return {
        "wavelengths": wavelengths,
        "thicknesses": thicknesses,
        "refractive_indices": refractive_indices,
        "probs": P,
        "absorption_spectra": absorption_spectra,
        "target": target,
        "best_indices": best_indices,
        "best_rmse": best_rmse,
        "weighted_rmse_all": weighted_rmse_all,
        "total_thickness": total_thickness,
        "pareto_indices": pareto_indices,
        "best_q": best_q,
        "pareto_q": pareto_q,
    }
