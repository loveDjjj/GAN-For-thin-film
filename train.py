import argparse
import os
import datetime

import torch
import numpy as np

from train.trainer import train_gan
from model.TMM.optical_calculator import cache_master_compatible_metal_refractive_indices
from utils.config_loader import Params, load_config
from data.myindex import MatDatabase


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spectral GAN")
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--output_dir",
        default="results/spectral_gan",
        help="Directory to save training outputs",
    )
    return parser.parse_args()


def setup_directories(output_dir):
    """Setup output directories."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    samples_dir = os.path.join(run_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    print(f"Output directory: {run_dir}")
    return run_dir, model_dir, samples_dir


def load_parameters(config_path, device):
    """Load model parameters."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    params = Params()
    config = load_config(config_path)

    def require(path):
        current = config
        for key in path:
            if key not in current:
                raise KeyError(f"Missing config key: {'/'.join(path)}")
            current = current[key]
        return current

    structure = require(("structure",))
    params.N_layers = structure["N_layers"]
    params.pol = structure["pol"]
    params.thickness_sup = structure["thickness_sup"]
    params.thickness_bot = structure["thickness_bot"]

    materials = require(("materials",))
    params.materials = materials["materials_list"]

    optics = require(("optics",))
    params.wavelength_range = optics["wavelength_range"]
    params.samples_total = optics["samples_total"]
    params.theta = optics["theta"]
    params.n_top = optics["n_top"]
    params.n_bot = optics["n_bot"]
    params.lorentz_width = optics["lorentz_width"]
    params.lorentz_center_range = optics["lorentz_center_range"]
    params.metal_name = optics["metal_name"]

    generator = require(("generator",))
    params.thickness_noise_dim = generator["thickness_noise_dim"]
    params.material_noise_dim = generator["material_noise_dim"]
    params.alpha_min = generator["alpha_min"]
    params.alpha_max = generator["alpha_max"]

    training = require(("training",))
    params.epochs = training["epochs"]
    params.batch_size = training["batch_size"]
    params.save_interval = training["save_interval"]
    params.noise_level = training["noise_level"]
    params.lambda_gp = training["lambda_gp"]
    params.d_steps = training.get("d_steps", 1)
    params.g_steps = training.get("g_steps", 1)

    optimizer = require(("optimizer",))
    params.lr_gen = optimizer["lr_gen"]
    params.lr_disc = optimizer["lr_disc"]
    params.beta1 = optimizer["beta1"]
    params.beta2 = optimizer["beta2"]
    params.weight_decay = optimizer["weight_decay"]

    visualization = config.get("visualization", {})
    params.checkpoint_sample_count = visualization.get("checkpoint_sample_count", 8)
    params.sample_export_count = visualization.get("sample_export_count", 4)
    params.material_analysis_batch_size = visualization.get("material_analysis_batch_size", 64)
    params.thickness_histogram_bins = visualization.get("thickness_histogram_bins", 20)
    params.distribution_epoch_interval = visualization.get(
        "distribution_epoch_interval",
        max(1, params.epochs // 10),
    )
    params.heatmap_epoch_tick_step = visualization.get(
        "heatmap_epoch_tick_step",
        params.distribution_epoch_interval,
    )

    q_evaluation = config.get("q_evaluation", {})
    params.q_eval_interval = max(0, int(q_evaluation.get("interval", 0)))
    params.q_eval_num_samples = max(1, int(q_evaluation.get("num_samples", 1000)))
    params.q_eval_dominant_prob_threshold = float(
        q_evaluation.get("dominant_material_prob_threshold", 0.99)
    )
    params.q_eval_fom_q_ref = float(q_evaluation.get("fom_q_ref", 200.0))
    params.q_eval_fom_rmse_ref = float(q_evaluation.get("fom_rmse_ref", 0.05))
    params.q_eval_fom_weight = float(q_evaluation.get("fom_weight", 0.5))

    high_quality_collection = config.get("high_quality_collection", {})
    params.high_quality_collection_enabled = bool(high_quality_collection.get("enabled", False))
    params.high_quality_q_min = float(high_quality_collection.get("q_min", 200.0))
    params.high_quality_mse_max = float(high_quality_collection.get("mse_max", 0.0025))
    params.high_quality_peak_min = float(high_quality_collection.get("peak_min", 0.9))
    params.high_quality_dominant_prob_min = float(
        high_quality_collection.get("dominant_material_prob_min", 0.99)
    )

    reproducibility = config.get("reproducibility", {})
    params.seed = int(reproducibility.get("seed", 20260319))
    params.fix_training_target_centers = bool(reproducibility.get("fix_training_target_centers", True))
    params.center_pool_size = max(
        1,
        int(reproducibility.get("center_pool_size", max(params.batch_size * 128, params.q_eval_num_samples))),
    )
    params.fix_q_evaluation_noise = bool(reproducibility.get("fix_q_evaluation_noise", True))
    params.train_center_pool_size = params.center_pool_size if params.fix_training_target_centers else 0
    params.train_batches_per_epoch = (
        max(1, int(np.ceil(params.train_center_pool_size / params.batch_size)))
        if params.train_center_pool_size > 0
        else 1
    )

    params.user_define = False
    print(f"Configuration loaded from {config_path}")

    params.k = 2 * np.pi / torch.linspace(
        params.wavelength_range[0], params.wavelength_range[1], params.samples_total
    )

    params.theta = torch.tensor([params.theta]).to(device)
    params.n_top = torch.tensor([params.n_top])
    params.n_bot = torch.tensor([params.n_bot])

    if hasattr(params, 'materials') and params.materials:
        params.matdatabase = MatDatabase(params.materials)
        params.n_database, params.k_database = params.matdatabase.interp_wv(
            2 * np.pi / params.k, params.materials, False
        )
        params.M_materials = params.n_database.size(0)
        print(f"Using materials: {', '.join(params.materials)}")

    # Keep the master branch metal interpolation logic, but cache the full wavelength grid once on GPU.
    params.metal_refractive_indices = cache_master_compatible_metal_refractive_indices(params, device)
    print(f"Using bottom metal: {params.metal_name}")

    print(f"Training parameters: batch_size={params.batch_size}, epochs={params.epochs}")
    print(
        "Optimizer parameters: "
        f"generator_lr={params.lr_gen}, discriminator_lr={params.lr_disc}, "
        f"weight_decay={params.weight_decay}"
    )
    print(f"GAN stabilization: noise_level={params.noise_level}, lambda_gp={params.lambda_gp}")
    print(
        "Visualization parameters: "
        f"checkpoint_sample_count={params.checkpoint_sample_count}, "
        f"sample_export_count={params.sample_export_count}, "
        f"material_analysis_batch_size={params.material_analysis_batch_size}, "
        f"thickness_histogram_bins={params.thickness_histogram_bins}, "
        f"distribution_epoch_interval={params.distribution_epoch_interval}, "
        f"heatmap_epoch_tick_step={params.heatmap_epoch_tick_step}"
    )
    print(
        "Q/MSE evaluation parameters: "
        f"interval={params.q_eval_interval}, "
        f"num_samples={params.q_eval_num_samples}, "
        f"lorentz_width={params.lorentz_width}, "
        f"dominant_material_prob_threshold={params.q_eval_dominant_prob_threshold}, "
        f"fom_q_ref={params.q_eval_fom_q_ref}, "
        f"fom_rmse_ref={params.q_eval_fom_rmse_ref}, "
        f"fom_weight={params.q_eval_fom_weight}"
    )
    print(
        "High-quality collection parameters: "
        f"enabled={params.high_quality_collection_enabled}, "
        f"q_min={params.high_quality_q_min}, "
        f"mse_max={params.high_quality_mse_max}, "
        f"peak_min={params.high_quality_peak_min}, "
        f"dominant_material_prob_min={params.high_quality_dominant_prob_min}"
    )
    print(
        "Reproducibility parameters: "
        f"seed={params.seed}, "
        f"fix_training_target_centers={params.fix_training_target_centers}, "
        f"center_pool_size={params.center_pool_size}, "
        f"fix_q_evaluation_noise={params.fix_q_evaluation_noise}"
    )
    print(
        "Full-pool epoch parameters: "
        f"train_center_pool_size={params.train_center_pool_size}, "
        f"train_batches_per_epoch={params.train_batches_per_epoch}, "
        f"train_batch_size={params.batch_size}"
    )

    return params


def main():
    args = parse_args()
    train_gan(
        args.config,
        args.output_dir,
        load_parameters=load_parameters,
        setup_directories=setup_directories,
    )


if __name__ == "__main__":
    main()
