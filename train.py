import argparse
import os
import datetime

import torch
import numpy as np

from train.trainer import train_gan
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
