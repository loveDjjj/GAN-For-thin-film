import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np

from inference.inferer import run_inference
from model.net import Generator, Discriminator
from utils.config_loader import Params, load_config
from data.myindex import MatDatabase


torch.serialization.add_safe_globals([Generator, Discriminator])


def apply_config_overrides(args, config_path):
    config_file = Path(config_path)
    if not config_file.exists():
        return args

    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        return args

    cli_args = set(sys.argv[1:])
    for key, value in config.items():
        flag = f"--{key}"
        if flag not in cli_args and hasattr(args, key):
            setattr(args, key, value)

    return args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load trained GAN model and generate samples')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained generator model (.pth file)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to the configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate for screening')
    parser.add_argument('--infer_batch_size', type=int, default=None,
                        help='Batch size for inference generation')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Alpha parameter for material selection sharpness')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize generated samples')
    parser.add_argument('--target_center', type=float, default=None,
                        help='Target Lorentzian center wavelength (μm)')
    parser.add_argument('--target_width', type=float, default=None,
                        help='Target Lorentzian width (μm)')
    parser.add_argument('--center_region', type=float, default=None,
                        help='Width of central region (μm) for increased weighting')
    parser.add_argument('--weight_factor', type=float, default=None,
                        help='Weight multiplier for central region')
    parser.add_argument('--best_samples', type=int, default=None,
                        help='Number of best samples to save')
    return parser.parse_args()


def load_parameters(config_path, device):
    """Load model parameters from configuration file."""
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
    params.metal_name = optics["metal_name"]

    generator = require(("generator",))
    params.thickness_noise_dim = generator["thickness_noise_dim"]
    params.material_noise_dim = generator["material_noise_dim"]
    params.alpha_sup = generator["alpha_sup"]
    params.alpha = generator["alpha"]

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
            2 * np.pi/params.k, params.materials, False
        )
        params.M_materials = params.n_database.size(0)
        print(f"Using materials: {', '.join(params.materials)}")

    return params


def load_model(model_path, params, device):
    """Load trained generator model."""
    generator = Generator(params).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")

    generator.eval()
    return generator


def main():
    args = parse_args()
    args = apply_config_overrides(args, "config/inference_config.yaml")
    missing = [key for key, value in vars(args).items() if value is None]
    if missing:
        raise ValueError(
            "Missing inference parameters in config or CLI: "
            + ", ".join(sorted(missing))
        )
    run_inference(args, load_parameters=load_parameters, load_model=load_model)


if __name__ == "__main__":
    main()
