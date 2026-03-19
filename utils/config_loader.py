"""Configuration helpers."""

import json
import logging
import os

import yaml


class Params:
    """Container for runtime parameters."""

    def __init__(self, json_path=None):
        if json_path is not None:
            self.update(json_path)

    def save(self, json_path):
        """Save parameters to a JSON file."""
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(self.__dict__, file, indent=4, ensure_ascii=False)

    def update(self, json_path):
        """Load parameters from a JSON file."""
        with open(json_path, encoding="utf-8") as file:
            params = json.load(file)
        self.__dict__.update(params)

    @property
    def dict(self):
        """Expose params as a dictionary."""
        return self.__dict__


def load_config(config_path):
    """Load a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config or {}


def _update_from_mapping(params, config_section, mapping):
    for key, attr_name in mapping.items():
        if key in config_section:
            setattr(params, attr_name, config_section[key])


def update_params(params, config):
    """Update a params object from a config dictionary."""
    if "structure" in config:
        _update_from_mapping(
            params,
            config["structure"],
            {
                "N_layers": "N_layers",
                "pol": "pol",
                "thickness_sup": "thickness_sup",
                "thickness_bot": "thickness_bot",
            },
        )

    if "materials" in config and "materials_list" in config["materials"]:
        params.materials = config["materials"]["materials_list"]

    if "optics" in config:
        _update_from_mapping(
            params,
            config["optics"],
            {
                "wavelength_range": "wavelength_range",
                "samples_total": "samples_total",
                "theta": "theta",
                "n_top": "n_top",
                "n_bot": "n_bot",
                "lorentz_width": "lorentz_width",
                "lorentz_center_range": "lorentz_center_range",
                "metal_name": "metal_name",
            },
        )

    if "generator" in config:
        _update_from_mapping(
            params,
            config["generator"],
            {
                "thickness_noise_dim": "thickness_noise_dim",
                "material_noise_dim": "material_noise_dim",
                "alpha_min": "alpha_min",
                "alpha_max": "alpha_max",
                "alpha_sup": "alpha_sup",
                "alpha": "alpha",
            },
        )

    if "training" in config:
        _update_from_mapping(
            params,
            config["training"],
            {
                "epochs": "epochs",
                "batch_size": "batch_size",
                "save_interval": "save_interval",
                "noise_level": "noise_level",
                "lambda_gp": "lambda_gp",
                "d_steps": "d_steps",
                "g_steps": "g_steps",
            },
        )

    if "optimizer" in config:
        _update_from_mapping(
            params,
            config["optimizer"],
            {
                "lr_gen": "lr_gen",
                "lr_disc": "lr_disc",
                "beta1": "beta1",
                "beta2": "beta2",
                "weight_decay": "weight_decay",
            },
        )

    if "visualization" in config:
        _update_from_mapping(
            params,
            config["visualization"],
            {
                "checkpoint_sample_count": "checkpoint_sample_count",
                "sample_export_count": "sample_export_count",
                "material_analysis_batch_size": "material_analysis_batch_size",
                "thickness_histogram_bins": "thickness_histogram_bins",
                "distribution_epoch_interval": "distribution_epoch_interval",
                "heatmap_epoch_tick_step": "heatmap_epoch_tick_step",
            },
        )

    if "q_evaluation" in config:
        _update_from_mapping(
            params,
            config["q_evaluation"],
            {
                "interval": "q_eval_interval",
                "num_samples": "q_eval_num_samples",
            },
        )

    if "high_quality_collection" in config:
        _update_from_mapping(
            params,
            config["high_quality_collection"],
            {
                "enabled": "high_quality_collection_enabled",
                "q_min": "high_quality_q_min",
                "mse_max": "high_quality_mse_max",
                "peak_min": "high_quality_peak_min",
                "dominant_material_prob_min": "high_quality_dominant_prob_min",
            },
        )

    params.user_define = False
    return params


def set_logger(log_path):
    """Set up logging to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
