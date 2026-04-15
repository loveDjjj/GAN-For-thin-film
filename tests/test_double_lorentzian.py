import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch
import yaml

from model.Lorentzian import lorentzian_curves


TRAIN_MODULE_PATH = Path(__file__).resolve().parents[1] / "train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("train_entry_module", TRAIN_MODULE_PATH)
TRAIN_MODULE = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN_MODULE)
load_train_parameters = TRAIN_MODULE.load_parameters


class DoubleLorentzianTests(unittest.TestCase):
    def test_generate_double_lorentzian_curves_has_two_target_peaks(self):
        generate_double_lorentzian_curves = getattr(
            lorentzian_curves,
            "generate_double_lorentzian_curves",
            None,
        )
        self.assertIsNotNone(generate_double_lorentzian_curves)

        wavelengths = torch.linspace(3.0, 6.0, 1200)
        curve = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            width=0.02,
            center1=3.8,
            center2=5.1,
        )

        left_mask = (wavelengths >= 3.75) & (wavelengths <= 3.85)
        right_mask = (wavelengths >= 5.05) & (wavelengths <= 5.15)

        self.assertGreater(float(curve[left_mask].max()), 0.95)
        self.assertGreater(float(curve[right_mask].max()), 0.95)
        self.assertAlmostEqual(float(curve.max()), 1.0, places=5)

    def test_train_load_parameters_reads_double_target_fields(self):
        payload = {
            "structure": {
                "N_layers": 10,
                "pol": 0,
                "thickness_sup": 0.5,
                "thickness_bot": 0.05,
            },
            "materials": {
                "materials_list": ["Ge", "SiO2"],
            },
            "optics": {
                "wavelength_range": [3, 6],
                "samples_total": 100,
                "theta": 0.0,
                "n_top": 1.0,
                "n_bot": 1.0,
                "lorentz_width": 0.02,
                "lorentz_center_range_1": [3.4, 4.2],
                "lorentz_center_range_2": [4.8, 5.6],
                "min_peak_spacing": 0.4,
                "max_peak_spacing": 2.5,
                "peak_height_ratio": 1.0,
                "metal_name": "Au",
            },
            "generator": {
                "thickness_noise_dim": 8,
                "material_noise_dim": 8,
                "alpha_min": 1,
                "alpha_max": 5,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "save_interval": 1,
                "noise_level": 0.0,
                "lambda_gp": 1.0,
            },
            "optimizer": {
                "lr_gen": 1e-3,
                "lr_disc": 1e-4,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0,
            },
        }

        config_dir = Path(__file__).resolve().parent / ".tmp"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "training.yaml"
        try:
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            params = load_train_parameters(str(config_path), torch.device("cpu"))
        finally:
            if config_path.exists():
                config_path.unlink()

        self.assertEqual(params.lorentz_center_range_1, [3.4, 4.2])
        self.assertEqual(params.lorentz_center_range_2, [4.8, 5.6])
        self.assertAlmostEqual(params.min_peak_spacing, 0.4)
        self.assertAlmostEqual(params.max_peak_spacing, 2.5)
        self.assertAlmostEqual(params.peak_height_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
