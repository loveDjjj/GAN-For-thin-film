import unittest
from pathlib import Path

import pandas as pd
import torch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from train import q_evaluator
from train.high_quality_solution_collector import initialize_high_quality_collection


class DualMetricTests(unittest.TestCase):
    def test_dual_q_metrics_returns_two_valid_q_values(self):
        compute_dual_q_metrics_torch = getattr(
            q_evaluator,
            "compute_dual_q_metrics_torch",
            None,
        )
        self.assertIsNotNone(compute_dual_q_metrics_torch)

        wavelengths = torch.linspace(3.0, 6.0, 2000)
        spectra = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            width=0.02,
            center1=3.7,
            center2=5.1,
        ).unsqueeze(0)

        metrics = compute_dual_q_metrics_torch(
            wavelengths=wavelengths,
            absorption_spectra=spectra,
            target_center_1=3.7,
            target_center_2=5.1,
            half_window=0.08,
        )

        self.assertTrue(bool(metrics["dual_valid_mask"][0].item()))
        self.assertGreater(float(metrics["q1_values"][0]), 10.0)
        self.assertGreater(float(metrics["q2_values"][0]), 10.0)
        self.assertAlmostEqual(
            float(metrics["q_min_pair_values"][0]),
            min(float(metrics["q1_values"][0]), float(metrics["q2_values"][0])),
            places=5,
        )

    def test_double_lorentzian_mse_near_zero_for_matching_target(self):
        compute_double_lorentzian_mse_torch = getattr(
            q_evaluator,
            "compute_double_lorentzian_mse_torch",
            None,
        )
        self.assertIsNotNone(compute_double_lorentzian_mse_torch)

        wavelengths = torch.linspace(3.0, 6.0, 2000)
        spectra = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            width=0.02,
            center1=3.7,
            center2=5.1,
        ).unsqueeze(0)

        mse_results = compute_double_lorentzian_mse_torch(
            wavelengths=wavelengths,
            absorption_spectra=spectra,
            peak_wavelengths_1=torch.tensor([3.7]),
            peak_wavelengths_2=torch.tensor([5.1]),
            width=0.02,
        )

        self.assertLess(float(mse_results["double_lorentz_mse_values"][0]), 1e-6)

    def test_dual_fom_uses_q_min_pair(self):
        compute_dual_fom_scores_torch = getattr(
            q_evaluator,
            "compute_dual_fom_scores_torch",
            None,
        )
        self.assertIsNotNone(compute_dual_fom_scores_torch)

        scores = compute_dual_fom_scores_torch(
            q_min_pair_values=torch.tensor([200.0]),
            rmse_values=torch.tensor([0.01]),
            dual_valid_mask=torch.tensor([True]),
            q_ref=200.0,
            rmse_ref=0.05,
            weight=0.5,
        )

        self.assertGreater(float(scores["fom_values"][0]), 0.0)

    def test_save_q_evaluation_history_accepts_dual_semantic_columns(self):
        save_dir = Path(__file__).resolve().parent / ".tmp" / "dual_history"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            history = [
                {
                    "epoch": 1,
                    "mean_q1": 10.0,
                    "mean_q2": 12.0,
                    "mean_q_min_pair": 10.0,
                    "median_q_min_pair": 9.5,
                    "max_q": 18.0,
                    "dual_valid_ratio": 0.75,
                    "mean_double_mse": 0.02,
                    "median_double_mse": 0.019,
                    "min_double_mse": 0.01,
                    "max_double_mse": 0.03,
                    "std_double_mse": 0.005,
                    "epoch_best_fom": 0.25,
                    "fully_fixed_ratio": 0.0,
                    "mean_fixed_layer_ratio": 0.0,
                    "mean_fixed_layer_count": 0.0,
                    "median_fixed_layer_count": 0.0,
                    "mean_min_dominant_material_probability": 0.3,
                    "median_min_dominant_material_probability": 0.3,
                    "dominant_material_prob_threshold": 0.95,
                }
            ]
            q_evaluator.save_q_evaluation_history(history, str(save_dir))
            self.assertTrue((save_dir / "q_mse_evaluation_summary.csv").exists())
            self.assertTrue((save_dir / "q_mse_evaluation_curves.png").exists())
            summary_df = pd.read_csv(save_dir / "q_mse_evaluation_summary.csv")
            self.assertIn("mean_q_min_pair", summary_df.columns)
            self.assertIn("dual_valid_ratio", summary_df.columns)
            self.assertNotIn("mean_q", summary_df.columns)
            self.assertNotIn("valid_ratio", summary_df.columns)

            curve_df = pd.read_csv(save_dir / "global_max_q_curve.csv")
            self.assertIn("epoch_max_q_min_pair", curve_df.columns)
            self.assertIn("global_max_q_min_pair", curve_df.columns)
            self.assertNotIn("epoch_max_q", curve_df.columns)
        finally:
            if save_dir.exists():
                for child in sorted(save_dir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                save_dir.rmdir()

    def test_high_quality_registry_uses_dual_field_names(self):
        save_dir = Path(__file__).resolve().parent / ".tmp" / "high_quality"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            initialize_high_quality_collection(str(save_dir), {"enabled": True})
            registry_df = pd.read_csv(save_dir / "summary" / "high_quality_solutions.csv", nrows=0)
            columns = set(registry_df.columns.tolist())
            self.assertIn("q1", columns)
            self.assertIn("q2", columns)
            self.assertIn("q_min_pair", columns)
            self.assertIn("double_lorentz_mse", columns)
            self.assertIn("peak_wavelength_1_um", columns)
            self.assertIn("peak_wavelength_2_um", columns)
            self.assertIn("peak_absorption_1", columns)
            self.assertIn("peak_absorption_2", columns)
            self.assertIn("fwhm_1_um", columns)
            self.assertIn("fwhm_2_um", columns)
            self.assertNotIn("q_value", columns)
            self.assertNotIn("lorentz_mse", columns)
            self.assertNotIn("peak_wavelength_um", columns)
            self.assertNotIn("peak_absorption", columns)
            self.assertNotIn("fwhm_um", columns)
        finally:
            if save_dir.exists():
                for child in sorted(save_dir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                save_dir.rmdir()

    def test_save_q_evaluation_epoch_details_use_dual_columns(self):
        save_dir = Path(__file__).resolve().parent / ".tmp" / "epoch_details"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            q_results = {
                "q_values": torch.tensor([10.0]),
                "q1_values": torch.tensor([10.0]),
                "q2_values": torch.tensor([12.0]),
                "q_min_pair_values": torch.tensor([10.0]),
                "mse_values": torch.tensor([0.02]),
                "rmse_values": torch.tensor([0.14142136]),
                "double_lorentz_mse_values": torch.tensor([0.02]),
                "double_lorentz_rmse_values": torch.tensor([0.14142136]),
                "fom_lorentz_mse_values": torch.tensor([0.01]),
                "fom_lorentz_rmse_values": torch.tensor([0.1]),
                "q_score_values": torch.tensor([0.2]),
                "rmse_score_values": torch.tensor([0.8]),
                "fom_values": torch.tensor([0.4]),
                "valid_mask": torch.tensor([True]),
                "dual_valid_mask": torch.tensor([True]),
                "peak_wavelengths": torch.tensor([4.4]),
                "peak_wavelengths_1": torch.tensor([3.8]),
                "peak_wavelengths_2": torch.tensor([5.0]),
                "peak_absorptions": torch.tensor([0.9]),
                "peak_absorptions_1": torch.tensor([0.9]),
                "peak_absorptions_2": torch.tensor([0.92]),
                "left_wavelengths": torch.tensor([4.35]),
                "left_wavelengths_1": torch.tensor([3.75]),
                "left_wavelengths_2": torch.tensor([4.95]),
                "right_wavelengths": torch.tensor([4.45]),
                "right_wavelengths_1": torch.tensor([3.85]),
                "right_wavelengths_2": torch.tensor([5.05]),
                "fwhm": torch.tensor([0.1]),
                "fwhm_1": torch.tensor([0.1]),
                "fwhm_2": torch.tensor([0.1]),
                "fixed_layer_count": torch.tensor([1]),
                "fixed_layer_ratio": torch.tensor([0.5]),
                "fully_fixed_mask": torch.tensor([False]),
                "min_dominant_material_probability": torch.tensor([0.4]),
                "mean_dominant_material_probability": torch.tensor([0.5]),
                "dominant_material_probabilities": torch.tensor([[0.6, 0.4]]),
                "dominant_material_indices": torch.tensor([[0, 1]]),
                "fixed_layer_mask": torch.tensor([[True, False]]),
            }
            summary = {
                "epoch": 1,
                "mean_q_min_pair": 10.0,
                "mean_double_mse": 0.02,
                "dual_valid_ratio": 1.0,
                "dominant_material_prob_threshold": 0.95,
                "fully_fixed_ratio": 0.0,
                "mean_fixed_layer_count": 1.0,
                "mean_min_dominant_material_probability": 0.4,
            }
            q_evaluator.save_q_evaluation_epoch(q_results, summary, str(save_dir), materials=["Ge", "SiO2"])
            details_df = pd.read_csv(save_dir / "q_mse_metrics_epoch_1.csv")
            columns = set(details_df.columns.tolist())
            self.assertIn("q1", columns)
            self.assertIn("q2", columns)
            self.assertIn("q_min_pair", columns)
            self.assertIn("double_lorentz_mse", columns)
            self.assertIn("double_lorentz_rmse", columns)
            self.assertIn("peak_wavelength_1_um", columns)
            self.assertIn("peak_wavelength_2_um", columns)
            self.assertIn("peak_absorption_1", columns)
            self.assertIn("peak_absorption_2", columns)
            self.assertIn("fwhm_1_um", columns)
            self.assertIn("fwhm_2_um", columns)
            self.assertNotIn("q_value", columns)
            self.assertNotIn("lorentz_mse", columns)
            self.assertNotIn("lorentz_rmse", columns)
            self.assertNotIn("peak_wavelength_um", columns)
            self.assertNotIn("peak_absorption", columns)
            self.assertNotIn("fwhm_um", columns)
        finally:
            if save_dir.exists():
                for child in sorted(save_dir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                save_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
