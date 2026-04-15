import unittest
from pathlib import Path

import torch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from train import q_evaluator


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
