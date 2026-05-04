import unittest
from pathlib import Path

import pandas as pd
import torch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from train import q_evaluator
from train.high_quality_solution_collector import (
    initialize_high_quality_collection,
    update_high_quality_collection_summary,
)


class DualMetricTests(unittest.TestCase):
    def _build_minimal_dual_q_results(self):
        return {
            "q1_values": torch.tensor([10.0]),
            "q2_values": torch.tensor([12.0]),
            "q_min_pair_values": torch.tensor([10.0]),
            "dual_valid_mask": torch.tensor([True]),
            "double_lorentz_mse_values": torch.tensor([0.02]),
            "double_lorentz_rmse_values": torch.tensor([0.14142136]),
            "fom_lorentz_mse_values": torch.tensor([0.01]),
            "fom_lorentz_rmse_values": torch.tensor([0.1]),
            "q_score_values": torch.tensor([0.2]),
            "rmse_score_values": torch.tensor([0.8]),
            "fom_values": torch.tensor([0.4]),
            "peak_wavelengths_1": torch.tensor([3.8]),
            "peak_wavelengths_2": torch.tensor([5.0]),
            "peak_absorptions_1": torch.tensor([0.9]),
            "peak_absorptions_2": torch.tensor([0.92]),
            "left_wavelengths_1": torch.tensor([3.75]),
            "left_wavelengths_2": torch.tensor([4.95]),
            "right_wavelengths_1": torch.tensor([3.85]),
            "right_wavelengths_2": torch.tensor([5.05]),
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

    def test_summarize_q_results_uses_dual_summary_names(self):
        summary = q_evaluator.summarize_q_results(
            q_results=self._build_minimal_dual_q_results(),
            epoch=1,
            alpha=5.0,
            num_samples=1,
            lorentz_width=0.02,
            dominant_prob_threshold=0.95,
            fom_q_ref=200.0,
            fom_lorentz_width=0.01,
            fom_rmse_ref=0.05,
            fom_weight=0.5,
        )
        self.assertIn("mean_q_min_pair", summary)
        self.assertIn("max_q_min_pair", summary)
        self.assertIn("mean_double_mse", summary)
        self.assertIn("mean_double_rmse", summary)
        self.assertNotIn("mean_q", summary)
        self.assertNotIn("max_q", summary)
        self.assertNotIn("mean_mse", summary)
        self.assertNotIn("mean_rmse", summary)

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
                    "max_q_min_pair": 18.0,
                    "std_q_min_pair": 1.0,
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
            self.assertTrue((save_dir / "global_max_q_min_pair_curve.csv").exists())
            self.assertTrue((save_dir / "global_max_q_min_pair_curve.png").exists())
            self.assertFalse((save_dir / "global_max_q_curve.csv").exists())
            self.assertFalse((save_dir / "global_max_q_curve.png").exists())
            summary_df = pd.read_csv(save_dir / "q_mse_evaluation_summary.csv")
            self.assertIn("mean_q_min_pair", summary_df.columns)
            self.assertIn("dual_valid_ratio", summary_df.columns)
            self.assertNotIn("mean_q", summary_df.columns)
            self.assertNotIn("valid_ratio", summary_df.columns)
            self.assertIn("global_max_q_min_pair", summary_df.columns)
            self.assertNotIn("global_max_q", summary_df.columns)

            curve_df = pd.read_csv(save_dir / "global_max_q_min_pair_curve.csv")
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

    def test_save_global_best_sample_histories_uses_dual_file_stem(self):
        save_dir = Path(__file__).resolve().parent / ".tmp" / "global_best"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            q_results = self._build_minimal_dual_q_results()
            summary = {
                "epoch": 1,
                "alpha": 5.0,
                "epoch_best_fom": 0.4,
            }

            wavelengths = torch.linspace(3.0, 6.0, 10)
            absorption_spectra = torch.linspace(0.1, 1.0, 10).unsqueeze(0)
            thicknesses = torch.tensor([[0.1, 0.2]])
            material_probabilities = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])

            class Params:
                materials = ["Ge", "SiO2"]
                q_eval_fom_lorentz_width = 0.01
                lorentz_width = 0.02

            q_evaluator.save_global_best_sample_histories(
                summary=summary,
                q_results=q_results,
                wavelengths=wavelengths,
                absorption_spectra=absorption_spectra,
                thicknesses=thicknesses,
                material_probabilities=material_probabilities,
                params=Params(),
                save_dir=str(save_dir),
            )

            tracking_dir = save_dir / "global_best_samples"
            self.assertTrue((tracking_dir / "global_max_q_min_pair_epoch_0001_structure.txt").exists())
            self.assertTrue((tracking_dir / "global_max_q_min_pair_epoch_0001_spectrum.csv").exists())
            self.assertTrue((tracking_dir / "global_max_q_min_pair_epoch_0001_spectrum.png").exists())
            self.assertFalse((tracking_dir / "global_max_q_epoch_0001_structure.txt").exists())
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
            self.assertIn("merged_structure_1nm_key", columns)
            self.assertIn("merged_structure_10nm_key", columns)
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

    def test_high_quality_summary_dedup_by_10nm_key_with_q_then_mse_priority(self):
        save_dir = Path(__file__).resolve().parent / ".tmp" / "high_quality_dedup"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            initialize_high_quality_collection(str(save_dir), {"enabled": True})
            records = [
                {
                    "sample_id": "a",
                    "epoch": 1,
                    "alpha": 5.0,
                    "evaluation_sample_index": 1,
                    "q1": 120.0,
                    "q2": 100.0,
                    "q_min_pair": 100.0,
                    "double_lorentz_mse": 0.0040,
                    "peak_wavelength_1_um": 3.8,
                    "peak_wavelength_2_um": 5.2,
                    "peak_absorption_1": 0.95,
                    "peak_absorption_2": 0.94,
                    "fwhm_1_um": 0.01,
                    "fwhm_2_um": 0.01,
                    "total_thickness_um": 0.35,
                    "min_dominant_material_probability": 0.98,
                    "merged_layer_count": 3,
                    "merged_structure_1nm_key": "Ge:120|Si:80",
                    "merged_structure_10nm_key": "Ge:12|Si:8",
                    "sample_dir": "x/a",
                },
                {
                    "sample_id": "b",
                    "epoch": 1,
                    "alpha": 5.0,
                    "evaluation_sample_index": 2,
                    "q1": 130.0,
                    "q2": 110.0,
                    "q_min_pair": 110.0,
                    "double_lorentz_mse": 0.0060,
                    "peak_wavelength_1_um": 3.8,
                    "peak_wavelength_2_um": 5.2,
                    "peak_absorption_1": 0.96,
                    "peak_absorption_2": 0.95,
                    "fwhm_1_um": 0.01,
                    "fwhm_2_um": 0.01,
                    "total_thickness_um": 0.36,
                    "min_dominant_material_probability": 0.98,
                    "merged_layer_count": 3,
                    "merged_structure_1nm_key": "Ge:121|Si:79",
                    "merged_structure_10nm_key": "Ge:12|Si:8",
                    "sample_dir": "x/b",
                },
                {
                    "sample_id": "c",
                    "epoch": 1,
                    "alpha": 5.0,
                    "evaluation_sample_index": 3,
                    "q1": 130.0,
                    "q2": 110.0,
                    "q_min_pair": 110.0,
                    "double_lorentz_mse": 0.0030,
                    "peak_wavelength_1_um": 3.8,
                    "peak_wavelength_2_um": 5.2,
                    "peak_absorption_1": 0.96,
                    "peak_absorption_2": 0.95,
                    "fwhm_1_um": 0.01,
                    "fwhm_2_um": 0.01,
                    "total_thickness_um": 0.36,
                    "min_dominant_material_probability": 0.98,
                    "merged_layer_count": 3,
                    "merged_structure_1nm_key": "Ge:122|Si:78",
                    "merged_structure_10nm_key": "Ge:12|Si:8",
                    "sample_dir": "x/c",
                },
            ]

            result = update_high_quality_collection_summary(str(save_dir), records)
            self.assertEqual(result["total_high_quality_count"], 1)

            registry_df = pd.read_csv(save_dir / "summary" / "high_quality_solutions.csv")
            self.assertEqual(len(registry_df), 1)
            winner = registry_df.iloc[0]
            self.assertEqual(winner["sample_id"], "c")
            self.assertEqual(winner["merged_structure_10nm_key"], "Ge:12|Si:8")
            self.assertEqual(winner["merged_structure_1nm_key"], "Ge:122|Si:78")
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
            q_results = self._build_minimal_dual_q_results()
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
