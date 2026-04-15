import unittest
from pathlib import Path

import torch

from inference import filtering, qfactor
from inference import visualization
from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves


class DoubleInferenceTests(unittest.TestCase):
    def test_dual_window_weighted_rmse_prefers_matching_double_peak(self):
        wavelengths = torch.linspace(3.0, 6.0, 1500)
        target = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            width=0.02,
            center1=3.8,
            center2=5.0,
        )
        good = target.unsqueeze(0)
        bad = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            width=0.02,
            center1=3.8,
            center2=4.1,
        ).unsqueeze(0)
        spectra = torch.cat([bad, good], dim=0)

        indices, _ = filtering.select_best_samples(
            absorption_spectra=spectra,
            wavelengths=wavelengths,
            target=target,
            centers=(3.8, 5.0),
            region_width=0.05,
            weight_factor=10.0,
            num_best=1,
        )

        self.assertEqual(int(indices[0].item()), 1)

    def test_compute_dual_q_for_spectra_returns_q_min_pair(self):
        compute_dual_q_for_spectra = getattr(qfactor, "compute_dual_q_for_spectra", None)
        self.assertIsNotNone(compute_dual_q_for_spectra)

        wavelengths = torch.linspace(3.0, 6.0, 1500)
        spectra = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            width=0.02,
            center1=3.8,
            center2=5.0,
        ).unsqueeze(0)

        results = compute_dual_q_for_spectra(
            wavelengths,
            spectra,
            center_1=3.8,
            center_2=5.0,
            half_window=0.08,
        )

        self.assertTrue(bool(results["dual_valid_mask"][0].item()))
        self.assertGreater(float(results["q_min_pair_values"][0]), 10.0)

    def test_save_best_results_writes_target_spectrum_file(self):
        class Params:
            materials = ["Ge", "SiO2"]

        wavelengths = torch.linspace(3.0, 6.0, 20).numpy()
        target = generate_double_lorentzian_curves(
            wavelengths=torch.linspace(3.0, 6.0, 20),
            width=0.02,
            center1=3.8,
            center2=5.0,
        ).numpy()
        absorption = target.reshape(1, -1)
        thicknesses = torch.tensor([[0.1, 0.2]])
        probs = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])

        tmpdir = Path(__file__).resolve().parent / ".tmp" / "best_results"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            save_dir = visualization.save_best_results(
                output_dir=str(tmpdir),
                wavelengths=wavelengths,
                thicknesses=thicknesses,
                P=probs,
                absorption_spectra=absorption,
                best_indices=[0],
                best_rmse=[0.1],
                target=target,
                params=Params(),
                original_indices=[0],
            )
            self.assertTrue((Path(save_dir) / "target_spectrum.xlsx").exists())
        finally:
            if tmpdir.exists():
                for child in sorted(tmpdir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                tmpdir.rmdir()


if __name__ == "__main__":
    unittest.main()
