import unittest

import torch

from inference import filtering, qfactor
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


if __name__ == "__main__":
    unittest.main()
