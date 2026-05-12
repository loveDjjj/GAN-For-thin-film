import unittest

import pandas as pd

from analyze_high_quality_peak_clusters import (
    assign_peak_clusters,
    build_global_peak_cluster_tables,
    build_peak_cluster_tables,
)


class HighQualityPeakClusterTests(unittest.TestCase):
    def test_rounds_to_0p1_then_clusters_by_0p5(self):
        df = pd.DataFrame(
            {
                "epoch": [31, 31],
                "q_min_pair": [100.0, 101.0],
                "double_lorentz_mse": [0.001, 0.001],
                "peak_wavelength_1_um": [3.511, 3.57],
                "peak_wavelength_2_um": [5.49, 5.52],
            }
        )

        clustered = assign_peak_clusters(df, round_width=0.1, cluster_width=0.5)

        self.assertEqual(clustered["peak1_rounded_0p1_um"].tolist(), [3.5, 3.6])
        self.assertEqual(clustered["peak1_cluster_um"].tolist(), [3.5, 3.5])
        self.assertEqual(clustered["peak2_cluster_um"].tolist(), [5.5, 5.5])
        self.assertEqual(clustered["peak_pair_cluster_key"].tolist(), ["3.5_5.5", "3.5_5.5"])

    def test_selects_one_representative_per_epoch_cluster_by_q_then_mse(self):
        df = pd.DataFrame(
            {
                "sample_id": ["low_q", "high_q_high_mse", "high_q_low_mse", "other_cluster"],
                "epoch": [31, 31, 31, 31],
                "q_min_pair": [100.0, 120.0, 120.0, 90.0],
                "double_lorentz_mse": [0.001, 0.004, 0.002, 0.001],
                "peak_wavelength_1_um": [3.51, 3.57, 3.55, 4.02],
                "peak_wavelength_2_um": [5.49, 5.52, 5.55, 5.97],
                "sample_dir": ["a", "b", "c", "d"],
            }
        )

        clustered = assign_peak_clusters(df, round_width=0.1, cluster_width=0.5)
        epoch_summary, cluster_summary, representatives = build_peak_cluster_tables(clustered)

        self.assertEqual(int(epoch_summary.loc[0, "unique_peak_pair_clusters"]), 2)
        selected = representatives[representatives["peak_pair_cluster_key"] == "3.5_5.5"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["sample_id"], "high_q_low_mse")
        self.assertEqual(int(cluster_summary.loc[cluster_summary["peak_pair_cluster_key"] == "3.5_5.5", "sample_count"].iloc[0]), 3)

    def test_selects_one_global_representative_per_cluster_across_epochs(self):
        df = pd.DataFrame(
            {
                "sample_id": [
                    "epoch31_lower_q",
                    "epoch32_high_q_high_mse",
                    "epoch33_high_q_low_mse",
                    "other_cluster_best",
                ],
                "epoch": [31, 32, 33, 33],
                "q_min_pair": [100.0, 130.0, 130.0, 80.0],
                "double_lorentz_mse": [0.001, 0.006, 0.002, 0.001],
                "peak_wavelength_1_um": [3.51, 3.57, 3.55, 4.02],
                "peak_wavelength_2_um": [5.49, 5.52, 5.55, 5.97],
                "sample_dir": ["a", "b", "c", "d"],
            }
        )

        clustered = assign_peak_clusters(df, round_width=0.1, cluster_width=0.5)
        global_summary, global_representatives = build_global_peak_cluster_tables(clustered)

        selected = global_representatives[global_representatives["peak_pair_cluster_key"] == "3.5_5.5"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["sample_id"], "epoch33_high_q_low_mse")
        same_cluster_summary = global_summary[global_summary["peak_pair_cluster_key"] == "3.5_5.5"].iloc[0]
        self.assertEqual(int(same_cluster_summary["sample_count"]), 3)
        self.assertEqual(int(same_cluster_summary["epoch_count"]), 3)


if __name__ == "__main__":
    unittest.main()
