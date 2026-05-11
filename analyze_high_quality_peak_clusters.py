"""Analyze per-epoch dual-peak coverage in collected high-quality solutions."""

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "epoch",
    "q_min_pair",
    "double_lorentz_mse",
    "peak_wavelength_1_um",
    "peak_wavelength_2_um",
}


def _round_to_step(values, step):
    return (values / step).round() * step


def _cluster_floor(values, step):
    return (values / step).apply(math.floor) * step


def _format_cluster(value):
    return f"{float(value):.1f}"


def assign_peak_clusters(df, round_width=0.1, cluster_width=0.5):
    """Add rounded peak positions and 0.5um-bin dual-peak cluster keys."""
    if round_width <= 0 or cluster_width <= 0:
        raise ValueError("round_width and cluster_width must be positive")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clustered = df.copy()
    clustered["peak1_rounded_0p1_um"] = _round_to_step(clustered["peak_wavelength_1_um"].astype(float), round_width)
    clustered["peak2_rounded_0p1_um"] = _round_to_step(clustered["peak_wavelength_2_um"].astype(float), round_width)
    clustered["peak1_cluster_um"] = _cluster_floor(clustered["peak1_rounded_0p1_um"], cluster_width)
    clustered["peak2_cluster_um"] = _cluster_floor(clustered["peak2_rounded_0p1_um"], cluster_width)
    clustered["peak_pair_cluster_key"] = (
        clustered["peak1_cluster_um"].map(_format_cluster)
        + "_"
        + clustered["peak2_cluster_um"].map(_format_cluster)
    )
    return clustered


def build_peak_cluster_tables(clustered_df):
    """Build epoch, cluster, and representative tables."""
    sort_columns = ["epoch", "peak_pair_cluster_key", "q_min_pair", "double_lorentz_mse"]
    sorted_df = clustered_df.sort_values(sort_columns, ascending=[True, True, False, True])
    representatives = sorted_df.drop_duplicates(subset=["epoch", "peak_pair_cluster_key"], keep="first").copy()

    epoch_summary = (
        clustered_df.groupby("epoch")
        .agg(
            total_high_quality_samples=("sample_id", "count"),
            unique_peak_pair_clusters=("peak_pair_cluster_key", "nunique"),
            best_q_min_pair=("q_min_pair", "max"),
            lowest_double_lorentz_mse=("double_lorentz_mse", "min"),
        )
        .reset_index()
    )

    cluster_summary = (
        clustered_df.groupby(["epoch", "peak_pair_cluster_key", "peak1_cluster_um", "peak2_cluster_um"])
        .agg(
            sample_count=("sample_id", "count"),
            best_q_min_pair=("q_min_pair", "max"),
            best_double_lorentz_mse=("double_lorentz_mse", "min"),
            peak1_min_um=("peak_wavelength_1_um", "min"),
            peak1_max_um=("peak_wavelength_1_um", "max"),
            peak2_min_um=("peak_wavelength_2_um", "min"),
            peak2_max_um=("peak_wavelength_2_um", "max"),
        )
        .reset_index()
    )

    representative_columns = [
        "epoch",
        "peak_pair_cluster_key",
        "peak1_cluster_um",
        "peak2_cluster_um",
        "sample_id",
        "q_min_pair",
        "double_lorentz_mse",
        "peak_wavelength_1_um",
        "peak_wavelength_2_um",
        "sample_dir",
    ]
    representative_columns = [column for column in representative_columns if column in representatives.columns]
    representatives = representatives[representative_columns].reset_index(drop=True)

    return epoch_summary, cluster_summary, representatives


def _load_from_structure_json(high_quality_dir):
    rows = []
    for structure_path in sorted(high_quality_dir.glob("epoch_*/epoch_*_sample_*/structure.json")):
        with structure_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", {})
        sample_dir = structure_path.parent
        rows.append(
            {
                "sample_id": payload.get("sample_id", sample_dir.name),
                "epoch": int(payload.get("epoch")),
                "q_min_pair": float(metrics.get("q_min_pair", 0.0)),
                "double_lorentz_mse": float(metrics.get("double_lorentz_mse", 0.0)),
                "peak_wavelength_1_um": float(metrics.get("peak_wavelength_1_um")),
                "peak_wavelength_2_um": float(metrics.get("peak_wavelength_2_um")),
                "sample_dir": str(sample_dir),
            }
        )
    return pd.DataFrame(rows)


def load_high_quality_solutions(high_quality_dir):
    """Load high-quality solutions from summary CSV or per-sample JSON files."""
    high_quality_dir = Path(high_quality_dir)
    summary_csv = high_quality_dir / "summary" / "high_quality_solutions.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
    else:
        df = _load_from_structure_json(high_quality_dir)

    if df.empty:
        raise ValueError(f"No high-quality solutions found under {high_quality_dir}")
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"sample_{index:05d}" for index in range(len(df))]
    if "sample_dir" not in df.columns:
        df["sample_dir"] = ""

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    return df


def _plot_epoch_counts(epoch_summary, output_dir):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(epoch_summary["epoch"].astype(str), epoch_summary["unique_peak_pair_clusters"], color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Unique Dual-Peak Clusters")
    ax.set_title("Unique Dual-Peak Clusters per Epoch")
    ax.grid(True, axis="y", alpha=0.3)
    if len(epoch_summary) > 12:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "epoch_unique_peak_pair_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_peak_pair_scatter(cluster_summary, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        cluster_summary["peak1_cluster_um"],
        cluster_summary["peak2_cluster_um"],
        s=cluster_summary["sample_count"].clip(lower=1) * 20,
        c=cluster_summary["epoch"],
        cmap="viridis",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Peak 1 Cluster (um)")
    ax.set_ylabel("Peak 2 Cluster (um)")
    ax.set_title("Dual-Peak Cluster Coverage")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Epoch")
    fig.tight_layout()
    fig.savefig(output_dir / "peak_pair_cluster_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_epoch_cluster_heatmap(cluster_summary, output_dir):
    pivot = cluster_summary.pivot_table(
        index="epoch",
        columns="peak_pair_cluster_key",
        values="sample_count",
        aggfunc="sum",
        fill_value=0,
    )
    fig_width = max(8, min(24, 0.45 * len(pivot.columns) + 4))
    fig_height = max(4, min(18, 0.35 * len(pivot.index) + 3))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Dual-Peak Cluster")
    ax.set_ylabel("Epoch")
    ax.set_title("High-Quality Sample Counts by Epoch and Dual-Peak Cluster")
    fig.colorbar(image, ax=ax, label="Sample Count")
    fig.tight_layout()
    fig.savefig(output_dir / "epoch_peak_pair_cluster_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_quality(representatives, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        representatives["q_min_pair"],
        representatives["double_lorentz_mse"],
        c=representatives["epoch"],
        cmap="plasma",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Representative Q_min_pair")
    ax.set_ylabel("Representative Double-Lorentzian MSE")
    ax.set_title("Representative Quality by Epoch")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Epoch")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_q_mse_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plots(epoch_summary, cluster_summary, representatives, output_dir):
    _plot_epoch_counts(epoch_summary, output_dir)
    _plot_peak_pair_scatter(cluster_summary, output_dir)
    _plot_epoch_cluster_heatmap(cluster_summary, output_dir)
    _plot_representative_quality(representatives, output_dir)


def copy_representative_samples(representatives, output_dir):
    rep_dir = output_dir / "representative_solutions"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for row in representatives.itertuples(index=False):
        sample_dir = Path(getattr(row, "sample_dir", ""))
        if not sample_dir.exists():
            continue
        target_dir = rep_dir / f"epoch_{int(row.epoch):04d}" / f"peak_{row.peak_pair_cluster_key}" / sample_dir.name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(sample_dir, target_dir)


def analyze_high_quality_peak_clusters(
    high_quality_dir,
    output_dir=None,
    round_width=0.1,
    cluster_width=0.5,
    copy_representatives=False,
):
    high_quality_dir = Path(high_quality_dir)
    output_dir = Path(output_dir) if output_dir else high_quality_dir / "peak_cluster_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_high_quality_solutions(high_quality_dir)
    clustered = assign_peak_clusters(df, round_width=round_width, cluster_width=cluster_width)
    epoch_summary, cluster_summary, representatives = build_peak_cluster_tables(clustered)

    clustered.to_csv(output_dir / "high_quality_solutions_with_peak_clusters.csv", index=False)
    epoch_summary.to_csv(output_dir / "epoch_peak_cluster_summary.csv", index=False)
    cluster_summary.to_csv(output_dir / "epoch_peak_pair_cluster_summary.csv", index=False)
    representatives.to_csv(output_dir / "representative_solutions.csv", index=False)

    save_plots(epoch_summary, cluster_summary, representatives, output_dir)
    if copy_representatives:
        copy_representative_samples(representatives, output_dir)

    summary = {
        "input_high_quality_dir": str(high_quality_dir),
        "output_dir": str(output_dir),
        "round_width_um": float(round_width),
        "cluster_width_um": float(cluster_width),
        "total_samples": int(len(clustered)),
        "epochs_with_samples": int(epoch_summary["epoch"].nunique()),
        "total_epoch_peak_pair_clusters": int(len(cluster_summary)),
        "total_representative_solutions": int(len(representatives)),
    }
    with (output_dir / "peak_cluster_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dual-peak position clusters in high-quality solutions.")
    parser.add_argument("--high_quality_dir", required=True, help="Path to high_quality_solutions directory.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to high_quality_dir/peak_cluster_analysis.")
    parser.add_argument("--round_width", type=float, default=0.1, help="Peak rounding width in um before clustering.")
    parser.add_argument("--cluster_width", type=float, default=0.5, help="Peak cluster width in um.")
    parser.add_argument("--copy_representatives", action="store_true", help="Copy representative sample folders.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = analyze_high_quality_peak_clusters(
        high_quality_dir=args.high_quality_dir,
        output_dir=args.output_dir,
        round_width=args.round_width,
        cluster_width=args.cluster_width,
        copy_representatives=args.copy_representatives,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
