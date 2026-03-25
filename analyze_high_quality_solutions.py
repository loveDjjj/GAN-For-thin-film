import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate and visualize collected high-quality solutions."
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to high_quality_solutions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save deduplicated CSVs and plots. Defaults to <csv_dir>/deduplicated_analysis",
    )
    parser.add_argument(
        "--q_threshold",
        type=float,
        default=250.0,
        help="Optional horizontal Q threshold line shown in plots.",
    )
    return parser.parse_args()


def resolve_sample_dir(sample_dir_value, csv_path):
    sample_dir = Path(str(sample_dir_value))
    candidates = []

    if sample_dir.is_absolute():
        candidates.append(sample_dir)
    else:
        candidates.append(Path.cwd() / sample_dir)
        candidates.append(csv_path.parent / sample_dir)
        candidates.append(csv_path.parent.parent / sample_dir)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def quantize_thickness_nm(thickness_um):
    return int(np.floor(float(thickness_um) * 1000.0 + 1e-9))


def build_structure_sequence(structure_payload):
    merged_layers = structure_payload.get("merged_layers") or []
    if merged_layers:
        sequence_items = []
        for layer in merged_layers:
            material = str(layer["material"])
            thickness_nm = quantize_thickness_nm(layer["merged_thickness_um"])
            sequence_items.append(f"{material}_{thickness_nm}nm")
        return sequence_items

    original_layers = structure_payload.get("original_layers") or []
    if original_layers:
        sequence_items = []
        for layer in original_layers:
            material = str(layer["dominant_material"])
            thickness_nm = quantize_thickness_nm(layer["thickness_um"])
            sequence_items.append(f"{material}_{thickness_nm}nm")
        return sequence_items

    return []


def load_structure_sequences(csv_df, csv_path):
    enriched_rows = []
    missing_structure_count = 0

    for _, row in csv_df.iterrows():
        sample_dir = resolve_sample_dir(row["sample_dir"], csv_path)
        if sample_dir is None:
            missing_structure_count += 1
            continue

        structure_path = sample_dir / "structure.json"
        if not structure_path.exists():
            missing_structure_count += 1
            continue

        with structure_path.open("r", encoding="utf-8") as file:
            structure_payload = json.load(file)

        sequence_items = build_structure_sequence(structure_payload)
        if not sequence_items:
            missing_structure_count += 1
            continue

        sequence_total_thickness_nm = int(
            sum(quantize_thickness_nm(layer["merged_thickness_um"]) for layer in structure_payload.get("merged_layers", []))
        )
        if sequence_total_thickness_nm <= 0:
            sequence_total_thickness_nm = int(
                sum(quantize_thickness_nm(layer["thickness_um"]) for layer in structure_payload.get("original_layers", []))
            )

        enriched_row = row.to_dict()
        enriched_row["sample_dir"] = str(sample_dir)
        enriched_row["structure_sequence"] = "[" + ", ".join(sequence_items) + "]"
        enriched_row["structure_sequence_key"] = "|".join(sequence_items)
        enriched_row["sequence_layer_count"] = len(sequence_items)
        enriched_row["sequence_total_thickness_nm"] = sequence_total_thickness_nm
        enriched_rows.append(enriched_row)

    return pd.DataFrame(enriched_rows), missing_structure_count


def deduplicate_sequences(sequence_df):
    if sequence_df.empty:
        return sequence_df

    ranking_df = sequence_df.sort_values(
        by=["q_value", "peak_absorption", "lorentz_mse"],
        ascending=[False, False, True],
    ).copy()

    dedup_rows = []
    for _, group in ranking_df.groupby("structure_sequence_key", sort=False):
        representative = group.iloc[0]
        dedup_rows.append(
            {
                "structure_sequence": representative["structure_sequence"],
                "structure_sequence_key": representative["structure_sequence_key"],
                "occurrence_count": int(len(group)),
                "sequence_layer_count": int(representative["sequence_layer_count"]),
                "sequence_total_thickness_nm": int(representative["sequence_total_thickness_nm"]),
                "representative_sample_id": representative["sample_id"],
                "representative_epoch": int(representative["epoch"]),
                "representative_alpha": float(representative["alpha"]),
                "representative_sample_dir": representative["sample_dir"],
                "q_value": float(representative["q_value"]),
                "lorentz_mse": float(representative["lorentz_mse"]),
                "peak_wavelength_um": float(representative["peak_wavelength_um"]),
                "peak_absorption": float(representative["peak_absorption"]),
                "fwhm_um": float(representative["fwhm_um"]),
                "total_thickness_um": float(representative["total_thickness_um"]),
                "min_dominant_material_probability": float(representative["min_dominant_material_probability"]),
                "merged_layer_count": int(representative["merged_layer_count"]),
                "mean_q_value": float(group["q_value"].mean()),
                "max_q_value": float(group["q_value"].max()),
                "mean_lorentz_mse": float(group["lorentz_mse"].mean()),
                "min_lorentz_mse": float(group["lorentz_mse"].min()),
                "mean_peak_wavelength_um": float(group["peak_wavelength_um"].mean()),
                "mean_peak_absorption": float(group["peak_absorption"].mean()),
                "max_peak_absorption": float(group["peak_absorption"].max()),
                "mean_total_thickness_um": float(group["total_thickness_um"].mean()),
            }
        )

    dedup_df = pd.DataFrame(dedup_rows)
    return dedup_df.sort_values(
        by=["q_value", "peak_absorption", "lorentz_mse"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_plot_dataframe(dedup_df):
    plot_df = dedup_df.copy()
    plot_df = plot_df.sort_values(by=["peak_wavelength_um", "q_value"]).reset_index(drop=True)
    return plot_df


def scatter_with_marginals(
    plot_df,
    output_path,
    color_column,
    color_label,
    title,
    q_threshold=250.0,
    cmap="jet",
    vmin=None,
    vmax=None,
    norm=None,
):
    if plot_df.empty:
        return

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[7.0, 1.2, 0.18],
        height_ratios=[1.2, 7.0],
        wspace=0.08,
        hspace=0.08,
    )

    ax_top = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_main)
    ax_color = fig.add_subplot(grid[1, 2])

    scatter = ax_main.scatter(
        plot_df["peak_wavelength_um"],
        plot_df["q_value"],
        c=plot_df[color_column],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        s=18,
        alpha=0.9,
        edgecolors="red",
        linewidths=0.4,
    )

    ax_main.set_xlabel("Peak Wavelength (um)")
    ax_main.set_ylabel("Q-factor")
    ax_main.set_title(title)
    ax_main.grid(True, alpha=0.25)
    if q_threshold is not None:
        ax_main.axhline(y=q_threshold, color="red", linestyle="--", linewidth=1.4, dashes=(5, 4))

    ax_top.hist(
        plot_df["peak_wavelength_um"],
        bins=min(60, max(10, len(plot_df) // 5)),
        color="#20d9df",
        edgecolor="#20d9df",
        alpha=0.95,
    )
    ax_top.set_ylabel("Count")
    ax_top.grid(True, axis="y", alpha=0.2)
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_right.hist(
        plot_df["q_value"],
        bins=min(60, max(10, len(plot_df) // 5)),
        orientation="horizontal",
        color="#3cff39",
        edgecolor="#3cff39",
        alpha=0.95,
    )
    ax_right.set_xlabel("Count")
    ax_right.grid(True, axis="x", alpha=0.2)
    ax_right.tick_params(axis="y", labelleft=False)

    colorbar = fig.colorbar(scatter, cax=ax_color)
    colorbar.set_label(color_label)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_json(output_dir, original_count, sequence_count, missing_structure_count):
    payload = {
        "original_solution_count": int(original_count),
        "deduplicated_sequence_count": int(sequence_count),
        "missing_structure_count": int(missing_structure_count),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (csv_path.parent / "deduplicated_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(csv_path)
    sequence_df, missing_structure_count = load_structure_sequences(raw_df, csv_path)
    if sequence_df.empty:
        raise RuntimeError("No valid structures were found. Please check sample_dir and structure.json paths.")

    sequence_df = sequence_df.sort_values(
        by=["peak_wavelength_um", "q_value"],
        ascending=[True, False],
    ).reset_index(drop=True)
    dedup_df = deduplicate_sequences(sequence_df)
    plot_df = build_plot_dataframe(dedup_df)

    sequence_csv_path = output_dir / "high_quality_solutions_with_sequences.csv"
    dedup_csv_path = output_dir / "high_quality_solutions_deduplicated.csv"
    sequence_df.to_csv(sequence_csv_path, index=False)
    dedup_df.to_csv(dedup_csv_path, index=False)

    absorption_plot_path = output_dir / "peak_wavelength_q_absorption.png"
    thickness_plot_path = output_dir / "peak_wavelength_q_total_thickness.png"
    mse_plot_path = output_dir / "peak_wavelength_q_mse.png"

    scatter_with_marginals(
        plot_df=plot_df,
        output_path=absorption_plot_path,
        color_column="peak_absorption",
        color_label="Peak Absorption",
        title="Unique High-Quality Structures: Wavelength vs Q (Color = Peak Absorption)",
        q_threshold=args.q_threshold,
        cmap="jet",
        vmin=0.8,
        vmax=1.0,
    )
    scatter_with_marginals(
        plot_df=plot_df,
        output_path=thickness_plot_path,
        color_column="total_thickness_um",
        color_label="Total Thickness (um)",
        title="Unique High-Quality Structures: Wavelength vs Q (Color = Total Thickness)",
        q_threshold=args.q_threshold,
        cmap="viridis",
    )

    mse_positive = plot_df["lorentz_mse"].clip(lower=np.finfo(np.float64).tiny)
    mse_min = float(mse_positive.min())
    mse_max = float(mse_positive.max())
    if np.isclose(mse_min, mse_max):
        mse_max = mse_min * 1.01
    mse_norm = mcolors.LogNorm(vmin=mse_min, vmax=mse_max)
    plot_df = plot_df.assign(_lorentz_mse_plot=mse_positive)
    scatter_with_marginals(
        plot_df=plot_df,
        output_path=mse_plot_path,
        color_column="_lorentz_mse_plot",
        color_label="Lorentzian MSE",
        title="Unique High-Quality Structures: Wavelength vs Q (Color = Lorentzian MSE)",
        q_threshold=args.q_threshold,
        cmap="plasma_r",
        norm=mse_norm,
    )

    save_summary_json(
        output_dir=output_dir,
        original_count=len(raw_df),
        sequence_count=len(dedup_df),
        missing_structure_count=missing_structure_count,
    )

    print(f"Annotated CSV saved to: {sequence_csv_path}")
    print(f"Deduplicated CSV saved to: {dedup_csv_path}")
    print(f"Absorption plot saved to: {absorption_plot_path}")
    print(f"Thickness plot saved to: {thickness_plot_path}")
    print(f"MSE plot saved to: {mse_plot_path}")
    print(f"Summary JSON saved to: {output_dir / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
