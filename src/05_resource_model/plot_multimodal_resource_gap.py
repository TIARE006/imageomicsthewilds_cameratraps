from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/benchmark/multimodal_resource_table.csv")
DEFAULT_FIGURE_DIR = Path("outputs/figures")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return pd.read_csv(path)


def plot_multimodal_utilization(df: pd.DataFrame, out_path: Path) -> None:
    summary = df.copy()
    summary["aggregate_util_percent"] = summary["aggregate_util"] * 100.0
    summary = summary.sort_values(["scenario", "policy"])

    scenarios = summary["scenario"].unique().tolist()
    policies = summary["policy"].unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.25
    x_positions = list(range(len(scenarios)))

    all_vals = summary["aggregate_util_percent"].tolist()
    max_val = max(all_vals) if all_vals else 1.0
    y_upper = max(max_val * 1.25, 1.0)

    for idx, policy in enumerate(policies):
        values = []
        for scenario in scenarios:
            row = summary[
                (summary["scenario"] == scenario) & (summary["policy"] == policy)
            ]
            values.append(
                float(row["aggregate_util_percent"].iloc[0]) if not row.empty else 0.0
            )

        offset = (idx - (len(policies) - 1) / 2.0) * width
        x = [x0 + offset for x0 in x_positions]
        ax.bar(x, values, width=width, label=policy)

    ax.set_ylim(0, y_upper)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Aggregate utilization (%)")
    ax.set_xlabel("Multimodal arrival scenario")
    ax.set_title("Multimodal aggregate utilization across policies")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cpu_gpu_split(df: pd.DataFrame, out_path: Path) -> None:
    summary = df.copy()
    summary["cpu_util_percent"] = summary["total_cpu_util"] * 100.0
    summary["gpu_util_percent"] = summary["total_gpu_util"] * 100.0
    summary = summary.sort_values(["scenario", "policy"])

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"{row['scenario']}\n{row['policy']}" for _, row in summary.iterrows()]
    x = list(range(len(summary)))

    ax.bar(x, summary["cpu_util_percent"], label="CPU (audio)")
    ax.bar(
        x,
        summary["gpu_util_percent"],
        bottom=summary["cpu_util_percent"],
        label="GPU (camera)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Utilization (%)")
    ax.set_xlabel("Scenario and policy")
    ax.set_title("Multimodal CPU/GPU utilization split")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multimodal resource-model figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to multimodal_resource_table.csv",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for output figures.",
    )
    args = parser.parse_args()

    ensure_dir(args.figure_dir)

    df = load_table(args.input)

    plot_multimodal_utilization(
        df,
        args.figure_dir / "multimodal_capacity_gap.png",
    )
    plot_cpu_gpu_split(
        df,
        args.figure_dir / "multimodal_cpu_gpu_split.png",
    )

    print(f"Saved figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()