from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESOURCE_INPUT = Path("outputs/tables/benchmark/camera_resource_table.csv")
DEFAULT_POLICY_INPUT = Path("outputs/tables/benchmark/camera_policy_comparison.csv")
DEFAULT_FIGURE_DIR = Path("outputs/figures")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return pd.read_csv(path)


def plot_mean_utilization(resource_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        resource_df.groupby(["scenario", "policy"], as_index=False)["utilization"]
        .mean()
        .sort_values(["scenario", "policy"])
    )

    summary["utilization_percent"] = summary["utilization"] * 100.0

    scenarios = summary["scenario"].unique().tolist()
    policies = summary["policy"].unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.25
    x_positions = list(range(len(scenarios)))

    max_percent = max(float(summary["utilization_percent"].max()), 0.01)
    y_upper = max_percent * 1.35

    safe_threshold_percent = 70.0
    stability_threshold_percent = 100.0

    for idx, policy in enumerate(policies):
        values = []
        for scenario in scenarios:
            row = summary[
                (summary["scenario"] == scenario) & (summary["policy"] == policy)
            ]
            values.append(
                float(row["utilization_percent"].iloc[0]) if not row.empty else 0.0
            )

        offset = (idx - (len(policies) - 1) / 2.0) * width
        x = [x0 + offset for x0 in x_positions]
        ax.bar(x, values, width=width, label=policy)

    if y_upper > safe_threshold_percent * 0.25:
        ax.axhline(
            safe_threshold_percent,
            linestyle="--",
            linewidth=1,
            label="safe threshold (70%)",
        )
    if y_upper > stability_threshold_percent * 0.25:
        ax.axhline(
            stability_threshold_percent,
            linestyle="--",
            linewidth=1,
            label="stability threshold (100%)",
        )

    ax.set_ylim(0, y_upper)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Mean utilization (%)")
    ax.set_xlabel("Arrival scenario")
    ax.set_title("Camera policy utilization across arrival scenarios")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_estimated_p95_latency(policy_df: pd.DataFrame, out_path: Path) -> None:
    summary = policy_df.sort_values(["scenario", "policy"]).copy()
    scenarios = summary["scenario"].unique().tolist()
    policies = summary["policy"].unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.25
    x_positions = list(range(len(scenarios)))

    all_values = []

    for idx, policy in enumerate(policies):
        values = []
        for scenario in scenarios:
            row = summary[
                (summary["scenario"] == scenario) & (summary["policy"] == policy)
            ]
            value = (
                float(row["estimated_p95_latency_ms"].iloc[0]) if not row.empty else 0.0
            )
            values.append(value)
            all_values.append(value)

        offset = (idx - (len(policies) - 1) / 2.0) * width
        x = [x0 + offset for x0 in x_positions]
        ax.bar(x, values, width=width, label=policy)

    max_latency = max(all_values) if all_values else 1.0
    ax.set_ylim(0, max_latency * 1.15)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Estimated P95 latency (ms)")
    ax.set_xlabel("Arrival scenario")
    ax.set_title("Estimated camera policy P95 latency")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot utilization and latency figures for the camera resource model."
    )
    parser.add_argument(
        "--resource_table",
        type=Path,
        default=DEFAULT_RESOURCE_INPUT,
        help="Path to camera_resource_table.csv",
    )
    parser.add_argument(
        "--policy_table",
        type=Path,
        default=DEFAULT_POLICY_INPUT,
        help="Path to camera_policy_comparison.csv",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where figures will be saved.",
    )
    args = parser.parse_args()

    ensure_dir(args.figure_dir)

    resource_df = load_csv(args.resource_table)
    policy_df = load_csv(args.policy_table)

    util_fig = args.figure_dir / "camera_policy_utilization.png"
    latency_fig = args.figure_dir / "camera_policy_p95_latency.png"

    plot_mean_utilization(resource_df, util_fig)
    plot_estimated_p95_latency(policy_df, latency_fig)

    print(f"Saved figure: {util_fig}")
    print(f"Saved figure: {latency_fig}")


if __name__ == "__main__":
    main()