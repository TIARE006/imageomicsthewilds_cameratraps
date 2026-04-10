from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS_INPUT = Path("outputs/tables/arrivals/stress_replay_metrics.csv")
DEFAULT_FIGURE_DIR = Path("outputs/figures")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stress replay metrics file does not exist: {path}")
    return pd.read_csv(path)


def plot_metric_lines(
    df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for policy in sorted(df["policy"].unique()):
        policy_df = df[df["policy"] == policy].sort_values("stress_factor")
        ax.plot(
            policy_df["stress_factor"],
            policy_df[metric_col],
            marker="o",
            linewidth=2,
            label=policy,
        )

    ax.set_xlabel("Arrival stress factor")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stress replay policy comparison figures."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_INPUT,
        help="Path to stress_replay_metrics.csv",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for output figures.",
    )
    args = parser.parse_args()

    ensure_dir(args.figure_dir)
    df = load_metrics(args.metrics)

    plot_metric_lines(
        df=df,
        metric_col="p95_latency_ms",
        y_label="P95 latency (ms)",
        title="Stress replay: P95 latency vs arrival stress",
        output_path=args.figure_dir / "stress_replay_p95_latency.png",
    )

    plot_metric_lines(
        df=df,
        metric_col="compute_cost_sec",
        y_label="Compute cost (s)",
        title="Stress replay: total compute cost vs arrival stress",
        output_path=args.figure_dir / "stress_replay_compute_cost.png",
    )

    plot_metric_lines(
        df=df,
        metric_col="slo_compliance",
        y_label="SLO compliance",
        title="Stress replay: SLO compliance vs arrival stress",
        output_path=args.figure_dir / "stress_replay_slo_compliance.png",
    )

    plot_metric_lines(
        df=df,
        metric_col="peak_queue_length",
        y_label="Peak queue length",
        title="Stress replay: peak queue length vs arrival stress",
        output_path=args.figure_dir / "stress_replay_peak_queue.png",
    )

    print(f"Saved figures to: {args.figure_dir}")
    
if __name__ == "__main__":
    main()