from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/arrivals/stress_replay_metrics.csv")
DEFAULT_OUTPUT = Path("outputs/tables/arrivals/exp4_summary_table.csv")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)

    required = {
        "stress_factor",
        "policy",
        "p95_latency_ms",
        "slo_compliance",
        "peak_queue_length",
        "compute_cost_sec",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Stress replay metrics table is missing required columns: {sorted(missing)}"
        )

    return df.copy()


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[
        [
            "stress_factor",
            "policy",
            "p95_latency_ms",
            "slo_compliance",
            "peak_queue_length",
            "compute_cost_sec",
        ]
    ].copy()

    summary = summary.sort_values(["stress_factor", "policy"]).reset_index(drop=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact Exp 4 summary table from stress replay metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to stress_replay_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output exp4_summary_table.csv",
    )
    args = parser.parse_args()

    ensure_parent(args.output)

    df = load_table(args.input)
    summary = build_summary_table(df)
    summary.to_csv(args.output, index=False)

    print(f"Saved Exp 4 summary table: {args.output}")
    print(f"Rows written: {len(summary)}")


if __name__ == "__main__":
    main()