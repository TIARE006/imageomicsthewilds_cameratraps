from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/benchmark/multimodal_resource_table.csv")
DEFAULT_OUTPUT = Path("outputs/tables/benchmark/exp3_summary_table.csv")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)

    required = {
        "scenario",
        "policy",
        "camera_rate_per_hour",
        "audio_rate_per_hour",
        "total_cpu_util",
        "total_gpu_util",
        "aggregate_util",
        "single_node_feasibility",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Multimodal resource table is missing required columns: {sorted(missing)}"
        )

    return df.copy()


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[
        [
            "scenario",
            "policy",
            "camera_rate_per_hour",
            "audio_rate_per_hour",
            "total_cpu_util",
            "total_gpu_util",
            "aggregate_util",
            "single_node_feasibility",
        ]
    ].copy()

    summary["cpu_util_percent"] = summary["total_cpu_util"] * 100.0
    summary["gpu_util_percent"] = summary["total_gpu_util"] * 100.0
    summary["aggregate_util_percent"] = summary["aggregate_util"] * 100.0

    summary = summary[
        [
            "scenario",
            "policy",
            "camera_rate_per_hour",
            "audio_rate_per_hour",
            "cpu_util_percent",
            "gpu_util_percent",
            "aggregate_util_percent",
            "single_node_feasibility",
        ]
    ].sort_values(["scenario", "policy"]).reset_index(drop=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact Exp 3 summary table from the multimodal resource table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to multimodal_resource_table.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output exp3_summary_table.csv",
    )
    args = parser.parse_args()

    ensure_parent(args.output)

    df = load_table(args.input)
    summary = build_summary_table(df)
    summary.to_csv(args.output, index=False)

    print(f"Saved Exp 3 summary table: {args.output}")
    print(f"Rows written: {len(summary)}")


if __name__ == "__main__":
    main()