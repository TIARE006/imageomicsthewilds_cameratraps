from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_BENCHMARK_INPUT = Path("outputs/tables/benchmark/benchmark_summary.csv")
DEFAULT_ARRIVAL_INPUT = Path("outputs/tables/arrivals/arrival_summary.csv")
DEFAULT_OUTPUT = Path("outputs/tables/benchmark/camera_resource_table.csv")


MODEL_COLUMN_CANDIDATES = ["model", "Model", "benchmark", "Benchmark"]
LATENCY_COLUMN_CANDIDATES = [
    "avg_latency_ms",
    "average_latency_ms",
    "mean_latency_ms",
    "latency_ms",
    "avg_ms",
]
THROUGHPUT_COLUMN_CANDIDATES = [
    "throughput_per_sec",
    "throughput_hz",
    "throughput",
    "samples_per_sec",
]
DEVICE_COLUMN_CANDIDATES = ["device", "Device"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a {label} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_benchmark_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark summary file does not exist: {path}")

    df = pd.read_csv(path)

    model_col = find_column(df, MODEL_COLUMN_CANDIDATES, "model")
    latency_col = find_column(df, LATENCY_COLUMN_CANDIDATES, "latency")
    device_col: Optional[str] = None
    for candidate in DEVICE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            device_col = candidate
            break

    keep_cols = [model_col, latency_col]
    if device_col is not None:
        keep_cols.append(device_col)

    result = df[keep_cols].copy()
    result = result.rename(columns={model_col: "model", latency_col: "mean_latency_ms"})

    if device_col is not None:
        result = result.rename(columns={device_col: "device"})
    else:
        result["device"] = "unknown"

    result["model"] = result["model"].astype(str)
    result["device"] = result["device"].astype(str)
    result["mean_latency_ms"] = pd.to_numeric(result["mean_latency_ms"], errors="coerce")
    result = result.dropna(subset=["mean_latency_ms"]).reset_index(drop=True)

    return result


def load_arrival_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arrival summary file does not exist: {path}")

    df = pd.read_csv(path)

    required = {
        "site_cam",
        "total_events",
        "avg_rate_per_hour",
        "p95_rate_per_hour",
        "peak_rate_per_hour",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Arrival summary is missing required columns: {sorted(missing)}"
        )

    result = df.copy()
    for col in [
        "total_events",
        "avg_rate_per_hour",
        "p95_rate_per_hour",
        "peak_rate_per_hour",
    ]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(
        subset=["total_events", "avg_rate_per_hour", "p95_rate_per_hour", "peak_rate_per_hour"]
    ).reset_index(drop=True)

    return result


def classify_feasibility(utilization: float) -> str:
    if utilization < 0.7:
        return "safe"
    if utilization < 1.0:
        return "risky"
    return "not_feasible"


def expand_policy_rows(arrival_df: pd.DataFrame, model_name: str, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    matches = benchmark_df[benchmark_df["model"].str.lower().str.contains(model_name.lower(), na=False)]
    if matches.empty:
        raise ValueError(
            f"Could not find a benchmark row for model '{model_name}'. "
            f"Available models: {benchmark_df['model'].tolist()}"
        )
    return matches.iloc[[0]].copy()


def build_camera_resource_table(
    benchmark_df: pd.DataFrame,
    arrival_df: pd.DataFrame,
    selective_hit_rate: float,
) -> pd.DataFrame:
    yolo_row = expand_policy_rows(arrival_df, "yolo", benchmark_df).iloc[0]
    bioclip_row = expand_policy_rows(arrival_df, "bioclip", benchmark_df).iloc[0]

    rows = []

    for _, site_row in arrival_df.iterrows():
        site_cam = site_row["site_cam"]

        for rate_label, rate_hour_col in [
            ("average", "avg_rate_per_hour"),
            ("p95", "p95_rate_per_hour"),
            ("peak", "peak_rate_per_hour"),
        ]:
            arrival_rate_per_hour = float(site_row[rate_hour_col])
            arrival_rate_per_sec = arrival_rate_per_hour / 3600.0

            policies = [
                {
                    "policy": "process_all",
                    "service_time_ms": float(yolo_row["mean_latency_ms"]) + float(bioclip_row["mean_latency_ms"]),
                    "device": "mixed_or_unknown",
                    "coverage_estimate": 1.0,
                },
                {
                    "policy": "detection_only",
                    "service_time_ms": float(yolo_row["mean_latency_ms"]),
                    "device": str(yolo_row["device"]),
                    "coverage_estimate": 1.0,
                },
                {
                    "policy": "selective_classification",
                    "service_time_ms": float(yolo_row["mean_latency_ms"])
                    + selective_hit_rate * float(bioclip_row["mean_latency_ms"]),
                    "device": "mixed_or_unknown",
                    "coverage_estimate": 1.0,
                },
            ]

            for policy in policies:
                service_time_sec = policy["service_time_ms"] / 1000.0
                service_rate_per_sec = 1.0 / service_time_sec if service_time_sec > 0 else 0.0
                service_rate_per_hour = service_rate_per_sec * 3600.0
                utilization = arrival_rate_per_sec * service_time_sec
                feasibility = classify_feasibility(utilization)

                rows.append(
                    {
                        "site_cam": site_cam,
                        "modality": "camera",
                        "scenario": rate_label,
                        "policy": policy["policy"],
                        "arrival_rate_per_hour": arrival_rate_per_hour,
                        "arrival_rate_per_sec": arrival_rate_per_sec,
                        "service_time_ms": policy["service_time_ms"],
                        "service_time_sec": service_time_sec,
                        "service_rate_per_sec": service_rate_per_sec,
                        "service_rate_per_hour": service_rate_per_hour,
                        "utilization": utilization,
                        "feasibility": feasibility,
                        "coverage_estimate": policy["coverage_estimate"],
                        "notes": (
                            "Selective classification assumes a fixed detection hit rate."
                            if policy["policy"] == "selective_classification"
                            else ""
                        ),
                    }
                )

    result = pd.DataFrame(rows).sort_values(["site_cam", "scenario", "policy"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a camera resource table from benchmark and arrival summaries."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Path to benchmark_summary.csv",
    )
    parser.add_argument(
        "--arrivals",
        type=Path,
        default=DEFAULT_ARRIVAL_INPUT,
        help="Path to arrival_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output camera_resource_table.csv",
    )
    parser.add_argument(
        "--selective_hit_rate",
        type=float,
        default=0.20,
        help="Fraction of images that trigger BioCLIP in the selective classification policy.",
    )
    args = parser.parse_args()

    ensure_parent(args.output)

    benchmark_df = load_benchmark_summary(args.benchmark)
    arrival_df = load_arrival_summary(args.arrivals)

    resource_table = build_camera_resource_table(
        benchmark_df=benchmark_df,
        arrival_df=arrival_df,
        selective_hit_rate=args.selective_hit_rate,
    )

    resource_table.to_csv(args.output, index=False)

    print(f"Saved resource table: {args.output}")
    print(f"Rows written: {len(resource_table)}")
    print("Policies included: process_all, detection_only, selective_classification")


if __name__ == "__main__":
    main()