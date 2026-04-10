from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_BENCHMARK_INPUT = Path("outputs/tables/benchmark/benchmark_summary.csv")
DEFAULT_ARRIVAL_INPUT = Path("outputs/tables/arrivals/arrival_summary.csv")
DEFAULT_OUTPUT = Path("outputs/tables/benchmark/multimodal_resource_table.csv")


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
    for col in ["avg_rate_per_hour", "p95_rate_per_hour", "peak_rate_per_hour"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(
        subset=["avg_rate_per_hour", "p95_rate_per_hour", "peak_rate_per_hour"]
    ).reset_index(drop=True)

    return result


def get_model_latency_ms(benchmark_df: pd.DataFrame, model_keyword: str) -> float:
    matches = benchmark_df[
        benchmark_df["model"].str.lower().str.contains(model_keyword.lower(), na=False)
    ]
    if matches.empty:
        raise ValueError(
            f"Could not find benchmark row for model keyword '{model_keyword}'. "
            f"Available models: {benchmark_df['model'].tolist()}"
        )
    return float(matches.iloc[0]["mean_latency_ms"])


def classify_feasibility(util: float) -> str:
    if util < 0.7:
        return "safe"
    if util < 1.0:
        return "risky"
    return "not_feasible"


def scenario_multiplier(name: str) -> float:
    mapping = {
        "nominal": 1.0,
        "busy": 2.0,
        "worst_case": 5.0,
    }
    if name not in mapping:
        raise ValueError(f"Unknown scenario '{name}'. Expected one of {list(mapping.keys())}")
    return mapping[name]


def build_multimodal_resource_table(
    benchmark_df: pd.DataFrame,
    arrival_df: pd.DataFrame,
    audio_nominal_rate_per_hour: float,
    audio_busy_rate_per_hour: float,
    audio_worst_rate_per_hour: float,
    selective_hit_rate: float,
) -> pd.DataFrame:
    yolo_ms = get_model_latency_ms(benchmark_df, "yolo")
    bioclip_ms = get_model_latency_ms(benchmark_df, "bioclip")
    birdnet_ms = get_model_latency_ms(benchmark_df, "birdnet")

    camera_avg_rate = float(arrival_df["avg_rate_per_hour"].mean())
    camera_p95_rate = float(arrival_df["p95_rate_per_hour"].mean())
    camera_peak_rate = float(arrival_df["peak_rate_per_hour"].mean())

    camera_rate_map = {
        "nominal": camera_avg_rate,
        "busy": camera_p95_rate,
        "worst_case": camera_peak_rate,
    }
    audio_rate_map = {
        "nominal": audio_nominal_rate_per_hour,
        "busy": audio_busy_rate_per_hour,
        "worst_case": audio_worst_rate_per_hour,
    }

    policy_defs = [
        {
            "policy": "process_all",
            "camera_service_ms": yolo_ms + bioclip_ms,
            "audio_service_ms": birdnet_ms,
            "camera_notes": "All camera images run detection and classification.",
            "audio_notes": "All audio clips run BirdNET immediately.",
        },
        {
            "policy": "priority_based",
            "camera_service_ms": yolo_ms,
            "audio_service_ms": birdnet_ms,
            "camera_notes": "Camera is limited to detection-only.",
            "audio_notes": "Audio is represented as the same service time but lower scheduling priority.",
        },
        {
            "policy": "adaptive",
            "camera_service_ms": yolo_ms + selective_hit_rate * bioclip_ms,
            "audio_service_ms": birdnet_ms,
            "camera_notes": "Secondary camera classification triggers only on estimated positives.",
            "audio_notes": "Audio is processed normally; camera load is reduced adaptively.",
        },
    ]

    rows = []

    for scenario in ["nominal", "busy", "worst_case"]:
        camera_rate_per_hour = camera_rate_map[scenario]
        audio_rate_per_hour = audio_rate_map[scenario]

        for policy in policy_defs:
            camera_rate_per_sec = camera_rate_per_hour / 3600.0
            audio_rate_per_sec = audio_rate_per_hour / 3600.0

            camera_service_sec = policy["camera_service_ms"] / 1000.0
            audio_service_sec = policy["audio_service_ms"] / 1000.0

            camera_required_throughput = camera_rate_per_sec
            audio_required_throughput = audio_rate_per_sec

            camera_service_rate = 1.0 / camera_service_sec if camera_service_sec > 0 else 0.0
            audio_service_rate = 1.0 / audio_service_sec if audio_service_sec > 0 else 0.0

            total_cpu_util = audio_rate_per_sec * audio_service_sec
            total_gpu_util = camera_rate_per_sec * camera_service_sec
            aggregate_util = total_cpu_util + total_gpu_util

            feasibility = classify_feasibility(max(total_cpu_util, total_gpu_util))

            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy["policy"],
                    "camera_rate_per_hour": camera_rate_per_hour,
                    "audio_rate_per_hour": audio_rate_per_hour,
                    "camera_rate_per_sec": camera_rate_per_sec,
                    "audio_rate_per_sec": audio_rate_per_sec,
                    "camera_service_ms": policy["camera_service_ms"],
                    "audio_service_ms": policy["audio_service_ms"],
                    "camera_required_throughput_per_sec": camera_required_throughput,
                    "audio_required_throughput_per_sec": audio_required_throughput,
                    "camera_service_rate_per_sec": camera_service_rate,
                    "audio_service_rate_per_sec": audio_service_rate,
                    "total_gpu_util": total_gpu_util,
                    "total_cpu_util": total_cpu_util,
                    "aggregate_util": aggregate_util,
                    "single_node_feasibility": feasibility,
                    "camera_notes": policy["camera_notes"],
                    "audio_notes": policy["audio_notes"],
                }
            )

    result = pd.DataFrame(rows).sort_values(["scenario", "policy"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a multimodal edge resource approximation table."
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
        help="Path to output multimodal_resource_table.csv",
    )
    parser.add_argument(
        "--audio_nominal_rate_per_hour",
        type=float,
        default=12.0,
        help="Approximate audio clip arrival rate for the nominal scenario.",
    )
    parser.add_argument(
        "--audio_busy_rate_per_hour",
        type=float,
        default=30.0,
        help="Approximate audio clip arrival rate for the busy scenario.",
    )
    parser.add_argument(
        "--audio_worst_rate_per_hour",
        type=float,
        default=60.0,
        help="Approximate audio clip arrival rate for the worst-case scenario.",
    )
    parser.add_argument(
        "--selective_hit_rate",
        type=float,
        default=0.20,
        help="Estimated fraction of camera frames that trigger secondary classification.",
    )
    args = parser.parse_args()

    ensure_parent(args.output)

    benchmark_df = load_benchmark_summary(args.benchmark)
    arrival_df = load_arrival_summary(args.arrivals)

    multimodal_df = build_multimodal_resource_table(
        benchmark_df=benchmark_df,
        arrival_df=arrival_df,
        audio_nominal_rate_per_hour=args.audio_nominal_rate_per_hour,
        audio_busy_rate_per_hour=args.audio_busy_rate_per_hour,
        audio_worst_rate_per_hour=args.audio_worst_rate_per_hour,
        selective_hit_rate=args.selective_hit_rate,
    )

    multimodal_df.to_csv(args.output, index=False)

    print(f"Saved multimodal resource table: {args.output}")
    print(f"Rows written: {len(multimodal_df)}")


if __name__ == "__main__":
    main()