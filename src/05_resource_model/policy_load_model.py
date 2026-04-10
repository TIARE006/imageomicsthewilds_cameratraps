from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RESOURCE_INPUT = Path("outputs/tables/benchmark/camera_resource_table.csv")
DEFAULT_OUTPUT = Path("outputs/tables/benchmark/camera_policy_comparison.csv")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_resource_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Resource table does not exist: {path}")

    df = pd.read_csv(path)

    required = {
        "site_cam",
        "scenario",
        "policy",
        "arrival_rate_per_hour",
        "service_time_ms",
        "utilization",
        "feasibility",
        "coverage_estimate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Resource table is missing required columns: {sorted(missing)}"
        )

    return df.copy()


def classify_latency_risk(mean_utilization: float) -> str:
    if mean_utilization < 0.5:
        return "low"
    if mean_utilization < 0.8:
        return "medium"
    if mean_utilization < 1.0:
        return "high"
    return "critical"


def estimate_slo_compliance(mean_utilization: float) -> float:
    if mean_utilization < 0.5:
        return 0.99
    if mean_utilization < 0.7:
        return 0.95
    if mean_utilization < 0.85:
        return 0.85
    if mean_utilization < 1.0:
        return 0.65
    return 0.30


def estimate_p95_latency_ms(service_time_ms: float, mean_utilization: float) -> float:
    if mean_utilization >= 1.0:
        queue_multiplier = 5.0
    else:
        queue_multiplier = 1.0 / max(1e-6, (1.0 - mean_utilization))
        queue_multiplier = min(queue_multiplier, 5.0)

    return service_time_ms * queue_multiplier


def build_policy_comparison(resource_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = resource_df.groupby(["scenario", "policy"], as_index=False)

    for (scenario, policy), group in grouped:
        mean_arrival_rate_per_hour = group["arrival_rate_per_hour"].mean()
        mean_service_time_ms = group["service_time_ms"].mean()
        mean_utilization = group["utilization"].mean()
        peak_utilization = group["utilization"].max()
        coverage_estimate = group["coverage_estimate"].mean()

        estimated_cpu_load = mean_utilization
        estimated_gpu_load = mean_utilization

        latency_risk = classify_latency_risk(mean_utilization)
        slo_compliance = estimate_slo_compliance(mean_utilization)
        estimated_p95_latency_ms = estimate_p95_latency_ms(
            service_time_ms=mean_service_time_ms,
            mean_utilization=mean_utilization,
        )

        rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "num_sites": int(group["site_cam"].nunique()),
                "mean_arrival_rate_per_hour": mean_arrival_rate_per_hour,
                "mean_service_time_ms": mean_service_time_ms,
                "estimated_cpu_load": estimated_cpu_load,
                "estimated_gpu_load": estimated_gpu_load,
                "mean_utilization": mean_utilization,
                "peak_utilization": peak_utilization,
                "coverage_estimate": coverage_estimate,
                "estimated_p95_latency_ms": estimated_p95_latency_ms,
                "slo_compliance_estimate": slo_compliance,
                "latency_risk": latency_risk,
                "comments": (
                    "This table is a first-order analytical estimate derived from arrival rates "
                    "and average service time."
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(["scenario", "policy"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a theoretical camera policy comparison table from a resource table."
    )
    parser.add_argument(
        "--resource_table",
        type=Path,
        default=DEFAULT_RESOURCE_INPUT,
        help="Path to camera_resource_table.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output camera_policy_comparison.csv",
    )
    args = parser.parse_args()

    ensure_parent(args.output)

    resource_df = load_resource_table(args.resource_table)
    comparison_df = build_policy_comparison(resource_df)
    comparison_df.to_csv(args.output, index=False)

    print(f"Saved policy comparison table: {args.output}")
    print(f"Rows written: {len(comparison_df)}")


if __name__ == "__main__":
    main()