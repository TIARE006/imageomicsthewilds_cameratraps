from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from replay_engine import load_replay_trace, simulate_policy


DEFAULT_TRACE_INPUT = Path("outputs/tables/arrivals/replay_trace.csv")
DEFAULT_RESULTS_OUTPUT = Path("outputs/tables/arrivals/stress_replay_results.csv")
DEFAULT_METRICS_OUTPUT = Path("outputs/tables/arrivals/stress_replay_metrics.csv")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_stress_factors(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("At least one stress factor must be provided.")
    return values


def build_policy_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "yolo_latency_ms": args.yolo_latency_ms,
        "bioclip_latency_ms": args.bioclip_latency_ms,
        "detection_probability": args.detection_probability,
        "audio_detection_probability": args.audio_detection_probability,
        "audio_conf_threshold": args.audio_conf_threshold,
        "birdnet_latency_ms": args.birdnet_latency_ms,
        "backlog_threshold": args.backlog_threshold,
        "high_activity_threshold": args.high_activity_threshold,
        "idle_backlog_threshold_hours": args.idle_backlog_threshold_hours,
    }


def compress_interarrival_times(trace_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    if factor <= 0:
        raise ValueError("Stress factor must be positive.")

    stressed = trace_df.copy().sort_values("event_time").reset_index(drop=True)

    base_time = stressed.loc[0, "event_time"]
    relative_seconds = (stressed["event_time"] - base_time).dt.total_seconds()
    stressed_seconds = relative_seconds / factor
    stressed["event_time"] = base_time + pd.to_timedelta(stressed_seconds, unit="s")

    return stressed


def build_policy_metrics(replay_df: pd.DataFrame, slo_ms: float) -> pd.DataFrame:
    rows = []

    for policy, group in replay_df.groupby("policy"):
        total_events = int(len(group))
        processed = group[group["action"] != "skip"].copy()
        skipped = group[group["action"] == "skip"].copy()

        processed_events = int(len(processed))
        skipped_events = int(len(skipped))

        coverage = processed_events / total_events if total_events > 0 else 0.0
        mean_latency_ms = float(processed["latency_ms"].mean()) if processed_events > 0 else 0.0
        p95_latency_ms = float(processed["latency_ms"].quantile(0.95)) if processed_events > 0 else 0.0
        max_latency_ms = float(processed["latency_ms"].max()) if processed_events > 0 else 0.0
        compute_cost_ms = float(processed["compute_cost_ms"].sum()) if processed_events > 0 else 0.0
        mean_queue_length = (
            float(processed["queue_length_at_arrival"].mean()) if processed_events > 0 else 0.0
        )
        peak_queue_length = (
            int(processed["queue_length_at_arrival"].max()) if processed_events > 0 else 0
        )
        slo_compliance = (
            float((processed["latency_ms"] <= slo_ms).mean()) if processed_events > 0 else 0.0
        )
        classification_trigger_rate = (
            float(processed["classification_triggered"].astype(bool).mean())
            if processed_events > 0
            else 0.0
        )

        rows.append(
            {
                "policy": policy,
                "total_events": total_events,
                "processed_events": processed_events,
                "skipped_events": skipped_events,
                "coverage": coverage,
                "mean_latency_ms": mean_latency_ms,
                "p95_latency_ms": p95_latency_ms,
                "max_latency_ms": max_latency_ms,
                "compute_cost_ms": compute_cost_ms,
                "compute_cost_sec": compute_cost_ms / 1000.0,
                "mean_queue_length": mean_queue_length,
                "peak_queue_length": peak_queue_length,
                "slo_ms": slo_ms,
                "slo_compliance": slo_compliance,
                "classification_trigger_rate": classification_trigger_rate,
            }
        )

    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run replay experiments under multiple arrival-rate stress factors."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE_INPUT,
        help="Path to replay_trace.csv",
    )
    parser.add_argument(
        "--results_output",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT,
        help="Path to stress_replay_results.csv",
    )
    parser.add_argument(
        "--metrics_output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Path to stress_replay_metrics.csv",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["process_all", "event_triggered", "adaptive"],
        help="Policies to evaluate.",
    )
    parser.add_argument(
        "--stress_factors",
        type=str,
        default="1,2,5,10",
        help="Comma-separated list of arrival stress factors.",
    )
    parser.add_argument(
        "--yolo_latency_ms",
        type=float,
        default=38.6,
        help="Detector service time in milliseconds.",
    )
    parser.add_argument(
        "--bioclip_latency_ms",
        type=float,
        default=60.8,
        help="Classifier service time in milliseconds.",
    )
    parser.add_argument(
        "--detection_probability",
        type=float,
        default=0.20,
        help="Probability that selective classification triggers the second stage.",
    )
    parser.add_argument(
        "--birdnet_latency_ms",
        type=float,
        default=19057.5,
        help="Service time for BirdNET audio processing in milliseconds.",
    )
    parser.add_argument(
        "--audio_detection_probability",
        type=float,
        default=0.25,
        help="Fallback probability that an audio event exceeds the confidence threshold when no confidence column is available.",
    )
    parser.add_argument(
        "--audio_conf_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for event-triggered audio processing.",
    )
    parser.add_argument(
        "--backlog_threshold",
        type=int,
        default=25,
        help="Queue threshold used by the adaptive policy.",
    )
    parser.add_argument(
        "--high_activity_threshold",
        type=int,
        default=10,
        help="Detection count threshold in a 1-hour window for the adaptive policy.",
    )
    parser.add_argument(
        "--idle_backlog_threshold_hours",
        type=float,
        default=2.0,
        help="Idle duration threshold in hours for the adaptive policy.",
    )
    parser.add_argument(
        "--slo_ms",
        type=float,
        default=5000.0,
        help="Latency SLO in milliseconds.",
    )
    args = parser.parse_args()

    ensure_parent(args.results_output)
    ensure_parent(args.metrics_output)

    trace_df = load_replay_trace(args.trace)
    config = build_policy_config(args)
    stress_factors = parse_stress_factors(args.stress_factors)

    all_results = []
    all_metrics = []

    for factor in stress_factors:
        stressed_trace = compress_interarrival_times(trace_df, factor=factor)

        policy_results = []
        for policy_name in args.policies:
            replay_df = simulate_policy(stressed_trace, policy_name, config)
            replay_df["stress_factor"] = factor
            policy_results.append(replay_df)

        combined_results = pd.concat(policy_results, ignore_index=True)
        metrics_df = build_policy_metrics(combined_results, slo_ms=args.slo_ms)
        metrics_df["stress_factor"] = factor

        all_results.append(combined_results)
        all_metrics.append(metrics_df)

    final_results = pd.concat(all_results, ignore_index=True)
    final_metrics = pd.concat(all_metrics, ignore_index=True)
    final_metrics = final_metrics.sort_values(["stress_factor", "policy"]).reset_index(drop=True)

    final_results.to_csv(args.results_output, index=False)
    final_metrics.to_csv(args.metrics_output, index=False)

    print(f"Saved stress replay results: {args.results_output}")
    print(f"Saved stress replay metrics: {args.metrics_output}")
    print(f"Stress factors evaluated: {stress_factors}")
    print(f"Policies evaluated: {args.policies}")
    
if __name__ == "__main__":
    main()