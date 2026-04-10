from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


DEFAULT_REPLAY_INPUT = Path("outputs/tables/arrivals/replay_results.csv")
DEFAULT_OUTPUT = Path("outputs/tables/arrivals/replay_metrics.csv")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_replay_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Replay results file does not exist: {path}")

    df = pd.read_csv(path)

    required = {
        "policy",
        "event_index",
        "action",
        "latency_ms",
        "service_time_ms",
        "compute_cost_ms",
        "classification_triggered",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Replay results are missing required columns: {sorted(missing)}")

    return df.copy()


def build_policy_metrics(
    replay_df: pd.DataFrame,
    slo_ms: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for policy, group in replay_df.groupby("policy"):
        total_events = int(len(group))
        processed = group[group["action"] != "skip"].copy()
        skipped = group[group["action"] == "skip"].copy()

        processed_events = int(len(processed))
        skipped_events = int(len(skipped))

        coverage = processed_events / total_events if total_events > 0 else 0.0
        compute_cost_ms = float(processed["compute_cost_ms"].sum()) if processed_events > 0 else 0.0
        mean_latency_ms = float(processed["latency_ms"].mean()) if processed_events > 0 else 0.0
        p95_latency_ms = float(processed["latency_ms"].quantile(0.95)) if processed_events > 0 else 0.0
        max_latency_ms = float(processed["latency_ms"].max()) if processed_events > 0 else 0.0

        if processed_events > 0:
            slo_compliance = float((processed["latency_ms"] <= slo_ms).mean())
        else:
            slo_compliance = 0.0

        classification_trigger_rate = (
            float(processed["classification_triggered"].astype(bool).mean())
            if processed_events > 0
            else 0.0
        )
        classification_trigger_count = (
            int(processed["classification_triggered"].astype(bool).sum())
            if processed_events > 0
            else 0
        )

        detection_positive_count = (
            int(processed["detection_positive"].astype(bool).sum())
            if "detection_positive" in processed.columns and processed_events > 0
            else 0
        )

        operator_effort_estimate = classification_trigger_count

        dominant_policy_state = ""
        if "policy_state" in group.columns and not group["policy_state"].dropna().empty:
            dominant_policy_state = str(group["policy_state"].mode().iloc[0])

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
                "slo_ms": slo_ms,
                "slo_compliance": slo_compliance,
                "classification_trigger_rate": classification_trigger_rate,
                "classification_trigger_count": classification_trigger_count,
                "detection_positive_count": detection_positive_count,
                "operator_effort_estimate": operator_effort_estimate,
                "dominant_policy_state": dominant_policy_state,
            }
        )

    result = pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate replay results and produce policy-level metrics."
    )
    parser.add_argument(
        "--replay_results",
        type=Path,
        default=DEFAULT_REPLAY_INPUT,
        help="Path to replay_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output replay_metrics.csv",
    )
    parser.add_argument(
        "--slo_ms",
        type=float,
        default=5000.0,
        help="Latency SLO threshold in milliseconds.",
    )
    args = parser.parse_args()

    ensure_parent(args.output)

    replay_df = load_replay_results(args.replay_results)
    metrics_df = build_policy_metrics(
        replay_df=replay_df,
        slo_ms=args.slo_ms,
    )
    metrics_df.to_csv(args.output, index=False)

    print(f"Saved replay metrics: {args.output}")
    print(f"Rows written: {len(metrics_df)}")


if __name__ == "__main__":
    main()