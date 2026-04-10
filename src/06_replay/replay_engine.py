from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from policies import POLICY_REGISTRY


DEFAULT_TRACE_INPUT = Path("outputs/tables/arrivals/replay_trace.csv")
DEFAULT_OUTPUT = Path("outputs/tables/arrivals/replay_results.csv")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_replay_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Replay trace file does not exist: {path}")

    df = pd.read_csv(path)

    required = {
        "event_time",
        "site_cam",
        "session",
        "path",
        "modality",
        "event_type",
        "sensor_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Replay trace is missing required columns: {sorted(missing)}")

    result = df.copy()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="coerce")
    result = result.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)
    result["_event_index"] = range(len(result))

    return result


def get_policy_config(args: argparse.Namespace) -> Dict[str, Any]:
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


def _count_recent_detections(
    records: List[Dict[str, Any]],
    policy_name: str,
    current_time: pd.Timestamp,
    window_hours: float = 1.0,
) -> int:
    if not records:
        return 0

    window_start = current_time - pd.to_timedelta(window_hours, unit="h")
    count = 0
    for record in records:
        if record["policy"] != policy_name:
            continue
        completion_time = record.get("completion_time")
        if pd.isna(completion_time):
            continue
        if completion_time >= window_start and bool(record.get("detection_positive", False)):
            count += 1
    return count


def _compute_idle_duration_hours(
    last_processed_time: pd.Timestamp | None,
    current_time: pd.Timestamp,
) -> float:
    if last_processed_time is None:
        return 0.0
    delta = current_time - last_processed_time
    return max(0.0, delta.total_seconds() / 3600.0)


def simulate_policy(
    trace_df: pd.DataFrame,
    policy_name: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Available policies: {sorted(POLICY_REGISTRY.keys())}"
        )

    policy_fn = POLICY_REGISTRY[policy_name]
    records: List[Dict[str, Any]] = []

    busy_until = pd.Timestamp.min
    queue_length = 0
    last_processed_completion_time = None

    for event in trace_df.to_dict(orient="records"):
        event_time = pd.Timestamp(event["event_time"])

        if event_time >= busy_until:
            queue_length = 0

        recent_detection_count = _count_recent_detections(
            records=records,
            policy_name=policy_name,
            current_time=event_time,
            window_hours=1.0,
        )
        idle_duration_hours = _compute_idle_duration_hours(
            last_processed_time=last_processed_completion_time,
            current_time=event_time,
        )

        state = {
            "queue_length": queue_length,
            "busy_until": busy_until,
            "idle_mode": queue_length == 0,
            "recent_detection_count": recent_detection_count,
            "idle_duration_hours": idle_duration_hours,
        }

        decision = policy_fn(event, state, config)

        if decision.action == "skip":
            records.append(
                {
                    "policy": policy_name,
                    "event_index": int(event["_event_index"]),
                    "event_time": event_time,
                    "site_cam": event["site_cam"],
                    "sensor_id": event["sensor_id"],
                    "path": event["path"],
                    "action": "skip",
                    "processing_start_time": pd.NaT,
                    "completion_time": pd.NaT,
                    "latency_ms": 0.0,
                    "service_time_ms": 0.0,
                    "compute_cost_ms": 0.0,
                    "queue_length_at_arrival": queue_length,
                    "classification_triggered": bool(
                        decision.metadata.get("classification_triggered", False)
                    ),
                    "detection_positive": bool(
                        decision.metadata.get("detection_positive", False)
                    ),
                    "policy_stage": decision.metadata.get("policy_stage", ""),
                    "policy_state": decision.metadata.get("policy_state", ""),
                    "skip_reason": decision.metadata.get("skip_reason", ""),
                    "recent_detection_count": recent_detection_count,
                    "idle_duration_hours": idle_duration_hours,
                }
            )
            continue

        processing_start_time = max(event_time, busy_until)
        waiting_ms = max(
            0.0,
            (processing_start_time - event_time).total_seconds() * 1000.0,
        )
        service_time_ms = float(decision.service_time_ms)
        completion_time = processing_start_time + pd.to_timedelta(service_time_ms, unit="ms")
        latency_ms = waiting_ms + service_time_ms

        queue_length_at_arrival = queue_length

        if event_time < busy_until:
            queue_length += 1
        else:
            queue_length = 1

        busy_until = completion_time
        last_processed_completion_time = completion_time

        records.append(
            {
                "policy": policy_name,
                "event_index": int(event["_event_index"]),
                "event_time": event_time,
                "site_cam": event["site_cam"],
                "sensor_id": event["sensor_id"],
                "path": event["path"],
                "action": "process_now",
                "processing_start_time": processing_start_time,
                "completion_time": completion_time,
                "latency_ms": latency_ms,
                "service_time_ms": service_time_ms,
                "compute_cost_ms": float(decision.compute_cost_ms),
                "queue_length_at_arrival": queue_length_at_arrival,
                "classification_triggered": bool(
                    decision.metadata.get("classification_triggered", False)
                ),
                "detection_positive": bool(
                    decision.metadata.get("detection_positive", False)
                ),
                "policy_stage": decision.metadata.get("policy_stage", ""),
                "policy_state": decision.metadata.get("policy_state", ""),
                "skip_reason": "",
                "recent_detection_count": recent_detection_count,
                "idle_duration_hours": idle_duration_hours,
            }
        )

    result = pd.DataFrame(records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a camera trace under one or more policy rules."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE_INPUT,
        help="Path to replay_trace.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output replay_results.csv",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["process_all", "detection_only", "selective_classification", "adaptive"],
        help="List of policy names to replay.",
    )
    parser.add_argument(
        "--yolo_latency_ms",
        type=float,
        default=38.6,
        help="Service time for the detector stage.",
    )
    parser.add_argument(
        "--bioclip_latency_ms",
        type=float,
        default=60.8,
        help="Service time for the classifier stage.",
    )
    parser.add_argument(
        "--detection_probability",
        type=float,
        default=0.20,
        help="Estimated probability that an image triggers classification.",
    )
    parser.add_argument(
        "--birdnet_latency_ms",
        type=float,
        default=19057.5,
        help="Service time for the BirdNET audio stage.",
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
    args = parser.parse_args()

    ensure_parent(args.output)

    trace_df = load_replay_trace(args.trace)
    config = get_policy_config(args)

    all_results = []
    for policy_name in args.policies:
        policy_result = simulate_policy(trace_df, policy_name, config)
        all_results.append(policy_result)

    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(args.output, index=False)

    print(f"Saved replay results: {args.output}")
    print(f"Policies simulated: {', '.join(args.policies)}")
    print(f"Rows written: {len(result_df)}")


if __name__ == "__main__":
    main()