from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/events.csv")
DEFAULT_TRACE_OUTPUT = Path("outputs/tables/arrivals/replay_trace.csv")
DEFAULT_ARRIVAL_OUTPUT = Path("outputs/tables/arrivals/arrival_summary.csv")
DEFAULT_HOURLY_OUTPUT = Path("outputs/tables/arrivals/hourly_counts.csv")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_events(events_path: Path) -> pd.DataFrame:
    if not events_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {events_path}")

    df = pd.read_csv(events_path)

    required_columns = {"timestamp", "site_cam", "session", "path"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input events file is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df


def build_replay_trace(df: pd.DataFrame) -> pd.DataFrame:
    trace = df.copy()

    trace["event_time"] = trace["timestamp"]
    trace["sensor_id"] = trace["site_cam"].astype(str)
    trace["modality"] = "camera"
    trace["event_type"] = "camera_image"

    trace["session"] = trace["session"].astype(str)
    trace["path"] = trace["path"].astype(str)

    replay_trace = trace[
        [
            "event_time",
            "site_cam",
            "session",
            "path",
            "modality",
            "event_type",
            "sensor_id",
        ]
    ].copy()

    replay_trace = replay_trace.sort_values("event_time").reset_index(drop=True)
    return replay_trace


def build_hourly_counts(replay_trace: pd.DataFrame) -> pd.DataFrame:
    hourly = replay_trace.copy()
    hourly["hour_bucket"] = hourly["event_time"].dt.floor("h")

    hourly_counts = (
        hourly.groupby(["site_cam", "hour_bucket"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
        .sort_values(["site_cam", "hour_bucket"])
        .reset_index(drop=True)
    )

    return hourly_counts


def build_arrival_summary(hourly_counts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        hourly_counts.groupby("site_cam", as_index=False)
        .agg(
            total_events=("event_count", "sum"),
            avg_rate_per_hour=("event_count", "mean"),
            p95_rate_per_hour=("event_count", lambda s: float(s.quantile(0.95))),
            peak_rate_per_hour=("event_count", "max"),
            active_hours=("event_count", "count"),
        )
        .sort_values("site_cam")
        .reset_index(drop=True)
    )

    summary["avg_rate_per_sec"] = summary["avg_rate_per_hour"] / 3600.0
    summary["p95_rate_per_sec"] = summary["p95_rate_per_hour"] / 3600.0
    summary["peak_rate_per_sec"] = summary["peak_rate_per_hour"] / 3600.0

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a replay trace and arrival summary from camera events."
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input events CSV file.",
    )
    parser.add_argument(
        "--trace_out",
        type=Path,
        default=DEFAULT_TRACE_OUTPUT,
        help="Path to the output replay trace CSV file.",
    )
    parser.add_argument(
        "--arrival_out",
        type=Path,
        default=DEFAULT_ARRIVAL_OUTPUT,
        help="Path to the output arrival summary CSV file.",
    )
    parser.add_argument(
        "--hourly_out",
        type=Path,
        default=DEFAULT_HOURLY_OUTPUT,
        help="Path to the output hourly counts CSV file.",
    )
    args = parser.parse_args()

    ensure_parent(args.trace_out)
    ensure_parent(args.arrival_out)
    ensure_parent(args.hourly_out)

    events_df = load_events(args.events)
    replay_trace = build_replay_trace(events_df)
    hourly_counts = build_hourly_counts(replay_trace)
    arrival_summary = build_arrival_summary(hourly_counts)

    replay_trace.to_csv(args.trace_out, index=False)
    hourly_counts.to_csv(args.hourly_out, index=False)
    arrival_summary.to_csv(args.arrival_out, index=False)

    print(f"Saved replay trace: {args.trace_out}")
    print(f"Saved hourly counts: {args.hourly_out}")
    print(f"Saved arrival summary: {args.arrival_out}")
    print(f"Total replay events: {len(replay_trace)}")
    print(f"Number of camera sites: {arrival_summary['site_cam'].nunique()}")


if __name__ == "__main__":
    main()