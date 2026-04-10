from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_REPLAY_INPUT = Path("outputs/tables/arrivals/replay_results.csv")
DEFAULT_FIGURE_DIR = Path("outputs/figures")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_replay_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Replay results file does not exist: {path}")

    df = pd.read_csv(path)
    required = {"policy", "event_time", "action", "latency_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Replay results are missing required columns: {sorted(missing)}")

    result = df.copy()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="coerce")
    result = result.dropna(subset=["event_time"]).reset_index(drop=True)
    result["hour_bucket"] = result["event_time"].dt.floor("h")

    return result


def plot_hourly_events(df: pd.DataFrame, out_path: Path) -> None:
    hourly = (
        df.groupby(["hour_bucket", "policy"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for policy in sorted(hourly["policy"].unique()):
        subset = hourly[hourly["policy"] == policy].sort_values("hour_bucket")
        ax.plot(
            subset["hour_bucket"],
            subset["event_count"],
            label=policy,
            linewidth=1.5,
        )

    ax.set_title("Replay timeline: hourly event count by policy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Events per hour")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hourly_latency(df: pd.DataFrame, out_path: Path) -> None:
    processed = df[df["action"] != "skip"].copy()

    hourly = (
        processed.groupby(["hour_bucket", "policy"], as_index=False)
        .agg(mean_latency_ms=("latency_ms", "mean"))
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for policy in sorted(hourly["policy"].unique()):
        subset = hourly[hourly["policy"] == policy].sort_values("hour_bucket")
        ax.plot(
            subset["hour_bucket"],
            subset["mean_latency_ms"],
            label=policy,
            linewidth=1.5,
        )

    ax.set_title("Replay timeline: hourly mean latency by policy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hourly_policy_state(df: pd.DataFrame, out_path: Path) -> None:
    if "policy_state" not in df.columns:
        return

    adaptive = df[df["policy"] == "adaptive"].copy()
    if adaptive.empty:
        return

    state_codes = {
        "normal": 0,
        "high_activity": 1,
        "idle_backlog": 2,
        "overloaded": 3,
        "event_triggered": 4,
        "full_pipeline": 5,
        "detection_only": 6,
        "selective_classification": 7,
        "timelapse_filtered": 8,
    }

    adaptive["state_code"] = adaptive["policy_state"].map(state_codes).fillna(-1)
    hourly = (
        adaptive.groupby("hour_bucket", as_index=False)
        .agg(mean_state_code=("state_code", "mean"))
        .sort_values("hour_bucket")
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hourly["hour_bucket"], hourly["mean_state_code"], linewidth=1.5)

    ax.set_title("Replay timeline: adaptive policy state over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean state code")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot replay timeline visualizations."
    )
    parser.add_argument(
        "--replay_results",
        type=Path,
        default=DEFAULT_REPLAY_INPUT,
        help="Path to replay_results.csv",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for output figures.",
    )
    args = parser.parse_args()

    ensure_dir(args.figure_dir)
    df = load_replay_results(args.replay_results)

    plot_hourly_events(df, args.figure_dir / "replay_timeline_events.png")
    plot_hourly_latency(df, args.figure_dir / "replay_timeline_latency.png")
    plot_hourly_policy_state(df, args.figure_dir / "replay_timeline_policy_state.png")

    print(f"Saved timeline figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()