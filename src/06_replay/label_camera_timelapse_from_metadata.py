from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_metadata_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    session = path.parent.name if path.parent.name.startswith("SD") else ""
    site_cam = path.parent.parent.name if session else path.parent.name

    lower = text.lower()

    has_timelapse = "timelapse" in lower
    timelapse_every_hour = "timelapse photo every hour" in lower
    motion_detect = "motion detect" in lower or "high pir" in lower
    hybrid_mode = "mode: hybrid" in lower or "hybrid" in lower

    return {
        "site_cam": site_cam,
        "session": session,
        "metadata_path": str(path),
        "has_timelapse": bool(has_timelapse),
        "timelapse_every_hour": bool(timelapse_every_hour),
        "motion_detect": bool(motion_detect),
        "hybrid_mode": bool(hybrid_mode),
        "raw_metadata": text.strip(),
    }


def build_metadata_table(root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in root.rglob("metadata.txt"):
        rows.append(parse_metadata_file(meta_path))
    if not rows:
        raise FileNotFoundError(f"No metadata.txt files found under {root}")
    return pd.DataFrame(rows)


def infer_timelapse(row: pd.Series) -> bool:
    if str(row.get("modality", "")) != "camera":
        return False

    if not bool(row.get("has_timelapse", False)):
        return False

    ts = pd.to_datetime(row["event_time"])
    minute = ts.minute
    second = ts.second

    # Rule for "timelapse photo every hour":
    # mark images captured at hh:00:00 ~ hh:00:59 as timelapse.
    if bool(row.get("timelapse_every_hour", False)):
        return minute == 0

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_root",
        default="thewilds_cameratraps_full",
        help="Root directory containing camera-trap metadata.txt files",
    )
    parser.add_argument(
        "--input_trace",
        default="outputs/tables/arrivals/replay_trace.csv",
        help="Input camera replay trace CSV",
    )
    parser.add_argument(
        "--output_trace",
        default="outputs/tables/arrivals/replay_trace_labeled.csv",
        help="Output replay trace CSV with metadata-driven is_timelapse",
    )
    parser.add_argument(
        "--output_metadata_table",
        default="outputs/tables/arrivals/camera_session_metadata.csv",
        help="Parsed session-level metadata table",
    )
    args = parser.parse_args()

    camera_root = Path(args.camera_root)
    trace_path = Path(args.input_trace)

    meta_df = build_metadata_table(camera_root)
    meta_df.to_csv(args.output_metadata_table, index=False)

    trace_df = pd.read_csv(trace_path)
    if "session" not in trace_df.columns:
        raise ValueError("Input trace must contain a 'session' column.")
    if "site_cam" not in trace_df.columns:
        raise ValueError("Input trace must contain a 'site_cam' column.")
    if "event_time" not in trace_df.columns:
        raise ValueError("Input trace must contain an 'event_time' column.")

    merged = trace_df.merge(
        meta_df[
            [
                "site_cam",
                "session",
                "has_timelapse",
                "timelapse_every_hour",
                "motion_detect",
                "hybrid_mode",
            ]
        ],
        on=["site_cam", "session"],
        how="left",
    )

    merged["has_timelapse"] = merged["has_timelapse"].fillna(False)
    merged["timelapse_every_hour"] = merged["timelapse_every_hour"].fillna(False)
    merged["motion_detect"] = merged["motion_detect"].fillna(False)
    merged["hybrid_mode"] = merged["hybrid_mode"].fillna(False)

    merged["is_timelapse"] = merged.apply(infer_timelapse, axis=1)

    merged.to_csv(args.output_trace, index=False)

    print(f"Saved parsed metadata table: {args.output_metadata_table}")
    print(f"Saved labeled replay trace: {args.output_trace}")
    print(f"Total rows: {len(merged)}")
    print(f"Timelapse-labeled rows: {int(merged['is_timelapse'].sum())}")


if __name__ == "__main__":
    main()