from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_INPUT = Path('outputs/tables/benchmark/benchmark_audio_100.csv')
DEFAULT_OUTPUT = Path('outputs/tables/arrivals/audio_replay_trace.csv')
DEFAULT_HOURLY_OUTPUT = Path('outputs/tables/arrivals/audio_hourly_counts.csv')
DEFAULT_SUMMARY_OUTPUT = Path('outputs/tables/arrivals/audio_arrival_summary.csv')

TS_PATTERNS = [
    re.compile(r'(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})'),
    re.compile(r'(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})'),
]
SITE_PATTERN = re.compile(r'(TW\d{2})[-_](SM\d{2,3})', re.IGNORECASE)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_timestamp(text: str) -> Optional[pd.Timestamp]:
    for idx, pattern in enumerate(TS_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        g = match.groups()
        if idx == 0:
            year, month, day, hour, minute, second = map(int, g)
        else:
            yy, month, day, hour, minute, second = map(int, g)
            year = 2000 + yy
        try:
            return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
        except ValueError:
            continue
    return None


def _extract_site_sensor(text: str) -> tuple[str, str]:
    match = SITE_PATTERN.search(text.replace('\\', '/'))
    if match:
        site = match.group(1).upper()
        sensor = match.group(2).upper()
        return site, f'{site}-{sensor}'
    return 'UNKNOWN', 'UNKNOWN'


def load_audio_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Input file does not exist: {path}')
    df = pd.read_csv(path)
    candidate_cols = ['audio_path', 'path', 'filepath', 'file', 'filename']
    path_col = next((c for c in candidate_cols if c in df.columns), None)
    if path_col is None:
        raise ValueError(f'Could not find an audio path column in {path}. Available columns: {list(df.columns)}')
    work = df.copy()
    work['audio_path'] = work[path_col].astype(str)
    work['event_time'] = work['audio_path'].map(_extract_timestamp)
    work[['site', 'sensor_id']] = work['audio_path'].apply(lambda s: pd.Series(_extract_site_sensor(s)))
    if 'confidence' not in work.columns:
        work['confidence'] = pd.NA
    work = work.dropna(subset=['event_time']).sort_values('event_time').reset_index(drop=True)
    work['modality'] = 'audio'
    work['event_type'] = 'audio_clip'
    work['site_cam'] = work['sensor_id']
    return work[['event_time', 'site_cam', 'audio_path', 'modality', 'event_type', 'sensor_id', 'confidence']]


def build_hourly_counts(trace: pd.DataFrame) -> pd.DataFrame:
    hourly = trace.copy()
    hourly['hour_bucket'] = hourly['event_time'].dt.floor('h')
    return (
        hourly.groupby(['site_cam', 'hour_bucket'], as_index=False)
        .size()
        .rename(columns={'size': 'event_count'})
        .sort_values(['site_cam', 'hour_bucket'])
        .reset_index(drop=True)
    )


def build_arrival_summary(hourly_counts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        hourly_counts.groupby('site_cam', as_index=False)
        .agg(
            total_events=('event_count', 'sum'),
            avg_rate_per_hour=('event_count', 'mean'),
            p95_rate_per_hour=('event_count', lambda s: float(s.quantile(0.95))),
            peak_rate_per_hour=('event_count', 'max'),
            active_hours=('event_count', 'count'),
        )
        .sort_values('site_cam')
        .reset_index(drop=True)
    )
    summary['avg_rate_per_sec'] = summary['avg_rate_per_hour'] / 3600.0
    summary['p95_rate_per_sec'] = summary['p95_rate_per_hour'] / 3600.0
    summary['peak_rate_per_sec'] = summary['peak_rate_per_hour'] / 3600.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build an audio replay trace from an audio manifest.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--trace_out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--hourly_out', type=Path, default=DEFAULT_HOURLY_OUTPUT)
    parser.add_argument('--summary_out', type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()

    ensure_parent(args.trace_out)
    ensure_parent(args.hourly_out)
    ensure_parent(args.summary_out)

    trace = load_audio_manifest(args.input)
    hourly = build_hourly_counts(trace)
    summary = build_arrival_summary(hourly)

    trace.to_csv(args.trace_out, index=False)
    hourly.to_csv(args.hourly_out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f'Saved audio replay trace: {args.trace_out}')
    print(f'Saved audio hourly counts: {args.hourly_out}')
    print(f'Saved audio arrival summary: {args.summary_out}')
    print(f'Audio events: {len(trace)}')


if __name__ == '__main__':
    main()
