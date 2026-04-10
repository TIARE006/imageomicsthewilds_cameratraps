from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CAMERA_TRACE = Path('outputs/tables/arrivals/replay_trace.csv')
DEFAULT_AUDIO_TRACE = Path('outputs/tables/arrivals/audio_replay_trace.csv')
DEFAULT_OUTPUT = Path('outputs/tables/arrivals/multimodal_replay_trace.csv')


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_trace(path: Path, modality: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'{modality} trace file does not exist: {path}')
    df = pd.read_csv(path)
    if 'event_time' not in df.columns:
        raise ValueError(f'{modality} trace is missing event_time column: {path}')
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df = df.dropna(subset=['event_time']).reset_index(drop=True)
    if 'modality' not in df.columns:
        df['modality'] = modality
    if 'sensor_id' not in df.columns:
        df['sensor_id'] = df.get('site_cam', 'UNKNOWN').astype(str)
    if 'site_cam' not in df.columns:
        df['site_cam'] = df['sensor_id'].astype(str)
    if 'path' not in df.columns:
        df['path'] = ''
    if 'session' not in df.columns:
        df['session'] = ''
    if 'event_type' not in df.columns:
        df['event_type'] = f'{modality}_event'
    if 'confidence' not in df.columns:
        df['confidence'] = pd.NA
    if 'is_timelapse' not in df.columns:
        df['is_timelapse'] = False
    cols = ['event_time', 'site_cam', 'session', 'path', 'modality', 'event_type', 'sensor_id', 'confidence', 'is_timelapse']
    return df[cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge camera and audio traces into a single multimodal replay trace.')
    parser.add_argument('--camera_trace', type=Path, default=DEFAULT_CAMERA_TRACE)
    parser.add_argument('--audio_trace', type=Path, default=DEFAULT_AUDIO_TRACE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ensure_parent(args.output)
    camera = load_trace(args.camera_trace, 'camera')
    audio = load_trace(args.audio_trace, 'audio')
    merged = pd.concat([camera, audio], ignore_index=True).sort_values(['event_time', 'modality', 'sensor_id']).reset_index(drop=True)
    merged.to_csv(args.output, index=False)
    print(f'Saved multimodal replay trace: {args.output}')
    print(f'Rows written: {len(merged)}')


if __name__ == '__main__':
    main()
