from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_BENCHMARK_INPUT = Path('outputs/tables/benchmark/benchmark_summary.csv')
DEFAULT_CAMERA_INPUT = Path('outputs/tables/arrivals/arrival_summary.csv')
DEFAULT_AUDIO_INPUT = Path('outputs/tables/arrivals/audio_arrival_summary.csv')
DEFAULT_OUTPUT = Path('outputs/tables/benchmark/trace_driven_multimodal_resource_table.csv')


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_latency(df: pd.DataFrame, keyword: str) -> float:
    match = df[df['model'].str.lower().str.contains(keyword.lower(), na=False)]
    if match.empty:
        raise ValueError(f'Model keyword not found: {keyword}')
    return float(match.iloc[0]['mean_latency_ms'])


def load_benchmarks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {'model', 'mean_latency_ms'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Benchmark file missing columns: {sorted(missing)}')
    return df.copy()


def load_arrivals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {'site_cam', 'avg_rate_per_hour', 'p95_rate_per_hour', 'peak_rate_per_hour'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Arrival summary missing columns: {sorted(missing)}')
    return df.copy()


def classify_feasibility(cpu_util: float, gpu_util: float) -> str:
    peak = max(cpu_util, gpu_util)
    if peak < 0.7:
        return 'safe'
    if peak < 1.0:
        return 'risky'
    return 'not_feasible'


def build_table(bench: pd.DataFrame, camera: pd.DataFrame, audio: pd.DataFrame, selective_hit_rate: float, audio_conf_threshold: float) -> pd.DataFrame:
    yolo_ms = get_latency(bench, 'yolo')
    bioclip_ms = get_latency(bench, 'bioclip')
    birdnet_ms = get_latency(bench, 'birdnet')

    scenarios = {
        'nominal': ('avg_rate_per_hour', 'avg_rate_per_hour'),
        'busy': ('p95_rate_per_hour', 'p95_rate_per_hour'),
        'worst_case': ('peak_rate_per_hour', 'peak_rate_per_hour'),
    }
    policies = {
        'process_all': {'camera_ms': yolo_ms + bioclip_ms, 'audio_ms': birdnet_ms},
        'event_triggered': {'camera_ms': yolo_ms + selective_hit_rate * bioclip_ms, 'audio_ms': birdnet_ms},
        'adaptive': {'camera_ms': yolo_ms + selective_hit_rate * bioclip_ms, 'audio_ms': birdnet_ms},
    }
    rows = []
    for scenario, (camera_col, audio_col) in scenarios.items():
        camera_rate_h = float(camera[camera_col].mean())
        audio_rate_h = float(audio[audio_col].mean())
        for policy, defs in policies.items():
            gpu_util = (camera_rate_h / 3600.0) * (defs['camera_ms'] / 1000.0)
            cpu_util = (audio_rate_h / 3600.0) * (defs['audio_ms'] / 1000.0)
            rows.append({
                'scenario': scenario,
                'policy': policy,
                'camera_rate_per_hour': camera_rate_h,
                'audio_rate_per_hour': audio_rate_h,
                'camera_service_ms': defs['camera_ms'],
                'audio_service_ms': defs['audio_ms'],
                'cpu_util_percent': 100.0 * cpu_util,
                'gpu_util_percent': 100.0 * gpu_util,
                'aggregate_util_percent': 100.0 * (cpu_util + gpu_util),
                'single_node_feasibility': classify_feasibility(cpu_util, gpu_util),
                'audio_conf_threshold': audio_conf_threshold,
            })
    return pd.DataFrame(rows).sort_values(['scenario', 'policy']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a trace-driven multimodal resource table using real camera and audio arrivals.')
    parser.add_argument('--benchmarks', type=Path, default=DEFAULT_BENCHMARK_INPUT)
    parser.add_argument('--camera_arrivals', type=Path, default=DEFAULT_CAMERA_INPUT)
    parser.add_argument('--audio_arrivals', type=Path, default=DEFAULT_AUDIO_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--selective_hit_rate', type=float, default=0.2)
    parser.add_argument('--audio_conf_threshold', type=float, default=0.7)
    args = parser.parse_args()

    ensure_parent(args.output)
    bench = load_benchmarks(args.benchmarks)
    camera = load_arrivals(args.camera_arrivals)
    audio = load_arrivals(args.audio_arrivals)
    out = build_table(bench, camera, audio, args.selective_hit_rate, args.audio_conf_threshold)
    out.to_csv(args.output, index=False)
    print(f'Saved trace-driven multimodal resource table: {args.output}')


if __name__ == '__main__':
    main()
