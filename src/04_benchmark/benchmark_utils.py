# src/04_benchmark/benchmark_utils.py
import time
import platform
import pandas as pd
from pathlib import Path

def load_manifest(csv_path):
    return pd.read_csv(csv_path)

def timer():
    return time.perf_counter()

def compute_stats(n_samples, total_time_s):
    avg_latency_ms = (total_time_s / n_samples) * 1000 if n_samples > 0 else None
    throughput = n_samples / total_time_s if total_time_s > 0 else None
    return avg_latency_ms, throughput

def save_result_row(out_csv, row_dict):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row_dict])
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(out_csv, index=False)

def get_system_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }