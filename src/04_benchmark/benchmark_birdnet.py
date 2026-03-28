from pathlib import Path
import birdnet

from benchmark_utils import (
    load_manifest,
    timer,
    compute_stats,
    save_result_row,
    get_system_info,
)

MANIFEST = "outputs/tables/benchmark/benchmark_audio_100.csv"
OUT_CSV = "outputs/tables/benchmark/birdnet_speed.csv"


def main():
    device = "cpu"
    batch_size = 1

    df = load_manifest(MANIFEST)
    audio_paths = df["audio_path"].tolist()[:100]

    print("Loading BirdNET model...")
    model = birdnet.load("acoustic", "2.4", "tf")
    print("BirdNET model loaded.")

    # warmup
    print(f"Warmup on: {audio_paths[0]}")
    _ = model.predict(str(audio_paths[0]))

    # timed run
    start = timer()
    for i, p in enumerate(audio_paths, 1):
        print(f"[{i}/{len(audio_paths)}] {p}")
        _ = model.predict(str(p))
    end = timer()

    total_time_s = end - start
    avg_latency_ms, throughput = compute_stats(len(audio_paths), total_time_s)

    row = {
        "model": "BirdNET",
        "variant": "acoustic / 2.4 / tf",
        "device": device,
        "batch_size": batch_size,
        "n_samples": len(audio_paths),
        "total_time_s": total_time_s,
        "avg_latency_ms": avg_latency_ms,
        "throughput_samples_per_s": throughput,
        "notes": "single-audio-file inference on 100 audio files, excludes manifest creation",
    }
    row.update(get_system_info())

    save_result_row(OUT_CSV, row)
    print(row)


if __name__ == "__main__":
    main()