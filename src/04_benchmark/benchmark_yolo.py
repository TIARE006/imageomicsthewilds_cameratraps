# src/04_benchmark/benchmark_yolo.py
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from benchmark_utils import load_manifest, timer, compute_stats, save_result_row, get_system_info

MANIFEST = "outputs/tables/benchmark/benchmark_images_500.csv"
OUT_CSV = "outputs/tables/benchmark/yolo_speed.csv"

def main():
    df = load_manifest(MANIFEST)
    image_paths = df["image_path"].tolist()

    model_name = "yolov8n.pt"
    device = "cpu"   
    batch_size = 1

    model = YOLO(model_name)

    warmup_paths = image_paths[:10]
    for p in warmup_paths:
        model.predict(source=p, device=device, verbose=False)

    start = timer()
    for p in image_paths:
        model.predict(source=p, device=device, verbose=False)
    end = timer()

    total_time_s = end - start
    avg_latency_ms, throughput = compute_stats(len(image_paths), total_time_s)

    row = {
        "model": "YOLO",
        "variant": model_name,
        "device": device,
        "batch_size": batch_size,
        "n_samples": len(image_paths),
        "total_time_s": total_time_s,
        "avg_latency_ms": avg_latency_ms,
        "throughput_samples_per_s": throughput,
        "notes": "single-image inference, excludes manifest creation"
    }
    row.update(get_system_info())
    save_result_row(OUT_CSV, row)
    print(row)

if __name__ == "__main__":
    main()