from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import open_clip

from benchmark_utils import (
    load_manifest,
    timer,
    compute_stats,
    save_result_row,
    get_system_info,
)

MANIFEST = "outputs/tables/benchmark/benchmark_images_500.csv"
OUT_CSV = "outputs/tables/benchmark/bioclip_speed.csv"


def main():
    # ===== config =====
    model_name = "ViT-B-32"
    pretrained = "openai"   
    device = "cpu"          
    batch_size = 1

    # ===== load manifest =====
    df = load_manifest(MANIFEST)
    image_paths = df["image_path"].tolist()

    # ===== load model =====
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    # 这里文本输入固定，主要是为了让图像编码路径完整走一遍
    text_inputs = tokenizer(["an animal", "a camera trap image"]).to(device)

    with torch.no_grad():
        _ = model.encode_text(text_inputs)

    # ===== warmup =====
    warmup_paths = image_paths[:10]
    with torch.no_grad():
        for p in warmup_paths:
            image = Image.open(p).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            _ = model.encode_image(image_tensor)

    # ===== timed run =====
    start = timer()
    with torch.no_grad():
        for p in image_paths:
            image = Image.open(p).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            _ = model.encode_image(image_tensor)
    end = timer()

    total_time_s = end - start
    avg_latency_ms, throughput = compute_stats(len(image_paths), total_time_s)

    row = {
        "model": "BioCLIP",
        "variant": f"{model_name} / {pretrained}",
        "device": device,
        "batch_size": batch_size,
        "n_samples": len(image_paths),
        "total_time_s": total_time_s,
        "avg_latency_ms": avg_latency_ms,
        "throughput_samples_per_s": throughput,
        "notes": "single-image image-encoder inference, excludes manifest creation",
    }
    row.update(get_system_info())

    save_result_row(OUT_CSV, row)
    print(row)


if __name__ == "__main__":
    main()