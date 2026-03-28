from pathlib import Path
import pandas as pd
import random

ROOT = Path("thewilds_cameratraps_full")
OUT = Path("outputs/tables/benchmark")
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    all_images = [p for p in ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    all_images = sorted(all_images)

    random.seed(42)
    sample_n = min(500, len(all_images))
    sampled = random.sample(all_images, sample_n)

    rows = []
    for p in sampled:
        rows.append({
            "image_path": str(p),
            "filename": p.name,
            "subset": "benchmark_500"
        })

    df = pd.DataFrame(rows)
    out_path = OUT / "benchmark_images_500.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()