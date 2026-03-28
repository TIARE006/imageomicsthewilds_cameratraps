from pathlib import Path
import pandas as pd
import random

ROOT = Path("thewilds_bioacousticmonitors_full")
OUT = Path("outputs/tables/benchmark")
OUT.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main():
    all_audio = [p for p in ROOT.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    all_audio = sorted(all_audio)

    random.seed(42)
    sample_n = min(100, len(all_audio))
    sampled = random.sample(all_audio, sample_n)

    rows = []
    for p in sampled:
        rows.append({
            "audio_path": str(p),
            "filename": p.name,
            "subset": "benchmark_audio_100"
        })

    df = pd.DataFrame(rows)
    out_path = OUT / "benchmark_audio_100.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()