from pathlib import Path
import pandas as pd


AUDIO_TRACE = "outputs/tables/arrivals/audio_replay_trace.csv"
BIRDNET_CLIP = "outputs/tables/baseline/birdnet_clip_confidence.csv"
OUT_AUDIO_TRACE = "outputs/tables/arrivals/audio_replay_trace_labeled.csv"


def main():
    audio_df = pd.read_csv(AUDIO_TRACE)
    bird_df = pd.read_csv(BIRDNET_CLIP)

    # normalize path separators
    audio_df["audio_path_norm"] = audio_df["audio_path"].astype(str).str.replace("\\", "/", regex=False)
    bird_df["audio_path_norm"] = bird_df["audio_path"].astype(str).str.replace("\\", "/", regex=False)

    merged = audio_df.merge(
        bird_df[["audio_path_norm", "top_species", "top_confidence", "top_start_time", "top_end_time"]],
        on="audio_path_norm",
        how="left",
    )

    merged["confidence"] = merged["top_confidence"]

    merged.drop(columns=["audio_path_norm"], inplace=True)
    Path(OUT_AUDIO_TRACE).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_AUDIO_TRACE, index=False)

    print(f"Saved labeled audio trace: {OUT_AUDIO_TRACE}")
    print("Matched rows with BirdNET confidence:", merged['top_confidence'].notna().sum())
    print("Total rows:", len(merged))


if __name__ == "__main__":
    main()