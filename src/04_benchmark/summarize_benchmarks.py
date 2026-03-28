from pathlib import Path
import pandas as pd

BENCHMARK_DIR = Path("outputs/tables/benchmark")
OUT_CSV = BENCHMARK_DIR / "benchmark_summary.csv"
OUT_REPORT_CSV = BENCHMARK_DIR / "benchmark_report_table.csv"


def load_all_benchmark_csvs(benchmark_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(benchmark_dir.glob("*_speed.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No benchmark CSV files matching '*_speed.csv' found in {benchmark_dir}"
        )

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def infer_modality(model_name: str) -> str:
    model_name = str(model_name).strip().lower()
    if model_name in {"yolo", "bioclip"}:
        return "image"
    if model_name in {"birdnet"}:
        return "audio"
    return "unknown"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["modality"] = df["model"].apply(infer_modality)

    group_cols = [
        "modality",
        "model",
        "variant",
        "device",
        "batch_size",
        "n_samples",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("model", "size"),
            mean_total_time_s=("total_time_s", "mean"),
            std_total_time_s=("total_time_s", "std"),
            mean_latency_ms=("avg_latency_ms", "mean"),
            std_latency_ms=("avg_latency_ms", "std"),
            mean_throughput=("throughput_samples_per_s", "mean"),
            std_throughput=("throughput_samples_per_s", "std"),
        )
        .reset_index()
    )

    return summary.sort_values(
        by=["modality", "model", "device", "batch_size", "n_samples"]
    ).reset_index(drop=True)


def make_report_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    report_df = summary_df.copy()

    def format_latency(row):
        if row["modality"] == "image":
            return f'{row["mean_latency_ms"]:.2f} ms/image'
        elif row["modality"] == "audio":
            return f'{row["mean_latency_ms"]:.2f} ms/file'
        return f'{row["mean_latency_ms"]:.2f} ms/sample'

    def format_throughput(row):
        if row["modality"] == "image":
            return f'{row["mean_throughput"]:.2f} images/s'
        elif row["modality"] == "audio":
            return f'{row["mean_throughput"]:.4f} files/s'
        return f'{row["mean_throughput"]:.2f} samples/s'

    report_df["mean_latency"] = report_df.apply(format_latency, axis=1)
    report_df["mean_throughput"] = report_df.apply(format_throughput, axis=1)

    keep_cols = [
        "modality",
        "model",
        "variant",
        "device",
        "batch_size",
        "n_samples",
        "runs",
        "mean_latency",
        "mean_throughput",
        "mean_total_time_s",
    ]

    return report_df[keep_cols]


def main():
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    all_df = load_all_benchmark_csvs(BENCHMARK_DIR)
    summary_df = summarize(all_df).round(4)
    report_df = make_report_table(summary_df)

    summary_df.to_csv(OUT_CSV, index=False)
    report_df.to_csv(OUT_REPORT_CSV, index=False)

    print("\n=== Benchmark summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Report table ===")
    print(report_df.to_string(index=False))

    print(f"\nSaved summary to: {OUT_CSV}")
    print(f"Saved report table to: {OUT_REPORT_CSV}")


if __name__ == "__main__":
    main()