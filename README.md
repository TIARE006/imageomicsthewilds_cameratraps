# Benchmark README

## `src/04_benchmark/`

- `benchmark_yolo.py`  
  Runs YOLO on the benchmark image subset and records runtime / speed results.

- `benchmark_bioclip.py`  
  Runs BioCLIP on the benchmark image subset and records runtime / speed results.

- `benchmark_birdnet.py`  
  Runs BirdNET on the benchmark audio subset and records runtime / speed results.

- `benchmark_utils.py`  
  Shared helper functions used by the benchmark scripts.

- `make_benchmark_manifest.py`  
  Builds the image benchmark manifest / subset list.

- `make_benchmark_audio_manifest.py`  
  Builds the audio benchmark manifest / subset list.

- `summarize_benchmarks.py`  
  Collects benchmark outputs and generates summary tables.

## `outputs/tables/benchmark/`

- `benchmark_audio_100.csv`  
  Audio benchmark subset or audio benchmark result table for the 100-sample run.

- `benchmark_images_500.csv`  
  Image benchmark subset or image benchmark result table for the 500-sample run.

- `yolo_speed.csv`  
  YOLO runtime / speed results.

- `bioclip_speed.csv`  
  BioCLIP runtime / speed results.

- `birdnet_speed.csv`  
  BirdNET runtime / speed results.

- `benchmark_summary.csv`  
  Combined benchmark summary across models.

- `benchmark_report_table.csv`  
  Cleaned report-ready table for slides / paper / summary reporting.
