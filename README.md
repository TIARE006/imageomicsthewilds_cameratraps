# Master README for `exp34_release`

## What this package is

This package is a **curated release focused on Experiment 3 and Experiment 4** from the broader SEC 2026 project plan, *Replay in the Wild: Trace-Driven Evaluation of Multi-Modal Autonomous Sensing Systems*.

It is **not** the full paper artifact for all six planned experiments. Instead, it packages:

- the final retained outputs for **Experiment 3: Edge Resource Model**,
- the final retained outputs for **Experiment 4: Replay-Based Policy Comparison**,
- the BirdNET confidence tables used to label the audio side of the replay,
- and the source scripts needed to understand or regenerate the released Exp3/Exp4 outputs.

In other words, this zip should be read as a **paper-facing subset for Exp3/Exp4**, not as the entire project repository.

---

## Relationship to the experimental plan

According to the project plan, Experiment 3 is supposed to answer whether a single mobile edge node can support concurrent multimodal inference under different scheduling policies, and Experiment 4 is supposed to compare policies by replaying identical real-world traces. The plan also expects a comparison table and timeline visualization for Experiment 4, and a table / gap analysis / policy comparison table for Experiment 3.

This release **does satisfy the core structure** of those two planned experiments:

- Exp3 includes benchmark summaries, resource-model tables, and a capacity-gap figure.
- Exp4 includes labeled replay traces, policy replay results, stress replay metrics, summary tables, and timeline figures.

However, this release is **best described as a completed analytical / replay package for the current available trace and labeling pipeline**, not as a perfect one-to-one finalization of every detail in the written plan. There are several important caveats:

1. **The package only covers Exp3/Exp4 (+ BirdNET support files)**, not the full plan.
2. **The packaged replay window is much shorter than the 24-day Campaign 2 replay described in the plan.** The released replay trace spans about **3.19 days** (from 2025-06-30 12:18:02 to 2025-07-03 16:55:14), not a 24-day Aug 4–28 window.
3. **Some retained artifacts come from slightly different policy-model versions.** In particular, some Exp3 files use `priority_based`, while others use `event_triggered`; those are related ideas but not identical implementations.
4. **Some Exp4 timeline figures appear to come from an earlier policy set** (`detection_only` / `selective_classification`) rather than the final three-policy summary table (`process_all` / `event_triggered` / `adaptive`).

So the right interpretation is:

> this zip is a strong, curated, reusable Exp3/Exp4 release,
> but it still contains a few version-alignment issues that should be stated explicitly in any paper or final repo README.

---

## Package layout

```text
exp34_release/
├── final_artifacts/
│   ├── README_FINAL.md
│   ├── birdnet/
│   │   ├── birdnet_predictions_full.csv
│   │   └── birdnet_clip_confidence.csv
│   ├── exp3/
│   │   ├── benchmark_report_table.csv
│   │   ├── benchmark_summary.csv
│   │   ├── exp3_summary_table.csv
│   │   ├── trace_driven_multimodal_resource_table.csv
│   │   └── multimodal_capacity_gap.png
│   └── exp4/
│       ├── replay_trace_labeled.csv
│       ├── audio_replay_trace_labeled.csv
│       ├── multimodal_replay_trace_fully_labeled.csv
│       ├── multimodal_replay_results_fully_labeled.csv
│       ├── multimodal_stress_replay_results_fully_labeled.csv
│       ├── multimodal_stress_replay_metrics_fully_labeled.csv
│       ├── exp4_summary_table_fully_labeled.csv
│       ├── replay_timeline_events.png
│       ├── replay_timeline_latency.png
│       └── replay_timeline_policy_state.png
└── src/
    ├── 05_resource_model/
    └── 06_replay/
```

---

## Directory-by-directory explanation

### `final_artifacts/`
This is the main results directory. If someone only wants the release outputs, this is where they should start.

### `final_artifacts/birdnet/`
These files support the **audio-side labeling** used by Experiment 4.

- `birdnet_predictions_full.csv`  
  Full BirdNET segment-level output. This is the most detailed retained BirdNET file and is mainly useful for provenance or debugging.

- `birdnet_clip_confidence.csv`  
  Clip-level BirdNET summary used to attach a single top confidence / species to each audio clip before replay.

This directory matters because Exp4’s event-triggered and adaptive logic depends on **audio confidence thresholding**.

### `final_artifacts/exp3/`
This directory contains the retained outputs for **Experiment 3 (Edge Resource Model)**.

- `benchmark_summary.csv`  
  Compact benchmark summary for the measured inference components.

- `benchmark_report_table.csv`  
  Cleaner report-style version of the benchmark summary.

- `trace_driven_multimodal_resource_table.csv`  
  Trace-driven multimodal utilization table that combines camera arrivals, audio arrivals, benchmark latencies, and policy assumptions.

- `exp3_summary_table.csv`  
  Compact policy/scenario summary intended for direct interpretation.

- `multimodal_capacity_gap.png`  
  Visualization of aggregate utilization across scenarios and policies.

### `final_artifacts/exp4/`
This directory contains the retained outputs for **Experiment 4 (Replay-Based Policy Comparison)**.

- `replay_trace_labeled.csv`  
  Camera replay trace after metadata-based timelapse labeling.

- `audio_replay_trace_labeled.csv`  
  Audio replay trace after BirdNET confidence merge.

- `multimodal_replay_trace_fully_labeled.csv`  
  Final merged replay input trace used by the replay engine.

- `multimodal_replay_results_fully_labeled.csv`  
  Event-level replay output for the final multimodal trace.

- `multimodal_stress_replay_results_fully_labeled.csv`  
  Event-level replay output under arrival-rate stress scaling.

- `multimodal_stress_replay_metrics_fully_labeled.csv`  
  Policy-level metrics across multiple stress factors.

- `exp4_summary_table_fully_labeled.csv`  
  The most useful final policy comparison table for the base replay.

- `replay_timeline_events.png`, `replay_timeline_latency.png`, `replay_timeline_policy_state.png`  
  Timeline figures intended to visualize volume, latency, and adaptive behavior over time.

### `src/05_resource_model/`
These scripts implement the **Exp3 analytical modeling pipeline**.

- `build_resource_table.py`  
  Builds a camera-only resource table from arrival summaries and benchmark latencies.

- `policy_load_model.py`  
  Builds a higher-level theoretical comparison from the resource table.

- `build_multimodal_resource_table.py`  
  Builds a multimodal resource model using camera arrivals plus manually parameterized audio rates.

- `build_trace_driven_multimodal_resource_table.py`  
  Builds a multimodal resource model using **trace-derived** camera and audio arrivals.

- `build_exp3_summary_table.py`  
  Compresses a full multimodal resource table into the compact Exp3 summary format.

- `plot_resource_gap.py` / `plot_multimodal_resource_gap.py`  
  Plotting scripts for utilization / capacity-gap figures.

### `src/06_replay/`
These scripts implement the **Exp4 replay pipeline**.

- `build_event_trace.py`  
  Builds a camera replay trace and camera arrival summary.

- `build_audio_event_trace.py`  
  Builds an audio replay trace and audio arrival summary.

- `merge_birdnet_confidence.py`  
  Merges BirdNET clip-level confidence into the audio replay trace.

- `label_camera_timelapse_from_metadata.py`  
  Parses `metadata.txt` files and labels likely timelapse camera frames.

- `build_multimodal_trace.py`  
  Merges the labeled camera trace and labeled audio trace into one replay stream.

- `policies.py`  
  Defines the policy logic (`process_all`, `detection_only`, `selective_classification`, `event_triggered`, `adaptive`).

- `replay_engine.py`  
  Runs the event stream through one or more policies and produces event-level replay output.

- `evaluate_replay.py`  
  Aggregates event-level replay output into policy metrics.

- `run_stress_replay.py`  
  Compresses inter-arrival times by stress factor and reruns replay.

- `plot_replay_timeline.py` / `plot_stress_replay.py`  
  Plotting scripts for time-series and stress-replay comparisons.

- `build_exp4_summary_table.py`  
  Builds a compact summary table from the stress replay metrics.

---

## Experiment 3: completion status and result interpretation

## What is complete

Experiment 3 is **substantially complete as a first-order analytical resource model**.

The release includes:

1. measured benchmark summaries,
2. compact report-form benchmark tables,
3. multimodal resource-model tables,
4. a compact policy/scenario summary table,
5. and a utilization figure.

This is enough to support the main Exp3 story:

> given the current measured service times and trace-derived arrival assumptions, how much of a single node’s compute budget would be consumed under different policies?

## What the packaged Exp3 results show

### Benchmarked service times
The retained benchmark summary shows the following measured mean latencies:

| Model | Mean latency | Mean throughput | Notes |
|------|-------------:|----------------:|------|
| BirdNET | 19057.49 ms/file | 0.0525 files/s | CPU; by far the slowest per item |
| BioCLIP | 60.85 ms/image | 16.46 images/s | CPU |
| YOLO | 38.64 ms/image | 25.92 images/s | CPU |

Two important conclusions follow immediately:

1. **BirdNET dominates per-item cost**.
2. The multimodal bottleneck in the packaged model is mostly **CPU-side audio**, not GPU-side image inference.

### Resource-model outcome
Across all retained Exp3 scenarios, the node remains classified as **`safe`**.

From `exp3_summary_table.csv`:

| Scenario | Policy | Aggregate utilization (%) | Feasibility |
|---------|--------|--------------------------:|-------------|
| nominal | adaptive | 6.50 | safe |
| nominal | priority_based | 6.46 | safe |
| nominal | process_all | 6.64 | safe |
| busy | adaptive | 16.10 | safe |
| busy | priority_based | 16.04 | safe |
| busy | process_all | 16.30 | safe |
| worst_case | adaptive | 32.01 | safe |
| worst_case | priority_based | 31.95 | safe |
| worst_case | process_all | 32.24 | safe |

### Interpretation
The core Exp3 conclusion from the retained package is:

- **under the current arrival assumptions and current benchmark latencies, the single-node system does not saturate**;
- even the worst retained case reaches only about **32.24% aggregate utilization**;
- because of that, Exp3 in this package is more of a **capacity headroom study** than a true “breaking-point” demonstration.

That is still useful, but the story is not:

> “the node fails here.”

Instead, the story is:

> “for the released arrival rates and measured latencies, the node still has substantial spare capacity.”

### What dominates utilization
In the retained Exp3 summary, CPU accounts for roughly **96%–99%** of aggregate utilization, depending on scenario and policy. That means the resource story is being driven primarily by **audio processing cost**, not by camera inference cost.

This matches the measured benchmark table: BirdNET is orders of magnitude slower per item than YOLO or BioCLIP.

## What is strong about the current Exp3 package

- It is internally interpretable.
- It includes both raw-ish benchmark summaries and paper-facing compact tables.
- It clearly supports a safe/headroom conclusion.
- It is reproducible from the included `src/05_resource_model/` scripts.

## Exp3 caveats

### 1. Two modeling variants are mixed in the retained outputs
This is the biggest Exp3 caveat.

- `exp3_summary_table.csv` and `multimodal_capacity_gap.png` come from the `multimodal_resource_table` path, which uses policies like `priority_based`.
- `trace_driven_multimodal_resource_table.csv` comes from a different script path and uses `event_triggered` instead.

These are **not the same policy definitions**.

So readers should **not** directly compare `exp3_summary_table.csv` row-for-row against `trace_driven_multimodal_resource_table.csv` as if they were a single perfectly synchronized output family.

### 2. The figure title says “capacity gap,” but the retained scenarios never become infeasible
The figure is still useful, but it visually presents **relative utilization**, not an actual failure gap where the system crosses 100% utilization.

### 3. The release does not include the upstream Exp1 arrival-extraction outputs
The model is clearly using arrival summaries, but the complete upstream extraction pipeline for all modalities is not bundled here. That is fine for a curated Exp3/Exp4 release, but it should be stated explicitly.

## Bottom-line Exp3 status

**Status:** analytically complete for the packaged release, but still slightly version-mixed.  
**Main conclusion:** all retained scenarios are safe; CPU-side audio dominates load; no retained case reaches overload.  
**What would strengthen it further:** one synchronized final table/figure set using one policy vocabulary and one modeling path.

---

## Experiment 4: completion status and result interpretation

## What is complete

Experiment 4 is the strongest part of the package.

The release includes the full chain needed to tell the story:

1. camera replay trace construction,
2. camera timelapse labeling from metadata,
3. audio replay trace construction,
4. BirdNET confidence merge,
5. merged multimodal replay trace,
6. event-level replay outputs,
7. stress replay outputs,
8. policy-level summary tables,
9. and timeline figures.

That means the package does not just contain final tables; it also preserves the **provenance** of the labeled replay inputs.

## What the packaged Exp4 trace actually contains

The retained fully labeled multimodal trace contains:

- **15,081 camera events**
- **110 audio events**
- **15,191 total events**
- **4 camera sites**
- **4 audio sites**
- trace window from **2025-06-30 12:18:02** to **2025-07-03 16:55:14**
- total duration of about **3.19 days**

Additional trace-level observations:

- **126 camera frames** are labeled as timelapse, all from `TW01-CT02`.
- On the audio side, **69 / 110 clips (62.7%)** have confidence `>= 0.7`.
- The median audio confidence is about **0.81**.

So the packaged Exp4 story is based on a **short, concrete, fully labeled replay slice**, not on the full 24-day Campaign 2 window described in the written plan.

## Base replay results

From `exp4_summary_table_fully_labeled.csv`:

| Policy | Coverage | Mean latency (ms) | P95 latency (ms) | Max latency (ms) | Compute cost (sec) | SLO compliance |
|-------|---------:|------------------:|-----------------:|-----------------:|-------------------:|---------------:|
| adaptive | 99.78% | 10977.40 | 497.0 | 1448370.0 | 2931.49 | 98.17% |
| event_triggered | 98.90% | 9054.86 | 253.8 | 1314967.5 | 2074.08 | 98.62% |
| process_all | 100.00% | 22988.23 | 497.0 | 2096325.0 | 3595.38 | 97.47% |

## Main Exp4 interpretation

### 1. `event_triggered` is the best efficiency tradeoff in the retained replay
Compared with `process_all`, `event_triggered`:

- reduces compute cost by about **42.3%**,
- reduces mean latency by about **60.6%**,
- improves SLO compliance by about **1.15 percentage points**,
- while sacrificing only about **1.10 percentage points** of coverage.

This is a strong result. In the retained release, `event_triggered` is the cleanest “best tradeoff” policy.

### 2. `adaptive` preserves more coverage, but behaves closer to full processing than expected
Compared with `process_all`, `adaptive`:

- reduces compute cost by about **18.5%**,
- reduces mean latency by about **52.2%**,
- improves SLO compliance by about **0.69 percentage points**,
- while sacrificing only about **0.22 percentage points** of coverage.

This makes `adaptive` a reasonable middle ground, but not necessarily the best efficiency winner.

### 3. `process_all` is the strongest coverage baseline but the weakest queueing baseline
`process_all` does exactly what it is supposed to do: it covers everything.

But that comes with:

- the highest total compute cost,
- the highest mean latency,
- and the worst long-tail backlog behavior.

That makes it a good baseline, but not the best deployed policy in the retained replay.

## Why the policy differences happen

### Skip behavior
The skip reasons show the policy semantics clearly:

- `event_triggered` skips **126 timelapse camera events** and **41 low-confidence audio events**, for a total of **167 skipped events**.
- `adaptive` skips only **34 low-confidence audio events** and skips **no camera events** in the retained run.

That means `adaptive` largely stopped behaving like strict event-triggering and spent most of its time in a more permissive mode.

### Adaptive-state behavior
The retained `adaptive` run is dominated by the `high_activity` state:

- `high_activity`: **14,775 events**
- `overloaded`: **235 events**
- `normal`: **179 events**
- `idle_backlog_flush`: **2 events**

This is extremely important for interpretation.

It means the adaptive controller is **not** spending most of the time in a lightweight event-triggered mode. Instead, once detections accumulate, it transitions into a high-activity regime and behaves much more like full processing.

That explains why:

- its coverage is very high,
- its compute cost is much closer to `process_all` than to `event_triggered`,
- and its latency improvements are smaller than the pure event-triggered policy.

### Audio cost matters more than event count suggests
Audio accounts for only **110 of 15,191 total events** (less than 1% of events), but because BirdNET is so expensive per clip, audio still contributes a large fraction of total compute cost:

- `process_all`: audio contributes about **58.3%** of total compute cost
- `event_triggered`: audio contributes about **63.4%** of total compute cost
- `adaptive`: audio contributes about **49.4%** of total compute cost

So the replay result is not just “camera-dominated because there are more images.”  
In compute terms, **audio is disproportionately expensive**.

## Stress replay results

The package also includes a stress-replay study, which is effectively a bridge from Exp4 into the Exp6-style overload question.

### Stress factors retained
The metrics table includes stress factors:

- 1×
- 2×
- 5×
- 10×
- 20×
- 50×
- 100×

### Main stress result
At every retained stress factor above 1×, `event_triggered` remains the most robust of the three released policies.

Examples:

| Stress factor | Best SLO compliance | Winner |
|--------------|--------------------:|--------|
| 1× | 98.62% | event_triggered |
| 5× | 94.99% | event_triggered |
| 10× | 90.26% | event_triggered |
| 20× | 80.08% | event_triggered |
| 50× | 38.43% | event_triggered |
| 100× | 19.22% | event_triggered ≈ adaptive |

### Breaking-point view
Using SLO compliance thresholds:

- all three policies fall below **95%** by **5×** stress,
- `event_triggered` is the last one to stay above **90%** and survives until **10×**,
- `process_all` degrades fastest,
- `adaptive` is better than `process_all`, but usually still behind `event_triggered`.

### Queue growth
Peak queue length grows very aggressively under stress:

- at **10×**, peak queue is already **1468** (`event_triggered`) vs **2469** (`process_all`)
- at **20×**, it becomes **2998** (`event_triggered`) vs **5443** (`process_all`)
- at **100×**, it reaches **12141** (`event_triggered`) vs **14929** (`process_all`)

So the main overload story is not just about latency; it is also about rapid queue explosion.

### Important interpretation detail
Total compute cost stays nearly constant for `process_all` and `event_triggered` across stress factors. That is expected: stress replay compresses inter-arrival times, but does not change the number of events or the per-event service-time model.

`adaptive` is the exception: its compute cost decreases under extreme stress because the overloaded-state behavior changes what gets fully processed.

## What is strong about the current Exp4 package

- It preserves provenance from BirdNET confidence and camera metadata to final replay metrics.
- It contains both event-level and policy-level outputs.
- It includes a stress replay extension, which makes the package richer than a single static table.
- It clearly supports a deployable policy conclusion: **event-triggered is the best practical tradeoff in the retained release**.

## Exp4 caveats

### 1. The replay window does not match the written plan
The plan says Exp4 should replay a **24-day Campaign 2 deployment (Aug 4–28)**. The retained package instead uses a trace lasting about **3.19 days**.

This does not invalidate the experiment, but it changes the scope. The current result should be described as:

> a fully labeled short-window replay study

rather than:

> the final full Campaign 2 replay study.

### 2. Timeline figures appear to be out of sync with the final summary table
The retained timeline figures show legends with:

- `adaptive`
- `detection_only`
- `process_all`
- `selective_classification`

But the final fully labeled summary table uses:

- `adaptive`
- `event_triggered`
- `process_all`

That strongly suggests the timeline figures were generated from an earlier replay-results file or a broader policy set, then copied into the final artifact folder.

So these figures are still informative, but they should **not** be treated as perfectly synchronized with `exp4_summary_table_fully_labeled.csv` until they are regenerated from the final fully labeled replay results.

### 3. Audio trace contains duplicate clip rows
The labeled audio replay trace has **110 rows but only 100 unique audio paths**, so a small number of audio clips are duplicated. This does not necessarily break the replay, but it should be documented because it affects trace cardinality and could slightly affect metrics.

## Bottom-line Exp4 status

**Status:** functionally complete and analytically strong for the packaged release.  
**Main conclusion:** `event_triggered` is the best overall policy tradeoff; `adaptive` preserves coverage but behaves too much like `process_all` in this trace; `process_all` is the strongest coverage baseline but the weakest efficiency choice.  
**What would strengthen it further:** rerun the final figures from the final three-policy fully labeled replay, and rerun on the intended full Campaign 2 window if strict plan alignment is required.

---

## What this package proves versus what it does not yet prove

## What it proves well

This release already proves several useful things:

1. A reproducible Exp3/Exp4 pipeline exists.
2. The replay now uses **real BirdNET-derived audio confidence**, not a placeholder-only audio trigger.
3. The camera replay now uses **metadata-driven timelapse labeling**, not only a synthetic default rule.
4. For the retained trace, `event_triggered` is a strong deployment candidate.
5. Under stress scaling, queueing and latency degrade rapidly, and full processing is the least robust choice.

## What it does not fully prove yet

1. It does not yet prove the result on the full planned 24-day Campaign 2 window.
2. It does not yet provide a single perfectly synchronized Exp3 artifact family under one policy vocabulary.
3. It does not yet provide final figure/table synchronization for all Exp4 visuals.
4. It does not include the upstream full-paper context for Experiments 1, 2, 5, and 6 as standalone release artifacts.

---

## Recommended use in the repo / paper

If this package is being used as the final shared release, the safest top-level description is:

> This archive contains the curated Exp3/Exp4 artifact subset for the SEC 2026 multimodal edge-sensing project. It includes benchmark-based edge resource modeling, metadata- and BirdNET-labeled multimodal replay traces, replay policy evaluation results, and stress-replay outputs. The package is sufficient to interpret the current Exp3/Exp4 results, but some artifacts reflect slightly different policy-model versions and the packaged replay window is shorter than the 24-day Campaign 2 target in the written plan.

---

## Best files to read first

If a collaborator only wants the minimum set of high-value outputs, start in this order:

1. `final_artifacts/exp3/benchmark_summary.csv`
2. `final_artifacts/exp3/exp3_summary_table.csv`
3. `final_artifacts/exp3/multimodal_capacity_gap.png`
4. `final_artifacts/exp4/exp4_summary_table_fully_labeled.csv`
5. `final_artifacts/exp4/multimodal_stress_replay_metrics_fully_labeled.csv`
6. `final_artifacts/exp4/multimodal_replay_trace_fully_labeled.csv`
7. `final_artifacts/birdnet/birdnet_clip_confidence.csv`

---

## Recommended cleanup before public/paper release

If you want this package to become the final public-facing repo artifact, the highest-value cleanup steps are:

1. regenerate the Exp4 timeline figures from `multimodal_replay_results_fully_labeled.csv`,
2. decide on one final Exp3 policy vocabulary and regenerate the summary table / figure accordingly,
3. document why the replay window is 3.19 days rather than the planned 24 days,
4. optionally deduplicate the audio clip trace if those repeated rows were unintentional,
5. keep this README as the authoritative explanation of what is actually in the release.
