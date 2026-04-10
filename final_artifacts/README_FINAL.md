# Final Artifacts README

## Project
**Replay in the Wild: Trace-Driven Evaluation of Multi-Modal Autonomous Sensing Systems**

This directory contains the **final curated artifacts** for the portions of the project related to:

- **Experiment 3: Edge Resource Model**
- **Experiment 4: Replay-Based Policy Comparison**
- **BirdNET-derived audio confidence outputs** used to support the final replay labeling pipeline

This folder was created after a non-destructive project cleanup.  
The goal is to separate:

- **final, paper-facing outputs**
from
- intermediate, old, test-only, and development-time files.

---

# Directory Overview

## `exp3/`
Final outputs for **Experiment 3 (Edge Resource Model)**.

These files support the resource modeling analysis that asks:

> Can a single mobile edge node handle concurrent multi-modal inference under SLO constraints, and under what policy assumptions?

This folder contains the final benchmark/resource-model tables and the final capacity-gap figure used to summarize feasibility.

### Files

#### `trace_driven_multimodal_resource_table.csv`
Final **trace-driven multimodal resource model table**.

This is one of the most important Experiment 3 outputs. It combines:
- measured or benchmarked inference service times,
- trace-derived arrival information,
- policy assumptions,
- and resulting resource-use estimates.

Typical uses:
- policy comparison,
- throughput feasibility analysis,
- determining whether a single edge node can sustain projected demand.

#### `exp3_summary_table.csv`
Final summary table for Experiment 3.

This is the most compact Experiment 3 result table and is the best candidate for:
- direct paper inclusion,
- summary in slides,
- or quick discussion with collaborators.

Typical contents include:
- scenario-level results,
- aggregate feasibility interpretation,
- policy-level comparison.

#### `multimodal_capacity_gap.png`
Final figure showing the **capacity gap** or resource shortfall / feasibility relationship.

This figure is useful for:
- illustrating when projected workload exceeds practical edge capacity,
- visually comparing policy assumptions,
- showing where adaptation or prioritization becomes necessary.

#### `benchmark_summary.csv`
Intermediate-to-final benchmark summary table retained because it supports the derivation of the final resource model.

This table is useful if someone needs to trace:
- where latency inputs came from,
- what benchmark values were used,
- or how service-time assumptions entered the final Experiment 3 model.

#### `benchmark_report_table.csv`
Formatted benchmark-oriented report table.

This is useful as a cleaner presentation layer over the benchmark measurements and may be used in:
- internal notes,
- result writeups,
- appendix tables,
- or discussion with collaborators.

---

## `exp4/`
Final outputs for **Experiment 4 (Replay-Based Policy Comparison)**.

This folder contains the final labeled replay inputs, final replay outputs, final stress-replay outputs, and final summarized policy metrics.

Experiment 4 asks:

> How do different sensing policies perform when replayed on identical real-world traces?

The final pipeline includes:
- **camera-side metadata-driven timelapse labeling**
- **audio-side BirdNET-derived confidence labeling**
- replay under multiple policies
- final replay and stress-replay evaluation

### Files

#### `replay_trace_labeled.csv`
Final camera replay trace with metadata-driven labeling.

This file reflects the camera-side labeling step in which:
- deployment/session metadata were used,
- timelapse-related semantics were incorporated,
- and camera events were prepared for replay.

This file is especially important because it is no longer using placeholder logic for timelapse handling.

#### `audio_replay_trace_labeled.csv`
Final audio replay trace with BirdNET-derived confidence attached.

This file is the audio-side counterpart to the camera labeled trace.  
It includes:
- audio clip events,
- the matched BirdNET confidence,
- optional top-species information,
- and any derived fields used by the replay framework.

This file is critical because it upgrades the replay from:
- placeholder or missing audio confidence
to
- **real BirdNET-derived audio confidence values**.

#### `multimodal_replay_trace_fully_labeled.csv`
The final **fully labeled multimodal replay trace**.

This is the main replay input for final Experiment 4 evaluation.

It combines:
- labeled camera events,
- labeled audio events,
- final multimodal ordering / synchronization,
- and the final event stream used by the replay engine.

This is the single best “final input trace” file for Experiment 4.

#### `multimodal_replay_results_fully_labeled.csv`
Final replay-engine output for the fully labeled multimodal trace.

This file contains event-level replay results across the final evaluated policies:
- `process_all`
- `event_triggered`
- `adaptive`

This file is the base input to the policy-level evaluation step.

#### `multimodal_stress_replay_results_fully_labeled.csv`
Final event-level output from the **stress replay** evaluation.

This is the stress-test counterpart to the plain replay output.  
It reflects how policies behave under scaled or intensified workload conditions.

Useful for:
- robustness analysis,
- degradation analysis,
- and identifying policy breakpoints.

#### `multimodal_stress_replay_metrics_fully_labeled.csv`
Final stress-replay policy metrics table.

This is one of the main final outputs for Experiment 4.  
It contains summarized metrics across stress factors and policies.

Typical uses:
- comparing policy resilience,
- plotting stress-compliance curves,
- evaluating whether latency/coverage degrade gracefully,
- and supporting discussion of overload behavior.

#### `exp4_summary_table_fully_labeled.csv`
Final policy-level comparison table for Experiment 4.

This is the **most important summary table** for the replay experiment.

It contains policy-level metrics such as:
- total events,
- processed events,
- skipped events,
- coverage,
- mean latency,
- P95 latency,
- max latency,
- compute cost,
- SLO compliance,
- classification trigger statistics,
- operator effort proxy,
- dominant policy state.

This file is the strongest candidate for:
- the main Experiment 4 comparison table in the paper,
- summary discussion in internal reports,
- and response to questions like “which policy is best under what tradeoff?”

#### `replay_timeline_events.png`
Final replay timeline figure focusing on event behavior over time.

Useful for showing:
- how event load evolves,
- when activity spikes occur,
- and how replay input volume varies.

#### `replay_timeline_latency.png`
Final replay timeline figure focusing on latency behavior.

Useful for:
- illustrating time-varying policy latency,
- spotting overload windows,
- showing where response-time tails worsen.

#### `replay_timeline_policy_state.png`
Final replay timeline figure showing policy-state evolution.

Useful especially for:
- interpreting adaptive behavior,
- showing when the policy enters high-activity or selective modes,
- and explaining why performance changes over time.

---

## `birdnet/`
Final BirdNET-derived outputs used to support the final audio labeling pipeline.

This directory exists because BirdNET confidence became a crucial input for the final version of Experiment 4.

### Files

#### `birdnet_predictions_full.csv`
Full segment-level BirdNET predictions.

This file contains the detailed per-segment prediction output from BirdNET, including:
- audio path,
- segment start time,
- segment end time,
- species name,
- confidence.

This is the highest-detail BirdNET output retained in the final curated set.

Use this when:
- debugging confidence assignment,
- checking which species caused a high-confidence trigger,
- or validating segment-level predictions.

#### `birdnet_clip_confidence.csv`
Clip-level BirdNET confidence summary.

This file is derived from the full segment-level BirdNET outputs and provides, per audio clip:
- the top predicted species,
- the top confidence,
- the segment interval where that top confidence occurred.

This is the actual bridge file used to attach BirdNET confidence to the audio replay trace.

This file is more directly relevant to Experiment 4 than `birdnet_predictions_full.csv`, because replay policy logic uses clip-level confidence more naturally than raw segment-level outputs.

---

# Final Pipeline Summary

## Experiment 3 pipeline summary
The final Experiment 3 flow can be understood as:

1. Benchmark or collect model service times
2. Estimate or summarize per-modality arrival behavior
3. Construct multimodal workload/resource tables
4. Compare policy assumptions under edge capacity constraints
5. Produce feasibility tables and the capacity-gap figure

The key final outputs retained are:
- `exp3/trace_driven_multimodal_resource_table.csv`
- `exp3/exp3_summary_table.csv`
- `exp3/multimodal_capacity_gap.png`

---

## Experiment 4 pipeline summary
The final Experiment 4 flow can be understood as:

1. Build camera replay trace
2. Parse camera metadata to identify or approximate timelapse-related semantics
3. Build audio replay trace
4. Run BirdNET on audio clips to obtain real confidence values
5. Merge BirdNET confidence into the audio replay trace
6. Combine labeled camera + labeled audio into a fully labeled multimodal trace
7. Run replay engine across policies
8. Run stress replay across policies and stress factors
9. Evaluate replay outputs into policy-level comparison metrics

The key final outputs retained are:
- `exp4/multimodal_replay_trace_fully_labeled.csv`
- `exp4/multimodal_replay_results_fully_labeled.csv`
- `exp4/multimodal_stress_replay_metrics_fully_labeled.csv`
- `exp4/exp4_summary_table_fully_labeled.csv`

---

# Which files matter most for the paper?

If someone only wants the **minimum set of final outputs** for writing the paper, the most important files are:

## For Experiment 3
- `exp3/exp3_summary_table.csv`
- `exp3/trace_driven_multimodal_resource_table.csv`
- `exp3/multimodal_capacity_gap.png`

## For Experiment 4
- `exp4/exp4_summary_table_fully_labeled.csv`
- `exp4/multimodal_stress_replay_metrics_fully_labeled.csv`
- `exp4/replay_timeline_events.png`
- `exp4/replay_timeline_latency.png`
- `exp4/replay_timeline_policy_state.png`

## For trace-labeling provenance
- `exp4/replay_trace_labeled.csv`
- `exp4/audio_replay_trace_labeled.csv`
- `birdnet/birdnet_clip_confidence.csv`

---

# Provenance Notes

## Camera-side labeling
The final replay pipeline includes camera-side labeling based on deployment/session metadata rather than relying only on placeholder defaults. This is important for timelapse-aware replay logic.

## Audio-side labeling
The final replay pipeline includes BirdNET-derived real confidence values. These were exported at segment level and then aggregated to clip level before being merged into the audio replay trace.

## Fully labeled replay
The “fully labeled” outputs refer to the version of the replay pipeline in which:
- camera metadata-driven labeling is included, and
- audio BirdNET-derived confidence is included.

---

# Interpretation Notes

## Experiment 3
Experiment 3 is intended to answer whether a single edge node can support multimodal inference workloads and what tradeoffs emerge under different policy assumptions.

The final retained files support:
- throughput feasibility analysis,
- resource gap analysis,
- and policy-level comparison under edge constraints.

## Experiment 4
Experiment 4 is intended to compare policy behavior under identical replay traces.

The final retained files support evaluation of:
- latency,
- coverage,
- compute cost,
- policy state behavior,
- and operator-effort proxy metrics.

---

# Current Caveat

The final curated outputs are sufficient for the implemented Experiment 3 and Experiment 4 pipelines.

However, if a later paper revision requires strict alignment with a different replay trace definition (for example, a different deployment window or a different campaign partition), the replay input trace may need to be regenerated and the final Experiment 4 outputs rerun.

This README documents the **current final curated version** of the project outputs, not every historical or alternate replay configuration.

---

# Relationship to `archive/`

The `archive/` directory contains:
- older replay outputs,
- intermediate versions,
- test-only files,
- one-off BirdNET testing artifacts,
- and development-stage scripts no longer needed for final result interpretation.

Those archived files were intentionally preserved for safety, but they are **not** the recommended starting point for reading final project results.

If you are trying to understand or cite the final outputs, start with:
- `final_artifacts/exp3/`
- `final_artifacts/exp4/`
- `final_artifacts/birdnet/`

---

# Recommended Reading Order

If you are new to the project, read files in this order:

1. `exp3/exp3_summary_table.csv`
2. `exp3/multimodal_capacity_gap.png`
3. `exp4/exp4_summary_table_fully_labeled.csv`
4. `exp4/multimodal_stress_replay_metrics_fully_labeled.csv`
5. `exp4/replay_timeline_policy_state.png`
6. `birdnet/birdnet_clip_confidence.csv`

This order gives:
- the Experiment 3 resource story,
- then the Experiment 4 policy tradeoff story,
- then the BirdNET audio-labeling provenance.

---

# Recommended Citation/Use Within Internal Drafts

When referring to the final project outputs internally, use language like:

- “final curated Experiment 3 artifacts”
- “fully labeled final replay outputs”
- “BirdNET-derived clip-level confidence table”
- “final policy comparison table for Experiment 4”

Avoid referencing archived outputs unless specifically discussing development history or intermediate versions.

---

# Contact / Ownership Context

This curated final artifact set is intended to support the paper draft and internal collaboration around:
- Experiment 3 resource modeling
- Experiment 4 replay-based policy comparison

If additional regeneration is needed, the authoritative workflow should begin from the project `src/` directory and rebuild outputs rather than editing the files in `final_artifacts/` manually.

---

# Practical Rule

If you only want the final answer and not the development history:

- use `final_artifacts/`
- ignore `archive/`

If you need to debug how a result was produced:

- use `archive/` only as a historical reference
- and go back to the corresponding scripts in `src/`

---