# Execution evidence — 13 October 2026

This is an implementation verification record, not a benchmark report or an answer to the research question.

## What actually ran

- All **20 unit/protocol tests passed**, including the original 7 StreamSpot tests. THEIA tests cover UUID canonicalisation, node-label alignment, time sorting, edge direction, duplicate events, second predicates, archive streaming without extraction, missing definitions, hard caps, conflicting labels, malformed records, unknown-label exclusion, fingerprint tampering, node permutation equivariance, gradients and checkpoint restoration.
- Changing future test events left training features unchanged under explicit cutoffs. Changing all ground-truth labels to unknown left all three models' saved anomaly scores unchanged. Scalers were checked against training-window data only. Thresholds correctly remained unavailable when validation lacked negatives.
- A complete synthetic smoke run executed preparation, all three model paths, validation thresholds, predictions, metric recomputation and dashboard creation.
- A larger **synthetic resource pilot** ran on this Mac's arm64 CPU. Its deliberately simple topology has two entity types and two event types; it is not representative of genuine THEIA graphs.
- The existing StreamSpot quick experiment completed all four original methods and generated its dashboard: `results/quick_20261013T063017021814Z`. Prior StreamSpot outputs were not overwritten.
- Generated dashboard JavaScript passed `node --check`. A local server returned HTTP 200 for the dashboard; it was stopped after the check. This was a static-script/HTTP check, not an automated interactive browser test.

## Saved THEIA-pipeline runs

- [Synthetic smoke dashboard](../results/theia/SYNTHETIC_smoke_NOT_BENCHMARK_20261013T063838189322Z/dashboard.html)
- [Synthetic pilot dashboard](../results/theia/SYNTHETIC_pilot_NOT_BENCHMARK_20261013T063911710468Z/dashboard.html)
- [Pilot metrics](../results/theia/SYNTHETIC_pilot_NOT_BENCHMARK_20261013T063911710468Z/metrics.json)
- [Pilot checkpoint audit](../results/theia/SYNTHETIC_pilot_NOT_BENCHMARK_20261013T063911710468Z/checkpoint_audit.json)
- [Pilot input manifest](../data/theia/synthetic_pilot_v1/processed/manifest.json)

The run names and saved `dataset: synthetic_cdm18` / `experiment_role: synthetic_pilot` distinguish these from real data. Classification scores in those artifacts test the software only; they must not be reported as THEIA detection performance. `binary_metrics_defined` is a mathematical availability flag, not benchmark approval.

## Measured synthetic pilot

Environment: macOS 15.2 arm64, Python 3.13.7, PyTorch 2.14.1, NumPy 2.5.3, scikit-learn 1.9.1, CPU, two threads, seed 42. These are the versions actually installed during execution; the dependency file retains compatible version ranges rather than forcing this exact environment on other machines.

105,000 CDM-shaped records contained 60,000 unique entities and 45,000 edges. There were 20,000 target entities and 15,000 edges in each of train/validation/test. Each partition used 10 windows. Largest actual window: 2,048 nodes / 1,532 edges (configured edge ceiling 4,096). Features: 8 dimensions. MAGIC used a 5,000-node training reference; RF used 20,000 training targets. Neural models ran 5 epochs.

| Model | Training seconds | Test inference seconds | Worker high-water RSS MiB |
| --- | ---: | ---: | ---: |
| FLASH-inspired node | 0.144 | 0.022 | 366.39 |
| MAGIC-inspired node | 0.252 | 0.131 | 388.27 |
| RF node-type anomaly baseline | 0.161 | 0.149 | 367.53 |

Preparation took 1.804 seconds with a preparation-process high-water RSS of 220.64 MiB. The preparation high-water mark includes this script's imported libraries. Model timings exclude process/library startup and shared preparation; training includes scaling and checkpoint serialization. Test inference includes shard I/O and scoring. Worker memory is measured after prediction CSV writing and is not total computer memory. Full definitions are in [THEIA.md](THEIA.md#resource-bounds-and-measurements).

The first execution of this same pilot used approximately 364–377 MiB across workers; the retained final execution above reached 388 MiB. These are single-machine engineering measurements, not confidence intervals or paper-runtime comparisons. Timing/allocator variation is expected.

Reloading the final pilot's checkpoints recomputed **120,000 validation/test scores** (40,000 per model) with maximum absolute difference **0.0**. The smoke checkpoint audit covers another 288 scores. Evaluation audits also verify identical targets/labels across models, validation-only thresholds, saved decisions and metric recomputation.

## Genuine-data status and remaining validation

The source repositories, original DARPA folder metadata and ThreaTrace label file were accessible. Both verified JSON archive downloads returned **Google Drive — Quota exceeded**. A binary-archive access attempt also failed. The final downloader's exact failure is saved in `data/theia/raw/download_attempts.jsonl`. No genuine THEIA events were downloaded or imported; no genuine-data pilot or benchmark result is claimed.

Obtain `ta1-theia-e3-official-1r.json.tar.gz` and/or `ta1-theia-e3-official-6r.json.tar.gz` from the verified links in [THEIA.md](THEIA.md#verified-sources-and-access), preserving entity-definition members. For the documented two-stream workflow obtain both. For meaningful binary evaluation, obtain and review the operational event log / ground-truth report and supply evidence-backed negatives, or explicitly adopt and report the scoped presumed-background convention. The included malicious UUID list alone does not establish negative labels.

The parser is tested against synthetic CDM18 structures derived from the official parser's field requirements; **compatibility with a genuine downloaded record stream remains unverified**. Upon access recovery, check actual unresolved endpoints, label matches, target coverage, event/type vocabularies, truncation, class presence and time boundaries before interpreting any metrics.

No larger/full-dataset recommendation follows from the synthetic pilot. The 24 GB machine was not stress-tested with the full dataset. `configs/theia_larger.json` remains an unmeasured candidate. Measure a genuine pilot and inspect system memory pressure/swap and disk usage before increasing limits. CPU is implemented and exercised; MPS support is not claimed.
