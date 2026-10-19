# Reproducible THEIA study

This study uses the local DGL exports copied from the official MAGIC repository. It is a controlled comparison of **FLASH-inspired**, **MAGIC-inspired**, and node-type Random Forest methods, not a reproduction of either paper. StreamSpot remains available through the original `main.py` commands and comparison dashboard; its graph-level results are not pooled with this node-level study.

## Old-run audit: why about one minute was plausible

The audited run is `results/theia/processed_full_20261018T132637258404Z`. The historical shell invocation was not stored, so an exact shell history cannot be asserted. The existing `bash scripts/train_theia.sh` launcher executes `.venv/bin/python -u -m ids.theia.processed`. The saved configuration, logs, manifest and checkpoints match that path. At audit time the saved hashes of `processed.py`, `worker.py`, the node models and graph layers exactly matched the inspected source. Evidence is in [THEIA_OLD_RUN_AUDIT.json](THEIA_OLD_RUN_AUDIT.json).

The loader decoded actual THEIA graphs; `smoke_nodes` was null and every stored node/edge was retained during preparation. Neural workers constructed fresh models and optimizers, with no checkpoint-loading path before training. Checkpoints were written after training. This was **not** evaluation-only execution or reuse of cached trained models.

| Old method | Actual fit | Parameters | Measured training time |
| --- | --- | ---: | ---: |
| FLASH-inspired | 15 epochs × 3 whole graphs = 45 optimizer updates; 890,093 nodes per epoch | 6,790 | 9.481 s |
| MAGIC-inspired | 15 epochs × 3 whole graphs = 45 updates; random masked reconstruction; 5,000-node sampled reference afterward | 3,528 | 21.861 s |
| Random Forest | Uniform reservoir of 100,000 of 890,093 eligible training nodes; 100 trees | Not comparable | 1.846 s |

The neural models had 36 input features, two layers and hidden width 32. The graphs were already preprocessed. Reported training + validation + test scoring times total 43.694 seconds; shared preparation took 5.832 seconds, with process startup/exports adding overhead. A roughly one-minute run was consistent with these measurements, not evidence of simulated training. Repeated simple feature vectors also limit diversity; 1.6 million node occurrences are not 1.6 million independent training examples.

## Data verification and fixed split

Each pickle contains one graph. The ZIP contains copies of the same five graphs plus metadata; it is not additional training data.

| Source file | Graphs | Nodes / graph-local occurrences | Edges | Role |
| --- | ---: | ---: | ---: | --- |
| train0.pkl | 1 | 167,265 | 284,562 | Training |
| train1.pkl | 1 | 380,204 | 674,723 | Training |
| train2.pkl | 1 | 342,624 | 587,111 | Training |
| train3.pkl | 1 | 389,106 | 700,318 | Validation |
| test0.pkl | 1 | 344,767 | 628,107 | Test |
| Total | 5 | 1,623,966 | 2,874,821 | All partitions |

- Training: **890,093 nodes, 1,546,396 edges**; positive/negative status is unknown for all nodes. All observed training behaviour is used; no retrospectively labelled nodes are removed.
- Validation: **389,106 nodes, 700,318 edges**; all presumed background under the declared convention, zero annotated positives.
- Test: **344,767 nodes, 628,107 edges**; **25,319 positive indices** and **319,448 presumed-background** indices.

`train3.pkl` is deliberately held out from the original four training exports. “All eligible training data” means all nodes in the three declared fitting graphs, not using held-out validation/test targets for fitting. No training node or edge reservoir remains in the full study. Whole-graph context is retained for the neural models; RF is intentionally a feature-only baseline.

Counts are unique **indices within each file**, then summed across files. Complete UUID mappings are missing: the same entity may occur in several files, and unique entities or entity-level split independence cannot be established. Graph-local IDs are explicitly named as such. No timestamps are invented; these are static-file splits, not verified chronological boundaries.

### Labels and authenticity

[Upstream MAGIC's loader](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/utils/loaddata.py) stores the malicious metadata with the exported graphs. Its [entity-level evaluation](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/eval.py) applies `metadata['malicious'][0]` as indices into concatenated test embeddings. There is exactly one test graph here, so indices address `test0.pkl` directly. The importer preserves node order and verifies every derived label, node type, endpoint and target-table row against the source exports. Invalid, repeated or out-of-range malicious indices fail rather than being silently repaired.

The archive's SHA-256 is `926b4e30a39b1680fe512ef015bfe03ab85760e12dc7d8ae88da40a346bd6864`, exactly matching the archive at the pinned official repository commit. The prior [quality audit](../results/data_quality/REPORT.md) records verification. Every study separately hashes all source files and retains that matching provenance evidence. This establishes artifact identity, not complete correctness of annotations.

Our importer removes no full-run nodes/edges. Changes already made by upstream preprocessing cannot be reconstructed from these exports. The earlier audit found no repeated `(source,destination,type)` triples, but raw event duplication cannot be assessed without timestamps. Names are often `NA`. Background labels are assumptions, not independently verified benign ground truth.

## Methods and deliberate simplifications

| Method | Implemented | Simplified or omitted |
| --- | --- | --- |
| FLASH-inspired | Two directed GraphSAGE layers predict observed node type. Anomaly score = 1 − model probability of that type. | Log incoming/outgoing edge-type counts replace semantic Word2Vec/positional features. No original iterative multi-model refinement or neighbourhood forgiveness. Not the FLASH notebook pipeline. |
| MAGIC-inspired | Two single-head graph-attention layers; random masked count-feature reconstruction; mean k-nearest embedding distance to **all** training node occurrences. | No original structure-reconstruction objective, full original multihead architecture, online adaptation or original type/edge encoder. Hidden width 32 and common count features are controlled choices. |
| Random Forest | Predict observed node type from the same scaled count features; anomaly score = 1 − probability of observed type. Every eligible training row is supplied to the forest. | Not a supervised malicious/benign classifier. Standard tree bootstrap sampling remains; there is no pre-fit node reservoir. No graph message passing. |

All scores are **uncalibrated anomaly scores**, not probabilities of attack. Entity type is a training target for FLASH/RF and is excluded from input features. Event-type and entity-type vocabularies and StandardScaler statistics are fitted using training data only. Unknown evaluation event types use a reserved count bin; unseen evaluation node types receive maximum type-anomaly score.

MAGIC reference compression merges only exactly identical embedding rows and saves their integer multiplicities plus the mapping of every original training occurrence. kNN scoring uses those multiplicities to recover the exact mean distance to the k nearest **occurrences**, including duplicates. It is not a representative subset or approximate nearest-neighbour scheme. A unit test compares the compressed scorer with the expanded reference.

## Training, validation and progress

The full preset is `configs/theia_study_full.json`: seeds **42, 43, 44**, CPU/four threads, hidden width 32, up to 30 neural epochs, patience 5 and min-delta 1e-5. Epochs are not increased merely to consume time. Early stopping/checkpoint selection uses:

- FLASH: lowest validation node-type cross-entropy.
- MAGIC: lowest validation masked reconstruction MSE, with the same fixed validation mask across epochs; training RNG is restored after validation.
- RF: prespecified tree count/depth; no epoch selection or test-based tuning.

The earliest best checkpoint under min-delta is retained. A checkpoint is reloaded before final neural scoring. Validation attack recall/F1/AP/ROC-AUC are undefined because no positives are supplied there; the report preserves null values. Final thresholds use the declared validation-background 99th percentile with conservative score-tie handling, frozen before opening test batches for scoring. Test labels are used only for final metrics and post-hoc context figures.

Progress prints the dataset fingerprint, source checksums, complete config, environment/seed, partition/class counts, epoch, completed graph batches, processed nodes/edges, loss-target counts, loss, elapsed time, epoch/remaining-budget estimates, validation objective, checkpoint-selection reason and final validation metrics. RF reports tree-building progress; MAGIC reports reference coverage. ETA is an estimate and excludes later reference/scoring work where stated. No artificial delays are used.

MAGIC visits every training node as context every epoch, while only the randomly masked subset contributes reconstruction targets in that epoch; actual masked-target counts are logged. RF sees every training row but uses normal bootstrap resampling within individual trees. There is no independently justified benign-only restriction for these exports.

## Memory control

One graph is loaded for a neural forward/backward batch at a time, with hard ceilings of **400,000 nodes / 750,000 edges**. All five supplied graphs fit these ceilings. Oversize graphs fail explicitly instead of silently truncating. This approach preserves all stored graph edges for this dataset; it does not promise arbitrary graph sizes fit.

RF inputs and the all-node embedding reference are assembled in disk-backed NumPy arrays. RF itself is a bounded-depth forest with four workers; sklearn may allocate internal working copies. Exact kNN uses a kd-tree over unique embeddings with **256-query batches** and occurrence multiplicity. One model/seed subprocess runs at a time. These are controlled bounds rather than a constant-memory guarantee. Temporary memmaps are removed after their final checkpoint/reference is saved.

Peak memory is the fresh worker's OS high-water resident set in MiB, including imports, preprocessing, model selection, reference and prediction work. It excludes other processes and GPU memory; the whole computer's memory pressure is a separate quantity. Explicit epoch resumes report the maximum high-water mark saved across attempts; timings aggregate completed checkpointed work and the current attempt, excluding uncheckpointed interrupted work.

## Saved outputs and reproducibility

Each study directory has:

- `invocation.json`, exact `config.json`, software versions, source-code hashes and a source snapshot (including the plotting module).
- `data/`: original source hashes, split definitions, class counts, graph-local target table, full NPZ graph batches and `alignment_audit.json`.
- One directory per model/seed: environment, preprocessing statistics, initial neural weights, best/last checkpoints, optimizer/RNG state, epoch history, worker log, selected-checkpoint rationale, frozen threshold, validation/test per-node CSVs and metrics.
- MAGIC reference arrays/counts and a row-to-training-node identity table; RF checkpoint includes its scaler.
- `metrics.csv/json`: one row per model/seed; `summary.csv/json`: mean and **sample standard deviation (ddof=1)** over defined values, with counts of defined seeds.
- `evaluation_audit.json`: metrics recomputed from CSVs, validation-only thresholds, exact agreement of held-out target identities/labels across all methods/seeds.
- `figures/`: PNG/SVG learning curves, ROC/PR curves, confusion matrices, comparison error bars, and graph context for correct detections, false positives and missed positives. Graph records are in `graph_context.json`.
- `dashboard.html`: an offline report clearly marked smoke or full study. Missing outcome categories and undefined metrics are displayed explicitly.

Training seconds include scaler fitting, optimization, epoch validation/checkpoint selection, checkpoint writes and reference construction. Shared source preparation and interpreter startup are excluded. Test inference seconds include forward/scoring but exclude graph reads, CSV export and metric calculation; export wall time is also saved. This differs slightly from the old worker's inference-time convention: do not directly compare their times without accounting for definitions.

The graph figures use the first test node in each outcome category for the first configured seed, at most 20 nodes and 30 stored incident edges. Coordinates are illustrative. They are observed context, **not proven model explanations or causal attack paths**. No image generation is used.

## Commands

From the project directory, a new full study (new output directory required):

```bash
bash scripts/run_theia_study.sh --output results/theia/my_full_study
```

A real-data smoke check (three seeds, two epochs, first 4,096 nodes per graph and their induced edges):

```bash
bash scripts/run_theia_study.sh --smoke --output results/theia/my_smoke_check
```

Smoke vocabularies are derived only from the original training files; scaling/fitting uses the retained training subset. The smoke run is not a smaller estimate of full-study detection performance. Removed prefix-complement nodes/edges are counted in its manifest.

Open that study's report:

```bash
open results/theia/my_full_study/dashboard.html
```

Regenerate reports from saved predictions, optionally reloading all checkpoints and verifying scores (no training):

```bash
.venv/bin/python -m ids.theia.study_report --run results/theia/my_full_study --rescore
```

Explicit resume after an interruption, using the exact unchanged config/data/code:

```bash
bash scripts/run_theia_study.sh --resume results/theia/my_full_study
```

Resume verifies hashes before reusing completed workers. Neural jobs restart from the last completed epoch with optimizer/RNG state; unfinished RF fitting restarts fresh with the same seed. No existing checkpoint is reused unless `--resume` is supplied. Source changes require a new run. Resume is supported after data preparation completes; a failure during preparation requires a new output directory.

Tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The old StreamSpot path remains:

```bash
.venv/bin/python main.py run --config configs/quick.json
open results/comparison/dashboard.html
```

The combined dashboard remains a snapshot of the earlier StreamSpot/THEIA comparison. Use the new study-specific dashboard for the new multi-seed results.

## Remaining scientific limitations

The held-out test set has already been inspected in this project, including the data-quality audit. The new configuration is fixed from the requested methodology, not selected by trying alternatives against test scores, but this is not a pristine, never-seen confirmatory holdout. A future confirmatory study needs a new independently chosen holdout.

Unique entity overlap, temporal leakage and independently verified negative labels remain unresolved without UUID/raw-event/annotation evidence. Nearly all positive test nodes are adjacent to one hub, and most annotations have one node type. Repeated features, source preprocessing and this concentration limit independence/generalisation. Three seeds quantify initialization/masking/forest variability, not diversity of independent attacks or datasets. StreamSpot's size shortcut and aggressive edge sampling remain separate concerns. No benchmark-quality or paper-reproduction claim follows just from passing software checks.
