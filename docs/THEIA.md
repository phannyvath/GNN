# THEIA extension: operating guide and protocol

## Data quality audit — 19 October 2026

The [completed quality audit](../results/data_quality/REPORT.md) verifies that the
local THEIA archive exactly matches the official MAGIC repository archive at the
recorded commit. This resolves the origin uncertainty recorded in earlier runs;
historical manifests retain their original statements. Structural checks passed,
but the audit found limited diversity in the simplified feature vectors, strongly
concentrated positive labels/topology, and a perfect sampled-node-count ranking
shortcut in the existing StreamSpot test split. These are evaluation limitations,
not evidence of corrupt or AI-generated files. Full numerical findings are in
[audit.json](../results/data_quality/audit.json). No models were retrained.

## Combined HTML dashboard

Open the comparison of the saved StreamSpot and full preprocessed THEIA runs:

```bash
open results/comparison/dashboard.html
```

The page works offline and includes an interactive graph explorer, ROC/PR figures,
training-loss plots, confusion matrices with row-normalised rates, metric selection,
dataset filtering, CSV export, source links and protocol differences. It compares
graph-level and node-level experiments descriptively, without pooling their test
sets or claiming equivalent model implementations. It uses the actual saved run
metrics, not the historical synthetic THEIA results.

Regenerate the snapshot with `.venv/bin/python scripts/build_comparison_dashboard.py`.
To select other completed runs, pass `--streamspot results/YOUR_RUN` and
`--theia results/theia/YOUR_FULL_RUN`. The default paths select the current
StreamSpot quick run and full THEIA run discussed here; new runs are not selected
automatically. Figures are generated from all labelled test predictions for the
first saved seed and checked against saved AP/AUC values. PNGs are embedded in the
HTML; downloadable PNG/SVG copies are in `results/comparison/figures/`.
`scripts/comparison_visuals.py` produces these figures and bounded graph excerpts.
StreamSpot examples are saved test-graph excerpts with whole-graph labels. THEIA
examples show the highest-degree test node from each label group and up to 28
nodes / 40 incident edges. They are deliberately selected illustrations, not
representative samples or causal attack explanations. Node selection exposes IDs;
THEIA IDs remain graph-local. Layout coordinates have no temporal meaning.

JavaScript syntax, rendering logic, graph switches, metric/curve controls, CSV
export and local artifact links were checked. Generated scientific plots were
visually inspected; a headless browser launch failed in this environment, so the
full page has not received screenshot-based browser QA.

## Train the newly supplied preprocessed graphs (19 October 2026)

From this project folder, run:

```bash
bash scripts/train_theia.sh
```

This entry point supports the current `data/theia/train0.pkl` through `train3.pkl`,
`test0.pkl`, and `metadata.json`. It uses the existing virtual environment (creating
one and installing requirements if necessary). DGL is not required: an allowlisted
reader reconstructs inert DGL records and CPU tensors with PyTorch's weights-only
storage loader. Unsupported pickle globals or graph schemas are rejected.

The default configuration is `configs/theia_processed.json`: CPU, four threads,
32 hidden units, 15 epochs, seed 42, and sequential FLASH-inspired, MAGIC-inspired,
and random-forest workers. Every node and edge in all five exports is processed;
one whole graph is loaded at a time. RF fits a uniform reservoir of up to 100,000
training nodes; MAGIC uses up to 5,000 training reference embeddings for kNN.
Those reservoirs are deliberate approximations, not full-population fits.

**Separate static-graph protocol:** `train0–2.pkl` fit the models; `train3.pkl` is
held out for validation; `test0.pkl` remains the test set. Thus one of the four
upstream training graphs is repurposed for validation. Validation nodes are
presumed background; test malicious indices come from the supplied metadata and
their complement is presumed background. These are dataset-convention assumptions,
not independently verified benign labels. Thresholds use the validation 99th
percentile (conservative tie handling); test labels never select thresholds.

Input features are log incoming/outgoing edge-type counts with a training-only
edge vocabulary and scaler. Node type is the FLASH/RF prediction target and is
excluded from features. The original one-hot type attributes are checked for
alignment. No raw timestamps or complete UUID mapping are available. IDs such as
`test0.pkl:node:42` are graph-local identifiers, not real UUIDs. Cross-graph entity
overlap cannot be audited, and this protocol makes no temporal causality or
entity-disjointness claim. It does not reproduce the original papers. Do not
interpret its results as equivalent to the raw-CDM protocol below or to the
600-graph StreamSpot experiment.

Each invocation creates a new `results/theia/processed_full_<timestamp>/` folder
with prepared data, source/code hashes, configuration, logs, epoch timings,
checkpoints, predictions, validation thresholds, resource measurements and audited
`metrics.csv`/`metrics.json`. The script prints that path. It does not create the
raw-CDM dashboard, whose chronological claims do not apply here. All saved metric
rows identify this as an exploratory static-graph comparison. Archive origin is
recorded in historical manifests as user-supplied. The subsequent quality audit
above independently matched the archive checksum to the official repository.

For an engineering check only, `bash scripts/train_theia.sh --smoke` takes an
induced prefix of 2,048 nodes per graph, uses one epoch, runs all three models,
and verifies checkpoint rescoring. Its metrics are not full-dataset results.

The remaining sections describe the **original raw-CDM workflow**. Its missing
raw/synthetic data folders were removed by the user; the new pickle exports do
not restore them. Historical synthetic timing results do not estimate full-graph
runtime.

This extends `main.py` and adds `ids/theia/`. The original StreamSpot modules, configurations, prepared data and prior results remain independent. These are simplified implementations for a controlled comparison, **not paper reproductions**. This document is software documentation, not assessment-report prose.

## Verified sources and access

Checked 13 October 2026. Exact reference commits and label SHA-256 are in [theia_sources/references.json](theia_sources/references.json).

- [MAGIC official repository](https://github.com/FDUDSDE/MAGIC): MIT code licence. Its [CDM parser](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/utils/trace_parser.py) expects CDM18 line-oriented JSON. Its THEIA recipe uses `ta1-theia-e3-official-6r.json.tar.gz`; training members are `.json`, `.json.1`–`.json.3`, test is `.json.8`. Other members provide definitions, so retain them. It offers processed `graphs.zip`/pickle artifacts, which this extension does **not** ingest: their format does not establish the timestamp-preserving shared protocol here. Its README directs users to ThreaTrace labels and discusses another convention in Appendix G.
- [FLASH official THEIA notebook](https://github.com/DART-Laboratory/Flash-IDS/blob/ccbafee7eb796c76aceaabdc259ebd1f12b867cc/Theia.ipynb) supplies the two archive IDs used by our downloader. It uses the 1r stream for training and 6r `.json.8` for testing, with parsing inside the notebook. It learns entity types using Word2Vec/positional features and GraphSAGE, and uses `data_files/theia.json` for evaluation. Its helper adjusts counts through two-hop neighbourhoods; **our metrics use exact target UUIDs, without that adjustment**. No LICENSE file was present in the inspected repository root. We do not redistribute or execute its notebook code or weights; the new model paths are independently written using this project's existing graph layers. Do not infer a code licence from public availability.
- [DARPA E3 release notice](https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md) releases the data into the public domain, as-is, with quality caveats. E3 uses CDM18; binary Avro and JSON archives are listed in the [original THEIA folder](https://drive.google.com/drive/folders/1AWXy7GFGJWeJPGzvkT935kTfwBYzjhfC). The release includes `CDM18.avdl`, `TCCDMDatum.avsc`, Java conversion tools, `operational_event_log.md`, and `ground truth/tc_ground_truth_report_e3_update.pdf`. A copy of the release notice is saved locally. The supplied importer supports JSONL, gzip JSONL and JSON tar.gz, **not binary Avro**. Obtain the JSON archive or use the release's conversion tools if you only have binary data.
- [ThreaTrace malicious UUID list](https://github.com/threaTrace-detector/threaTrace/blob/f5b1c02cb0c03373f11ea0378dc1491fa7d85b1c/groundtruth/theia.txt): included as `docs/theia_sources/theia_malicious.txt`, with the upstream MIT licence. It contains 25,358 unique canonical UUIDs. It supplies positive entity identities, not a complete benign/malicious label for every CDM node and not attack-onset timestamps. This is research-derived annotation; it is distinct from DARPA's narrative ground-truth report.

The original DARPA folder and FLASH notebook refer to the **same** JSON file IDs:

| Archive | Verified file link |
| --- | --- |
| `ta1-theia-e3-official-1r.json.tar.gz` | [1r](https://drive.google.com/file/d/10cecNtR3VsHfV0N-gNEeoVeB89kCnse5/view) |
| `ta1-theia-e3-official-6r.json.tar.gz` | [6r](https://drive.google.com/file/d/1Kadc6CUTb4opVSDE4x6RFFnEy0P1cRp0/view) |

Metadata and labels were accessible. Both JSON download attempts returned HTML instead of archives; both responses explicitly said **Google Drive — Quota exceeded**. The 1r binary alternative also returned HTML. No genuine raw events were obtained during verification. A login requirement was **not** established. Wait for quota recovery or download those exact files through Drive using access available to you. Do not rename an HTML response as an archive.

For a meaningful labelled experiment, also obtain the operational event log and E3 ground-truth PDF from the [official release folder](https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK). Review the intended temporal scope and derive/audit explicit negative labels. We have not read the inaccessible narrative report and have not invented attack dates or benign periods.

## Terminal workflow

Run from the existing `gnn_ids` folder:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

No DGL, PyG, MPS, GPU, notebook or external service is required for training. CPU is the supported device; `device` values other than `cpu` are rejected. `joblib` is now declared explicitly; other dependencies were already used by StreamSpot.

First try a **bounded genuine prefix** after Drive access recovers:

```bash
python main.py download --dataset theia --archive 1r \
  --sample-records 100000 --max-download-mib 128
python main.py prepare --dataset theia --config configs/theia_smoke.json \
  --input data/theia/raw/ta1-theia-e3-official-1r.json.sample.jsonl \
  --data-dir data/theia/processed/smoke
python main.py validate --dataset theia --data-dir data/theia/processed/smoke
python main.py run --dataset theia --config configs/theia_smoke.json \
  --data-dir data/theia/processed/smoke
```

The sample downloader streams a prefix of the first JSON archive member and stops; this is deliberately **not representative attack sampling**. It records the exact member, record count, downloaded-byte count, source URL and both prefix/file SHA-256 values. It does not claim to have verified the complete remote archive. If the sample lacks definitions, time variation or targets in any partition, preparation/run will report that limitation; obtain a broader scope rather than invent labels or graph nodes.

For complete local source archives, omit `--sample-records`:

```bash
python main.py download --dataset theia --archive 1r --max-download-mib 2048
python main.py download --dataset theia --archive 6r --max-download-mib 16384
```

These are **download ceilings, not verified archive sizes or RAM forecasts**. 1r's Drive confirmation reported 1.1G; inspect Drive and available disk space for 6r. Interrupted downloads retain `.part` files; inspect/remove your partial file before retrying. No automatic resume is implemented. Existing destinations are never overwritten. Full-download SHA-256 is a local reproducibility identifier, not a publisher-supplied authenticity checksum.

Manual downloads go in `data/theia/raw/` with the exact archive names above. You do not need to extract them. An importer run can accept several archives or extracted JSON members. Supply all relevant members when possible so entity definitions can resolve. Inputs are explicitly bounded; a prefix may omit later definitions and will report unresolved endpoints.

Measure a genuine pilot once you have suitable data and labels:

```bash
python main.py prepare --dataset theia --config configs/theia_pilot.json \
  --input data/theia/raw/ta1-theia-e3-official-1r.json.tar.gz \
          data/theia/raw/ta1-theia-e3-official-6r.json.tar.gz \
  --data-dir data/theia/processed/pilot
python main.py run --dataset theia --config configs/theia_pilot.json \
  --data-dir data/theia/processed/pilot
```

**The default record cap is global across inputs**, in the supplied file/member order; it can be reached inside 1r before any 6r events are imported. Caps are engineering scope controls, not balanced or representative sampling. For a study across both streams, select and document extracted members/chronological scopes, supply enough records to include them, and inspect the actual manifest before training. Do not claim these commands alone recreate the papers' train/test sets.

Each run prints a unique `results/theia/...` folder. Use that exact folder:

```bash
python main.py evaluate --dataset theia --run-dir results/theia/YOUR_RUN --rescore
python main.py demo --dataset theia --run-dir results/theia/YOUR_RUN --port 8766
```

Open `http://127.0.0.1:8766/dashboard.html`, or open that run's `dashboard.html` directly. `evaluate` recomputes metrics from saved predictions, verifies shared targets and validation thresholds; `--rescore` also reloads the local checkpoints and verifies every validation/test score within 1e-6. Only load joblib files produced locally; they are not a safe format for untrusted downloaded models.

StreamSpot commands still work without a dataset argument:

```bash
python main.py run --config configs/quick.json
python main.py demo --run-dir results/YOUR_STREAMSPOT_RUN
```

`--dataset streamspot` is also accepted. THEIA outputs reside one level deeper, so the old latest-StreamSpot dashboard selection will not accidentally select a THEIA run.

## Labels: chosen convention and alternatives

Default `label_policy: "positive_unknown"`:

1. Exact, canonical UUID membership in the supplied malicious list means positive **entity involvement**, not “actively malicious at this timestamp.” The list lacks onset times; this pipeline does not measure attack-onset detection delay.
2. Explicit curated labels may add verified negatives or positives, with a nonempty evidence field. Conflicting labels are rejected. No label is inherited from an enclosing graph, adjacent node or attack window.
3. Other nodes are **unknown (-1)**. Unknown targets receive scores, but are excluded from labelled metrics and threshold fitting. If validation has no negative labels, threshold and binary predictions remain unavailable (`null`/blank). The run still saves models and scores. AP/ROC-AUC require both labelled classes under this project's reporting convention.

Supply curated labels using `--curated-labels path/to/labels.csv` during preparation. Schema:

```text
uuid,label,evidence
```

Use actual CDM UUIDs, labels 0 or 1, and an auditable report/annotation reference. Do not fill an empty template with random or assumed negatives.

An optional **exploratory** convention is `label_policy: "scoped_background"`. Copy a configuration, change that field and enter a nonempty `background_rationale` describing the reviewed files/time scope and why the complement is a useful negative approximation. It marks *eligible retained process/file/network-flow targets* absent from positive/curated labels as **presumed background**, not verified benign. Memory objects, pipes, principals, unresolved nodes and other ineligible types are not silently made negatives. The assumption is saved in configuration, target label-source strings, metrics and dashboard. It may hide unannotated attacks and understate false positives; report sensitivity to this choice separately. It is never activated automatically.

This is the extension's common labelling protocol, not a claim that FLASH's original labels, MAGIC's labels and DARPA's narrative annotations agree. The same target CSV and labels are used for all models. Global malicious labels are **never used to clean training graphs**, choose vocabulary, fit a scaler, sample training nodes or select hyperparameters. Training uses all observed training-window behaviour, potentially contaminated. A verified benign training period would better match the papers, but must be established independently from the operational record, not by removing retrospectively test-labelled UUIDs.

## Chronology, units and leakage controls

- Events are disk-sorted by integer nanosecond timestamp, event UUID and predicate slot. Split intervals are `[start,end)`. Default smoke/pilot configurations use 60%/20%/20% of observed **time span**, not a random split or proportions of nodes. Boundaries use timestamps only, not labels. They are exploratory and depend on imported coverage.
- For a predeclared study, set `split_mode` to `explicit` with integer `start_ns`, `train_end_ns`, `validation_end_ns`, `end_ns`, then preserve that configuration across models/seeds. Choose actual cutoffs from the acquired release records; no unverifiable dates are supplied here. Check `splits.json` and per-partition counts before training. Test-only label/feature changes cannot alter explicit training partitions/features.
- One prediction per eligible entity UUID, at the end of its **first retained window**. That UUID cannot be a target in a later partition. Earlier entities can recur as context. This is target-disjoint evaluation, not a guarantee of completely disjoint context graphs. It excludes later alerts on previously scored long-lived entities and can miss attacks that start later in an entity's life. The static label convention does not resolve that limitation.
- Time windows are further cut at hard node/edge limits, without splitting an individual edge. Limits on windows may leave temporal gaps and produce a sampled first-retained target cohort, not full coverage. Each prediction records its actual cutoff. Timestamp ties may span bounded windows; no future timestamp is used, but event-by-event causality within a tied timestamp is not claimed.
- Read/receive/load edges point object → subject; other edges point subject → object. Both predicate objects are handled; duplicate event UUID/slot pairs are deduplicated and conflicting duplicates rejected. Null references are counted. Missing endpoint definitions are counted and excluded, not assigned invented types.
- All provided definitions are stored in SQLite. Only immutable entity **types** may resolve from a later definition. Later names, paths, command lines and mutable attributes are never used. Dynamic features depend only on the current window's events up to its cutoff. There are no cross-window message-passing edges.
- Features are log in/out counts per **training-only** event vocabulary, within-window activity duration and recency fractions. Unknown event types map to a reserved bin. UUIDs, attack labels, absolute dates, entity types, filenames and source-member IDs are not feature inputs. Entity type is a training target for FLASH/RF, so including it as an input would trivialise their task.
- Scaling is fitted incrementally on training windows only. Entity-type vocabulary is built from training event participants. Unseen evaluation entity types receive anomaly score 1 for FLASH/RF. Seeds govern weights, masks, RF and bounded uniform training reservoirs. Test data does not choose thresholds or epochs.

## Models and evaluation

| Path | Node score and deliberate simplifications |
| --- | --- |
| `flash_node` | Two directed GraphSAGE layers predict entity type. Score = 1 − probability assigned to the observed type. Structural/temporal counts replace Word2Vec and positional semantic encoders; one model replaces the multi-model refinement procedure. No neighbourhood forgiveness or graph pooling. |
| `magic_node` | Two graph-attention layers reconstruct randomly masked node features, then mean kNN distance in **node embedding** space to a bounded training reference reservoir. Omits structure reconstruction, original multihead settings, online adaptation and full benign reference population. |
| `random_forest_node` | Non-GNN node-feature baseline predicts entity type, with the same 1 − type-probability anomaly score as FLASH. This explicitly justified adaptation gives RF the same training information without requiring unavailable malicious training labels. It is **not** a supervised malicious/benign RF and not the old graph-summary RF. |

All are anomaly detectors with binary alert decisions only when a validation threshold can be fitted. No StreamSpot graph proxy is relabelled as a node baseline. MAGIC uses the same features/cohort, though a different self-supervised objective; scores are not calibrated probabilities of attack.

Validation with both classes selects maximum F1, highest threshold on ties. Validation with only negatives selects the configured upper quantile, moved strictly above its value to handle score ties conservatively; recall-based tuning is impossible there. Validation without negatives produces no threshold. Test precision, recall, F1, AP, ROC-AUC, TN/FP/FN/TP and FPR use exact node identities and exclude unknowns. Undefined ranking/recall metrics are null, not fabricated zeros. `binary_metrics_defined` only means the mathematical prerequisites are available; **it does not mean a valid benchmark was performed**. Repeated tuning against this test set invalidates a final held-out claim.

## Resource bounds and measurements

| Setting | Smoke | Pilot | Larger, unmeasured |
| --- | ---: | ---: | ---: |
| Raw records scanned | 100,000 | 500,000 | 5,000,000 |
| Imported entity definitions | 50,000 | 200,000 | 1,000,000 |
| Imported edges | 60,000 | 300,000 | 3,000,000 |
| Nodes / graph window | 512 | 2,048 | 4,096 |
| Edges / graph window | 1,024 | 4,096 | 8,192 |
| Windows / partition | 12 | 40 | 100 |
| RF training nodes | 4,096 | 20,000 | 100,000 |
| MAGIC reference nodes | 2,048 | 5,000 | 10,000 |
| Epochs / seeds | 2 / 1 | 5 / 1 | 15 / 3 |

Raw input is streamed into disk SQLite with a 32 MiB page-cache target and disk temporary sorting. Definitions/edges exceeding import limits are counted; raw-record and window truncation are explicit. Preparation never holds the raw dataset in a Python graph. Training loads one bounded graph tensor set at a time; RF and kNN retain only their declared reservoirs. kNN queries use batches of 256. Labels, target audit IDs, scalar prediction rows, model state and a reservoir still occupy RAM; this is memory-controlled, not constant-memory in every artifact count. RF cannot `partial_fit`, so its explicit reservoir is intentional. Only one model worker runs at a time.

`training_seconds` includes incremental scaling, graph loading/training, reference fitting and checkpoint serialization, but excludes Python/library startup and shared raw preparation. `validation_seconds` and `inference_seconds` include shard reading, scaling/representation extraction and scoring; inference is test-only. CSV export, metric computation and dashboard generation are outside inference time. `process_peak_rss_mib` is the OS high-water resident set measured after prediction CSV writing in a fresh worker: imports, training and inference included; other processes/GPU allocation excluded. macOS `ru_maxrss` bytes and Linux KiB are converted separately to MiB. Preparation has its own wall time and process high-water mark. This is not model-only tensor memory or system-wide RAM consumption.

See [THEIA_VERIFICATION.md](THEIA_VERIFICATION.md) for actual measurements. A synthetic workload can test bounded execution but cannot establish real THEIA degree/feature diversity, training cost or detection quality. **No larger or full-dataset setting is recommended yet.** On your 24 GB Mac, run and inspect the genuine pilot first, including Activity Monitor memory pressure, swap and disk growth. Increase one limit at a time only after measuring. `configs/theia_larger.json` is an explicitly unmeasured candidate, not a promise that the full dataset fits. Disk storage can substantially exceed compressed archive sizes.

## Saved evidence and reproducibility

Prepared directories are never overwritten. They contain the input SHA-256/provenance, import counts, dropped/unresolved counts, actual shard node/edge counts, labels aligned with UUIDs, target label-source strings, chronological boundaries, vocabulary, limits, and `events.sqlite`. Each NPZ uses non-object arrays and is loaded with `allow_pickle=False`. `manifest.json` fingerprints the data/configuration/shards/target table; `validate` checks content hashes and alignment.

Each run saves copied manifest/splits/target labels, run configuration, source-code fingerprints, seeds, environment versions, per-model logs, training histories/scaler statistics, reservoir UUIDs, checkpoints, frozen validation thresholds, all validation/test predictions, metrics CSV/JSON and evaluation audits. Dashboard rows are capped for display; CSVs are complete. Its observed incident edges are **context only, not an explanation or proof of malicious causation**.

To reproduce a synthetic engineering check without network access:

```bash
python tests/run_theia_smoke.py --size smoke --output data/theia/my_synthetic_smoke
python tests/run_theia_smoke.py --size pilot --output data/theia/my_synthetic_pilot
```

These directories must be new. The generated files, dataset metadata and run names explicitly say `SYNTHETIC`/`synthetic_cdm18`; their classification metrics are test fixtures, not research findings.
