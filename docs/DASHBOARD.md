# Study dashboard: evidence and HPC snapshots

## Open and refresh

From the project directory:

```bash
.venv/bin/python scripts/build_study_dashboard.py
open results/comparison/dashboard.html
```

The dashboard is offline HTML. There is no backend API or automatic live refresh. Rebuilding reads each saved experiment separately and writes `results/comparison/study_index.json` (schema version 1) and HTML. Run selection changes all tables, recommendations and figures; exported figures cover the selected run, while the seed selector applies to metrics and epoch records. The original training code and checkpoints are unchanged.

## Why the count appeared too small

The preceding dashboard displayed the first model's training target count, not source-wide totals. Its default full study used 890,093 training node occurrences. The historical two-run dashboard also defaulted to a different, earlier THEIA experiment. Selecting a smoke run displays an intentionally reduced subset. No additional training is needed to explain this discrepancy.

For `study_full_v1`, `data/manifest.json` records five original MAGIC-exported graph files with 1,623,966 node occurrences and 2,874,821 edges. All are retained by this run's preprocessing. Training uses train0–2 (890,093 nodes), validation train3 (389,106), and testing test0 (344,767). The source files are preprocessed graph exports; their node counts do not measure raw CDM events. Unique entities across graphs cannot be established from graph-local IDs. Those values show **Unavailable**.

Pipeline: local MAGIC exports → `ids/theia/processed.py` graph/features preparation → `study_data.py` alignment verification and saved shards/splits → `study_worker.py` training and predictions → `study_report.py` metric/audit/figure exports → dashboard evidence normalization → embedded static JSON and HTML. Source hashes, source/used counts and split identifiers appear in the dashboard. Train-only feature fitting and validation-only thresholds are documented by the study protocol and saved audit; the dashboard does not itself retrain or independently re-audit predictions.

StreamSpot is a separate graph-level experiment. The saved quick-run manifest records 600 graphs and 89,770,902 raw edge rows; reservoir preprocessing retained 307,200 edges. It is not directly comparable to node-level THEIA. Training restrictions may vary by model; consult the saved configuration and per-model artifacts.

## Research focus

FLASH-inspired and MAGIC-inspired are the primary methods in the headline chart, performance table and compute table. A separate baseline table compares both against Random Forest (and StreamSpot-inspired where implemented). Recommendations and original evaluation figures still include available baselines so measured advantages are not hidden.

## Comparison rules

F2 is calculated from confusion counts as `5 TP / (5 TP + 4 FN + FP)`. Zero denominator is unavailable. Throughput is held-out targets divided by recorded inference seconds, not end-to-end service throughput. Mean and sample standard deviation are calculated over defined seed measurements. A single seed has no SD. AP is average precision, not trapezoidal PR-AUC.

Recommendations require a study completion marker, matching dataset fingerprints, a full model/seed grid, and the saved audit affirming identical evaluation targets/labels, recomputed metrics and validation-only thresholds. Older runs without this combined evidence remain visible but receive no winner recommendation. This conservative gate does not prove that historical runs are incorrect. Imported audit claims are identified as supplied evidence, not freshly verified results.

Rules are explicit: highest F1 for detection balance, highest recall for missed attacks, lowest FPR for false alarms, lowest scoring seconds for speed, and lowest worker RSS for memory. Practical priority selects recall, FPR or scoring time directly, without an arbitrary composite score. Missing measurements prevent a winner for that criterion. These retrospective comparisons do not establish statistical significance, production suitability or a validated deployment policy.

The full THEIA study supports Random Forest on mean F1 and false-alarm rate; MAGIC has marginally higher mean recall with much higher false-alarm rate and scoring cost. FLASH has the lowest mean test-scoring time. Hardware/timing definitions and all seed values are visible. Smoke results are engineering checks only.

## Import HPC results

Copy the **whole result directory** from the cluster to your Mac using your university's documented file-transfer method. Keep manifests, configuration, audit files, metrics, predictions, worker logs, figures and checkpoints together. No cluster credentials or scheduler details are assumed here.

```bash
.venv/bin/python scripts/import_hpc_results.py /path/to/copied/study_run --name hpc_theia_001
.venv/bin/python scripts/build_study_dashboard.py
open results/comparison/dashboard.html
```

Imports are copied into `results/imported/hpc_theia_001`. Existing names are rejected. Files are not executed and checkpoints are not deserialized. Imports record source location, timestamp and hashes of structured evidence/log/prediction files. Structural consistency is checked for a claimed completed study; importing does not rerun the prediction audit. Use a new name for every refreshed partial snapshot. Imports remain distinct from local completed runs and never become the default study silently.

A scheduler wrapper may provide `run_status.json`:

```json
{"state":"running","updated_utc":"2026-10-19T12:00:00Z","scheduler":"slurm","job_id":"your-job-id"}
```

Supported declared states: queued, running, completed, failed, interrupted. This dashboard does not poll Slurm/PBS or infer process liveness. State is a timestamped snapshot when supplied. Completion claims alone do not enable recommendations; completion/audit evidence is still required. A study without state/completion evidence is shown as incomplete or completion unverified, not automatically failed. Full imported-study support targets this project's `ids.theia.study` output format; arbitrary third-party result formats require an adapter.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p test_study_dashboard.py -v
node tests/check_study_dashboard.cjs
```

Checks cover exact source/processed/split counts, unknown entities, F2, throughput, sampled data, missing/queued/failed/interrupted snapshots, import isolation/no overwrite, every experiment and seed selector, recommendation gating and priorities. The Node harness tests DOM interaction logic, not browser rendering. Desktop/mobile CSS retains the responsive layout and scrollable wide tables. No new full training was started for this presentation change. The previously executed real-data smoke/full studies remain available as evidence.

Final verification: all 39 Python tests passed, and the Node interaction harness passed. A Brave headless browser render exited with code 134 without producing a screenshot; desktop/mobile visual rendering remains unverified in this environment. No HPC connection was attempted.
