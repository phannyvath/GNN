# Study verification record — 19 October 2026

This record separates engineering checks from actual full-study execution. It does not claim replication of the original FLASH or MAGIC papers.

## Implemented checks

All **33 tests passed** in the verification run saved during this task. They include the original StreamSpot/raw-THEIA protocol checks and eight new study checks:

- Exact multiplicity-preserving kNN compression equals an expanded reference.
- Float32 score serialization and float64 validation-threshold recomputation agree.
- Known confusion counts, F1, average precision, ROC-AUC, undefined metrics and finite-score rejection.
- Sample standard deviation uses `ddof=1`.
- Three distinct seeds and fixed non-overlapping graph-file splits are enforced.
- Wrong malicious-node label alignment is rejected even if prepared file hashes are recomputed.
- Changes to test labels/edges do not change training features/types.
- Simulated interruption after epoch one resumes MAGIC to bit-identical model parameters and matching predictions versus uninterrupted two-epoch training.

These tests use small controlled fixtures where appropriate. They are distinct from the real-data runs below.

## Real-data smoke execution

`results/theia/study_smoke_v2` completed all **nine model/seed combinations** (three models × seeds 42, 43, 44), with two neural epochs and ten RF trees. Each graph was restricted to its first 4,096 nodes and induced edges. Vocabulary discovery read only the original training files; fitting/scaling used the retained training subset.

| Partition | Node occurrences | Edges | Positive | Presumed background | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 12,288 | 17,651 | 0 | 0 | 12,288 |
| Validation | 4,096 | 5,719 | 0 | 4,096 | 0 |
| Test | 4,096 | 5,260 | 56 | 4,040 | 0 |

The smoke report recomputed all metrics from saved predictions and verified identical evaluation targets/labels across methods/seeds. Reloaded checkpoints reproduced **73,728 scores**, with maximum absolute difference **2.220446049250313e-16**, within 1e-6. It generated all five figure families and an offline dashboard. The learning-curve image was visually inspected. These are engineering checks, not estimates of full-study performance.

The preceding `study_smoke_v1` attempt trained its workers but **failed final threshold verification** because float32 threshold selection and float64 CSV replay used different `nextafter` precision. The new inference path converts scores to float64 before selecting thresholds; a regression test covers this. Version 1 was not presented as a successful study. Version 2 started fresh rather than silently reusing those workers.

## Full-study execution

The full run was launched as:

```bash
bash scripts/run_theia_study.sh --output results/theia/study_full_v1
```

Its exact invocation, original configuration, code/source snapshot and software versions are stored in that directory. The run **completed all nine model/seed combinations**. Source snapshots matched the executed code hashes. All full-run node and edge coverage checks passed; no worker silently fell back to a smoke subset or a reservoir. Every neural run completed 30 epochs, selecting epoch 30 by the validation objective. The patience stopping condition was not reached; this is not a convergence claim.

The observed invocation-to-completion-marker wall interval was **445.335 seconds (about 7 min 25 s)**, including preparation, workers, exports, audits and plots. It is derived from the recorded UTC start and completion-marker file timestamp, not a monotonic whole-run timer. Later checkpoint rescoring is excluded. The maximum fresh-worker high-water RSS was **2,570.094 MiB** (about 2.51 GiB), not total machine memory.

Each seed used all **890,093 training node occurrences / 1,546,396 training edges**, with 389,106 validation and 344,767 test nodes. RF fit all 890,093 rows. MAGIC's reference represented all 890,093 occurrences, compressed to 28,718 / 29,252 / 29,263 distinct embeddings for seeds 42 / 43 / 44 respectively, with multiplicities retained. No benign-only filtering was applied.

| Method | Test F1 mean ± sample SD | Test ROC-AUC mean ± sample SD | Training seconds mean ± sample SD | Test scoring seconds mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| FLASH-inspired | 0.8629 ± 0.0736 | 0.9644 ± 0.0027 | 26.181 ± 0.136 | 0.144 ± 0.008 |
| MAGIC-inspired | 0.6650 ± 0.0709 | 0.9767 ± 0.0155 | 55.882 ± 0.514 | 18.018 ± 0.313 |
| Random Forest | 0.9465 ± 0.0003 | 0.9677 ± 0.0010 | 7.619 ± 0.129 | 0.163 ± 0.015 |

These figures are measured outputs, not targets. The new study did not improve every score. MAGIC's mean test false-positive rate was **0.07722 ± 0.03163**, despite a configured validation FPR budget of 0.01. That budget is a validation quantile convention, not a guaranteed test rate. No test-driven threshold correction was made. The RF logs contain sklearn's generic class-weight/warm-start warning; every tree-growth call used the same complete input rows and unchanged class distribution.

All metrics were recalculated from the saved per-node CSVs. Target identities and labels matched across all nine jobs, and thresholds matched the validation-only selection. The report generated five PNG/SVG figure families, including nine actual graph excerpts (TP/FP/FN for each model at seed 42). Comparison and graph-context plots were visually inspected. Browser screenshot testing was not available; the HTML's static content, artifact links and embedded figures were checked.

A separate post-run checkpoint audit reloaded all nine saved models and reproduced **6,604,857 validation/test scores**, with maximum absolute error **2.220446049250313e-16** against a tolerance of 1e-6. This check performed inference only; it did not retrain models or tune thresholds. See [checkpoint audit](../results/theia/study_full_v1/checkpoint_audit.json).

The environment recorded PyTorch 2.14.1, NumPy 2.5.3, scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0 and matplotlib 3.11.2. Each worker also records Python, operating system, CPU device, thread count and seed. These are observed installed versions, not asserted portable version guarantees.

Artifacts:

- [Full dashboard](../results/theia/study_full_v1/dashboard.html)
- [Per-seed metrics](../results/theia/study_full_v1/metrics.csv)
- [Means and sample SDs](../results/theia/study_full_v1/summary.csv)
- [Evaluation audit](../results/theia/study_full_v1/evaluation_audit.json)
- [Data alignment audit](../results/theia/study_full_v1/data/alignment_audit.json)
- [Execution/coverage evidence](../results/theia/study_full_v1/execution_verification.json)
- [Smoke dashboard](../results/theia/study_smoke_v2/dashboard.html)

The limitations in THEIA_STUDY.md remain: unknown entity overlap, assumed-background labels, concentrated positive topology, simplified model objectives, and a previously inspected test set. Three seeds measure run variability; they do not establish generalisation to new attacks.
