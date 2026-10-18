# Dataset quality audit — 19 October 2026

## Assessment

**The THEIA files are authentic copies of the published MAGIC archive and are structurally valid. The current datasets/features nevertheless have substantial evaluation limitations.** They are useful for an exploratory research project; the existing scores alone do not establish general APT detection performance.

This is an audit of the existing files and saved predictions. No source data, labels, model weights, splits or thresholds were changed, and no additional models were trained. Diagnostics using test labels must not become a new tuning loop against the same test set.

## 1. Authenticity and integrity: passed

The local `data/theia/graphs.zip` exactly matches [MAGIC's THEIA archive at commit aa0b647](https://raw.githubusercontent.com/FDUDSDE/MAGIC/aa0b647eea74b6faa0e52eb444370c4411a32cbe/data/theia/graphs.zip):

- Remote and local size: **7,877,492 bytes**.
- SHA-256: `926b4e30a39b1680fe512ef015bfe03ab85760e12dc7d8ae88da40a346bd6864`.
- ZIP CRC check passed; all six extracted members match their archive copies.
- All five graph exports and metadata match the saved training manifest.
- Prepared-data fingerprint and shard checks passed.

This resolves the previously unverified origin of this particular download: it is the repository's published preprocessed artifact, not the project's earlier synthetic fixture. Matching a published artifact does not prove every annotation or original recorded event is correct.

StreamSpot's local raw archive matches its saved source checksum; all prepared-file checksums passed. This audit did not redownload the StreamSpot upstream archive.

## 2. THEIA graph structure: passed the implemented checks

All 1,623,966 graph-local nodes and 2,874,821 edges were inspected. Endpoint indices, array alignment, finite features, and one-hot node/edge type attributes passed. No isolated nodes, self-loops or repeated `(source, destination, type)` triples were found in these exports. Without raw event timestamps, this is not an audit of duplicate raw events or preprocessing losses.

| Export | Nodes | Edges | Weak components | Unique prepared feature vectors |
| --- | ---: | ---: | ---: | ---: |
| train0 | 167,265 | 284,562 | 7 | 883 |
| train1 | 380,204 | 674,723 | 7 | 962 |
| train2 | 342,624 | 587,111 | 15 | 1,058 |
| train3 (validation) | 389,106 | 700,318 | 16 | 1,265 |
| test0 | 344,767 | 628,107 | 16 | 1,267 |

Two constant feature columns, 0 and 18, are the reserved incoming/outgoing unknown-edge-type bins. Their being zero here is expected, not corruption.

## 3. Feature diversity: limited by our representation

The 890,093 training nodes have **2,117 distinct feature vectors** across their three graphs. The 344,767 test nodes have **1,267 distinct vectors**. These are our simplified log edge-type count features, not the complete information content of the original data or graph topology.

**343,508 test nodes (99.635%)** have a feature vector that also occurs in training. This is evidence of repetitive features; it is **not proof that the same entities leaked between train and test**. UUIDs are needed to verify entity overlap.

Ten test feature groups contain both positive and presumed-background labels, covering 33,973 nodes. Any deterministic classifier using only these features, with the same decision rule for every identical row, must misclassify at least 946 test nodes. Graph context or the observed entity type used by the current RF scoring rule can distinguish some otherwise identical feature rows; this bound does not apply to those full scoring rules.

## 4. Label and topology concentration: major interpretation limit

The test set has **25,319 annotated positives (7.344%)** and **319,448 presumed-background nodes**. The latter are not independently verified benign.

- **25,291 of 25,319 positives (99.889%)** belong to raw node-type code 3. These integer codes are reported without inventing semantic type names.
- Test node **207781**, the highest-degree node, has 25,303 unique neighbours; **25,290 are positive**. Thus **99.885% of all positive test nodes are directly adjacent to this one hub**.
- This means many positive observations share one local structure. Millions of nodes do not imply millions of independent examples or diverse attacks. It does not by itself demonstrate incorrect labels, leakage or an attack path.
- 25,278 entries in the associated positive-name list are `NA`. Positive numeric indices are valid and unique, but the metadata does not provide a full auditable UUID/name mapping.

The current high aggregate scores obscure the tiny positive populations outside type 3:

| Saved model | Positives detected in type 3 / 25,291 | Positives detected outside type 3 / 28 |
| --- | ---: | ---: |
| FLASH-inspired | 18,044 | 0 |
| MAGIC-inspired | 25,283 | 22 |
| Random Forest | 24,490 | 2 |

These subgroup counts use the original saved thresholds. There are only 28 positives outside type 3, so uncertainty is large. Within-type-3 ROC-AUC remains high (FLASH 0.9743, MAGIC 0.9839, RF 0.9624); type imbalance alone does not explain all discrimination.

## 5. Simple shortcut diagnostics

The following scores were calculated directly, without training or choosing a score direction to maximise results. They are retrospective diagnostics, not independently validated new detectors.

| THEIA score | Test ROC-AUC | Test AP |
| --- | ---: | ---: |
| Higher total degree | 0.5296 | 0.0771 |
| Lower total degree | 0.4704 | 0.0695 |
| Higher incoming degree | 0.5743 | 0.0915 |
| Higher outgoing degree | 0.4369 | 0.0716 |
| Feature vector absent from training | 0.4987 | 0.0734 |

The positive prevalence is 0.0734. These simple scores do not explain the high model scores. They do not rule out other structural shortcuts, shared entity identities or the effect of the positive hub.

**StreamSpot has a particularly strong shortcut signal:** sampled node count alone achieves test ROC-AUC **1.000** and AP **1.000** on the existing 120-graph test split. This is perfect ordering, not a validation-selected classification threshold. It suggests that apparent success can reflect graph size/activity differences rather than sophisticated attack detection. Raw edge count has AUC 0.000 in its increasing direction, showing an equally strong inverse ordering on this split; it is not necessarily a model input.

## 6. StreamSpot preprocessing: heavy information reduction

- 600 graphs: 500 benign-labelled and 100 attack-labelled.
- Train/validation/test graph IDs are disjoint (360/120/120).
- All features/endpoints passed; no isolated nodes or exact tensor-order duplicate graphs were found.
- **307,200 of 89,770,902 original edges are retained: 0.3422% overall.**
- Per-graph retained fractions range from 0.0715% to 1.8813%, with median 0.5052%.

This is a property of our 512-edge sampling configuration, not evidence that the original StreamSpot source data is poor. Exact tensor equality also does not detect graph isomorphism or repeated underlying entities.

## What remains unresolved

1. Original UUIDs and raw events are needed to audit entity overlap, event chronology and preprocessing omissions.
2. Original annotation evidence and temporal scope are needed to verify assumed-background labels and interpret the large positive hub.
3. Repeated seeds and defensible alternative splits are needed to assess performance stability. They were not run in this audit.
4. Broader feature/context designs and size-controlled baselines should be predeclared and evaluated on a fresh holdout. Repeatedly adjusting against this already-inspected test set would weaken the final evaluation.

**Recommended report wording:** “We verified artifact provenance and structural integrity. Results are exploratory because of presumed-background labels, limited feature diversity, concentrated positive topology, and a graph-size shortcut in the sampled StreamSpot split.”

## Reproduce and inspect

Run `.venv/bin/python scripts/audit_data_quality.py --verify-upstream` to repeat the audit with the public archive checksum check. Omit `--verify-upstream` for a local audit; a successful prior upstream check is retained only when the local archive checksum still matches. The script saves `results/data_quality/audit.json` and leaves all source and training artifacts unchanged.

[Machine-readable findings](audit.json) include full graph statistics, per-type prediction metrics, checksums and diagnostic rankings. This narrative describes the supplied saved runs; inspect new JSON findings if the inputs change.
