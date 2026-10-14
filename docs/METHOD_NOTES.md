# Implementation notes for learning and experiment review

## Fixed question and operational interpretation

Retain the design plan's RQ1. This implementation operationalises malicious activity as **an attack-labelled whole graph**. It cannot answer an entity-level localisation question. The original plan deliberately left the detection unit open; document the final graph-level choice in your own work. If the facilitator requires complete FLASH semantic/temporal behaviour or DARPA node detection, this StreamSpot prototype must be extended or replaced; it is not sufficient for that requirement.

## Dataset and input schema

StreamSpot ALL contains six scenarios, 100 graphs per scenario. IDs 300–399 are the drive-by-download attack; the other five scenarios are benign. These labels come from the original repository's documented mapping. No labels are inferred from model scores.

Fields: `source-id`, `source-type`, `destination-id`, `destination-type`, `edge-type`, `graph-id`. The released format has removed timestamps and does not provide names/paths. Edge ordinals are evidence of record order only. Graph IDs and scenario names must not enter model features: their numeric ranges disclose the labels.

Use the measured `statistics.json` for your dataset table. Its node counts refer to sampled graphs, not original full graphs. Raw edge counts refer to the complete input. Never copy paper statistics into a table and describe them as measurements of this sampled experiment.

## Graph construction and sampling

Every original edge gets an independent uniform priority from a seeded, graph-specific generator. Keep the lowest K priorities per graph, then restore retained record order. This is uniform edge sampling without replacement. Both models and baselines use the same retained graphs. Local node indexes preserve direction; incoming and outgoing type counts are log-transformed. Edge types enter features and the typed-walk baseline; the GNN layers themselves do not learn relation-specific edge weights.

Sampling makes local experiments feasible but may disconnect nodes, remove attack paths, alter degree statistics and underrepresent rare actions. Report K and compare sample sizes only as a predeclared sensitivity experiment. Whole-graph attack labels remain attached after sampling even if the sampled excerpt omits the attack action; this can produce label-feature mismatch.

## Shared partitions and leakage controls

Fixed split seed 2026; within each scenario, 60 graphs train, 20 validate and 20 test. Totals: 360/120/120. There are 60/20/20 attack graphs and 300/100/100 benign graphs in those partitions. Disjoint whole graphs avoid transductive adjacency between train and test. Node/edge type vocabularies use only training graphs; unknown categories map to zero. MAGIC embedding scaling and benign references fit on training benign graphs only.

This is not a chronological split. It does not establish independence of underlying sessions or test attacks from unseen families. Similar scenario repetitions across partitions can inflate results. A later scenario-held-out design should hold out benign scenarios as well, but StreamSpot has only one attack scenario, limiting unseen-attack evaluation. Do not describe graph disjointness as proof that all benchmark leakage or distribution bias has been eliminated.

## Models and deviations

| Model in this code | Implemented | Important omitted/replaced elements |
| --- | --- | --- |
| FLASH-inspired | Two directed GraphSAGE layers, mean/max pooling, supervised graph binary head | No semantic text embeddings, no true temporal encoder, no original entity-level scoring or attack-evolution reconstruction |
| MAGIC-inspired | Two single-head attention layers, masked attribute reconstruction on benign training graphs, graph mean/max embedding, benign kNN mean distance | No original edge/structure reconstruction, adaptation, multihead settings or official preprocessing; masked MSE replaces original objectives |
| StreamSpot-inspired | Hash counts of typed directed walks of lengths 1/2, L2 normalisation, benign kNN distance | Offline proxy only: no original traversal shingles, streaming sketch maintenance or online cluster updates |
| Random Forest | Mean/std/max node features and sampled graph sizes; 100 trees with balanced class weights | Simple feature baseline, not Unicorn or ProvDetector |

Do not rename these as exact FLASH, MAGIC or StreamSpot. Unicorn and ProvDetector are not implemented. They can be discussed as related work, but no experimental scores should be attributed to them. The professor's screenshot lists alternatives; whether an exact named baseline is mandatory needs facilitator clarification.

Supervised models receive benign and attack training labels. Anomaly models receive benign training graphs. All four use labelled validation data to set thresholds, so the anomaly models are not fully unsupervised end to end. This comparison examines different learning strategies under a shared test protocol, not identical supervision budgets.

## Metrics and resource measurement

Higher scores mean more suspicious. FLASH/RF output classifier scores; MAGIC/StreamSpot-inspired output distances. They have different scales and are not comparably calibrated probabilities. Thresholds maximise validation F1, choosing the highest threshold when F1 ties. There are no test-driven changes in the scripted run.

Report precision, recall, F1, AP (the explicitly named PR-AUC convention), ROC-AUC, FPR and TN/FP/FN/TP. AP's random-ranking reference is approximately attack prevalence; the exact score still depends on ranking. For this test split prevalence is 20/120. Explain errors by graph context and sampling, without claiming individual causal explanations from a structural preview.

Each model/seed uses a fresh CPU worker to separate peak RSS measurements. All workers load the same 600 sampled graphs, even when fitting only benign training graphs. Timing and memory cover defined process stages, not production throughput. The StreamSpot-inspired baseline is deterministic, so its predictions repeat across training seeds; seed repetition does not create new baseline evidence.

## Original sources consulted

- Dataset schema and labels: https://github.com/sbustreamspot/sbustreamspot-data
- MAGIC official code and dataset guidance: https://github.com/FDUDSDE/MAGIC
- MAGIC paper: https://www.usenix.org/conference/usenixsecurity24/presentation/jia-zian
- FLASH official implementation: https://github.com/DART-Laboratory/Flash-IDS
- FLASH paper: https://doi.org/10.1109/SP54263.2024.00139
- StreamSpot core: https://github.com/sbustreamspot/sbustreamspot-core

Technical implementation notes are not a draft assessment report. Use them to inspect and understand the code and to decide which assumptions your team can defend.
