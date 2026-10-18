# Preprocessed THEIA verification — 19 October 2026

The supplied five DGL exports were decoded and checked without installing DGL.
All endpoints, array dimensions, integer type indices and one-hot node/edge
attributes passed validation. Totals are 1,623,966 graph-local nodes and
2,874,821 edges. The ZIP integrity check passed and every extracted file matched
its archive member. This does not certify upstream provenance or label accuracy.

The launcher is `bash scripts/train_theia.sh`. Full training was left for the user.
The full configuration processes every graph without a node/edge cap, using
train0–2 for fitting, train3 for validation and test0 for testing. See THEIA.md
for the explicit presumed-background convention and graph-local ID limitations.

Validation performed:

- All 25 unit/protocol tests passed, including restricted pickle global rejection,
  directional feature counts, isolated nodes, partition/label alignment, JSON
  fingerprint stability and tampering detection.
- A real-data prefix smoke run completed all three models and metric audits:
  `results/theia/processed_smoke_20261018T132428555340Z`.
- Saved checkpoints reproduced 12,288 validation/test scores with maximum absolute
  error 2.220446049250313e-16 (tolerance 1e-6).
- A separate single optimizer step on the largest complete graph (train3:
  389,106 nodes / 700,318 edges), hidden width 32 and four CPU threads, took
  approximately 0.301 seconds for FLASH and 0.822 seconds for MAGIC. The process
  high-water RSS reached 1,435.84 MiB after FLASH and 2,285.38 MiB after MAGIC.
  These were sequential in one process, so the latter is cumulative high-water
  memory, not isolated MAGIC memory. They are capacity checks, not full-run
  training/inference forecasts. No checkpoints from this one-step probe were saved.
- Bash launcher syntax passed `bash -n`.

Hardware: Apple M4 Pro, 12 CPU cores, 24 GB memory. Smoke metrics are engineering
checks only, not full-dataset detection results. Full-run memory, kNN evaluation
cost and final performance remain to be measured by the user's run.
