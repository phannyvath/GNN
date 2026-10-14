# Build verification — 6 October 2026

Executed in a hosted Linux CPU environment. These are build records, not the team's assessment report or individual contributions.

- Downloaded the original StreamSpot ALL archive, 87,860,166 bytes.
- SHA-256: `1551d7eba96fdea4958b9c0b6c15c773983a5ebdf9f1ffe4430cbb7ea85d23cf`.
- Parsed all 89,770,902 original edges and all 600 graphs.
- Prepared uniform 512-edge samples with reservoir seed 77 and scenario-stratified split seed 2026.
- Saved 360 training, 120 validation and 120 test graphs; 60/20/20 are attack-labelled.
- Executed all four models at training seeds 42, 43 and 44, eight epochs for the GNNs, hidden width 32.
- Checked all 12 runs for complete validation/test prediction files and 120 total test confusion counts.
- Seven protocol tests passed: split separation/scenario counts, incomplete dataset rejection, edge direction, threshold/confusion arithmetic, ID/label exclusion from baseline inputs, GNN permutation/gradient checks, attention on an empty-edge graph.
- Reloaded both GNN checkpoints and performed actual forward passes on held-out graph 308. Both produced finite scores and alerts for seed 42. Scores match the saved prediction values within floating-point precision.
- Local HTTP server returned 200 for the dashboard, both figures and its data JSON.
- Executed dashboard JavaScript in a DOM harness: no script errors, four model metric rows, all 120 test graphs, five MAGIC-inspired missed attacks for seed 43, selection and review state working.
- Inspected generated PR/ROC figures.
- A browser engine could not be downloaded in this environment; visual browser rendering and responsive layout are not verified.
- The larger `full.json` configuration has not been executed here. It is a planned extension, not a completed result.

## Executed pilot outputs

Folder: `results/pilot_20261006T031228745094Z`.

| Implementation | Test F1 mean | Sample standard deviation across training seeds |
| --- | ---: | ---: |
| FLASH-inspired | 0.9279 | 0.0338 |
| MAGIC-inspired | 0.9073 | 0.0803 |
| StreamSpot-inspired offline baseline | 0.8085 | 0.0000 |
| Random Forest | 0.9544 | 0.0582 |

These are results on one fixed, sampled StreamSpot split. They do not establish production detection performance, exact paper reproduction, model superiority with statistical significance, or unseen-attack generalisation. The deterministic StreamSpot-inspired baseline repeats identical predictions across seeds. The Random Forest result is a reason to examine simple feature separability and benchmark bias. Ranking can have AP=1 while the validation-selected operating threshold still misses some test attacks.
