# Provenance IDS experiment workspace

A runnable, CPU-based Python project with separate **graph-level StreamSpot** and **node-level DARPA TC E3-THEIA** experiments. It provides simplified FLASH-inspired and MAGIC-inspired models and Random Forest baselines, plus a StreamSpot-inspired graph baseline. This project is technical assistance, not an assessment report.

## Unified study dashboard

See [dashboard evidence, comparison rules and HPC import instructions](docs/DASHBOARD.md).

Open `results/comparison/dashboard.html` for the offline experiment explorer: THEIA and StreamSpot selectors, per-seed metrics, mean/sample-SD comparisons, compute costs, exported evaluation figures, and source artifacts. Historical, synthetic, smoke and incomplete runs are labelled separately. This view reads saved results; it does not train models or make scores from different protocols directly comparable.

```bash
.venv/bin/python scripts/build_study_dashboard.py
open results/comparison/dashboard.html
```

Rebuild after saving a new experiment. Figures and evidence links refer to their original directories, so keep the results tree together. Use `build_study_dashboard.py` for this unified view; the older `build_comparison_dashboard.py` produces the historical two-run comparison at the same output path.

## DARPA TC E3-THEIA (node-level extension)

The completed reproducible study uses the actual MAGIC-preprocessed DARPA E3-THEIA files in `data/theia`. It trained fresh FLASH-inspired, MAGIC-inspired and Random Forest models for seeds 42, 43 and 44, using all 890,093 eligible training node occurrences. The 1,623,966 total node occurrences include separate validation and test graphs; they are not all training examples.

Read the [study protocol and commands](docs/THEIA_STUDY.md), [measured execution and verification](docs/THEIA_STUDY_VERIFICATION.md), and [completed dashboard](results/theia/study_full_v1/dashboard.html). Labels and entity identity have documented limitations; these simplified models are not exact paper reproductions.

From the project directory, launch a fresh full study and open the already completed dashboard with:

```bash
bash scripts/run_theia_study.sh --output results/theia/my_full_study
open results/theia/study_full_v1/dashboard.html
```

Use a new output directory for fresh training. Explicit interrupted-run recovery is `bash scripts/run_theia_study.sh --resume results/theia/my_full_study`.

The original raw-CDM importer (`--dataset theia`) remains available in [THEIA.md](docs/THEIA.md). Its earlier synthetic engineering checks are separate from this executed real-data study. The StreamSpot workflow below remains a separate graph-level experiment.

## Start on your Mac

Use Python 3.11 or 3.12, Terminal and VS Code. This is a Python project, not a BlueJ project.

1. Extract this ZIP and open the `gnn_ids` folder.
2. In Terminal, enter that folder (`cd ` followed by dragging the folder into Terminal).
3. Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. The ZIP includes the real raw StreamSpot archive, processed 512-edge graphs, and an executed pilot. Open the included `results/pilot_.../dashboard.html` directly in your browser to explore it without installing anything. Figures beside it must stay in the same folder.
5. To serve the latest results locally:

```bash
python main.py demo
```

Open the localhost address printed in Terminal. Stop the server with Control+C.

6. To perform your own quick run:

```bash
python -m unittest discover -s tests -v
python main.py run --config configs/quick.json
```

7. For three training seeds on the small sampled graphs:

```bash
python main.py run --config configs/pilot.json
```

8. For the larger planned experiment:

```bash
python main.py run --config configs/full.json
```

The full configuration samples up to 2,048 edges per graph and trains for 30 epochs with three seeds. It is provided for further experimentation; check `BUILD_VERIFICATION.md` for what was actually executed here. CPU is intentional: sparse graph operations can behave differently on MPS, and all models need a consistent platform. Your M4 Pro can run the CPU path. No Mac execution claim is made for the bundled Linux pilot.

## If you need to download again

```bash
python main.py download
python main.py prepare --config configs/quick.json
```

The downloader retrieves `all.tar.gz` from the original StreamSpot repository. It does not run executable files from the dataset. The importer streams `all.tsv` from the archive without extracting arbitrary archive paths. It verifies all 600 graph IDs, six-column records, consistent node types and graph-level labels. Network access is needed only for installation and downloading.

## What each component does

| File | Purpose |
| --- | --- |
| `main.py` | Download, prepare, experiment and local demo commands |
| `ids/data.py` | Validate raw records, uniformly sample edges, construct directed graphs, save fixed splits |
| `ids/models.py` | Directed GraphSAGE classifier and masked graph attention encoder |
| `ids/baselines.py` | Typed-walk hashing baseline and graph statistics |
| `ids/metrics.py` | Validation-only thresholds and held-out metrics |
| `ids/worker.py` | Train/score one model in an isolated process |
| `ids/experiment.py` | Run all models/seeds and assemble evidence |
| `ids/demo.py` | Generate plots and an offline interactive results browser |
| `ids/predict.py` | Reload an executed GNN checkpoint and score one prepared graph |
| `configs/` | Quick, three-seed pilot and larger experimental settings |
| `tests/` | Split, feature, direction, gradient and metric checks |
| `docs/` | Methodology notes, assessment planning prompts, blank progress records |

## Output evidence

Every new experiment creates a new result folder rather than overwriting old results. It includes the dataset SHA-256, graph statistics, train/validation/test IDs, configuration, per-model training logs, checkpoint files, per-graph predictions, metric JSON/CSV, seed summaries, plots and an offline dashboard.

`summary.json` reports the mean and sample standard deviation across training seeds. A single-seed run has `std: null`, not a fabricated zero uncertainty. These seeds measure model variability on one fixed split, not variability across datasets or split choices.

`average_precision` is the stated PR-AUC convention. Each threshold maximises labelled validation F1 with deterministic tie handling. Test scores are computed after threshold selection. No early stopping or test-driven hyperparameter selection is implemented. Repeatedly changing settings after looking at test results makes those results exploratory; reserve an untouched final evaluation for a stronger study.

Process memory is the lifetime peak RSS of a fresh worker, including imports and all loaded graphs, not model-only RAM. Training time includes reference embeddings/fitting for anomaly models. Inference time includes representation/feature extraction for test graphs and scoring. Shared raw preprocessing is excluded. Do not compare the bundled timings with paper timings on different machines.

## Interpretation and model fidelity

Read `docs/METHOD_NOTES.md` before presenting. All model names deliberately say **inspired**. StreamSpot has whole-graph labels; a node inside an attack graph is not automatically malicious. Type codes are retained as codes rather than guessed process/file meanings. Scores are not calibrated probabilities across methods.

The dashboard is a playback of genuine saved test outputs. Its graph excerpt is context, not a faithful causal explanation or attack localisation. Review buttons record your assessment locally and can export JSON; they perform no blocking.

## Live checkpoint demonstration

For an actual model forward pass rather than saved-output playback, use:

```bash
python main.py predict --run-dir results/YOUR_RUN_FOLDER --model flash_inspired --seed 42 --graph-id 308
python main.py predict --run-dir results/YOUR_RUN_FOLDER --model magic_inspired --seed 42 --graph-id 308
```

Use any graph ID listed in that run's `splits.json` test partition for a held-out demonstration. The command states the partition. Checkpoints are loaded with PyTorch `weights_only=True`; numerical reference arrays use `allow_pickle=False`. Random Forest's `.joblib` is a locally produced checkpoint, not an input file to accept from untrusted sources.

## Academic use

The supplied brief prohibits AI-generated or AI-edited report prose. Write the report yourselves using your actual work and measured outputs. The included assessment notes are organisational prompts, not submission-ready report text. Confirm the unit's separate rules for AI-assisted code and disclose this assistance as required. Do not claim generated code, pilot execution or future work as a teammate's completed individual contribution.

Dataset: https://github.com/sbustreamspot/sbustreamspot-data (Apache-2.0)
MAGIC reference: https://github.com/FDUDSDE/MAGIC
FLASH reference: https://github.com/DART-Laboratory/Flash-IDS
StreamSpot reference: https://github.com/sbustreamspot/sbustreamspot-core
