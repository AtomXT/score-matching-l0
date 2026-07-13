# L0-penalized score matching

This repository contains the manuscript and experiment code for studying
L0-penalized score matching in undirected graphical models. The computational
workflow is organized as a reproducible generate--fit--summarize pipeline.

## Project layout

- `src/` contains the score-matching estimators and solver adapters.
- `src/sm_l1/` preserves the authors' uploaded R implementation as reference
  material; the registered SM--L1 estimator is the independent Python module
  `src/score_matching_l1.py`.
- `src/l0bnb2/` contains the bundled GraphL0Learn comparison implementation.
- `experiments/` contains reusable simulation models, data-generation entry
  points, estimator runners, and Quest job files.
- `analysis/` contains result aggregation scripts.
- `data/` contains generated instances and is excluded from version control.
- `results/` contains retained preliminary results from earlier experiments.
- `paper/` contains the LaTeX manuscript and compiled PDF.
- `references/` contains local literature files and research notes.

Detailed simulation settings and Quest commands are documented in
[`experiments/README.md`](experiments/README.md).

## Setup

Create a project-local Python environment and install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Gurobi and a valid license are required for the MIQP estimators. The active L1
score-matching estimator is implemented in Python/Numba and, like the data
generators, requires neither Gurobi nor R.

## Generate the registered experiment instances

From the project root:

```bash
.venv/bin/python experiments/generate_primary_graph_recovery_data.py
```

The 60 registered instances are written under
`data/gaussian_experiments/primary_graph_recovery/`.  Fitting programs and
Quest jobs only read these saved instances; they never generate data.

## Run a fitting runner directly

```bash
.venv/bin/python experiments/Run_gaussian_sample_size.py
```

The runner's command-line arguments have defaults: this invocation loads one
existing panel instance and fits SM--L1 at one constant.  Run it with `--help`
to see the settings you can change.  It creates no dataset and does not require
Gurobi.

## Run and summarize the full study

Choose each method's penalty constant manually with the panel runners, then set
that method and constant in the corresponding Quest job.  Submit the three
panel arrays with:

```bash
bash experiments/quest_jobs/gaussian_support_recovery/submit_all.sh
```

After evaluation finishes:

```bash
.venv/bin/python analysis/summarize_gaussian_experiments.py \
  --study primary_graph_recovery
```

The statistical CSV is intentionally compact.  Detailed solver diagnostics and
run metadata are written to linked JSONL and JSON files.
