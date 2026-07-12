# L0-penalized score matching

This repository contains the manuscript and experiment code for studying
L0-penalized score matching in undirected graphical models. The computational
workflow is organized as a reproducible generate--fit--summarize pipeline.

## Project layout

- `src/` contains the score-matching estimators and solver adapters.
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

Gurobi and a valid license are required for the MIQP estimators. The L1
score-matching estimator and data generators do not require Gurobi.

## Generate experiment instances

From the project root:

```bash
.venv/bin/python -m experiments.generate_gaussian_experiments \
  --study local_example \
  --topology-list chain,erdos_renyi \
  --p-list 20 \
  --n-list 40,100 \
  --degree-list 4 \
  --signal-list 0.2 \
  --condition-list 10 \
  --rep-list 0,1
```

Instances are written under `data/gaussian_experiments/<study>/`. Reusable
graph and precision-matrix constructions are defined in
`experiments/gaussian_models.py`; this module also retains the
lattice-with-hubs design of Lin, Drton, and Shojaie (2016).

## Fit the estimators

```bash
.venv/bin/python -m experiments.Run_gaussian_experiments \
  --study local_example \
  --method-list sm_l0,sm_l1 \
  --penalty-multiplier-list 0.25,0.5,1,2,4 \
  --time-limit 600 \
  --mip-gap 0.01 \
  --threads 8 \
  --job-name local_example \
  --overwrite-results
```

Raw results are written to `experiments_results/`. Each fitted method uses the
same saved instance and candidate-edge set.

## Summarize results

```bash
.venv/bin/python analysis/summarize_gaussian_experiments.py \
  --study local_example
```

The summary contains Monte Carlo means and standard errors for the requested
support-recovery, estimation, predictive-score, and computational metrics.
