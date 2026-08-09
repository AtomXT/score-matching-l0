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

## Validate the support-optimality MILP

The validation script compares the support-optimality MILP with both the original
convex MIQP and exhaustive support enumeration. It checks optimized values, supports,
coefficients, stationarity residuals, and the identity that replaces the quadratic
objective by `-trace(K) / 2`:

```bash
.venv/bin/python experiments/test_support_optimality_milp.py
```

By default it loads
`data/gaussian/m009_n030_comp01_side03_hubs01_deg04_seed000/dataset.npz`, a saved
9-variable, 30-observation grid/hub instance with 13 true edges. The test retains the
18 strongest empirical-correlation candidates (including true edges and nonedge
decoys) and enumerates all 262,144 supports. Use `--ridge 0.01` to test the
ridge-modified active stationarity equations as well, or `--dataset` to select another
saved `.npz` instance.

## Generate the registered ROC instances

From the project root:

```bash
.venv/bin/python experiments/generate_primary_graph_recovery_data.py
```

Ten independent `p=500, n=250` connected Erdős–Rényi instances are written
under `data/gaussian_experiments/primary_graph_recovery/`. The registered graph
has target average degree 4, minimum partial correlation 0.20, and condition
number 10. Fitting programs and Quest jobs only read these saved instances;
they never generate data.

## Run a fitting runner directly

```bash
.venv/bin/python experiments/Run_gaussian.py
```

The single-test runner has explicit defaults for topology, `p`, `n`, method,
penalty constant, screening, and solver settings. `Run_gaussian_roc.py` is the
separate registered runner that evaluates the full penalty path.

## Run and summarize the full study

Submit the SM--L0 CORe, support-optimality MILP, and SM--L1 ROC arrays:

```bash
bash experiments/quest_jobs/gaussian_support_recovery/submit_all.sh
```

After evaluation finishes, build the averaged metric table, ROC plot, and
precision–recall plot:

```bash
.venv/bin/python analysis/plot_gaussian_roc.py --p 500 --n 250
```

The result rows include TPR/FPR; Gurobi rows also include UB, LB, and gap.
Detailed solver diagnostics and run metadata are written to linked JSONL and
JSON files.
