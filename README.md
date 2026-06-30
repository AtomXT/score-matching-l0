# Score Matching Experiments

Utilities for preliminary experiments with regularized score matching methods for
graphical models.

## Setup

GraphL0Learn is included locally under `src/l0bnb2`. Create a project-local
environment that reuses the bundled NumPy, then install the missing packages:

```bash
/Users/tongxu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Gaussian Test Data

The first generator follows the Gaussian simulation procedure from Section 4.1 of
Lin, Drton, and Shojaie, "Estimation of High-Dimensional Graphical Models Using
Regularized Score Matching".

```bash
python scripts/generate_gaussian_data.py \
  --n 600 \
  --seed 1 \
  --out data/gaussian/gaussian_n600_m1000_seed1.npz
```

The saved `.npz` file contains:

- `X`: generated samples with shape `(n, m)`
- `Sigma`: population correlation matrix
- `precision`: sparse diagonally dominant matrix before inversion
- `adjacency`: graph adjacency matrix
- `params_json`: parameters used for the run

Default parameters match the paper's Gaussian setup: 10 disconnected `10 x 10`
lattice components, 3 hubs per component, hub degree 20, and sample size 600.
The CLI default seed is `0` for reproducibility.

## GraphL0BnB Test Run

Run a moderate support-recovery test with GraphL0Learn's `BNBTree`:

```bash
.venv/bin/python scripts/run_graphl0bnb_test.py
```

By default, this creates or reuses:

```text
data/gaussian/m025_n150_comp01_side05_hubs02_deg08_seed000/
```

The result row is appended to:

```text
results/graphl0bnb/results.csv
```

The CSV includes runtime, objective/gap information, selected edge count, true
edge count, TPR, FPR, precision, recall, and F1 score.

For an exact-size testing graph with multiple L0 penalties:

```bash
.venv/bin/python scripts/run_graphl0bnb_test.py \
  --m 10 \
  --n 500 \
  --target-edges 12 \
  --l0-values 0.005,0.01,0.02,0.05,0.1 \
  --overwrite-results
```

## Running Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
