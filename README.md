# Score Matching Experiments

Utilities for preliminary experiments with regularized score matching methods for
graphical models.

## Setup

GraphL0Learn is included locally under `src/l0bnb2`. The score-matching MIQP
runner uses Gurobi through `gurobipy`. Create a project-local environment that
reuses the bundled NumPy, then install the missing packages:

```bash
/Users/tongxu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

If you are using another interpreter with Gurobi already installed, replace
`.venv/bin/python` in the commands below with that interpreter. A quick check is:

```bash
.venv/bin/python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

On this machine, `/usr/bin/python3` can import the user-site `gurobipy` package.

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

Generate the default GraphL0Learn dataset first:

```bash
.venv/bin/python scripts/generate_graphl0learn_data.py
```

This script can be launched directly in PyCharm. Its defaults are `m=50`,
`n=500`, `model=banded_Toeplitz_precision`, `half_bandwidth=2`, `rho=0.5`,
`cond=2`, and `seed=0`.

Run a moderate support-recovery test with GraphL0Learn's `BNBTree`:

```bash
.venv/bin/python scripts/run_graphl0bnb_test.py
```

By default, the runner loads this existing dataset:

```text
data/graphl0learn/m050_n500_graphl0learn_banded_bw02_rho050_cond02_seed000/
```

The direct-run defaults are `m=50`, `n=500`, `l0=0.02`, `l2=0.05`, and a
300-second time limit. The result row is written fresh to:

```text
results/graphl0bnb/pycharm_default.csv
```

The method runners do not generate data. If the dataset folder is missing, run
`scripts/generate_graphl0learn_data.py` first.
The default data and result paths are fixed inside this project directory, so
these scripts can be launched from PyCharm even if the working directory is
different. The runners use `--verbose True` by default. To silence direct
PyCharm runs, add `--verbose False` to the run configuration.

The CSV includes runtime, objective/gap information, selected edge count, true
edge count, TPR, FPR, precision, recall, and F1 score.

For an exact-size testing graph with multiple L0 penalties:

```bash
.venv/bin/python scripts/run_graphl0bnb_test.py \
  --data-source exact \
  --m 10 \
  --n 500 \
  --target-edges 12 \
  --l0-values 0.005,0.01,0.02,0.05,0.1 \
  --overwrite-results
```

## Score-Matching MIQP Test Run

Run the Gaussian score-matching MIQP estimator:

```bash
/usr/bin/python3 scripts/run_score_matching_miqp_test.py
```

By default, this runner also loads the same existing GraphL0Learn-generated
banded Toeplitz dataset:

```text
data/graphl0learn/m050_n500_graphl0learn_banded_bw02_rho050_cond02_seed000/
```

The direct-run defaults are `m=50`, `n=500`, `lambda=0.012`, and a 300-second
time limit. It writes fresh results to:

```text
results/score_matching_miqp/pycharm_default.csv
```

Result columns include dataset size information, lambda, data-derived Big-M
range, runtime, objective, objective bound, MIP gap, node count, Gurobi status,
selected edge count, true edge count, TPR, FPR, precision, recall, and F1 score.

You can change the test size and lambda grid directly:

```bash
/usr/bin/python3 scripts/run_score_matching_miqp_test.py \
  --data-source exact \
  --m 10 \
  --n 500 \
  --target-edges 12 \
  --lambda-values 0.005,0.01,0.02,0.05,0.1 \
  --time-limit 60 \
  --overwrite-results
```

## Running Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
