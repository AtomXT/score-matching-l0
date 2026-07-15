# Gaussian graph-recovery experiment

The registered primary experiment compares graph-recovery methods by their ROC
curves, so it evaluates a complete penalty path instead of selecting one
penalty constant for each method.

## Data-generating procedure

The main experiment uses ten independently generated connected Erdős–Rényi
graphs at `p=500` and `n=1000`. Each graph has target average degree 4. Random
edge signs and magnitudes are calibrated to target minimum partial correlation
0.20 and precision-matrix condition number 10, after which the covariance is
standardized and mean-zero Gaussian observations are drawn.

## Generate the data

Run from the repository root:

```bash
python3 experiments/generate_primary_graph_recovery_data.py
```

The equivalent panel-specific entry point is:

```bash
python3 experiments/generate_gaussian_roc_data.py
```

Both have no-argument defaults and support explicit replication arguments in
the panel-specific runner. Existing datasets are retained unless `--overwrite`
is supplied. The saved `design.json`, `manifest.json`, and per-instance metadata
record the population targets and generation seeds.

## Run the penalty path

`Run_gaussian_roc.py` exposes `p`, `n`, method, penalty path, screening penalty,
solver limits, and output paths as explicit arguments. Its defaults load
replication 0 at `p=500, n=1000` and run SM–L1:

```bash
python3 experiments/Run_gaussian_roc.py
```

For example, an SM–L0 evaluation is:

```bash
python3 experiments/Run_gaussian_roc.py \
  --stage evaluation \
  --rep-list 0 \
  --topology erdos_renyi \
  --p 500 \
  --n 1000 \
  --method-list sm_l0 \
  --penalty-constant-list "0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,1,1.2,1.4,1.6,1.8,2,2.5,3,5,10,20,50,100" \
  --overwrite-results
```

The common constant path is multiplied by `log(p)/n` for SM–L0 and by
`sqrt(log(p)/n)` for SM–L1. The constants are not tuned or selected: every
point is retained for the ROC curve.

There are 124,750 possible undirected edges at `p=500`, which is too large for
the current research MIQP. The registered runner therefore gives both
score-matching methods the support selected by a lightly regularized graphical
lasso fit with `--screen-alpha 0.01`. Unscreened edges count as absent in
TPR/FPR, and each diagnostics record reports `candidate_recall`; this makes the
resulting plot a screened ROC curve rather than an unrestricted 124,750-edge
ROC curve.

Each result row includes TPR, FPR, the penalty constant and realized penalty.
Gurobi fits additionally record the incumbent objective (`UB`), best bound
(`LB`), and relative gap.

## Quest and plotting

Submit one ten-task array for SM–L0 and one for SM–L1:

```bash
bash experiments/quest_jobs/gaussian_support_recovery/submit_all.sh
```

After the arrays finish, average the metrics at each penalty point and draw the
ROC and precision–recall plots:

```bash
python3 analysis/plot_gaussian_roc.py
```

This writes `experiments_results/gaussian_roc_summary.csv`,
`experiments_results/gaussian_roc.png`, and
`experiments_results/gaussian_pr.png`. Failed fits are excluded; the summary
reports the mean and standard error of TPR, FPR, and precision, together with
the number of available replications for each point.

## Single tests

The old sample-size, dimension, and topology runners have been consolidated
into one directly runnable script:

```bash
python3 experiments/Run_gaussian.py
```

Its defaults fit SM–L1 once at `c=1` on replication 0 with `p=500, n=1000`. Change
`--topology`, `--p`, `--n`, `--method-list`, and
`--penalty-constant-list` directly for manual tests; the requested saved dataset
must already exist.
