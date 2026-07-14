# Gaussian graph-recovery experiment

The registered primary experiment compares graph-recovery methods by their ROC
curves, so it evaluates a complete penalty path instead of selecting one
penalty constant for each method.

## Data-generating procedure

For a direct comparison with the competing score-matching method, the generator
implements Section 4.1 of [Lin, Drton, and Shojaie
(2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5476334/):

1. Form five disconnected components, each a 10×10 four-neighbor lattice. The
   paper used ten such components for `p=1000`; five preserves the construction
   at the registered `p=500`.
2. Choose three hubs independently in each component and bring each hub to
   degree 20.
3. Draw every directed nonzero entry independently from `Uniform(0.5, 1)`,
   divide each row by 1.5 times its absolute off-diagonal row sum, average the
   matrix with its transpose, and set its diagonal to one.
4. Invert this diagonally dominant matrix and convert the resulting covariance
   to a correlation matrix.
5. Draw `n=1000` mean-zero Gaussian observations at `p=500`.

The paper averaged 100 independently generated datasets. This research
prototype registers ten independent datasets; change
`NUMBER_OF_REPLICATIONS` in `primary_graph_recovery_config.py` before generation
if the final study should use 100.

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
record the paper procedure and generation seeds.

## Run the penalty path

`Run_gaussian_roc.py` exposes `p`, `n`, method, penalty path, screening size,
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
score-matching methods the same 2,500 edges with largest absolute sample
correlations. Unscreened edges count as absent in TPR/FPR, and each diagnostics
record reports `candidate_recall`; this makes the resulting plot a screened ROC
curve rather than an unrestricted 499,500-edge ROC curve.

Each result row includes TPR, FPR, the penalty constant and realized penalty.
Gurobi fits additionally record the incumbent objective (`UB`), best bound
(`LB`), and relative gap.

## Quest and plotting

Submit one ten-task array for SM–L0 and one for SM–L1:

```bash
bash experiments/quest_jobs/gaussian_support_recovery/submit_all.sh
```

After the arrays finish, average TPR/FPR at each penalty point and draw the plot:

```bash
python3 analysis/plot_gaussian_roc.py
```

This writes `experiments_results/gaussian_roc_summary.csv` and
`experiments_results/gaussian_roc.png`. Failed fits are excluded; the summary
reports the number of available replications for each point.

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
