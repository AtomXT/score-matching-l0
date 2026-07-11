# Gaussian experiment workflow

The experiment code follows a generate--solve--summarize workflow. Data generation is
separated from estimation so that every formulation is evaluated on exactly the same
random instances. Each instance directory contains the training, validation, and test
samples, the population covariance and precision matrices, the true adjacency matrix,
and a complete JSON record of the random seeds and achieved graph properties. The solver
driver appends one result row immediately after each fit. Consequently, completed fits
remain available if a later instance reaches its time limit or fails.

## Experimental setup

For a graph with adjacency matrix \(A\), the generator first draws a symmetric signed
edge matrix \(W\). It then forms \(K=W+cI\), choosing \(c\) to attain the requested
spectral condition number before standardization. The lower endpoint of the edge-weight
interval is calibrated to make the minimum nonzero partial correlation as close as
possible to its target. Finally, the covariance is standardized to have unit diagonal.
The requested and achieved condition numbers and signal strengths are both retained;
this is important because the two targets need not be jointly attainable on every graph.

The controlled graph families are chains, four-neighbor square lattices, banded graphs,
hub graphs, connected Erdos--Renyi-type graphs, and Barabasi--Albert scale-free graphs.
Independent training, validation, and test samples of size \(n\) are generated from each
population matrix. Unless a job script states otherwise, the subset-selection penalty is
\(\rho=c\log(r)/n\), whereas the loss-matched \(\ell_1\) penalty is
\(c\sqrt{\log(r)/n}\). Here \(r\) is the number of candidate edges and
\(c\in\{0.25,0.5,1,2,4\}\).

Within a replication, configurations that differ only in sample size, signal, or
condition number use the same underlying graph and edge randomness. Thus each factor
study is paired rather than being confounded by a newly generated topology. By default,
the population graph changes across replications. Adding `--fixed-graph` holds it fixed
and varies only the observations, providing the complementary within-graph experiment
described in the paper.

The following command generates a small local example:

```bash
python3 -m experiments.generate_gaussian_experiments \
  --study "local_example" \
  --topology-list "chain,erdos_renyi" \
  --p-list "20" \
  --n-list "40,100" \
  --degree-list "4" \
  --signal-list "0.2" \
  --condition-list "10" \
  --rep-list "0,1"
```

## Compared methods

The runner currently includes three score-matching estimators. `sm_l0` is the profiled
big-\(M\) MIQP, `sm_l0_core` adds the coordinatewise optimality relations, and `sm_l1`
is the loss-matched regularized score-matching estimator solved by FISTA. `graphl0` is
also available as a Gaussian pseudolikelihood comparison when the complete candidate
graph is used. The two score-matching penalties use their respective theoretical rates;
their multiplier and realized numerical value are both recorded. Native tuning paths
should still be run separately when a method-specific calibration is desired.

```bash
/usr/bin/python3 -m experiments.Run_gaussian_experiments \
  --study "local_example" \
  --method-list "sm_l0,sm_l0_core,sm_l1" \
  --penalty-multiplier-list "0.25,0.5,1,2,4" \
  --time-limit "600" \
  --mip-gap "0.01" \
  --threads "8" \
  --job-name "local_example" \
  --overwrite-results
```

The complete graph is the default candidate set. For a larger pilot, empirical
correlation screening can be enabled with `--candidate-rule correlation --screen-size K`.
This screening is heuristic: the result file therefore records both the number of
candidate edges and the fraction of population edges retained. A screened incumbent
must not be described as the global solution over the complete graph.

Each result row reports support recovery, matrix estimation, held-out score, solver time,
objective bounds, optimality gap, node count, and whether the declared MIP tolerance was
met. It also records the Python, NumPy, and Gurobi versions, host, thread count, and SLURM
job identifiers. A time-limited incumbent is retained, but it is not labeled certified.

## Quest studies

The Quest scripts are grouped by purpose under `experiments/quest_jobs/`.

- `gaussian_support_recovery/sample_size.sh` varies \(n\) at fixed \(p\).
- `gaussian_support_recovery/problem_size.sh` varies \(p\) at fixed \(n/p=2\).
- `gaussian_support_recovery/degree.sh` varies the maximum degree of a hub graph.
- `gaussian_support_recovery/signal.sh` varies the minimum-partial-correlation target.
- `gaussian_computation/formulation.sh` compares the base and strengthened MIQPs on
  identical instances.

The statistical jobs use 100 array tasks, one for each independent replication. The
computational study uses 20. Statistical fits use a 600-second per-fit limit; the
formulation study uses 3600 seconds. Submit a job from the project root, for example:

```bash
mkdir -p experiments/quest_jobs/outlog
sbatch experiments/quest_jobs/gaussian_support_recovery/sample_size.sh
```

The scripts reproduce the software structure used in the `QP_indicator` project: they
purge modules, activate the `python39` environment, load Gurobi, generate the assigned
instances, and then run the requested methods. Account, partition, environment, and
email directives are intentionally explicit and should be checked before submission.

## Summaries

After the array jobs finish, aggregate Monte Carlo means and standard errors with:

```bash
python3 analysis/summarize_gaussian_experiments.py --study "sample_size"
```

Error rows are retained in the raw files and excluded from numerical averages. Their
frequency should be reported separately rather than silently treated as missing data.
