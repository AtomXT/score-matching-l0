# Experiment code guide

This folder contains the data generators and fitting programs for the first
experiment in the paper: Gaussian graph recovery.  Data generation, penalty
calibration, evaluation, and summarization are separate stages.  In particular,
the fitting programs and Quest jobs never create or modify datasets.

## Registered design

All configurations have target average degree 4, target minimum partial
correlation 0.20, target condition number 10, and ten independent replications.

| Use | Topology | Dimensions |
|---|---|---|
| Penalty calibration only | Erdős--Rényi | `(p,n)=(20,40)` |
| Sample-size comparison | Erdős--Rényi | `p=40`, `n=20,40,80,160` |
| Dimension comparison | Erdős--Rényi | `(p,n)=(40,80),(60,120)` |
| Topology comparison | Erdős--Rényi and scale-free | `(p,n)=(40,80)` |

The Erdős--Rényi `(40,80)` datasets are shared by all three evaluation panels
and are fit only once.  The saved design therefore has seven unique
configurations and 70 datasets: ten calibration datasets and 60 evaluation
datasets.  Calibration performance is not reported as evaluation performance.

## Complete workflow

Run all commands from the project root.

### 1. Generate the datasets once

In PyCharm, open `generate_primary_graph_recovery_data.py` and click **Run**, or
use:

```bash
python3 experiments/generate_primary_graph_recovery_data.py
```

This creates all 70 instances under:

```text
data/gaussian_experiments/primary_graph_recovery/
```

Existing instances are retained unless overwrite is requested explicitly.  Each
instance contains `dataset.npz` and `metadata.json`; the study folder also
contains `design.json` and `manifest.json`.

The Quest scripts assume that these files already exist on Quest.  They do not
run a generator.

### 2. Calibrate one constant per method

For each method, the calibration stage fits the following constants on all ten
Erdős--Rényi `(p,n)=(20,40)` datasets:

```text
0.03125, 0.044, 0.0625, 0.088, 0.125, 0.177, 0.25, 0.354,
0.5, 0.707, 1, 1.414, 2, 2.828, 4
```

The selection program chooses the constant with the largest mean F1 score.  It
breaks ties by smaller mean structural Hamming distance and then by the larger
constant.  A constant is eligible only if all ten calibration fits are
available.  The selected constants are saved in:

```text
experiments_results/penalty_calibration/selected_constants.json
```

The companion `calibration_summary.csv` records every candidate and identifies
the selected one.  It also counts available mixed-integer incumbents that did
not have a global-optimality certificate.

The method-specific penalty formulas are:

| Method | Penalty used at `(p,n)` | Source of order |
|---|---|---|
| SM--L0 | `c * log(p) / n` | same order as the theory in this paper |
| SM--L1 | `c * sqrt(log(p) / n)` | Lin, Drton, and Shojaie (2016), Corollary 1 |
| GraphL0 | `c * log(p) / n` | Behdin, Chen, and Mazumder, Theorem 5 |
| Graphical lasso | `c * sqrt(log(p) / n)` | Ravikumar et al. (2011) |

GraphL0's original numerical appendix used a square-root-order search and
jointly tuned its two penalties; this experiment instead uses its support-
recovery theorem's `log(p)/n` order.
Only GraphL0's sparsity constant is calibrated, while its ridge coefficient is
held fixed at `0.05`.  Numerical values of `c` should not be compared across
methods because their objectives use different normalizations.

### 3. Evaluate with the fixed constants

Each evaluation fit reads `selected_constants.json` and uses exactly one
constant for each method.  Constants are transferred without retuning across
sample sizes, dimensions, graph topologies, and replications.

The three evaluation arrays have nonoverlapping ownership:

- `sample_size.sh` fits the four Erdős--Rényi `p=40` configurations;
- `dimension.sh` fits only `(p,n)=(60,120)`, because `(40,80)` is already fit by
  the sample-size array; and
- `topology.sh` fits only the scale-free `(40,80)` configuration, because its
  Erdős--Rényi counterpart is already fit by the sample-size array.

The shared `(40,80)` results are reused when constructing the dimension and
topology panels.  This prevents duplicate fits and duplicate rows in summaries.

### 4. Summarize the evaluation

After all evaluation arrays finish, run:

```bash
python3 analysis/summarize_gaussian_experiments.py \
  --study primary_graph_recovery
```

The default summary includes only evaluation rows.  Its columns are the
configuration, method, selected constant and rate, number of available and
excluded fits, mean F1, TPR, FPR, and SHD with standard errors, exact-recovery
count, and the number of uncertified mixed-integer incumbents.

The paper's main table should use only mean F1 (standard error) and the exact-
recovery count in each method cell.  SHD is a secondary or supplementary table;
the remaining support measures can be reconstructed from `TP`, `FP`, and `FN`.

## Running the complete workflow on Quest

Generate the 70 datasets before submission.  Then submit the dependency-aware
workflow from the project root:

```bash
bash experiments/quest_jobs/gaussian_support_recovery/submit_all.sh
```

This submits, in order:

1. `calibration.sh`, a ten-task array with one replication per task;
2. `select_constants.sh`, which starts only after all calibration tasks succeed;
3. the sample-size, dimension, and topology arrays, which start only after the
   constants have been selected.

The array index selects the replication only.  It is not used as the job name.
Each evaluation runner receives the explicit job name `sample_size`,
`dimension`, or `topology`.

The scripts currently request account `p32811`, the `python-miniconda3` module,
the `python39` environment, the Gurobi module, eight CPU cores, 16 GB of memory,
and email notifications to `tongxu2027@u.northwestern.edu`.  These local Quest
values should be confirmed before submission.  The wall-time and memory requests
are provisional; inspect a representative task with `seff` before the final run.
The solver thread count should equal the allocated CPU count.

Quest output is written to `experiments/quest_jobs/outlog/`.  The fitting jobs
use verbose solver output, so Gurobi progress, graphical-lasso warnings, and fit
summaries appear in the Slurm log.  SM--L1 prints a final coordinate-descent
summary but does not print every sweep.

Northwestern documents Slurm in the
[Quest user guide](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html).

## Runner defaults

Each panel runner explicitly defines its command-line arguments and defaults,
so it can be opened in PyCharm and run directly while still allowing every
setting to be changed manually:

```text
Run_gaussian_sample_size.py
Run_gaussian_dimension.py
Run_gaussian_topology.py
```

The defaults fit SM--L1 at `c=1` on replication 0 of the first configuration
owned by that runner.  They do not generate data and do not require Gurobi.
Use `--help` to see and change the method, constant list, configuration,
replication list, solver limits, initial big-M, output path, and other settings.

The same checks can be launched from the project root:

```bash
python3 experiments/Run_gaussian_sample_size.py
python3 experiments/Run_gaussian_dimension.py
python3 experiments/Run_gaussian_topology.py
```

Pass `--overwrite-results` to repeat a run using the same output path.  The
runners never alter a dataset.

## Output files

For a result path such as `example.csv`, the runner writes three linked files:

- `example.csv` is the compact statistical record.  It contains identifiers,
  the selected constant and realized penalty, status and availability,
  certification and runtime, `TP`, `FP`, `FN`, `TPR`, `FPR`, `F1`, exact
  recovery, SHD, and any error message.
- `example.diagnostics.jsonl` contains one detailed record per fit, including
  solver objectives, bounds, gaps, node counts, convergence diagnostics,
  candidate-set information, big-M relaxation diagnostics, and tracebacks.
- `example.run.json` records arguments, software and scheduler information,
  the complete fit plan, selected instances, and start/finish status.

This separation keeps the results table readable without discarding numerical
diagnostics.

Pass `--overwrite-results` to replace the linked output files; otherwise the
CSV and JSONL records are appended.

SM--L0 starts from the configurable scalar `--big-m-init` (default `1000`),
solves the continuous relaxation with indicators in `[0,1]`, and uses
`min(M_init, max(1e-6 M_init, 2 max(abs(beta_relax))))` in the integer model.
The compact `certified` flag records whether that bounded MIQP reached the
requested MIP gap.

## Methods

- `sm_l0`: the proposed L0-penalized score-matching MIQP.  Gurobi and a valid
  license are required.
- `sm_l1`: the Gaussian L1 score-matching estimator of Lin, Drton, and Shojaie,
  implemented in Python with highscore-compatible cyclic coordinate descent.
- `graphl0`: the bundled L0-penalized Gaussian pseudolikelihood competitor.
- `glasso`: graphical lasso from scikit-learn.

SM--L1 uses the authors' full symmetric-matrix convention:

```text
0.5 tr(K S K) - tr(K) + lambda * sum(i != j) |K[i,j]|.
```

Thus one undirected edge contributes `2 * lambda * |K[i,j]|`.  The diagonal is
unpenalized, and an edge is reported when `abs(K[i,j]) > 1e-6`.  In an `n<p`
problem the empirical quadratic can be unbounded below at weak penalties.  Such
fits remain recorded with `fit_available=0` and cannot be used to select a
constant.

## File map

### Data generation

- `generate_primary_graph_recovery_data.py`: click **Run** to generate the
  complete registered design and master manifest.
- `generate_gaussian_sample_size_data.py`: click **Run** to generate only the
  sample-size panel datasets.
- `generate_gaussian_dimension_data.py`: click **Run** to generate only the
  dimension panel datasets.
- `generate_gaussian_topology_data.py`: click **Run** to generate only the
  topology panel datasets.
- `generate_gaussian_experiments.py`: general generator used by the dedicated
  entry points; direct use requires a study name and design arguments.
- `gaussian_models.py`: graph constructors and precision-matrix calibration
  routines; this is a library module, not a runnable experiment.

The three panel-specific generators do not create the separate calibration
configuration.  Use `generate_primary_graph_recovery_data.py` for the complete
registered workflow.

### Fitting and registered settings

- `Run_gaussian_experiments.py`: saved-data-only fitting engine.  It implements
  the calibration, evaluation, and local-check stages and writes the compact
  result plus diagnostic sidecars.
- `Run_gaussian_sample_size.py`: sample-size runner with direct-run defaults.
- `Run_gaussian_dimension.py`: dimension runner with direct-run defaults.
- `Run_gaussian_topology.py`: topology runner with direct-run defaults.
- `primary_panel_workflow.py`: shared execution logic used by the dedicated
  generators and runners.
- `primary_graph_recovery_config.py`: the single source of truth for methods,
  calibration grid, calibration configuration, replications, and nonoverlapping
  evaluation ownership.
- `penalty_rates.py`: method-specific rate formulas and selected-constant JSON
  loading.
- `common.py`: shared parsing, atomic storage, CSV, screening, and metric
  helpers; this is a library module.
- `__init__.py`: marks `experiments` as a Python package.

### Analysis

- `analysis/select_penalty_constants.py`: selects one constant per method and
  writes the JSON consumed by all evaluations.
- `analysis/summarize_gaussian_experiments.py`: writes the compact Monte Carlo
  summary.

### Quest jobs

- `quest_jobs/gaussian_support_recovery/calibration.sh`: ten-task calibration
  array.
- `quest_jobs/gaussian_support_recovery/select_constants.sh`: selects constants
  after calibration succeeds.
- `quest_jobs/gaussian_support_recovery/sample_size.sh`: sample-size evaluation
  array.
- `quest_jobs/gaussian_support_recovery/dimension.sh`: dimension evaluation
  array.
- `quest_jobs/gaussian_support_recovery/topology.sh`: topology evaluation array.
- `quest_jobs/gaussian_support_recovery/submit_all.sh`: submits all five stages
  with the required Slurm dependencies.

Do not edit generated instances or selected constants manually.  Regenerate
data explicitly when the design changes, and rerun calibration whenever the
method implementation, preprocessing, candidate graph, or penalty grid changes.
