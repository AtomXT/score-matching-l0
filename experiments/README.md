# Experiment code guide

This folder contains the data generators and fitting programs for the first
experiment in the paper: Gaussian graph recovery.  Data generation, fitting,
and summarization are separate stages.  The fitting programs and Quest jobs
never create or modify datasets.

## Registered design

All configurations have target average degree 4, target minimum partial
correlation 0.20, target condition number 10, and ten independent replications.

| Use | Topology | Dimensions |
|---|---|---|
| Sample-size comparison | Erdős--Rényi | `p=40`, `n=20,40,80,160` |
| Dimension comparison | Erdős--Rényi | `(p,n)=(40,80),(60,120)` |
| Topology comparison | Erdős--Rényi and scale-free | `(p,n)=(40,80)` |

The Erdős--Rényi `(40,80)` datasets are shared by all three evaluation panels
and are fit only once.  The saved design therefore has six unique
configurations and 60 datasets.

## Complete workflow

Run all commands from the project root.

### 1. Generate the datasets once

In PyCharm, open `generate_primary_graph_recovery_data.py` and click **Run**, or
use:

```bash
python3 experiments/generate_primary_graph_recovery_data.py
```

This creates all 60 instances under:

```text
data/gaussian_experiments/primary_graph_recovery/
```

Existing instances are retained unless overwrite is requested explicitly.  Each
instance contains `dataset.npz` and `metadata.json`; the study folder also
contains `design.json` and `manifest.json`.

The Quest scripts assume that these files already exist on Quest.  They do not
run a generator.

### 2. Test penalty constants manually

Each panel runner accepts a comma-separated `--penalty-constant-list`.  Run one
method at several constants, inspect the resulting F1, TPR, FPR, and SHD, and
then choose the constant you want to use.  For example:

```bash
python3 experiments/Run_gaussian_sample_size.py \
  --p 40 \
  --n 80 \
  --method-list sm_l1 \
  --penalty-constant-list "0.5,1,2" \
  --results-csv experiments_results/sm_l1_constant_check.csv \
  --overwrite-results
```

The method-specific penalty formulas are:

| Method | Penalty used at `(p,n)` | Source of order |
|---|---|---|
| SM--L0 | `c * log(p) / n` | same order as the theory in this paper |
| SM--L0 CORe | `c * log(p) / n` | alternative formulation of SM--L0 |
| SM--L1 | `c * sqrt(log(p) / n)` | Lin, Drton, and Shojaie (2016), Corollary 1 |
| GraphL0 | `c * log(p) / n` | Behdin, Chen, and Mazumder, Theorem 5 |
| Graphical lasso | `c * sqrt(log(p) / n)` | Ravikumar et al. (2011) |

GraphL0's original numerical appendix used a square-root-order search and
jointly tuned its two penalties; this experiment instead uses its support-
recovery theorem's `log(p)/n` order.
Only GraphL0's sparsity constant is varied, while its ridge coefficient is
held fixed at `0.05`.  Numerical values of `c` should not be compared across
methods because their objectives use different normalizations.

### 3. Evaluate with manually chosen constants

Pass the chosen constant directly to the runner.  Run methods separately when
they use different constants, because `--penalty-constant-list` applies every
listed constant to every method in `--method-list`.

```bash
python3 experiments/Run_gaussian_dimension.py \
  --stage evaluation \
  --p 60 \
  --n 120 \
  --method-list sm_l1 \
  --penalty-constant-list "1"
```

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
configuration, method, penalty constant and rate, number of available and
excluded fits, mean F1, TPR, FPR, and SHD with standard errors, exact-recovery
count, and the number of uncertified mixed-integer incumbents.

The paper's main table should use only mean F1 (standard error) and the exact-
recovery count in each method cell.  SHD is a secondary or supplementary table;
the remaining support measures can be reconstructed from `TP`, `FP`, and `FN`.

## Running the complete workflow on Quest

Generate the 60 datasets and set the method and penalty constant inside each
panel job before submission.  Then submit the jobs from the project root:

```bash
bash experiments/quest_jobs/gaussian_support_recovery/submit_all.sh
```

This submits the sample-size, dimension, and topology arrays independently.

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
explicit `p` and `n`, replication list, solver limits, initial big-M, output
path, and other settings.

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
  the supplied constant and realized penalty, status and availability,
  certification and runtime, the Gurobi `UB`, `LB`, and relative `gap`, `TP`,
  `FP`, `FN`, `TPR`, `FPR`, `F1`, exact recovery, SHD, and any error message.
  The three Gurobi fields are blank for methods not solved by Gurobi.
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
The CORe formulation also enforces its minimum active-coefficient threshold, so
its final bound is the larger of this relaxation-derived value and the largest
CORe threshold.
The compact `certified` flag records whether that bounded MIQP reached the
requested MIP gap.

## Methods

- `sm_l0`: the proposed L0-penalized score-matching MIQP.  Gurobi and a valid
  license are required.
- `sm_l0_core`: the historical CORe inactive/positive/negative formulation of
  the same SM--L0 problem. Select it with `--method-list sm_l0_core`.
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
fits remain recorded with `fit_available=0` and should not guide the manual
constant choice.

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

### Fitting and registered settings

- `Run_gaussian_experiments.py`: saved-data-only fitting engine.  It implements
  the evaluation and local-check stages and writes the compact result plus
  diagnostic sidecars.
- `Run_gaussian_sample_size.py`: sample-size runner with direct-run defaults.
- `Run_gaussian_dimension.py`: dimension runner with direct-run defaults.
- `Run_gaussian_topology.py`: topology runner with direct-run defaults.
- `primary_panel_workflow.py`: shared execution logic used by the dedicated
  generators and runners.
- `primary_graph_recovery_config.py`: the single source of truth for
  replications and nonoverlapping evaluation ownership.
- `penalty_rates.py`: method-specific penalty-rate formulas.
- `../src/score_matching_core_miqp.py`: the alternative CORe formulation of
  SM--L0.
- `common.py`: shared parsing, atomic storage, CSV, screening, and metric
  helpers; this is a library module.
- `__init__.py`: marks `experiments` as a Python package.

### Analysis

- `analysis/summarize_gaussian_experiments.py`: writes the compact Monte Carlo
  summary.

### Quest jobs

- `quest_jobs/gaussian_support_recovery/sample_size.sh`: sample-size evaluation
  array.
- `quest_jobs/gaussian_support_recovery/dimension.sh`: dimension evaluation
  array.
- `quest_jobs/gaussian_support_recovery/topology.sh`: topology evaluation array.
- `quest_jobs/gaussian_support_recovery/submit_all.sh`: submits all three panel
  arrays.

Do not edit generated instances manually.  Regenerate data explicitly when the
design changes, and record the manually chosen penalty constant with each run.
