# Experiment code guide

This folder contains the data generators, fitting programs, and Quest submission files
for the first experiment in the paper: Gaussian graph recovery.  The experiment has
three panels:

| Panel | Question | Settings |
|---|---|---|
| Sample size | Does recovery improve as more observations are available? | `p=40`, `n=20,40,80,160` |
| Dimension | How does recovery change with problem dimension at a fixed ratio? | `(p,n)=(20,40),(40,80),(60,120)` |
| Topology | Is the comparison sensitive to degree heterogeneity? | Erdős--Rényi-type and scale-free graphs at `(p,n)=(40,80)` |

Every setting uses target average degree `4`, target minimum signal `0.20`, target
condition number `10`, and ten independent replications.  The central
Erdős--Rényi-type setting `(p,n)=(40,80)` is shared by all three panels and is stored only
once.

The workflow separates data generation from fitting:

```text
generate saved datasets  ->  run all methods on the same datasets  ->  summarize CSV files
```

## Quick local test

Each panel runner can be opened in PyCharm and run with the green **Run** button.  No
arguments are needed:

```text
Run_gaussian_sample_size.py
Run_gaussian_dimension.py
Run_gaussian_topology.py
```

A no-argument run does **not** launch the full experiment.  It creates one deterministic
`p=8,n=16` test instance, fits SM--L1 at one penalty value, and writes a small result
file.  SM--L1 is used for this smoke test so that it finishes in a few seconds and does
not require a local Gurobi installation.  The automatically assigned job names are:

```text
sample_size_test_run
dimension_test_run
topology_test_run
```

The test data are saved below `data/gaussian_experiments/<panel>_test_run/`, and results
are saved below `experiments_results/`.  Repeating a no-argument run safely replaces its
test result file.

The same tests can be run from the project root:

```bash
python3 experiments/Run_gaussian_sample_size.py
python3 experiments/Run_gaussian_dimension.py
python3 experiments/Run_gaussian_topology.py
```

## Panel entry points

Each panel has one data generator and one runner.  The small wrapper files contain only
the panel identity; shared behavior is implemented in `primary_panel_workflow.py` so that
the settings cannot drift between panels.

### Sample-size panel

Generate all four settings and all ten replications:

```bash
python3 experiments/generate_gaussian_sample_size_data.py
```

Run a quick no-argument test:

```bash
python3 experiments/Run_gaussian_sample_size.py
```

The full Quest job is:

```text
quest_jobs/gaussian_support_recovery/sample_size.sh
```

### Dimension panel

Generate all three settings and all ten replications:

```bash
python3 experiments/generate_gaussian_dimension_data.py
```

Run a quick no-argument test:

```bash
python3 experiments/Run_gaussian_dimension.py
```

The full Quest job is:

```text
quest_jobs/gaussian_support_recovery/dimension.sh
```

### Topology panel

Generate both settings and all ten replications:

```bash
python3 experiments/generate_gaussian_topology_data.py
```

Run a quick no-argument test:

```bash
python3 experiments/Run_gaussian_topology.py
```

The full Quest job is:

```text
quest_jobs/gaussian_support_recovery/topology.sh
```

### Generate the complete first experiment at once

`generate_primary_graph_recovery_data.py` is a convenience entry point that generates
all seven unique settings and all ten replications in one local run:

```bash
python3 experiments/generate_primary_graph_recovery_data.py
```

It writes `design.json` and a 70-instance `manifest.json` under:

```text
data/gaussian_experiments/primary_graph_recovery/
```

Existing datasets are not overwritten unless the relevant generator receives
`--overwrite` or `OVERWRITE_EXISTING` is changed deliberately.

## Running the full experiment on Quest

Submit the three array scripts from the project root so that the relative log path
resolves correctly:

```bash
sbatch experiments/quest_jobs/gaussian_support_recovery/sample_size.sh
sbatch experiments/quest_jobs/gaussian_support_recovery/dimension.sh
sbatch experiments/quest_jobs/gaussian_support_recovery/topology.sh
```

Each script declares `#SBATCH --array=0-9`, so one submission creates ten independent
Slurm tasks.  Task 0 generates and fits replication 0, task 1 handles replication 1, and
so on through task 9.  The fixed runner job names are `sample_size`, `dimension`, and
`topology`; the array index appears only in the replication selector, manifest filename,
result filename, and log filename.  Thus parallel tasks cannot overwrite one another.

The Quest scripts follow the conventions used in the QP_indicator project:

- account `p32811`;
- the `python-miniconda3` module and `python39` environment;
- the Gurobi module;
- one node with eight allocated CPU cores; and
- email notifications to `tongxu2027@u.northwestern.edu`.

The account, environment name, and email should be confirmed before submission.  The
current per-task wall-time limits are 24 hours for one sample-size replication, 18 hours
for one dimension replication, and 12 hours for one topology replication.  These are
maximum limits rather than expected runtimes.  The 16-GB and eight-core requests remain
provisional.  Run one representative array task first, inspect it with `seff`, and then
adjust wall time and memory.  The solver thread count must remain equal to the allocated
CPU count.  The `experiments/quest_jobs/outlog/` directory is already tracked because
Slurm requires the output directory to exist before submission.

Northwestern's current references are the
[Quest Slurm guide](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html)
and the
[Mamba/Conda guide](https://rcdsdocs.it.northwestern.edu/tutorials/software-management/conda-mamba-quest/mamba-conda-quest.html).

## Full runner options

Supplying `--job-name` switches a panel runner from its small local test to the complete
panel design.  For example, this command fits replication zero of the sample-size panel:

```bash
python3 -m experiments.Run_gaussian_sample_size \
  --rep-list 0 \
  --job-name sample_size \
  --results-csv experiments_results/gaussian_primary_graph_recovery_sample_size_rep0.csv \
  --method-list sm_l0,sm_l1,graphl0,glasso \
  --penalty-multiplier-list 0.03125,0.044,0.0625,0.088,0.125,0.177,0.25,0.354,0.5,0.707,1,1.414,2,2.828,4 \
  --time-limit 600 \
  --mip-gap 0.01 \
  --threads 8
```

The available method names are:

- `sm_l0`: proposed L0-penalized score-matching MIQP;
- `sm_l1`: loss-matched L1 score matching;
- `graphl0`: bundled L0-penalized Gaussian pseudolikelihood method; and
- `glasso`: graphical lasso from scikit-learn.

Gurobi and a valid license are required for `sm_l0`.  Install the packages in
`requirements.txt` in the Quest environment before submitting the full jobs.

## Generated instance contents

Each instance directory contains:

```text
dataset.npz
metadata.json
```

`dataset.npz` contains the training, validation, and test observations, the population
covariance and precision matrices, and the true adjacency matrix.  `metadata.json`
records the requested and achieved signal, condition number, degree, random seeds, and
other generation diagnostics.  Panel-specific manifest files list all generated
instances.

Do not edit generated instances manually.  If a design changes, use a new study name or
regenerate it explicitly with overwrite enabled.

## Shared implementation files

### `primary_panel_workflow.py`

Defines the three manuscript panels, the full four-method penalty path, the local test
behavior, panel filtering, and the command-line interfaces used by the six small panel
entry points.

### `generate_gaussian_experiments.py`

General Gaussian data generator used by every dedicated generator.  It constructs one
population graph and independent training, validation, and test samples, then records all
seeds and diagnostics.  It can also be used directly for pilot studies; unlike the panel
generators, direct use requires a `--study` argument.

### `Run_gaussian_experiments.py`

General fitting engine used by every dedicated runner.  It loads saved instances,
standardizes using training-sample quantities, applies one common candidate graph, fits
each requested method and penalty, and appends one result row immediately.  Direct use
requires a `--study` argument; normal users should use the panel runners.

### `gaussian_models.py`

Contains graph constructors and precision-matrix calibration routines.  It is a library
module and is not intended to be run directly.

### `common.py`

Contains shared parsing, atomic data storage, CSV output, screening, and evaluation
metrics.  It is a library module and is not intended to be run directly.

### `__init__.py`

Marks `experiments` as a Python package.  It has no runnable experiment.

## Summarizing results

After the Quest jobs finish, combine their CSV files with:

```bash
python3 analysis/summarize_gaussian_experiments.py --study primary_graph_recovery
```

The summarizer aggregates Monte Carlo means and standard errors.  Check error rows and
solver statuses before interpreting a summary; failures and uncertified time-limited
fits must not be silently discarded.
