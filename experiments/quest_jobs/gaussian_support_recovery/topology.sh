#!/bin/bash
# Evaluate all ten replications of the primary topology panel.
# Each array task handles exactly one replication.  Generate the datasets and
# select the penalty constants before submitting this job.
# Confirm the p32811 account and python39 environment before submission.
# This resource request is provisional; revise it after checking a pilot with seff.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --array=0-9
#SBATCH --job-name=sm_topology
#SBATCH --output=experiments/quest_jobs/outlog/%x_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -e

module purge all
module load python-miniconda3
source activate python39
module load gurobi

REPLICATION="${SLURM_ARRAY_TASK_ID}"

python3 -u -m experiments.Run_gaussian_topology \
  --stage "evaluation" \
  --rep-list "${REPLICATION}" \
  --job-name "topology" \
  --results-csv "experiments_results/gaussian_primary_graph_recovery_topology_rep${REPLICATION}.csv" \
  --method-list "sm_l0,sm_l1,graphl0,glasso" \
  --penalty-constants-json "experiments_results/penalty_calibration/selected_constants.json" \
  --time-limit "600" \
  --mip-gap "0.01" \
  --threads "8" \
  --verbose \
  --overwrite-results
