#!/bin/bash
# Calibrate one penalty constant for each method on the smallest configuration.
# Each array task handles one of the ten pre-generated replications.
# Confirm the p32811 account and python39 environment before submission.
# This resource request is provisional; revise it after checking a pilot with seff.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --array=0-9
#SBATCH --job-name=sm_calibration
#SBATCH --output=experiments/quest_jobs/outlog/%x_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -e

module purge all
module load python-miniconda3
source activate python39
module load gurobi

REPLICATION="${SLURM_ARRAY_TASK_ID}"
PENALTY_CONSTANTS="$(python3 -m experiments.primary_graph_recovery_config)"

python3 -u -m experiments.Run_gaussian_experiments \
  --stage "calibration" \
  --study "primary_graph_recovery" \
  --configuration-list "erdos_renyi:20:40" \
  --rep-list "${REPLICATION}" \
  --job-name "penalty_calibration" \
  --results-csv "experiments_results/penalty_calibration/gaussian_primary_graph_recovery_calibration_rep${REPLICATION}.csv" \
  --method-list "sm_l0,sm_l1,graphl0,glasso" \
  --penalty-constant-list "${PENALTY_CONSTANTS}" \
  --time-limit "600" \
  --mip-gap "0.01" \
  --threads "8" \
  --verbose \
  --overwrite-results
