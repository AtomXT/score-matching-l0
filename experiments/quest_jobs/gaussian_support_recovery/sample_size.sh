#!/bin/bash
# Evaluate all ten replications of the primary sample-size panel.
# Each array task handles exactly one replication.  Set the method and manually
# chosen penalty constant below before submitting this job.
# Confirm the p32811 account and python39 environment before submission.
# This resource request is provisional; revise it after checking a pilot with seff.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --array=0-9
#SBATCH --job-name=sm_sample_size
#SBATCH --output=experiments/quest_jobs/outlog/%x_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -e

module purge all
module load python-miniconda3
source activate python39
module load gurobi

REPLICATION="${SLURM_ARRAY_TASK_ID}"
METHOD="sm_l1"  # sm_l0, sm_l0_core, sm_l1, graphl0, or glasso
PENALTY_CONSTANT="1"

python3 -u -m experiments.Run_gaussian_sample_size \
  --stage "evaluation" \
  --rep-list "${REPLICATION}" \
  --job-name "sample_size" \
  --configuration-list "erdos_renyi:40:20;erdos_renyi:40:40;erdos_renyi:40:80;erdos_renyi:40:160" \
  --results-csv "experiments_results/gaussian_primary_graph_recovery_sample_size_${METHOD}_rep${REPLICATION}.csv" \
  --method-list "${METHOD}" \
  --penalty-constant-list "${PENALTY_CONSTANT}" \
  --time-limit "600" \
  --mip-gap "0.01" \
  --threads "8" \
  --verbose \
  --overwrite-results
