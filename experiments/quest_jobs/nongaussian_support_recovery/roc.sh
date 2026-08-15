#!/bin/bash
# Fit one method over the common ROC path on one multivariate-t dataset.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --job-name=sm_t3_roc
#SBATCH --output=experiments/quest_jobs/outlog/%x_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -euo pipefail

module purge all
module load python-miniconda3
source activate python39
module load gurobi

REPLICATION="${SLURM_ARRAY_TASK_ID}"
METHOD="${METHOD:-sm_l0_milp}"
# Supported METHOD values include sm_l0, sm_l0_core, sm_l0_milp, and sm_l1.
TOPOLOGY="${TOPOLOGY:-erdos_renyi}"
P="${P:-500}"
N="${N:-1000}"
PENALTY_CONSTANTS="${PENALTY_CONSTANTS:-0.1,0.2,0.5,1,1.2,1.4,1.6,1.8,2,2.2,2.5,3,5,10}"
SCREEN_ALPHA="${SCREEN_ALPHA:-0.1}"
TIME_LIMIT="${TIME_LIMIT:-1200}"
MIP_GAP="${MIP_GAP:-0.0001}"
RESULTS_DIR="experiments_results/nongaussian_robustness_roc/topology=${TOPOLOGY}_p=${P}_n=${N}"

python3 -u -m experiments.Run_nongaussian_roc \
  --stage "evaluation" \
  --rep-list "${REPLICATION}" \
  --job-name "roc" \
  --topology "${TOPOLOGY}" \
  --p "${P}" \
  --n "${N}" \
  --method-list "${METHOD}" \
  --penalty-constant-list "${PENALTY_CONSTANTS}" \
  --candidate-rule "spearman_graphical_lasso" --screen-alpha "${SCREEN_ALPHA}" \
  --results-csv "${RESULTS_DIR}/${METHOD}_rep${REPLICATION}.csv" \
  --time-limit "${TIME_LIMIT}" \
  --mip-gap "${MIP_GAP}" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --verbose \
  --overwrite-results
