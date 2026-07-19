#!/bin/bash
# Fit one method over the common ROC penalty path; each task uses one dataset.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --job-name=sm_roc
#SBATCH --output=experiments/quest_jobs/outlog/%x_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -e

module purge all
module load python-miniconda3
source activate python39
module load gurobi

REPLICATION="${SLURM_ARRAY_TASK_ID}"
METHOD="${METHOD:-sm_l1}"
TOPOLOGY="${TOPOLOGY:-erdos_renyi}"
P="${P:-500}"
N="${N:-250}"
PENALTY_CONSTANTS="0.1,0.2,0.5,1,1.2,1.4,1.6,1.8,2,2.2,2.5,3,5,10"
RESULTS_DIR="experiments_results/gaussian_primary_graph_recovery/topology=${TOPOLOGY}_p=${P}_n=${N}"

# To enable screening, replace the candidate rule below with:
#   --candidate-rule "graphical_lasso" --screen-alpha "0.08"
python3 -u -m experiments.Run_gaussian_roc \
  --stage "evaluation" \
  --rep-list "${REPLICATION}" \
  --job-name "roc" \
  --topology "${TOPOLOGY}" \
  --p "${P}" \
  --n "${N}" \
  --method-list "${METHOD}" \
  --penalty-constant-list "${PENALTY_CONSTANTS}" \
  --candidate-rule "graphical_lasso" --screen-alpha "0.15" \
  --results-csv "${RESULTS_DIR}/${METHOD}_rep${REPLICATION}.csv" \
  --time-limit "1200" \
  --mip-gap "0.001" \
  --threads "8" \
  --verbose \
  --overwrite-results
