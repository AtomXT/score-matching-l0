#!/bin/bash
# Compare the base and CORe-strengthened MIQPs on identical instances.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --array=0-19
#SBATCH --job-name=sm_formulation
#SBATCH --output=experiments/quest_jobs/outlog/sm_formulation_%A_%a.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

module purge all
module load python-miniconda3
source activate python39
module load gurobi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"

for P in 10 20 30 40; do
  N=$((2 * P))
  python3 -m experiments.generate_gaussian_experiments \
    --study "formulation" \
    --topology-list "chain,erdos_renyi" \
    --p-list "${P}" \
    --n-list "${N}" \
    --degree-list "4" \
    --signal-list "0.2" \
    --condition-list "3,10,30" \
    --rep-list "${SLURM_ARRAY_TASK_ID}" \
    --manifest-name "manifest_p${P}_rep${SLURM_ARRAY_TASK_ID}"
done

python3 -m experiments.Run_gaussian_experiments \
  --study "formulation" \
  --rep-list "${SLURM_ARRAY_TASK_ID}" \
  --method-list "sm_l0,sm_l0_core" \
  --penalty-multiplier-list "1" \
  --time-limit "3600" \
  --mip-gap "0.01" \
  --threads "8" \
  --job-name "formulation_rep${SLURM_ARRAY_TASK_ID}" \
  --overwrite-results
