#!/bin/bash
# Vary p at n/p = 2; one array task contains one independent replication.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --array=0-99
#SBATCH --job-name=sm_problem_size
#SBATCH --output=experiments/quest_jobs/outlog/sm_problem_size_%A_%a.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

module purge all
module load python-miniconda3
source activate python39
module load gurobi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"

# Separate calls preserve n/p = 2 without creating an unintended Cartesian product.
for P in 10 20 30 40; do
  N=$((2 * P))
  python3 -m experiments.generate_gaussian_experiments \
    --study "problem_size" \
    --topology-list "erdos_renyi" \
    --p-list "${P}" \
    --n-list "${N}" \
    --degree-list "4" \
    --signal-list "0.2" \
    --condition-list "10" \
    --rep-list "${SLURM_ARRAY_TASK_ID}" \
    --manifest-name "manifest_p${P}_rep${SLURM_ARRAY_TASK_ID}"
done

python3 -m experiments.Run_gaussian_experiments \
  --study "problem_size" \
  --rep-list "${SLURM_ARRAY_TASK_ID}" \
  --method-list "sm_l0,sm_l0_core,sm_l1" \
  --penalty-multiplier-list "0.25,0.5,1,2,4" \
  --time-limit "600" \
  --mip-gap "0.01" \
  --threads "8" \
  --job-name "problem_size_rep${SLURM_ARRAY_TASK_ID}" \
  --overwrite-results
