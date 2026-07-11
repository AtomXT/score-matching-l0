#!/bin/bash
# Vary n while holding the graph family, dimension, degree, signal, and condition fixed.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --array=0-99
#SBATCH --job-name=sm_sample_size
#SBATCH --output=experiments/quest_jobs/outlog/sm_sample_size_%A_%a.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

module purge all
module load python-miniconda3
source activate python39
module load gurobi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"

python3 -m experiments.generate_gaussian_experiments \
  --study "sample_size" \
  --topology-list "erdos_renyi" \
  --p-list "30" \
  --n-list "15,30,60,150" \
  --degree-list "4" \
  --signal-list "0.2" \
  --condition-list "10" \
  --rep-list "${SLURM_ARRAY_TASK_ID}" \
  --manifest-name "manifest_rep${SLURM_ARRAY_TASK_ID}"

python3 -m experiments.Run_gaussian_experiments \
  --study "sample_size" \
  --rep-list "${SLURM_ARRAY_TASK_ID}" \
  --method-list "sm_l0,sm_l0_core,sm_l1" \
  --penalty-multiplier-list "0.25,0.5,1,2,4" \
  --time-limit "600" \
  --mip-gap "0.01" \
  --threads "8" \
  --job-name "sample_size_rep${SLURM_ARRAY_TASK_ID}" \
  --overwrite-results
