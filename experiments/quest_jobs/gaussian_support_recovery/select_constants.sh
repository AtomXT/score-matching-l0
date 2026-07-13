#!/bin/bash
# Select one constant per method after all calibration array tasks finish.
# Submit through submit_all.sh so the dependency is installed automatically.
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --job-name=sm_select_constants
#SBATCH --output=experiments/quest_jobs/outlog/%x_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -e

module purge all
module load python-miniconda3
source activate python39

python3 -u analysis/select_penalty_constants.py
