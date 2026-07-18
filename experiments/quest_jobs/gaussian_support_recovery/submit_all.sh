#!/bin/bash
# Submit the SM-L0 and SM-L1 ROC arrays from the project root.

set -e

JOB_DIR="experiments/quest_jobs/gaussian_support_recovery"

# L0_JOB_ID=$(sbatch --parsable --export=ALL,METHOD=sm_l0 "${JOB_DIR}/roc.sh")
L0_core_JOB_ID=$(sbatch --parsable --export=ALL,METHOD=sm_l0_core "${JOB_DIR}/roc.sh")
L1_JOB_ID=$(sbatch --parsable --export=ALL,METHOD=sm_l1 "${JOB_DIR}/roc.sh")
# echo "SM-L0 ROC array: ${L0_JOB_ID%%;*}"
echo "SM-L0 core ROC array: ${L0_core_JOB_ID%%;*}"
echo "SM-L1 ROC array: ${L1_JOB_ID%%;*}"
