#!/bin/bash
# Submit the mixed-integer and SM-L1 multivariate-t ROC arrays.

set -euo pipefail

JOB_DIR="experiments/quest_jobs/nongaussian_support_recovery"

# L0_JOB_ID=$(sbatch --parsable --export=ALL,METHOD=sm_l0 "${JOB_DIR}/roc.sh")
L0_core_JOB_ID=$(sbatch --parsable --export=ALL,METHOD=sm_l0_core "${JOB_DIR}/roc.sh")
L0_MILP_JOB_ID=$(sbatch --parsable --export=ALL,METHOD=sm_l0_milp "${JOB_DIR}/roc.sh")
L1_JOB_ID=$(sbatch --parsable --export=ALL,METHOD=sm_l1 "${JOB_DIR}/roc.sh")
# echo "SM-L0 ROC array: ${L0_JOB_ID%%;*}"
echo "SM-L0 core ROC array: ${L0_core_JOB_ID%%;*}"
echo "SM-L0 support MILP ROC array: ${L0_MILP_JOB_ID%%;*}"
echo "SM-L1 ROC array: ${L1_JOB_ID%%;*}"
