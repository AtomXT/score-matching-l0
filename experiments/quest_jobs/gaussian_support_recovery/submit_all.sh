#!/bin/bash
# Submit the calibration, selection, and evaluation stages from the project root.
# The afterok dependencies prevent evaluation with missing or failed calibration.

set -e

JOB_DIR="experiments/quest_jobs/gaussian_support_recovery"

CALIBRATION_JOB_ID=$(sbatch --parsable "${JOB_DIR}/calibration.sh")
CALIBRATION_JOB_ID="${CALIBRATION_JOB_ID%%;*}"

SELECTION_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB_ID}" \
  "${JOB_DIR}/select_constants.sh")
SELECTION_JOB_ID="${SELECTION_JOB_ID%%;*}"

SAMPLE_SIZE_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${SELECTION_JOB_ID}" \
  "${JOB_DIR}/sample_size.sh")
SAMPLE_SIZE_JOB_ID="${SAMPLE_SIZE_JOB_ID%%;*}"

DIMENSION_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${SELECTION_JOB_ID}" \
  "${JOB_DIR}/dimension.sh")
DIMENSION_JOB_ID="${DIMENSION_JOB_ID%%;*}"

TOPOLOGY_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${SELECTION_JOB_ID}" \
  "${JOB_DIR}/topology.sh")
TOPOLOGY_JOB_ID="${TOPOLOGY_JOB_ID%%;*}"

echo "calibration array: ${CALIBRATION_JOB_ID}"
echo "constant selection: ${SELECTION_JOB_ID}"
echo "sample-size array: ${SAMPLE_SIZE_JOB_ID}"
echo "dimension array: ${DIMENSION_JOB_ID}"
echo "topology array: ${TOPOLOGY_JOB_ID}"
