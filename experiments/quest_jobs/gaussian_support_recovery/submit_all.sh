#!/bin/bash
# Submit the three evaluation panels from the project root.
# Set each job's method and manually chosen penalty constant before running this.

set -e

JOB_DIR="experiments/quest_jobs/gaussian_support_recovery"

SAMPLE_SIZE_JOB_ID=$(sbatch --parsable "${JOB_DIR}/sample_size.sh")
SAMPLE_SIZE_JOB_ID="${SAMPLE_SIZE_JOB_ID%%;*}"

DIMENSION_JOB_ID=$(sbatch --parsable "${JOB_DIR}/dimension.sh")
DIMENSION_JOB_ID="${DIMENSION_JOB_ID%%;*}"

TOPOLOGY_JOB_ID=$(sbatch --parsable "${JOB_DIR}/topology.sh")
TOPOLOGY_JOB_ID="${TOPOLOGY_JOB_ID%%;*}"

echo "sample-size array: ${SAMPLE_SIZE_JOB_ID}"
echo "dimension array: ${DIMENSION_JOB_ID}"
echo "topology array: ${TOPOLOGY_JOB_ID}"
