#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Generic runner used by cluster/submit_all.sh. Runs "$@" from the clone the job
# was submitted from, with that clone's src/ first on the path, and records the
# exact code version in the log.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-retromorgan}"

export PYTHONPATH="$(pwd)/src"
export PYTHONUNBUFFERED=1
THREADS="${THREADS:-1}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

echo "job:       ${SLURM_JOB_NAME:-NA} (${SLURM_JOB_ID:-NA}) on $(hostname)"
echo "start:     $(date)"
echo "commit:    $(git rev-parse HEAD)"
python -c "import morganrxn; print('morganrxn:', morganrxn.__file__)"
echo "command:   $*"
echo "============================================================"

"$@"

echo "============================================================"
echo "end:       $(date)"
echo "Job finished."
