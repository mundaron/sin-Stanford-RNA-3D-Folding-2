#!/usr/bin/env bash
#SBATCH --job-name=ptx-native-prep
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_native_prep_%j.out
#SBATCH --error=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_native_prep_%j.err

set -euo pipefail

REPO_ROOT="/scratch/work/sethih1/RNA-prediction"
SOLUTION_ROOT="${REPO_ROOT}/sin-Stanford-RNA-3D-Folding-2/kaggle-part-2-solution"
CODE_DIR="${SOLUTION_ROOT}/code"
mkdir -p "${REPO_ROOT}/slurm_logs_rna"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-/scratch/phys/sin/rna-dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/phys/sin/rna-dataset/preprocessed_data}"

CMD=(
  "${PYTHON_BIN}"
  "${CODE_DIR}/build_kaggle_native_cache.py"
  --data-dir "${DATA_DIR}"
  --output-root "${OUTPUT_ROOT}"
)

echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
