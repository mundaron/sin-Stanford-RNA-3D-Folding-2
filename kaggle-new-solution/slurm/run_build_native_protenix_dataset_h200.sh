#!/usr/bin/env bash
#SBATCH --job-name=ptx-v1-native-prep
#SBATCH --partition=batch-milan
#SBATCH -c 32
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_v1_native_prep_%j.out
#SBATCH --error=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_v1_native_prep_%j.err

set -euo pipefail

REPO_ROOT="/scratch/work/sethih1/RNA-prediction"
SOLUTION_ROOT="${REPO_ROOT}/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution"
CODE_DIR="${SOLUTION_ROOT}/code"
mkdir -p "${REPO_ROOT}/slurm_logs_rna"

VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/phys/sin/rna-dataset/venv/px-kaggle/bin/activate}"
if [[ -n "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PROTENIX_REPO_DIR="${PROTENIX_REPO_DIR:-${REPO_ROOT}/sin-Stanford-RNA-3D-Folding-2/external/Protenix-v1-official}"
RAW_STRUCTURE_ROOT="${RAW_STRUCTURE_ROOT:-/scratch/phys/sin/rna-dataset/PDB_RNA}"
MSA_ROOT="${MSA_ROOT:-/scratch/phys/sin/rna-dataset/MSA}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-/scratch/phys/sin/rna-dataset}"
CCD_ROOT="${CCD_ROOT:-/scratch/phys/sin/rna-dataset/protenix_ccd_cache}"
OFFICIAL_TEMPLATE="${OFFICIAL_TEMPLATE:-${PROTENIX_REPO_DIR}/examples/examples_with_rna_msa/example_9gmw_2.json}"
OUT_ROOT="${OUT_ROOT:-/scratch/phys/sin/rna-dataset/protenix_native_kaggle_full}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-0}"
DEFAULT_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-32}"
if [[ "${DEFAULT_NUM_WORKERS}" =~ ^[0-9]+$ ]] && [[ "${DEFAULT_NUM_WORKERS}" -gt 2 ]]; then
  DEFAULT_NUM_WORKERS="$((DEFAULT_NUM_WORKERS - 2))"
fi
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"

mkdir -p "${OUT_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  "${CODE_DIR}/build_native_protenix_dataset.py"
  --protenix-repo "${PROTENIX_REPO_DIR}"
  --raw-structure-root "${RAW_STRUCTURE_ROOT}"
  --msa-root "${MSA_ROOT}"
  --split-manifest "${SPLIT_MANIFEST}"
  --ccd-root "${CCD_ROOT}"
  --out-root "${OUT_ROOT}"
  --official-template "${OFFICIAL_TEMPLATE}"
  --num-workers "${NUM_WORKERS}"
)

if [[ "${LIMIT_PER_SPLIT}" =~ ^[0-9]+$ ]] && [[ "${LIMIT_PER_SPLIT}" -gt 0 ]]; then
  CMD+=(--limit-per-split "${LIMIT_PER_SPLIT}")
fi

echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Official Protenix repo: ${PROTENIX_REPO_DIR}"
echo "Raw structure root: ${RAW_STRUCTURE_ROOT}"
echo "MSA root: ${MSA_ROOT}"
echo "Split manifest: ${SPLIT_MANIFEST}"
echo "CCD root: ${CCD_ROOT}"
echo "Official template: ${OFFICIAL_TEMPLATE}"
echo "Output root: ${OUT_ROOT}"
echo "LIMIT_PER_SPLIT: ${LIMIT_PER_SPLIT}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
