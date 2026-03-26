#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/work/sethih1/RNA-prediction"
SOLUTION_ROOT="${REPO_ROOT}/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution"
SLURM_SCRIPT="${SOLUTION_ROOT}/slurm/run_build_native_protenix_dataset_h200.sh"

VENV_ACTIVATE="${VENV_ACTIVATE:-/scratch/phys/sin/rna-dataset/venv/px-kaggle/bin/activate}"
PROTENIX_REPO_DIR="${PROTENIX_REPO_DIR:-${REPO_ROOT}/sin-Stanford-RNA-3D-Folding-2/external/Protenix-v1-official}"
RAW_STRUCTURE_ROOT="${RAW_STRUCTURE_ROOT:-/scratch/phys/sin/rna-dataset/PDB_RNA}"
MSA_ROOT="${MSA_ROOT:-/scratch/phys/sin/rna-dataset/MSA}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-/scratch/phys/sin/rna-dataset}"
CCD_ROOT="${CCD_ROOT:-/scratch/phys/sin/rna-dataset/protenix_ccd_cache}"
OFFICIAL_TEMPLATE="${OFFICIAL_TEMPLATE:-${PROTENIX_REPO_DIR}/examples/examples_with_rna_msa/example_9gmw_2.json}"
OUT_ROOT="${OUT_ROOT:-/scratch/phys/sin/rna-dataset/protenix_native_kaggle_full}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-0}"
NUM_WORKERS="${NUM_WORKERS:-30}"

SBATCH_EXPORTS=(
  "ALL"
  "VENV_ACTIVATE=${VENV_ACTIVATE}"
  "PROTENIX_REPO_DIR=${PROTENIX_REPO_DIR}"
  "RAW_STRUCTURE_ROOT=${RAW_STRUCTURE_ROOT}"
  "MSA_ROOT=${MSA_ROOT}"
  "SPLIT_MANIFEST=${SPLIT_MANIFEST}"
  "CCD_ROOT=${CCD_ROOT}"
  "OFFICIAL_TEMPLATE=${OFFICIAL_TEMPLATE}"
  "OUT_ROOT=${OUT_ROOT}"
  "LIMIT_PER_SPLIT=${LIMIT_PER_SPLIT}"
  "NUM_WORKERS=${NUM_WORKERS}"
)

EXPORT_STRING="$(IFS=,; printf '%s' "${SBATCH_EXPORTS[*]}")"

echo "Submitting ${SLURM_SCRIPT}"
echo "PARTITION=batch-milan"
echo "RAW_STRUCTURE_ROOT=${RAW_STRUCTURE_ROOT}"
echo "MSA_ROOT=${MSA_ROOT}"
echo "SPLIT_MANIFEST=${SPLIT_MANIFEST}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LIMIT_PER_SPLIT=${LIMIT_PER_SPLIT}"
echo "NUM_WORKERS=${NUM_WORKERS}"

sbatch --export="${EXPORT_STRING}" "$@" "${SLURM_SCRIPT}"
