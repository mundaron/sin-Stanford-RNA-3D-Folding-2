#!/usr/bin/env bash
#SBATCH --job-name=ptx-v1-rna-ft
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --partition=gpu-h100-80g-short
#SBATCH --partition=gpu-a100-80g
# SBATCH --partition=gpu-debug
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_v1_rna_train_%j.out
#SBATCH --error=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_v1_rna_train_%j.err

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
DATA_MODE="${DATA_MODE:-coarse}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/phys/sin/rna-dataset/preprocessed_data}"
NATIVE_DATA_ROOT="${NATIVE_DATA_ROOT:-/scratch/phys/sin/rna-dataset/protenix_native_kaggle_full_chainfix}"
CCD_ROOT="${CCD_ROOT:-/scratch/phys/sin/rna-dataset/protenix_ccd_cache}"
RUN_BASE_DIR="${RUN_BASE_DIR:-/scratch/phys/sin/rna-dataset/protenix_runs}"
RUN_NAME="${RUN_NAME:-protenix_native_chainfix_stage1}"
CHECKPOINT_SAVE_DIR="${CHECKPOINT_SAVE_DIR:-/scratch/phys/sin/rna-dataset/models_stage1_rna_chainfix}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch/phys/sin/rna-dataset/protenix_checkpoint/protenix_base_20250630_v1.0.0.pt}"
EMA_CHECKPOINT_PATH="${EMA_CHECKPOINT_PATH:-}"
USE_MSA="${USE_MSA:-true}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-protenix-v1-rna}"
WANDB_ID="${WANDB_ID:-}"
WANDB_MODE="${WANDB_MODE:-online}"
MAX_STEPS="${MAX_STEPS:-5000}"
TRAIN_CROP_SIZE="${TRAIN_CROP_SIZE:-416}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-4}"
SEED="${SEED:-42}"
EMA_DECAY="${EMA_DECAY:-0.995}"
PROTENIX_DISABLE_EVAL="${PROTENIX_DISABLE_EVAL:-0}"
PROTENIX_ENABLE_VALIDATION_EVAL="${PROTENIX_ENABLE_VALIDATION_EVAL:-1}"
LAYERNORM_TYPE="${LAYERNORM_TYPE:-openfold}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-}"
PROTENIX_NATIVE_TRAIN_MAX_N_TOKEN="${PROTENIX_NATIVE_TRAIN_MAX_N_TOKEN:-768}"
PROTENIX_NATIVE_TRAIN_EXCLUDED_MOL_GROUPS="${PROTENIX_NATIVE_TRAIN_EXCLUDED_MOL_GROUPS:-prot_prot,intra_prot,intra_ligand,ligand_prot,ligand_ligand}"

if [[ -z "${CC:-}" ]]; then
  if ! [[ -x /usr/bin/gcc || -x /usr/bin/cc ]] && ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    fi
    if type module >/dev/null 2>&1; then
      module load triton/2025.1-gcc >/dev/null 2>&1 || true
      module load gcc/13.3.0 >/dev/null 2>&1 || true
    fi
  fi

  if [[ -x /usr/bin/gcc ]]; then
    CC="/usr/bin/gcc"
  elif [[ -x /usr/bin/cc ]]; then
    CC="/usr/bin/cc"
  elif command -v gcc >/dev/null 2>&1; then
    CC="$(command -v gcc)"
  elif command -v cc >/dev/null 2>&1; then
    CC="$(command -v cc)"
  fi
fi

if [[ -z "${CC:-}" ]]; then
  echo "No C compiler found on the node even after trying triton/2025.1-gcc + gcc/13.3.0. Set CC=/path/to/gcc before launching." >&2
  exit 1
fi

export CC

mkdir -p "${RUN_BASE_DIR}"
if [[ -n "${CHECKPOINT_SAVE_DIR}" ]]; then
  mkdir -p "${CHECKPOINT_SAVE_DIR}"
fi

if [[ -z "${NUM_GPUS}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a CUDA_DEVICE_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS="${#CUDA_DEVICE_IDS[@]}"
  else
    NUM_GPUS=1
  fi
fi

LAUNCH_PREFIX=("${PYTHON_BIN}")
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCH_PREFIX=(
    "${PYTHON_BIN}"
    -m
    torch.distributed.run
    --standalone
    "--nproc_per_node=${NUM_GPUS}"
  )
fi

CMD=(
  "${LAUNCH_PREFIX[@]}"
  "${CODE_DIR}/launch_official_protenix_train.py"
  --protenix-repo "${PROTENIX_REPO_DIR}"
  --data-mode "${DATA_MODE}"
  --cache-root "${CACHE_ROOT}"
  --native-data-root "${NATIVE_DATA_ROOT}"
  --ccd-root "${CCD_ROOT}"
  --
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --base_dir "${RUN_BASE_DIR}"
  --dtype bf16
  --project "${WANDB_PROJECT}"
  --use_wandb "${USE_WANDB}"
  --diffusion_batch_size "${DIFFUSION_BATCH_SIZE}"
  --eval_interval "${EVAL_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
  --ema_decay "${EMA_DECAY}"
  --train_crop_size "${TRAIN_CROP_SIZE}"
  --max_steps "${MAX_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --iters_to_accumulate "${ITERS_TO_ACCUMULATE}"
  --lr "${LR}"
  --sample_diffusion.N_step 20
  --loss.weight.alpha_confidence 0.0001
  --loss.weight.alpha_distogram 0.03
  --loss.weight.alpha_bond 0.0
  --loss.weight.smooth_lddt 1.0
)

if [[ -n "${WANDB_ID}" ]]; then
  CMD+=(--wandb_id "${WANDB_ID}")
fi

if [[ -f "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--load_checkpoint_path "${CHECKPOINT_PATH}")
else
  echo "Checkpoint not found at ${CHECKPOINT_PATH}; training will start from scratch."
fi

if [[ -n "${EMA_CHECKPOINT_PATH}" && -f "${EMA_CHECKPOINT_PATH}" ]]; then
  CMD+=(--load_ema_checkpoint_path "${EMA_CHECKPOINT_PATH}")
fi

export PROTENIX_DISABLE_EVAL
export PROTENIX_ENABLE_VALIDATION_EVAL
export LAYERNORM_TYPE
export PROTENIX_NATIVE_TRAIN_MAX_N_TOKEN
export PROTENIX_NATIVE_TRAIN_EXCLUDED_MOL_GROUPS
export PROTENIX_USE_MSA="${USE_MSA}"
export PROTENIX_RNA_NUM_WORKERS="${NUM_WORKERS}"
export PROTENIX_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
if [[ -n "${CHECKPOINT_SAVE_DIR}" ]]; then
  export PROTENIX_CHECKPOINT_DIR="${CHECKPOINT_SAVE_DIR}"
fi
if [[ "${USE_WANDB}" == "true" ]]; then
  export WANDB_MODE
fi

echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Official Protenix repo: ${PROTENIX_REPO_DIR}"
echo "DATA_MODE: ${DATA_MODE}"
echo "RNA cache root: ${CACHE_ROOT}"
echo "Native data root: ${NATIVE_DATA_ROOT}"
echo "CCD root: ${CCD_ROOT}"
echo "Run base dir: ${RUN_BASE_DIR}"
echo "Checkpoint save dir: ${CHECKPOINT_SAVE_DIR:-<default under run dir>}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "USE_WANDB: ${USE_WANDB}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "PROTENIX_DISABLE_EVAL: ${PROTENIX_DISABLE_EVAL}"
echo "PROTENIX_ENABLE_VALIDATION_EVAL: ${PROTENIX_ENABLE_VALIDATION_EVAL}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "DIFFUSION_BATCH_SIZE: ${DIFFUSION_BATCH_SIZE}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "WARMUP_STEPS: ${WARMUP_STEPS}"
echo "ITERS_TO_ACCUMULATE: ${ITERS_TO_ACCUMULATE}"
echo "PROTENIX_NATIVE_TRAIN_MAX_N_TOKEN: ${PROTENIX_NATIVE_TRAIN_MAX_N_TOKEN}"
echo "PROTENIX_NATIVE_TRAIN_EXCLUDED_MOL_GROUPS: ${PROTENIX_NATIVE_TRAIN_EXCLUDED_MOL_GROUPS}"
echo "Effective complexes/update: $(( NUM_GPUS * ITERS_TO_ACCUMULATE ))"
echo "CC: ${CC}"
if [[ "${DATA_MODE}" == "native" ]]; then
  echo "Note: official native training uses one complex per rank per step."
  echo "      TRAIN_BATCH_SIZE does not create a true 32-complex batch in native mode."
  echo "      Use NUM_GPUS and ITERS_TO_ACCUMULATE to scale the effective batch instead."
fi
echo "Command: ${CMD[*]}"

"${CMD[@]}"
