#!/usr/bin/env bash
#SBATCH --job-name=ptx-native-train
#SBATCH --partition=gpu-h200-141g-short
# SBATCH --partition=gpu-b300-288g-short
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_native_train_%j.out
#SBATCH --error=/scratch/work/sethih1/RNA-prediction/slurm_logs_rna/ptx_native_train_%j.err

set -euo pipefail

REPO_ROOT="/scratch/work/sethih1/RNA-prediction"
SOLUTION_ROOT="${REPO_ROOT}/sin-Stanford-RNA-3D-Folding-2/kaggle-part-2-solution"
SOLUTION_ROOT="/scratch/phys/sin/rna-dataset/proteinix_outputs"
CODE_DIR="${SOLUTION_ROOT}/code"
REF_DIR="${SOLUTION_ROOT}/references"
mkdir -p "${REPO_ROOT}/slurm_logs_rna"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PROTENIX_REPO_DIR="${PROTENIX_REPO_DIR:-${REF_DIR}/Protenix-RNA-Kaggle}"
CACHE_ROOT="${CACHE_ROOT:-${REF_DIR}/preprocessed_data}"
RUN_BASE_DIR="${RUN_BASE_DIR:-${REF_DIR}/protenix_runs}"
RUN_NAME="${RUN_NAME:-kaggle_native_ft}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REF_DIR}/protenix_checkpoint/protenix_base_20250630_v1.0.0.pt}"
EMA_CHECKPOINT_PATH="${EMA_CHECKPOINT_PATH:-}"
USE_MSA="${USE_MSA:-false}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-protenix-kaggle-native}"
WANDB_ID="${WANDB_ID:-}"
WANDB_MODE="${WANDB_MODE:-online}"
MAX_STEPS="${MAX_STEPS:-5000}"
TRAIN_CROP_SIZE="${TRAIN_CROP_SIZE:-416}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
LR="${LR:-5e-5}"
SEED="${SEED:-42}"
EMA_DECAY="${EMA_DECAY:-0.995}"
PROTENIX_DISABLE_EVAL="${PROTENIX_DISABLE_EVAL:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p "${RUN_BASE_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "${CODE_DIR}/launch_kaggle_native_train.py"
  --protenix-repo "${PROTENIX_REPO_DIR}"
  --cache-root "${CACHE_ROOT}"
  --split train
  --
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --base_dir "${RUN_BASE_DIR}"
  --dtype bf16
  --use_msa "${USE_MSA}"
  --project "${WANDB_PROJECT}"
  --use_wandb "${USE_WANDB}"
  --diffusion_batch_size 8
  --eval_interval "${EVAL_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
  --ema_decay "${EMA_DECAY}"
  --train_crop_size "${TRAIN_CROP_SIZE}"
  --max_steps "${MAX_STEPS}"
  --warmup_steps 50
  --lr "${LR}"
  --sample_diffusion.N_step 20
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
export PROTENIX_KAGGLE_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
export PROTENIX_KAGGLE_NUM_WORKERS="${NUM_WORKERS}"
if [[ "${USE_WANDB}" == "true" ]]; then
  export WANDB_MODE
fi

echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Protenix repo: ${PROTENIX_REPO_DIR}"
echo "Cache root: ${CACHE_ROOT}"
echo "Run base dir: ${RUN_BASE_DIR}"
echo "USE_WANDB: ${USE_WANDB}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "PROTENIX_DISABLE_EVAL: ${PROTENIX_DISABLE_EVAL}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
