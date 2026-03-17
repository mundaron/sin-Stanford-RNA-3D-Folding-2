# kaggle-part-2-solution

This folder is the cleaned-up view of the Kaggle Part 2 Protenix finetuning setup we actually used.
It does not move your heavy data or the external Protenix fork. Instead, it keeps the working code here and links the large runtime folders into one readable place.

## What Is In Here

- `code/build_kaggle_native_cache.py`
  - Preprocesses the Kaggle Part 2 CSVs into the cache used for finetuning.
- `code/kaggle_rna_dataset_comp_native_runtime.py`
  - The working runtime dataset loader used by training.
- `code/launch_kaggle_native_train.py`
  - Launches the Kaggle Protenix fork with the runtime loader, CCD cache, and zero-worker DataLoader setup.
- `slurm/run_kaggle_native_prep_h200.sh`
  - H200 job for preprocessing the Kaggle CSVs.
- `slurm/run_kaggle_native_train_h200.sh`
  - H200 job for finetuning.
- `references/`
  - Symlinks to the external fork, dataset, cache, checkpoints, runs, CCD cache, and venv.

## Folder Layout

```text
kaggle-part-2-solution/
  code/
    build_kaggle_native_cache.py
    kaggle_rna_dataset_comp_native_runtime.py
    launch_kaggle_native_train.py
  slurm/
    run_kaggle_native_prep_h200.sh
    run_kaggle_native_train_h200.sh
  references/
    Protenix-RNA-Kaggle -> ../external/Protenix-RNA-Kaggle
    dataset -> /scratch/phys/sin/rna-dataset
    preprocessed_data -> /scratch/phys/sin/rna-dataset/preprocessed_data
    protenix_checkpoint -> /scratch/phys/sin/rna-dataset/protenix_checkpoint
    protenix_ccd_cache -> /scratch/phys/sin/rna-dataset/protenix_ccd_cache
    protenix_runs -> /scratch/phys/sin/rna-dataset/protenix_runs
    protenix-kaggle-lite-venv -> /scratch/phys/sin/rna-dataset/protenix-kaggle-lite-venv
```

## How The Imports Work

The important import trick is in `code/launch_kaggle_native_train.py`.
It first puts the Kaggle Protenix fork on `sys.path`, then it injects our local runtime loader as:

- `protenix.data.rna_dataset_comp`

That means the fork still imports its expected module name, but it receives our Kaggle-native loader.
This is why the code is collected here, but the actual Protenix model code still lives in:

- `references/Protenix-RNA-Kaggle`

## What The Training Uses

The finetuning path reads from these linked runtime folders:

- Dataset root: `references/dataset`
- Preprocessed cache: `references/preprocessed_data`
- Pretrained checkpoint: `references/protenix_checkpoint/model_v0.2.0.pt`
- CCD cache: `references/protenix_ccd_cache`
- Output runs: `references/protenix_runs`
- External fork: `references/Protenix-RNA-Kaggle`

## Typical Commands

### 1. Preprocess the Kaggle CSVs

```bash
sbatch --export=ALL,VENV_ACTIVATE=/scratch/phys/sin/rna-dataset/protenix-kaggle-lite-venv/bin/activate /scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-part-2-solution/slurm/run_kaggle_native_prep_h200.sh
```

### 2. Start finetuning

```bash
sbatch --export=ALL,VENV_ACTIVATE=/scratch/phys/sin/rna-dataset/protenix-kaggle-lite-venv/bin/activate /scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-part-2-solution/slurm/run_kaggle_native_train_h200.sh
```

### 3. Resume safely from the latest non-broken checkpoint

For the interrupted run, the latest usable normal checkpoint is:

- `/scratch/phys/sin/rna-dataset/protenix_runs/kaggle_native_ft_stage1_nomsa_1000_20260317_132920/checkpoints/599.pt`

A safer resume example is:

```bash
sbatch --export=ALL,VENV_ACTIVATE=/scratch/phys/sin/rna-dataset/protenix-kaggle-lite-venv/bin/activate,CHECKPOINT_PATH=/scratch/phys/sin/rna-dataset/protenix_runs/kaggle_native_ft_stage1_nomsa_1000_20260317_132920/checkpoints/599.pt,EMA_CHECKPOINT_PATH=,EMA_DECAY=-1,CHECKPOINT_INTERVAL=300,RUN_NAME=kaggle_native_ft_resume_599 /scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-part-2-solution/slurm/run_kaggle_native_train_h200.sh
```

This resume recipe does three things to reduce storage pressure:

- loads from `599.pt`
- disables EMA checkpointing with `EMA_DECAY=-1`
- saves less often with `CHECKPOINT_INTERVAL=300`

## Notes

- Validation is currently disabled in the launcher by default with `PROTENIX_DISABLE_EVAL=1` because the fork's eval path still expects a different batch shape.
- `wandb` logging is supported through `USE_WANDB=true`, `WANDB_PROJECT=...`, and optionally `WANDB_ID=...` and `WANDB_MODE=online|offline`.
- The runtime loader is copied here from the working version under `/scratch/phys/sin/rna-dataset` so this folder stays understandable and self-contained.
