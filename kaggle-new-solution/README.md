# Kaggle New Solution

This solution uses the official Protenix `v1.0.0` codepath with the
`protenix_base_20250630_v1.0.0.pt` checkpoint family. It now supports two data
modes:

- `coarse`: reuse the existing cached Kaggle RNA adapter with coarse `C1'` supervision
- `native`: build official-format `bioassembly + indices + RNA MSA` assets from the raw Kaggle mmCIF/MSA files and train through the untouched official loss

Why this exists:

- The older `Protenix-RNA-Kaggle` fork cannot load the official `v1.0.0`
  checkpoint because the model architecture changed.
- The checkpoint is a state dict, not a self-contained serialized model, so it
  still needs the matching official codebase to instantiate the model before
  loading weights.
- Our RNA cache only contains residue-level supervision. In practice that means
  this solution performs coarse RNA finetuning by supervising only the expanded
  `C1'` anchor positions rather than pretending to have full-atom RNA labels.

What is reused unchanged:

- `samples/train_samples.json`
- `samples/validation_samples.json`
- `labels/{train,validation}/*.npz`
- Precomputed RNA MSA paths embedded in the cached sample manifests

What is intentionally different from the old Kaggle fork solution:

- The launcher defaults `LAYERNORM_TYPE=openfold` so the official repo uses its built-in non-fused layer norm path instead of compiling the optional CUDA fused layer norm extension at import time

- Training runs through the official repo clone at
  `external/Protenix-v1-official`
- The runtime loader is local to this solution and adapts the cached RNA schema
  to the official trainer batch contract
- Chain permutation is bypassed in the launcher because the coarse RNA labels do
  not carry the full metadata that the official chain-permutation path expects
- The effective finetuning objective is coarse and explicit:
  `diffusion MSE + smooth-LDDT`

What is intentionally unsupported in this coarse setup:

- Full-atom RNA supervision
- Bond loss as a meaningful atom-level objective
- Distogram loss as a meaningful atom-level objective
- Confidence-style losses as primary finetuning supervision

The new entry points are:

- `code/launch_official_protenix_train.py`
- `code/kaggle_rna_official_runtime.py`
- `code/build_native_protenix_dataset.py`
- `slurm/run_official_protenix_train_h200.sh`
- `slurm/submit_official_protenix_train_h200.sh`

Which SLURM script should you use:

- use `slurm/submit_official_protenix_train_h200.sh` for normal runs
- it is the wrapper that calls `sbatch` and passes environment overrides through
- `slurm/run_official_protenix_train_h200.sh` is the actual batch script that runs on the node
- if you want the simplest path, use the submit wrapper instead of calling `sbatch` yourself

How to use the Kaggle dataset with official Protenix:

1. Decide which data mode you want.
   - `DATA_MODE=coarse`: use the existing Kaggle cache at `/scratch/phys/sin/rna-dataset/preprocessed_data`
   - `DATA_MODE=native`: convert the raw Kaggle mmCIF + MSA files into official Protenix-style assets and train with the official loss
2. Use the `px-kaggle` environment.
   - default env activate path: `/scratch/phys/sin/rna-dataset/venv/px-kaggle/bin/activate`
3. Make sure the official repo clone and checkpoint exist.
   - repo: `/scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/external/Protenix-v1-official`
   - checkpoint: `/scratch/phys/sin/rna-dataset/protenix_checkpoint/protenix_base_20250630_v1.0.0.pt`
4. For native mode only, build the official-format dataset first.
5. Submit the training job with the wrapper script.

Coarse mode:

- This is the quickest path if you want to reuse the existing Kaggle cache.
- It uses the local RNA adapter and coarse `C1'` supervision.

Example:

```bash
DATA_MODE=coarse \
RUN_NAME=kaggle_coarse_ft \
MAX_STEPS=5000 \
/scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution/slurm/submit_official_protenix_train_h200.sh
```

Native mode:

- This is the path to use if you want raw Kaggle structures plus existing MSA to go through the official Protenix data pipeline and official loss.
- Raw inputs expected by the builder:
  - structures: `/scratch/phys/sin/rna-dataset/PDB_RNA/*.cif`
  - MSA: `/scratch/phys/sin/rna-dataset/MSA/*.MSA.fasta`

Build the native dataset:

```bash
source /scratch/phys/sin/rna-dataset/venv/px-kaggle/bin/activate
python /scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution/code/build_native_protenix_dataset.py
```

Audited subset build:

```bash
source /scratch/phys/sin/rna-dataset/venv/px-kaggle/bin/activate
python /scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution/code/build_native_protenix_dataset.py --limit-per-split 12
```

Default native output root:

- `/scratch/phys/sin/rna-dataset/protenix_native_kaggle_subset`

Submit native-mode training:

```bash
DATA_MODE=native \
NATIVE_DATA_ROOT=/scratch/phys/sin/rna-dataset/protenix_native_kaggle_subset \
RUN_NAME=kaggle_native_ft \
MAX_STEPS=5000 \
/scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution/slurm/submit_official_protenix_train_h200.sh
```

How to submit with `sbatch`:

- preferred: run the wrapper directly

```bash
/scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution/slurm/submit_official_protenix_train_h200.sh
```

- direct `sbatch` form if you want it explicitly:

```bash
sbatch --export=ALL,DATA_MODE=native,NATIVE_DATA_ROOT=/scratch/phys/sin/rna-dataset/protenix_native_kaggle_subset \
/scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution/slurm/run_official_protenix_train_h200.sh
```

Useful overrides:

- `RUN_NAME`: name prefix for the run
- `CHECKPOINT_PATH`: checkpoint to load at startup
- `CHECKPOINT_SAVE_DIR`: where new checkpoints are written
- `USE_MSA=true|false`: enable or disable MSA usage
- `USE_WANDB=true|false`: enable or disable `wandb`
- `WANDB_PROJECT`: `wandb` project name
- `MAX_STEPS`: number of training steps
- `EVAL_INTERVAL`: validation interval
- `TRAIN_BATCH_SIZE`: dataset batch size control for the adapter path
- `DIFFUSION_BATCH_SIZE`: official diffusion batch size

Where outputs go:

- run directory root: `RUN_BASE_DIR`
- default run base dir: `/scratch/phys/sin/rna-dataset/protenix_runs`
- saved checkpoints:
  - if `CHECKPOINT_SAVE_DIR` is set, checkpoints are written there
  - otherwise they go under `RUN_BASE_DIR/<run_name_timestamp>/checkpoints`

Wandb:

- the submit wrapper defaults to `USE_WANDB=true`
- if you use the wrapper and want online logging, export `WANDB_API_KEY` first

Example:

```bash
export WANDB_API_KEY=...
DATA_MODE=native \
USE_WANDB=true \
WANDB_PROJECT=protenix-v1-rna \
/scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/kaggle-new-solution/slurm/submit_official_protenix_train_h200.sh
```

Current native conversion status:

- the audited native subset currently converts successfully for most samples
- the remaining failures are explicit `sequence_mismatch` cases listed in:
  `/scratch/phys/sin/rna-dataset/protenix_native_kaggle_subset/manifests/conversion_manifest.csv`

Default local paths used by this solution:

- Official repo:
  `/scratch/work/sethih1/RNA-prediction/sin-Stanford-RNA-3D-Folding-2/external/Protenix-v1-official`
- RNA cache:
  `/scratch/phys/sin/rna-dataset/preprocessed_data`
- Official checkpoint:
  `/scratch/phys/sin/rna-dataset/protenix_checkpoint/protenix_base_20250630_v1.0.0.pt`
- CCD cache:
  `/scratch/phys/sin/rna-dataset/protenix_ccd_cache`

Official repo pin used for this solution:

- tag: `v1.0.0`
- commit: `4d089c7c6c10d72af2cec4c6b0586249c993a79b`
