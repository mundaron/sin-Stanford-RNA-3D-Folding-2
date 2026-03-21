# Kaggle New Solution

This solution uses the official Protenix `v1.0.0` codepath with the
`protenix_base_20250630_v1.0.0.pt` checkpoint family while reusing the existing
RNA cache and MSA pipeline from this workspace.

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
- `slurm/run_official_protenix_train_h200.sh`
- `slurm/submit_official_protenix_train_h200.sh`

Default `sbatch` submit wrapper:

- `slurm/submit_official_protenix_train_h200.sh` enables `wandb` by default with
  `USE_WANDB=true`, `WANDB_PROJECT=protenix-v1-rna`, and `WANDB_MODE=online`

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
