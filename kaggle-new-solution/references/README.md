This solution does not duplicate large external assets inside `references/`.

By default it points at these shared paths:

- Official Protenix repo:
  `../../external/Protenix-v1-official`
- RNA cache:
  `/scratch/phys/sin/rna-dataset/preprocessed_data`
- Official checkpoint:
  `/scratch/phys/sin/rna-dataset/protenix_checkpoint/protenix_base_20250630_v1.0.0.pt`
- CCD cache:
  `/scratch/phys/sin/rna-dataset/protenix_ccd_cache`

You can override all of them with environment variables in the SLURM launcher.
