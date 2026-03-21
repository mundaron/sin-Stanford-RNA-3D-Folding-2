#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

SOLUTION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTENIX_REPO = SOLUTION_ROOT.parent / "external" / "Protenix-v1-official"
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "PROTENIX_RNA_CACHE_ROOT",
        "/scratch/phys/sin/rna-dataset/preprocessed_data",
    )
)
DEFAULT_CCD_ROOT = Path(
    os.environ.get(
        "PROTENIX_CCD_CACHE_ROOT",
        "/scratch/phys/sin/rna-dataset/protenix_ccd_cache",
    )
)
RUNTIME_LOADER = Path(
    os.environ.get(
        "PROTENIX_RNA_RUNTIME_LOADER",
        str(Path(__file__).resolve().with_name("kaggle_rna_official_runtime.py")),
    )
)


os.environ.setdefault("LAYERNORM_TYPE", "openfold")


def wandb_requested(train_args: list[str]) -> bool:
    for index, arg in enumerate(train_args):
        if arg != "--use_wandb":
            continue
        if index + 1 >= len(train_args):
            return False
        return train_args[index + 1].lower() == "true"
    return False


def maybe_stub_wandb(allow_stub: bool) -> None:
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError:
        if not allow_stub:
            raise RuntimeError(
                "wandb logging was requested, but the wandb package is not installed."
            )
        wandb = types.ModuleType("wandb")
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb
        os.environ.setdefault("WANDB_DISABLED", "true")


def load_module_alias(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module alias {alias} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def configure_ccd_paths(ccd_root: Path) -> None:
    import configs.configs_data as configs_data_module

    components_file = ccd_root / "components.v20240608.cif"
    rdkit_pkl = ccd_root / "components.v20240608.cif.rdkit_mol.pkl"
    configs_data_module.data_configs["ccd_components_file"] = str(components_file)
    configs_data_module.data_configs["ccd_components_rdkit_mol_file"] = str(rdkit_pkl)

    try:
        import protenix.data.core.ccd as ccd

        ccd.COMPONENTS_FILE = str(components_file)
        ccd.RKDIT_MOL_PKL = rdkit_pkl
    except Exception:
        pass


def patch_permutation_for_coarse_rna() -> None:
    from protenix.utils.permutation import atom_permutation
    from protenix.utils.permutation.permutation import SymmetricPermutation

    def _atom_only_permute_label_to_match_mini_rollout(
        self,
        mini_coord,
        input_feature_dict,
        label_dict,
        label_full_dict,
    ):
        del label_full_dict
        log_dict = {}
        if "atom_perm_list" not in input_feature_dict:
            log_dict["minirollout_perm/Atom-skipped"] = torch.tensor(1.0)
            return label_dict, log_dict
        permuted_label_dict, atom_perm_log_dict, _ = atom_permutation.run(
            pred_coord=mini_coord[0],
            true_coord=label_dict["coordinate"],
            true_coord_mask=label_dict["coordinate_mask"],
            ref_space_uid=input_feature_dict["ref_space_uid"],
            atom_perm_list=input_feature_dict["atom_perm_list"],
            permute_label=True,
            error_dir=self.atom_error_dir,
            global_align_wo_symmetric_atom=(
                self.configs.atom_permutation.global_align_wo_symmetric_atom
            ),
        )
        if self.configs.atom_permutation.train.mini_rollout:
            label_dict.update(permuted_label_dict)
            log_dict.update(
                {f"minirollout_perm/Atom-{k}": v for k, v in atom_perm_log_dict.items()}
            )
        else:
            log_dict.update(
                {
                    f"minirollout_perm/Atom.F-{k}": v
                    for k, v in atom_perm_log_dict.items()
                }
            )
        return label_dict, log_dict

    def _atom_only_permute_diffusion_sample_to_match_label(
        self,
        input_feature_dict,
        pred_dict,
        label_dict,
        stage,
        permute_by_pocket=False,
    ):
        del permute_by_pocket
        log_dict = {}
        if "atom_perm_list" not in input_feature_dict:
            log_dict["sample_perm/Atom-skipped"] = torch.tensor(1.0)
            return pred_dict, log_dict, None, None
        permuted_pred_dict, atom_perm_log_dict, atom_perm_pred_indices = (
            atom_permutation.run(
                pred_coord=pred_dict["coordinate"],
                true_coord=label_dict["coordinate"],
                true_coord_mask=label_dict["coordinate_mask"],
                ref_space_uid=input_feature_dict["ref_space_uid"],
                atom_perm_list=input_feature_dict["atom_perm_list"],
                permute_label=False,
                error_dir=self.atom_error_dir,
                global_align_wo_symmetric_atom=(
                    self.configs.atom_permutation.global_align_wo_symmetric_atom
                ),
            )
        )
        if self.configs.atom_permutation.get(stage).diffusion_sample:
            pred_dict.update(permuted_pred_dict)
            log_dict.update(
                {f"sample_perm/Atom-{k}": v for k, v in atom_perm_log_dict.items()}
            )
        else:
            log_dict.update(
                {f"sample_perm/Atom.F-{k}": v for k, v in atom_perm_log_dict.items()}
            )
        return pred_dict, log_dict, atom_perm_pred_indices, None

    SymmetricPermutation.permute_label_to_match_mini_rollout = (
        _atom_only_permute_label_to_match_mini_rollout
    )
    SymmetricPermutation.permute_diffusion_sample_to_match_label = (
        _atom_only_permute_diffusion_sample_to_match_label
    )


def patch_trainer(train_module, runtime_module, cache_root: Path) -> None:
    def _init_basics(self):
        self.step = 0
        self.global_step = 0
        self.start_step = 0
        self.iters_to_accumulate = self.configs.iters_to_accumulate

        self.run_name = self.configs.run_name + "_" + train_module.time.strftime("%Y%m%d_%H%M%S")
        run_names = train_module.DIST_WRAPPER.all_gather_object(
            self.run_name if train_module.DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        checkpoint_override = os.environ.get("PROTENIX_CHECKPOINT_DIR", "").strip()
        self.checkpoint_dir = checkpoint_override or f"{self.run_dir}/checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"

        if train_module.DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            train_module.save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, "
            f"checkpoint_dir: {self.checkpoint_dir}, "
            f"prediction_dir: {self.prediction_dir}, "
            f"structure_dir: {self.structure_dir}, "
            f"error_dir: {self.error_dir}"
        )

    def _init_data(self):
        os.environ["PROTENIX_RNA_CACHE_ROOT"] = str(cache_root)
        os.environ.setdefault("PROTENIX_RNA_NUM_WORKERS", "0")
        self.configs.data.num_dl_workers = int(os.environ["PROTENIX_RNA_NUM_WORKERS"])
        self.configs.use_msa = os.environ.get("PROTENIX_USE_MSA", "true").lower() == "true"
        self.train_dl = runtime_module.get_kaggle_rna_dataloader(
            configs=self.configs,
            cache_root=str(cache_root),
            split="train",
            shuffle=True,
        )
        if os.environ.get("PROTENIX_ENABLE_VALIDATION_EVAL", "1") == "1":
            self.test_dls = {
                "validation": runtime_module.get_kaggle_rna_dataloader(
                    configs=self.configs,
                    cache_root=str(cache_root),
                    split="validation",
                    shuffle=False,
                )
            }
        else:
            self.test_dls = {}

    def _coarse_rna_get_loss(self, batch, mode: str = "train"):
        assert mode in ["train", "eval"]
        with torch.no_grad():
            batch["label_dict"] = self.loss.calculate_label(
                feat_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
            )

        if mode == "train":
            diffusion_per_sample_scale = (
                batch["pred_dict"]["noise_level"] ** 2 + self.configs.sigma_data**2
            ) / (self.configs.sigma_data * batch["pred_dict"]["noise_level"]) ** 2
        else:
            diffusion_per_sample_scale = None

        mse_loss = self.loss.mse_loss(
            pred_coordinate=batch["pred_dict"]["coordinate"],
            true_coordinate=batch["label_dict"]["coordinate"],
            coordinate_mask=batch["label_dict"]["coordinate_mask"],
            is_rna=batch["input_feature_dict"]["is_rna"],
            is_dna=batch["input_feature_dict"]["is_dna"],
            is_ligand=batch["input_feature_dict"]["is_ligand"],
            per_sample_scale=diffusion_per_sample_scale,
        )

        if self.configs.loss.diffusion_lddt_loss_dense:
            smooth_lddt_loss = self.loss.smooth_lddt_loss.dense_forward(
                pred_coordinate=batch["pred_dict"]["coordinate"],
                true_coordinate=batch["label_dict"]["coordinate"],
                lddt_mask=batch["label_dict"]["lddt_mask"],
                diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
            )
        elif self.configs.loss.diffusion_sparse_loss_enable:
            smooth_lddt_loss = self.loss.smooth_lddt_loss.sparse_forward(
                pred_coordinate=batch["pred_dict"]["coordinate"],
                true_coordinate=batch["label_dict"]["coordinate"],
                lddt_mask=batch["label_dict"]["lddt_mask"],
                diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
            )
        else:
            batch["pred_dict"] = self.loss.calculate_prediction(batch["pred_dict"])
            smooth_lddt_loss = self.loss.smooth_lddt_loss(
                pred_distance=batch["pred_dict"]["distance"],
                true_distance=batch["label_dict"]["distance"],
                distance_mask=batch["label_dict"]["distance_mask"],
                lddt_mask=batch["label_dict"]["lddt_mask"],
                diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
            )

        weighted_mse_loss = self.loss.alpha_diffusion * mse_loss
        weighted_smooth_lddt_loss = (
            self.loss.alpha_diffusion
            * self.loss.weight_smooth_lddt
            * smooth_lddt_loss
        )
        total_loss = weighted_mse_loss + weighted_smooth_lddt_loss
        zero = total_loss.detach().clone() * 0.0
        loss_dict = {
            "mse_loss": mse_loss.detach().clone(),
            "weighted_mse_loss": weighted_mse_loss.detach().clone(),
            "smooth_lddt_loss": smooth_lddt_loss.detach().clone(),
            "weighted_smooth_lddt_loss": weighted_smooth_lddt_loss.detach().clone(),
            "bond_loss": zero,
            "weighted_bond_loss": zero,
            "distogram_loss": zero,
            "weighted_distogram_loss": zero,
            "plddt_loss": zero,
            "weighted_plddt_loss": zero,
            "pde_loss": zero,
            "weighted_pde_loss": zero,
            "resolved_loss": zero,
            "weighted_resolved_loss": zero,
            "pae_loss": zero,
            "weighted_pae_loss": zero,
            "loss": total_loss.detach().clone(),
        }
        return total_loss, loss_dict, batch

    train_module.AF3Trainer.init_basics = _init_basics
    train_module.AF3Trainer.init_data = _init_data
    train_module.AF3Trainer.get_loss = _coarse_rna_get_loss

    if os.environ.get("PROTENIX_DISABLE_EVAL", "0") == "1":
        def _skip_evaluate(self, mode: str = "eval"):
            self.print("Skipping evaluation in kaggle-new-solution runtime launcher.")
            return

        train_module.AF3Trainer.evaluate = _skip_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch official Protenix v1 training with the cached RNA pipeline."
    )
    parser.add_argument(
        "--protenix-repo",
        default=os.environ.get("PROTENIX_REPO_DIR", str(DEFAULT_PROTENIX_REPO)),
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("PROTENIX_RNA_CACHE_ROOT", str(DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument(
        "--ccd-root",
        default=os.environ.get("PROTENIX_CCD_CACHE_ROOT", str(DEFAULT_CCD_ROOT)),
    )
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    repo_dir = Path(args.protenix_repo).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    ccd_root = Path(args.ccd_root).expanduser().resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Missing official Protenix repo: {repo_dir}")
    if not cache_root.exists():
        raise FileNotFoundError(f"Missing RNA cache root: {cache_root}")
    if not ccd_root.exists():
        raise FileNotFoundError(f"Missing CCD cache root: {ccd_root}")
    if not RUNTIME_LOADER.exists():
        raise FileNotFoundError(f"Missing runtime loader: {RUNTIME_LOADER}")

    sys.path.insert(0, str(repo_dir))
    os.environ.setdefault("LAYERNORM_TYPE", "openfold")
    use_wandb = wandb_requested(train_args)
    maybe_stub_wandb(allow_stub=not use_wandb)
    configure_ccd_paths(ccd_root)
    runtime_module = load_module_alias(
        "kaggle_new_solution.rna_runtime",
        RUNTIME_LOADER,
    )

    import runner.train as train_module

    patch_permutation_for_coarse_rna()
    patch_trainer(train_module=train_module, runtime_module=runtime_module, cache_root=cache_root)

    sys.argv = [str(repo_dir / "runner" / "train.py"), *train_args]
    print(f"Using official Protenix repo: {repo_dir}")
    print(f"Using RNA cache root: {cache_root}")
    print(f"Using CCD cache root: {ccd_root}")
    print(f"Using runtime loader: {RUNTIME_LOADER}")
    print("Using coarse RNA supervision: diffusion MSE + smooth-LDDT")
    print("Chain permutation is disabled in the adapter; atom permutation remains enabled.")
    print(f"Validation eval enabled: {os.environ.get('PROTENIX_ENABLE_VALIDATION_EVAL', '1') == '1'}")
    print(f"wandb enabled: {use_wandb}")
    print(f"Forwarded training args: {train_args}")
    train_module.main()


if __name__ == "__main__":
    main()
