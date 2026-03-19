#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

SOLUTION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTENIX_REPO = SOLUTION_ROOT / "references" / "Protenix-RNA-Kaggle"
DEFAULT_CACHE_ROOT = SOLUTION_ROOT / "references" / "preprocessed_data"
RUNTIME_LOADER = Path(
    os.environ.get(
        "PROTENIX_KAGGLE_RUNTIME_LOADER",
        str(Path(__file__).resolve().with_name("kaggle_rna_dataset_comp_native_runtime.py")),
    )
)
CCD_ROOT = Path(
    os.environ.get(
        "PROTENIX_CCD_CACHE_ROOT",
        str(SOLUTION_ROOT / "references" / "protenix_ccd_cache"),
    )
)


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
                "wandb logging was requested, but the wandb package is not installed in the active environment."
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Kaggle-native Protenix training with the lhwcv RNA fork."
    )
    parser.add_argument(
        "--protenix-repo",
        default=os.environ.get("PROTENIX_REPO_DIR", str(DEFAULT_PROTENIX_REPO)),
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("PROTENIX_KAGGLE_CACHE_ROOT", str(DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument(
        "--split",
        default=os.environ.get("PROTENIX_KAGGLE_SPLIT", "train"),
    )
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    repo_dir = Path(args.protenix_repo).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Missing Protenix fork repo: {repo_dir}")
    if not cache_root.exists():
        raise FileNotFoundError(f"Missing Kaggle-native cache root: {cache_root}")
    if not RUNTIME_LOADER.exists():
        raise FileNotFoundError(f"Missing runtime dataset loader: {RUNTIME_LOADER}")
    if not CCD_ROOT.exists():
        raise FileNotFoundError(f"Missing CCD cache root: {CCD_ROOT}")

    sys.path.insert(0, str(repo_dir))
    os.environ["PROTENIX_KAGGLE_CACHE_ROOT"] = str(cache_root)
    os.environ["PROTENIX_KAGGLE_SPLIT"] = args.split
    os.environ.setdefault("PROTENIX_KAGGLE_NUM_WORKERS", "0")

    import configs.configs_data as configs_data_module

    use_wandb = wandb_requested(train_args)
    maybe_stub_wandb(allow_stub=not use_wandb)
    configs_data_module.data_configs["ccd_components_file"] = str(
        CCD_ROOT / "components.v20240608.cif"
    )
    configs_data_module.data_configs["ccd_components_rdkit_mol_file"] = str(
        CCD_ROOT / "components.v20240608.cif.rdkit_mol.pkl"
    )
    runtime_module = load_module_alias("protenix.data.rna_dataset_comp", RUNTIME_LOADER)

    import runner.train as train_module

    def _init_data(self):
        self.configs.input_json_path = "./examples/casp16_part.json"
        self.configs.dump_dir = "./output/"
        self.configs.num_workers = int(os.environ.get("PROTENIX_KAGGLE_NUM_WORKERS", "0"))
        train_batch_size = int(os.environ.get("PROTENIX_KAGGLE_TRAIN_BATCH_SIZE", "1"))
        self.train_dl = runtime_module.get_kaggle_rna_dataloader(
            configs=self.configs,
            cache_root=str(cache_root),
            split="train",
            shuffle=True,
            collate_mode="list",
            batch_size=train_batch_size,
        )
        if os.environ.get("PROTENIX_ENABLE_VALIDATION_EVAL", "1") == "1":
            validation_dl = runtime_module.get_kaggle_rna_dataloader(
                configs=self.configs,
                cache_root=str(cache_root),
                split="validation",
                shuffle=False,
                collate_mode="single",
                batch_size=1,
            )
            self.test_dls = {"validation": validation_dl}
        else:
            self.test_dls = {}

    train_module.AF3Trainer.init_data = _init_data

    if os.environ.get("PROTENIX_DISABLE_EVAL", "1") == "1":
        def _skip_evaluate(self):
            self.print("Skipping evaluation in runtime launcher.")
            return

        train_module.AF3Trainer.evaluate = _skip_evaluate

    sys.argv = [str(repo_dir / "runner" / "train.py"), *train_args]
    print(f"Using Protenix repo: {repo_dir}")
    print(f"Using cache root: {cache_root}")
    print(f"Using runtime loader: {RUNTIME_LOADER}")
    print(f"Using CCD cache root: {CCD_ROOT}")
    print(f"Train batch size: {os.environ.get('PROTENIX_KAGGLE_TRAIN_BATCH_SIZE', '1')}")
    print(f"Validation eval enabled: {os.environ.get('PROTENIX_ENABLE_VALIDATION_EVAL', '1') == '1'}")
    print("Forcing DataLoader num_workers=0")
    print(f"wandb enabled: {use_wandb}")
    print(f"Forwarded training args: {train_args}")
    train_module.main()


if __name__ == "__main__":
    main()
