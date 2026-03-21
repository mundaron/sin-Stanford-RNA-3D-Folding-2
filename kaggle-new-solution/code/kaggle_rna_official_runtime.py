#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from biotite.structure.io import pdbx as biotite_pdbx
from biotite.structure.io.pdbx import convert as biotite_pdbx_convert
from torch.utils.data import DataLoader, Dataset, DistributedSampler

if not hasattr(biotite_pdbx, "PDBX_BOND_TYPE_ID_TO_TYPE") and hasattr(
    biotite_pdbx, "PDBX_BOND_ORDER_TO_TYPE"
):
    biotite_pdbx.PDBX_BOND_TYPE_ID_TO_TYPE = dict(
        biotite_pdbx.PDBX_BOND_ORDER_TO_TYPE
    )
if not hasattr(biotite_pdbx_convert, "PDBX_BOND_TYPE_ID_TO_TYPE") and hasattr(
    biotite_pdbx_convert, "PDBX_BOND_ORDER_TO_TYPE"
):
    biotite_pdbx_convert.PDBX_BOND_TYPE_ID_TO_TYPE = dict(
        biotite_pdbx_convert.PDBX_BOND_ORDER_TO_TYPE
    )

from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.torch_utils import collate_fn_first, dict_to_tensor

DEFAULT_CACHE_ROOT = os.environ.get(
    "PROTENIX_RNA_CACHE_ROOT",
    "/scratch/phys/sin/rna-dataset/preprocessed_data",
)
DEFAULT_NUM_WORKERS = int(os.environ.get("PROTENIX_RNA_NUM_WORKERS", "0"))


def _resolve_protenix_repo_root() -> Path:
    env_repo_dir = os.environ.get("PROTENIX_REPO_DIR")
    search_roots = []
    if env_repo_dir:
        search_roots.append(Path(env_repo_dir))
    search_roots.extend(Path(entry) for entry in sys.path if entry)

    for root in search_roots:
        if (root / "protenix" / "data" / "inference" / "json_to_feature.py").exists():
            return root
    raise FileNotFoundError(
        "Could not locate the official Protenix repo on sys.path. "
        "Set PROTENIX_REPO_DIR before importing the RNA runtime loader."
    )


def _ensure_inference_package_stub() -> None:
    package_name = "protenix.data.inference"
    if package_name in sys.modules:
        return
    repo_root = _resolve_protenix_repo_root()
    package_dir = repo_root / "protenix" / "data" / "inference"
    package_module = types.ModuleType(package_name)
    package_module.__path__ = [str(package_dir)]
    sys.modules[package_name] = package_module


_ensure_inference_package_stub()

from protenix.data.inference.json_to_feature import SampleDictToFeatures


def _clone_tensor_dict(tensor_dict: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in tensor_dict.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def get_kaggle_rna_dataloader(
    configs: Any,
    cache_root: str,
    split: str = "train",
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = KaggleRNADatasetOfficial(
        cache_root=cache_root,
        split=split,
        use_msa=bool(getattr(configs, "use_msa", True)),
        crop_size=int(getattr(configs, "train_crop_size", 420)),
        msa_pair_as_unpair=bool(getattr(configs, "msa_pair_as_unpair", True)),
        use_rna_msa=bool(getattr(configs, "use_rna_msa", True)),
    )
    if shuffle is None:
        shuffle = split == "train"
    batch_size = int(
        os.environ.get(
            "PROTENIX_TRAIN_BATCH_SIZE",
            getattr(configs, "train_batch_size", 1),
        )
    )
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_first,
        num_workers=DEFAULT_NUM_WORKERS,
    )


class KaggleRNADatasetOfficial(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str = "train",
        use_msa: bool = True,
        crop_size: int = 420,
        msa_pair_as_unpair: bool = True,
        use_rna_msa: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.split = split
        self.is_train_split = split == "train"
        self.use_msa = use_msa
        self.crop_size = crop_size
        self.msa_pair_as_unpair = msa_pair_as_unpair
        self.use_rna_msa = use_rna_msa
        self._warned_cropped_msa_disabled = False

        sample_path = self.cache_root / "samples" / f"{split}_samples.json"
        label_dir = self.cache_root / "labels" / split
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample file: {sample_path}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label dir: {label_dir}")

        self.inputs = json.loads(sample_path.read_text(encoding="utf-8"))
        self.label_dir = label_dir

    def _load_label(self, target_id: str) -> dict[str, np.ndarray]:
        path = self.label_dir / f"{target_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing label cache: {path}")
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    def _prepare_cropped_sample(
        self,
        single_sample_dict: Mapping[str, Any],
        label: Mapping[str, np.ndarray],
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, bool]:
        sample_dict = json.loads(json.dumps(single_sample_dict))
        sequence = sample_dict["sequences"][0]["rnaSequence"]["sequence"]
        residue_coordinate_multi = np.asarray(
            label.get("coordinate_multi", label["coordinate"][None, ...]),
            dtype=np.float32,
        )
        residue_mask_multi = np.asarray(
            label.get("coordinate_mask_multi", label["coordinate_mask"][None, ...]),
            dtype=np.float32,
        )

        if len(sequence) != residue_coordinate_multi.shape[1]:
            raise ValueError(
                "Sequence length and residue label length disagree for "
                f"{sample_dict['name']}: {len(sequence)} vs {residue_coordinate_multi.shape[1]}"
            )

        was_cropped = False
        if len(sequence) > self.crop_size:
            was_cropped = True
            max_offset = len(sequence) - self.crop_size
            start = (
                np.random.randint(0, max_offset + 1)
                if self.is_train_split
                else max_offset // 2
            )
            end = start + self.crop_size
            sample_dict["sequences"][0]["rnaSequence"]["sequence"] = sequence[start:end]
            residue_coordinate_multi = residue_coordinate_multi[:, start:end, :]
            residue_mask_multi = residue_mask_multi[:, start:end]

        return sample_dict, residue_coordinate_multi, residue_mask_multi, was_cropped

    @staticmethod
    def _expand_residue_labels_to_atom_space(
        atom_array: Any,
        residue_coordinate_multi: np.ndarray,
        residue_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coordinate_list: list[list[list[float]]] = []
        coordinate_mask_list: list[float] = []
        c1_index = 0
        n_conformers = residue_coordinate_multi.shape[0]

        for atom in atom_array:
            if atom.atom_name == "C1'":
                coordinate_list.append(residue_coordinate_multi[:, c1_index].tolist())
                coordinate_mask_list.append(float(residue_mask[c1_index]))
                c1_index += 1
            else:
                coordinate_list.append([[0.0, 0.0, 0.0] for _ in range(n_conformers)])
                coordinate_mask_list.append(0.0)

        if c1_index != residue_coordinate_multi.shape[1]:
            raise ValueError(
                "Failed to align residue-level RNA labels to atom space: "
                f"used {c1_index} C1' atoms for {residue_coordinate_multi.shape[1]} residues."
            )

        coordinate_multi = np.ascontiguousarray(
            np.asarray(coordinate_list, dtype=np.float32).transpose(1, 0, 2)
        )
        coordinate_mask = np.asarray(coordinate_mask_list, dtype=np.float32)
        return coordinate_multi, coordinate_mask

    def process_one(self, single_sample_dict: Mapping[str, Any]) -> dict[str, Any]:
        label = self._load_label(single_sample_dict["name"])
        (
            sample_dict,
            residue_coordinate_multi,
            residue_mask_multi,
            was_cropped,
        ) = self._prepare_cropped_sample(single_sample_dict, label)

        sample2feat = SampleDictToFeatures(sample_dict)
        features_dict, atom_array, _ = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.as_tensor(
            atom_array.distogram_rep_atom_mask
        ).long()

        use_sample_msa = self.use_msa and not was_cropped
        if was_cropped and self.use_msa and not self._warned_cropped_msa_disabled:
            print(
                "MSA featurization is disabled for cropped sequences because the cached "
                "RNA MSA files are full-length and do not align with cropped token columns."
            )
            self._warned_cropped_msa_disabled = True

        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=sample_dict["sequences"],
                atom_array=atom_array,
                msa_pair_as_unpair=self.msa_pair_as_unpair,
                use_rna_msa=self.use_rna_msa,
            )
            if use_sample_msa
            else {}
        )

        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            features_dict.update(dict_to_tensor(msa_features))

        input_feature_dict = data_type_transform(
            feat_or_label_dict=make_dummy_feature(
                features_dict=features_dict,
                dummy_feats=dummy_feats,
            )
        )
        input_feature_dict["is_distillation"] = torch.tensor([0])
        input_feature_dict["resolution"] = torch.tensor([-1.0], dtype=torch.float32)

        residue_mask = residue_mask_multi[0]
        coordinate_multi, coordinate_mask = self._expand_residue_labels_to_atom_space(
            atom_array=atom_array,
            residue_coordinate_multi=residue_coordinate_multi,
            residue_mask=residue_mask,
        )
        label_dict = data_type_transform(
            feat_or_label_dict={
                "coordinate_multi": torch.from_numpy(coordinate_multi).float(),
                "coordinate": torch.from_numpy(coordinate_multi[0]).float(),
                "coordinate_mask": torch.from_numpy(coordinate_mask).float(),
            }
        )

        return {
            "input_feature_dict": input_feature_dict,
            "label_dict": _clone_tensor_dict(label_dict),
            "label_full_dict": _clone_tensor_dict(label_dict),
            "basic": {"pdb_id": sample_dict["name"]},
            "sample_name": sample_dict["name"],
        }

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.process_one(self.inputs[index])
