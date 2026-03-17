#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.torch_utils import dict_to_tensor

DEFAULT_CACHE_ROOT = os.environ.get(
    "PROTENIX_KAGGLE_CACHE_ROOT",
    "/scratch/phys/sin/rna-dataset/preprocessed_data",
)
DEFAULT_SPLIT = os.environ.get("PROTENIX_KAGGLE_SPLIT", "train")
DEFAULT_NUM_WORKERS = int(os.environ.get("PROTENIX_KAGGLE_NUM_WORKERS", "0"))


def get_rna_dataloader(configs: Any) -> DataLoader:
    cache_root = os.environ.get("PROTENIX_KAGGLE_CACHE_ROOT", DEFAULT_CACHE_ROOT)
    split = os.environ.get("PROTENIX_KAGGLE_SPLIT", DEFAULT_SPLIT)
    return get_kaggle_rna_dataloader(configs=configs, cache_root=cache_root, split=split)


def get_kaggle_rna_dataloader(
    configs: Any,
    cache_root: str,
    split: str = "train",
    shuffle: bool | None = None,
    collate_mode: str = "list",
) -> DataLoader:
    crop_size = 420
    data_config = configs.data
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        crop_size = config_dict["cropping_configs"]["crop_size"]

    dataset = KaggleRNADatasetCompNative(
        cache_root=cache_root,
        split=split,
        use_msa=configs.use_msa,
        crop_size=crop_size,
    )
    if shuffle is None:
        shuffle = split == "train"
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=shuffle,
    )
    if collate_mode == "list":
        collate_fn = lambda batch: batch
    elif collate_mode == "single":
        collate_fn = lambda batch: batch[0]
    else:
        raise ValueError(f"Unsupported collate_mode: {collate_mode}")
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=DEFAULT_NUM_WORKERS,
    )


class KaggleRNADatasetCompNative(Dataset):
    def __init__(self, cache_root: str, split: str = "train", use_msa: bool = True, crop_size: int = 420) -> None:
        self.cache_root = Path(cache_root)
        self.split = split
        self.is_train_split = split == "train"
        self.use_msa = use_msa
        self.crop_size = crop_size

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
            return {k: data[k] for k in data.files}

    def process_one(self, single_sample_dict: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], Any, dict[str, float]]:
        t0 = time.time()
        label = self._load_label(single_sample_dict["name"])
        xyz_multi = label["coordinate_multi"]
        residue_mask = label["coordinate_mask_multi"][0]
        seq = single_sample_dict["sequences"][0]["rnaSequence"]["sequence"]
        assert len(seq) == xyz_multi.shape[1]
        was_cropped = False

        if len(seq) > self.crop_size:
            was_cropped = True
            max_offset = len(seq) - self.crop_size
            if self.is_train_split:
                start = np.random.randint(0, max_offset + 1)
            else:
                start = max_offset // 2
            end = start + self.crop_size
            seq = seq[start:end]
            xyz_multi = xyz_multi[:, start:end, :]
            single_sample_dict = json.loads(json.dumps(single_sample_dict))
            single_sample_dict["sequences"][0]["rnaSequence"]["sequence"] = seq

        sample2feat = SampleDictToFeatures(single_sample_dict)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(atom_array.distogram_rep_atom_mask).long()
        entity_poly_type = sample2feat.entity_poly_type

        coordinate_list = []
        coordinate_mask_list = []
        c1_idx = 0

        for atom in atom_array:
            if atom.atom_name == "C1'":
                coordinate_list.append(xyz_multi[:, c1_idx].tolist())
                coordinate_mask_list.append(float(residue_mask[c1_idx]))
                c1_idx += 1
            else:
                coordinate_list.append([[0.0, 0.0, 0.0] for _ in range(len(xyz_multi))])
                coordinate_mask_list.append(0)

        t1 = time.time()
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        use_sample_msa = self.use_msa and not was_cropped
        if was_cropped and self.use_msa and not getattr(self, "_warned_cropped_msa_disabled", False):
            print(
                "MSA featurization is disabled for cropped sequences because the precomputed RNA MSA files are full-length and do not align with cropped token columns."
            )
            self._warned_cropped_msa_disabled = True
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                pdb_name=single_sample_dict["name"],
                bioassembly=single_sample_dict["sequences"],
                entity_to_asym_id=entity_to_asym_id,
                token_array=token_array,
                atom_array=atom_array,
            )
            if use_sample_msa
            else {}
        )

        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=dummy_feats)
        feat = data_type_transform(feat_or_label_dict=features_dict)
        t2 = time.time()

        data = {"input_feature_dict": feat}
        n_token = feat["token_index"].shape[0]
        n_atom = feat["atom_to_token_idx"].shape[0]
        n_msa = feat["msa"].shape[0]
        n_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([n_asym]),
                "N_token": torch.tensor([n_token]),
                "N_atom": torch.tensor([n_atom]),
                "N_msa": torch.tensor([n_msa]),
                "entity_poly_type": entity_poly_type,
            }
        )

        coordinate_multi = np.ascontiguousarray(np.array(coordinate_list, dtype=np.float32).transpose(1, 0, 2))
        coordinate_mask = np.array(coordinate_mask_list, dtype=np.float32)
        data["coordinate_multi"] = torch.from_numpy(coordinate_multi).float()
        data["coordinate"] = data["coordinate_multi"][0]
        data["coordinate_mask"] = torch.from_numpy(coordinate_mask).float()

        t3 = time.time()
        return data, atom_array, {"crop": t1 - t0, "featurizer": t2 - t1, "added_feature": t3 - t2}

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        single_sample_dict = json.loads(json.dumps(self.inputs[index]))
        data, _, _ = self.process_one(single_sample_dict)
        data["basic"] = {"pdb_id": single_sample_dict["name"]}
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data
