#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR_CANDIDATES = [
    Path(os.environ.get("RNA_3D_DATA_DIR", "")) if os.environ.get("RNA_3D_DATA_DIR") else None,
    Path("/scratch/phys/sin/rna-dataset"),
    Path("/kaggle/input/stanford-rna-3d-folding-2"),
]
SOLUTION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("PROTENIX_KAGGLE_CACHE_ROOT", SOLUTION_ROOT / "references" / "preprocessed_data"))
SENTINEL_THRESHOLD = -1e17


def resolve_data_dir(data_dir: str | None) -> Path:
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    candidates.extend([p for p in DEFAULT_DATA_DIR_CANDIDATES if p is not None])
    for candidate in candidates:
        if candidate.exists() and (candidate / "train_sequences.csv").exists():
            return candidate
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve RNA dataset dir. Checked:\n{checked}")


def target_from_label_id(label_id: str) -> str:
    return str(label_id).rsplit("_", 1)[0]


def infer_conformer_indices(columns: list[str]) -> list[int]:
    indices = []
    for col in columns:
        if col.startswith("x_"):
            try:
                indices.append(int(col.split("_")[1]))
            except ValueError:
                continue
    return sorted(set(indices))


def read_msa_lines(msa_path: Path) -> str:
    lines = []
    with msa_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                lines.append(line)
            else:
                lines.append(line.replace(" ", ""))
    return "\n".join(lines) + "\n"


def build_sample_entry(target_id: str, sequence: str, msa_path: Path | None, msa_root: Path | None) -> dict:
    rna_sequence = {
        "sequence": sequence,
        "count": 1,
    }
    if msa_path is not None and msa_root is not None:
        rna_sequence["msa"] = {
            "precomputed_msa_dir": str(msa_root),
            "pairing_db": "",
        }
        rna_sequence["unpairedMsaPath"] = str(msa_path)

    return {
        "name": target_id,
        "covalent_bonds": [],
        "sequences": [
            {
                "rnaSequence": rna_sequence,
            }
        ],
    }


def sequence_from_rows(group: pd.DataFrame) -> str:
    return "".join(group["resname"].astype(str).tolist())


def choose_row_order(group: pd.DataFrame, expected_sequence: str) -> pd.DataFrame:
    candidates = []

    base = group.reset_index(drop=True).copy()
    candidates.append(base)

    numeric_suffix = group["ID"].astype(str).str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    by_id = group.iloc[np.argsort(numeric_suffix, kind="stable")].reset_index(drop=True).copy()
    candidates.append(by_id)

    sort_cols = [col for col in ["copy", "chain", "resid"] if col in group.columns]
    if sort_cols:
        candidates.append(group.sort_values(sort_cols + ["ID"]).reset_index(drop=True).copy())

    seen = set()
    for candidate in candidates:
        key = tuple(candidate["ID"].astype(str).tolist())
        if key in seen:
            continue
        seen.add(key)
        if sequence_from_rows(candidate) == expected_sequence:
            return candidate

    return candidates[0]


def build_coordinate_tensor(group: pd.DataFrame, conformer_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    masks = []
    for idx in conformer_indices:
        xyz = group[[f"x_{idx}", f"y_{idx}", f"z_{idx}"]].to_numpy(dtype=np.float32)
        mask = ~(xyz <= SENTINEL_THRESHOLD).any(axis=1)
        if mask.sum() == 0:
            continue
        xyz = xyz.copy()
        xyz[~mask] = 0.0
        coords.append(xyz)
        masks.append(mask.astype(np.int64))

    if not coords:
        zero_xyz = np.zeros((1, len(group), 3), dtype=np.float32)
        zero_mask = np.zeros((1, len(group)), dtype=np.int64)
        return zero_xyz, zero_mask

    return np.stack(coords, axis=0), np.stack(masks, axis=0)


def process_split(
    data_dir: Path,
    split: str,
    output_root: Path,
    msa_input_root: Path,
    copy_msa: bool,
) -> list[dict]:
    seq_df = pd.read_csv(data_dir / f"{split}_sequences.csv")
    labels_df = pd.read_csv(data_dir / f"{split}_labels.csv", low_memory=False)
    target_ids = labels_df["ID"].astype(str).str.rsplit("_", n=1).str[0]

    conformer_indices = infer_conformer_indices(labels_df.columns.tolist())
    if not conformer_indices:
        raise ValueError(f"No conformer columns found in {split}_labels.csv")

    samples_dir = output_root / "samples"
    split_label_dir = output_root / "labels" / split
    msa_a3m_dir = output_root / "msa_a3m"
    samples_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)
    if copy_msa:
        msa_a3m_dir.mkdir(parents=True, exist_ok=True)

    sample_entries = []
    summary_rows = []

    seq_map = seq_df.set_index("target_id")["sequence"].to_dict()
    desc_map = seq_df.set_index("target_id")["description"].to_dict()
    usage_map = seq_df.set_index("target_id").to_dict("index")

    for target_id, group in labels_df.groupby(target_ids, sort=True):
        target_id = str(target_id)
        expected_sequence = seq_map.get(target_id)
        if expected_sequence is None:
            continue

        ordered_group = choose_row_order(group, expected_sequence)
        label_sequence = sequence_from_rows(ordered_group)
        sequence_match = label_sequence == expected_sequence

        coords, masks = build_coordinate_tensor(ordered_group, conformer_indices)

        out_npz = split_label_dir / f"{target_id}.npz"
        np.savez_compressed(
            out_npz,
            coordinate_multi=coords,
            coordinate=coords[0],
            coordinate_mask=masks[0],
            coordinate_mask_multi=masks,
            sequence=np.array(expected_sequence),
            target_id=np.array(target_id),
        )

        msa_source = msa_input_root / f"{target_id}.MSA.fasta"
        sample_msa_path = None
        sample_msa_root = None
        if msa_source.exists():
            if copy_msa:
                sample_msa_path = msa_a3m_dir / f"{target_id}.a3m"
                sample_msa_path.write_text(read_msa_lines(msa_source), encoding="utf-8")
                sample_msa_root = msa_a3m_dir
            else:
                sample_msa_path = msa_source
                sample_msa_root = msa_input_root

        sample_entries.append(build_sample_entry(target_id, expected_sequence, sample_msa_path, sample_msa_root))
        summary_rows.append(
            {
                "split": split,
                "target_id": target_id,
                "length": len(expected_sequence),
                "sequence_match": sequence_match,
                "n_conformers": int(coords.shape[0]),
                "msa_exists": msa_source.exists(),
                "description": desc_map.get(target_id, ""),
                "stoichiometry": usage_map.get(target_id, {}).get("stoichiometry", ""),
            }
        )

    sample_json_path = samples_dir / f"{split}_samples.json"
    sample_json_path.write_text(json.dumps(sample_entries, indent=2), encoding="utf-8")
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Kaggle-native Protenix preprocessing cache from the RNA competition CSVs.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--copy-msa",
        action="store_true",
        help="Copy local MSA files into output_root/msa_a3m as .a3m files. By default the builder references the existing MSA files in-place.",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    msa_input_root = data_dir / "MSA"
    all_summary_rows = []
    for split in ["train", "validation"]:
        all_summary_rows.extend(process_split(data_dir, split, output_root, msa_input_root, args.copy_msa))

    summary_df = pd.DataFrame(all_summary_rows).sort_values(["split", "target_id"]).reset_index(drop=True)
    summary_df.to_csv(output_root / "cache_summary.csv", index=False)

    summary = {
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "copy_msa": bool(args.copy_msa),
        "msa_root": str((output_root / "msa_a3m") if args.copy_msa else msa_input_root),
        "n_train_targets": int((summary_df["split"] == "train").sum()),
        "n_validation_targets": int((summary_df["split"] == "validation").sum()),
        "n_msa_present": int(summary_df["msa_exists"].sum()),
        "n_sequence_matches": int(summary_df["sequence_match"].sum()),
        "n_sequence_mismatches": int((~summary_df["sequence_match"]).sum()),
    }
    (output_root / "cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
