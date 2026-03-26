from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch

import protenix_v1_rna_inference_helpers as base


REPO_ROOT = base.REPO_ROOT
SOLUTION_ROOT = base.SOLUTION_ROOT
DEFAULT_PROTENIX_REPO = base.DEFAULT_PROTENIX_REPO
DEFAULT_NOTEBOOK_OUTPUT_DIR = (
    SOLUTION_ROOT / "notebooks_output" / "protenix_v1_rna_raw_inference"
)

configure_notebook_logging = base.configure_notebook_logging
discover_checkpoints = base.discover_checkpoints
choose_checkpoint = base.choose_checkpoint
plot_rna_3d_interactive = base.plot_rna_3d_interactive
save_prediction_artifacts = base.save_prediction_artifacts
get_result = base.get_result

LOGGER = logging.getLogger(__name__)


def _load_module_from_path(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {alias} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_rna_data_dir(data_dir: str | Path | None = None) -> Path:
    return base.resolve_rna_data_dir(data_dir)


def resolve_sequences_csv(
    data_dir: str | Path | None = None,
    sequences_csv: str | Path | None = None,
    split: str = "validation",
) -> Path:
    if sequences_csv is not None:
        resolved = Path(sequences_csv).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Sequences CSV does not exist: {resolved}")
        return resolved

    resolved_data_dir = resolve_rna_data_dir(data_dir)
    split_name = str(split).strip().lower()
    candidate = resolved_data_dir / f"{split_name}_sequences.csv"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find sequences CSV for split {split_name!r}: {candidate}")


def _first_present_value(record: Mapping[str, Any], names: Iterable[str]) -> str | None:
    for name in names:
        if name not in record:
            continue
        value = record[name]
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def load_sequences_table(
    data_dir: str | Path | None = None,
    *,
    sequences_csv: str | Path | None = None,
    split: str = "validation",
) -> tuple[Path, pd.DataFrame, Path]:
    csv_path = resolve_sequences_csv(data_dir=data_dir, sequences_csv=sequences_csv, split=split)
    resolved_data_dir = csv_path.parent.resolve()
    sequences_df = pd.read_csv(csv_path)

    records = []
    for record in sequences_df.to_dict("records"):
        target_id = _first_present_value(record, ["target_id", "id", "ID", "name"])
        sequence = _first_present_value(record, ["sequence", "seq", "rna_sequence"])
        if target_id is None or sequence is None:
            raise ValueError(
                "Could not infer target_id/sequence columns from sequences CSV. "
                f"Columns were: {list(sequences_df.columns)}"
            )
        enriched = dict(record)
        enriched["target_id"] = target_id
        enriched["sequence"] = sequence
        records.append(enriched)

    normalized_df = pd.DataFrame(records)
    return resolved_data_dir, normalized_df, csv_path


def select_sequence_rows(
    sequences_df: pd.DataFrame,
    *,
    target_ids: Iterable[str] | None = None,
    n_examples: int = 4,
) -> pd.DataFrame:
    frame = sequences_df.copy()
    frame["target_id"] = frame["target_id"].astype(str)
    frame["sequence"] = frame["sequence"].astype(str)

    if target_ids is None:
        return frame.head(int(n_examples)).reset_index(drop=True)

    wanted = [str(target_id) for target_id in target_ids]
    wanted_order = {target_id: index for index, target_id in enumerate(wanted)}
    subset = frame[frame["target_id"].isin(wanted)].copy()
    if subset.empty:
        raise KeyError(f"None of the requested target IDs were found: {wanted}")
    subset["__order"] = subset["target_id"].map(wanted_order)
    return subset.sort_values(["__order", "target_id"]).drop(columns=["__order"]).reset_index(drop=True)


def resolve_ccd_root(
    ccd_root: str | Path | None = None,
    *,
    protenix_repo: str | Path | None = None,
) -> Path:
    repo_dir = Path(protenix_repo or DEFAULT_PROTENIX_REPO).expanduser().resolve()
    candidates: list[Path] = []
    if ccd_root is not None:
        candidates.append(Path(ccd_root).expanduser())
    for env_name in ["PROTENIX_CCD_CACHE_ROOT", "CCD_ROOT"]:
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            Path("/scratch/phys/sin/rna-dataset/protenix_ccd_cache"),
            SOLUTION_ROOT / "references" / "protenix_ccd_cache",
            repo_dir / "release_data" / "ccd_cache",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if ccd_root is not None:
        return Path(ccd_root).expanduser().resolve()
    return candidates[0].resolve()


def _find_ccd_files(ccd_root: Path) -> tuple[Path | None, Path | None]:
    if not ccd_root.exists():
        return None, None

    candidates: list[tuple[bool, bool, str, Path, Path]] = []
    for cif_path in ccd_root.glob("components*.cif"):
        if not cif_path.is_file():
            continue
        rdkit_pkl = cif_path.with_name(f"{cif_path.name}.rdkit_mol.pkl")
        candidates.append(
            (
                not rdkit_pkl.exists(),
                cif_path.name == "components.cif",
                cif_path.name,
                cif_path.resolve(),
                rdkit_pkl.resolve(),
            )
        )

    if not candidates:
        return None, None

    _, _, _, components_file, rdkit_pkl = sorted(candidates)[0]
    return components_file, rdkit_pkl


def configure_ccd_paths(
    components_file: str | Path,
    rdkit_pkl: str | Path,
    *,
    protenix_repo: str | Path | None = None,
) -> tuple[Path, Path]:
    base.configure_official_repo(protenix_repo)

    components_path = Path(components_file).expanduser().resolve()
    rdkit_path = Path(rdkit_pkl).expanduser().resolve()
    if not components_path.exists():
        raise FileNotFoundError(f"Missing CCD components file: {components_path}")
    if not rdkit_path.exists():
        raise FileNotFoundError(f"Missing CCD rdkit cache file: {rdkit_path}")

    import configs.configs_data as configs_data_module

    configs_data_module.data_configs["ccd_components_file"] = str(components_path)
    configs_data_module.data_configs["ccd_components_rdkit_mol_file"] = str(rdkit_path)

    try:
        import protenix.data.core.ccd as ccd

        ccd.COMPONENTS_FILE = str(components_path)
        ccd.RKDIT_MOL_PKL = rdkit_path
    except Exception:
        pass

    return components_path, rdkit_path


def ensure_ccd_cache(
    ccd_root: str | Path | None = None,
    *,
    protenix_repo: str | Path | None = None,
    build_if_needed: bool = True,
    download_if_missing: bool = False,
    num_cpu: int = 1,
) -> dict[str, Any]:
    repo_dir = base.configure_official_repo(protenix_repo)
    resolved_ccd_root = resolve_ccd_root(ccd_root, protenix_repo=repo_dir)
    components_file, rdkit_pkl = _find_ccd_files(resolved_ccd_root)
    status_parts: list[str] = []

    gen_ccd_module = None

    def load_gen_ccd_module():
        nonlocal gen_ccd_module
        if gen_ccd_module is None:
            gen_ccd_module = _load_module_from_path(
                "_protenix_gen_ccd_cache",
                repo_dir / "scripts" / "gen_ccd_cache.py",
            )
        return gen_ccd_module

    if components_file is None:
        if not download_if_missing:
            raise FileNotFoundError(
                "Could not find a CCD components file under "
                f"{resolved_ccd_root}. Expected a file like components.cif or components.v20240608.cif."
            )
        resolved_ccd_root.mkdir(parents=True, exist_ok=True)
        load_gen_ccd_module().download_ccd_cif(resolved_ccd_root)
        components_file, rdkit_pkl = _find_ccd_files(resolved_ccd_root)
        status_parts.append("downloaded_components_cif")

    if components_file is None:
        raise FileNotFoundError(f"Failed to locate a CCD components file in {resolved_ccd_root}")

    if rdkit_pkl is None:
        rdkit_pkl = components_file.with_name(f"{components_file.name}.rdkit_mol.pkl")

    if not rdkit_pkl.exists():
        if not build_if_needed:
            raise FileNotFoundError(
                f"Missing CCD rdkit cache file: {rdkit_pkl}. Set build_if_needed=True to generate it."
            )
        load_gen_ccd_module().precompute_ccd_mol(components_file, rdkit_pkl, num_cpu=int(num_cpu))
        status_parts.append("generated_rdkit_cache")
    else:
        status_parts.append("existing_rdkit_cache")

    components_file, rdkit_pkl = configure_ccd_paths(
        components_file,
        rdkit_pkl,
        protenix_repo=repo_dir,
    )

    return {
        "ccd_root": resolved_ccd_root,
        "components_file": components_file,
        "rdkit_pkl": rdkit_pkl,
        "status": ", ".join(status_parts),
    }


def load_runner(
    checkpoint_path: str | Path,
    *,
    ccd_root: str | Path | None = None,
    protenix_repo: str | Path | None = None,
    build_ccd_cache_if_needed: bool = True,
    download_ccd_if_missing: bool = False,
    ccd_cache_num_cpu: int = 1,
    model_name: str = "protenix_base_default_v1.0.0",
    dtype: str = "bf16",
    use_msa: bool = False,
    use_rna_msa: bool = True,
    n_diffusion_samples: int = 5,
    n_diffusion_steps: int = 20,
    mc_dropout_apply_rate: float = 0.0,
    output_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    ensure_ccd_cache(
        ccd_root=ccd_root,
        protenix_repo=protenix_repo,
        build_if_needed=build_ccd_cache_if_needed,
        download_if_missing=download_ccd_if_missing,
        num_cpu=ccd_cache_num_cpu,
    )
    return base.load_runner(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        protenix_repo=protenix_repo,
        dtype=dtype,
        use_msa=use_msa,
        use_rna_msa=use_rna_msa,
        n_diffusion_samples=n_diffusion_samples,
        n_diffusion_steps=n_diffusion_steps,
        mc_dropout_apply_rate=mc_dropout_apply_rate,
        output_dir=output_dir,
    )


def resolve_msa_path(target_id: str, msa_root: str | Path | None = None) -> Path | None:
    if msa_root is None:
        return None
    resolved_msa_root = Path(msa_root).expanduser().resolve()
    if not resolved_msa_root.exists():
        return None

    direct_candidates = [
        resolved_msa_root / f"{target_id}.a3m",
        resolved_msa_root / f"{target_id}.MSA.fasta",
        resolved_msa_root / f"{target_id}.fasta",
        resolved_msa_root / f"{target_id}.fa",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    nested_candidates = [
        resolved_msa_root / target_id / "non_pairing.a3m",
        resolved_msa_root / target_id / "pairing.a3m",
    ]
    for candidate in nested_candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def build_rna_sample_entry(
    target_id: str,
    sequence: str,
    *,
    msa_path: str | Path | None = None,
) -> dict[str, Any]:
    rna_sequence: dict[str, Any] = {
        "sequence": str(sequence),
        "count": 1,
    }
    if msa_path is not None:
        resolved_msa_path = Path(msa_path).expanduser().resolve()
        rna_sequence["msa"] = {
            "precomputed_msa_dir": str(resolved_msa_path.parent),
            "pairing_db": "",
        }
        rna_sequence["unpairedMsaPath"] = str(resolved_msa_path)

    return {
        "name": str(target_id),
        "covalent_bonds": [],
        "sequences": [{"rnaSequence": rna_sequence}],
    }


def _prepare_sample_dict_for_inference(
    sample_dict: Mapping[str, Any],
    *,
    crop_size: int | None = None,
) -> tuple[dict[str, Any], bool]:
    prepared = json.loads(json.dumps(sample_dict))
    sequence = prepared["sequences"][0]["rnaSequence"]["sequence"]
    was_cropped = False
    if crop_size is not None and len(sequence) > int(crop_size):
        max_offset = len(sequence) - int(crop_size)
        start = max_offset // 2
        end = start + int(crop_size)
        prepared["sequences"][0]["rnaSequence"]["sequence"] = sequence[start:end]
        was_cropped = True
    return prepared, was_cropped


def _extract_row_target_and_sequence(row: Mapping[str, Any]) -> tuple[str, str]:
    target_id = _first_present_value(row, ["target_id", "id", "ID", "name"])
    sequence = _first_present_value(row, ["sequence", "seq", "rna_sequence"])
    if target_id is None or sequence is None:
        raise ValueError(f"Could not extract target_id/sequence from row: {row}")
    return target_id, sequence


def prepare_raw_example(
    row: Mapping[str, Any],
    *,
    msa_root: str | Path | None = None,
    protenix_repo: str | Path | None = None,
    use_msa: bool = False,
    msa_pair_as_unpair: bool = True,
    use_rna_msa: bool = True,
    crop_size: int | None = None,
) -> dict[str, Any]:
    base.configure_official_repo(protenix_repo)

    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
    from protenix.data.utils import data_type_transform, make_dummy_feature
    from protenix.utils.torch_utils import dict_to_tensor

    target_id, sequence = _extract_row_target_and_sequence(row)
    msa_path = resolve_msa_path(target_id, msa_root) if use_msa else None
    sample_dict = build_rna_sample_entry(target_id, sequence, msa_path=msa_path)
    prepared_sample_dict, was_cropped = _prepare_sample_dict_for_inference(
        sample_dict,
        crop_size=crop_size,
    )

    sample2feat = SampleDictToFeatures(prepared_sample_dict)
    features_dict, atom_array, _ = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.as_tensor(
        atom_array.distogram_rep_atom_mask
    ).long()

    use_sample_msa = bool(use_msa) and msa_path is not None and not was_cropped
    if was_cropped and use_msa:
        LOGGER.warning(
            "MSA featurization is disabled for cropped sequences because the raw MSA files are full length."
        )

    msa_features = (
        InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=prepared_sample_dict["sequences"],
            atom_array=atom_array,
            msa_pair_as_unpair=bool(msa_pair_as_unpair),
            use_rna_msa=bool(use_rna_msa),
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

    return {
        "target_id": target_id,
        "sequence": prepared_sample_dict["sequences"][0]["rnaSequence"]["sequence"],
        "original_sequence": sequence,
        "sample_dict": prepared_sample_dict,
        "entity_poly_type": sample2feat.entity_poly_type,
        "atom_array": atom_array,
        "data": {"input_feature_dict": input_feature_dict},
        "n_token": int(input_feature_dict["token_index"].shape[0]),
        "msa_path": str(msa_path) if msa_path is not None else None,
        "used_msa": bool(use_sample_msa),
        "was_cropped": bool(was_cropped),
    }


def prepare_raw_examples(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    msa_root: str | Path | None = None,
    protenix_repo: str | Path | None = None,
    use_msa: bool = False,
    msa_pair_as_unpair: bool = True,
    use_rna_msa: bool = True,
    crop_size: int | None = None,
) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict("records")
    else:
        records = list(rows)

    return [
        prepare_raw_example(
            row,
            msa_root=msa_root,
            protenix_repo=protenix_repo,
            use_msa=use_msa,
            msa_pair_as_unpair=msa_pair_as_unpair,
            use_rna_msa=use_rna_msa,
            crop_size=crop_size,
        )
        for row in records
    ]


def _summary_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    return value


def summarize_prediction(prediction: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prediction_index, summary_confidence in enumerate(prediction["summary_confidence"]):
        row = {
            "prediction_index": int(prediction_index),
            "ranking_score": base._tensor_to_float(summary_confidence["ranking_score"]),
            "plddt": base._tensor_to_float(summary_confidence["plddt"]),
        }
        for optional_key in ["has_clash", "num_recycles", "ptm", "iptm", "gpde", "disorder"]:
            if optional_key in summary_confidence:
                value = _summary_value(summary_confidence[optional_key])
                if not isinstance(value, (list, tuple, dict)):
                    row[optional_key] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_columns = [col for col in ["ranking_score", "plddt"] if col in frame.columns]
    return frame.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)


def predict_raw_bundle(runner: Any, bundle: dict[str, Any]) -> dict[str, Any]:
    return base.predict_validation_bundle(runner, bundle)


def run_raw_inference(
    runner: Any,
    bundles: Iterable[dict[str, Any]],
    *,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    bundle_list = list(bundles)
    iterator: Iterable[dict[str, Any]] = bundle_list
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(bundle_list, desc="Raw RNA inference")
        except Exception:
            iterator = bundle_list

    results = []
    for bundle in iterator:
        prediction = predict_raw_bundle(runner, bundle)
        results.append(
            {
                "target_id": bundle["target_id"],
                "bundle": bundle,
                "prediction": prediction,
                "metrics": summarize_prediction(prediction),
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def summarize_raw_results(results: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for result in results:
        frame = result["metrics"].copy()
        frame.insert(0, "target_id", result["target_id"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_prediction_c1_trace(
    result: dict[str, Any],
    *,
    prediction_row: int = 0,
) -> tuple[Any, int]:
    row = result["metrics"].iloc[int(prediction_row)]
    prediction_index = int(row["prediction_index"])
    c1_coords = base.extract_c1_coordinates(
        result["bundle"]["atom_array"],
        result["prediction"]["coordinate"],
    )
    return c1_coords[prediction_index], prediction_index
