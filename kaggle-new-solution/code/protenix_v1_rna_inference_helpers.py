from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SOLUTION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTENIX_REPO = REPO_ROOT / "external" / "Protenix-v1-official"
DEFAULT_NOTEBOOK_OUTPUT_DIR = (
    SOLUTION_ROOT / "notebooks_output" / "protenix_v1_rna_inference"
)


def configure_notebook_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
        force=True,
    )


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _tensor_to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def resolve_rna_data_dir(data_dir: str | Path | None = None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    env_dir = os.environ.get("RNA_3D_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/scratch/phys/sin/rna-dataset"),
            Path("/kaggle/input/stanford-rna-3d-folding-2"),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and (candidate / "validation_sequences.csv").exists():
            return candidate.resolve()
    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve the RNA competition data directory. Checked:\n"
        f"{checked}"
    )


def resolve_cache_root(cache_root: str | Path | None = None) -> Path:
    candidates = []
    if cache_root is not None:
        candidates.append(Path(cache_root))
    env_root = os.environ.get("PROTENIX_RNA_CACHE_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            Path("/scratch/phys/sin/rna-dataset/preprocessed_data"),
            SOLUTION_ROOT / "references" / "preprocessed_data",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and (candidate / "samples" / "validation_samples.json").exists():
            return candidate.resolve()
    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve the Kaggle RNA cache root. Checked:\n"
        f"{checked}"
    )


def configure_official_repo(protenix_repo: str | Path | None = None) -> Path:
    repo_dir = Path(protenix_repo or DEFAULT_PROTENIX_REPO).expanduser().resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Missing official Protenix repo: {repo_dir}")

    os.environ.setdefault("PROTENIX_REPO_DIR", str(repo_dir))
    os.environ.setdefault("LAYERNORM_TYPE", "openfold")

    for path in (SOLUTION_ROOT / "code", repo_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_dir


def load_validation_tables(
    data_dir: str | Path | None = None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    resolved_data_dir = resolve_rna_data_dir(data_dir)
    validation_sequences = pd.read_csv(resolved_data_dir / "validation_sequences.csv")
    validation_labels = pd.read_csv(resolved_data_dir / "validation_labels.csv")
    if "pdb_id" not in validation_labels.columns:
        validation_labels["pdb_id"] = (
            validation_labels["ID"].astype(str).str.rsplit("_", n=1).str[0]
        )
    return resolved_data_dir, validation_sequences, validation_labels


def discover_checkpoints(
    protenix_repo: str | Path | None = None,
    extra_roots: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    repo_dir = Path(protenix_repo or DEFAULT_PROTENIX_REPO).expanduser().resolve()
    roots = [
        ("official_train_dir", Path("/scratch/phys/sin/rna-dataset/models")),
        ("official_release", Path("/scratch/phys/sin/rna-dataset/protenix_checkpoint")),
        ("official_release", repo_dir / "release_data" / "checkpoint"),
        ("run_checkpoint", Path("/scratch/phys/sin/rna-dataset/protenix_runs")),
    ]
    if extra_roots is not None:
        for root in extra_roots:
            roots.append(("extra", Path(root)))

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for source, root in roots:
        if not root.exists():
            continue
        if source == "run_checkpoint":
            pattern_paths = list(root.glob("*/checkpoints/*.pt")) + list(
                root.glob("*/checkpoints/*.ckpt")
            )
        else:
            pattern_paths = list(root.glob("*.pt")) + list(root.glob("*.ckpt"))

        for checkpoint_path in pattern_paths:
            resolved = checkpoint_path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            stat = resolved.stat()
            step_prefix = resolved.stem.split("_", 1)[0]
            step = int(step_prefix) if step_prefix.isdigit() else -1
            rows.append(
                {
                    "source": source,
                    "path": str(resolved),
                    "filename": resolved.name,
                    "parent": resolved.parent.name,
                    "is_ema": "_ema_" in resolved.name,
                    "step": step,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "mtime": pd.Timestamp.fromtimestamp(stat.st_mtime),
                }
            )

    checkpoints = pd.DataFrame(rows)
    if checkpoints.empty:
        return checkpoints

    source_order = {
        "official_train_dir": 0,
        "official_release": 1,
        "run_checkpoint": 2,
        "extra": 3,
    }
    checkpoints["source_rank"] = checkpoints["source"].map(source_order).fillna(99)
    checkpoints = checkpoints.sort_values(
        ["source_rank", "is_ema", "step", "mtime", "filename"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    return checkpoints


def choose_checkpoint(
    checkpoint_path: str | Path | None = None,
    checkpoints: pd.DataFrame | None = None,
) -> Path:
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {resolved}")
        return resolved

    if checkpoints is None:
        checkpoints = discover_checkpoints()
    if checkpoints.empty:
        raise FileNotFoundError("No Protenix checkpoints were discovered.")

    for source in ["official_train_dir", "official_release", "run_checkpoint", "extra"]:
        subset = checkpoints[checkpoints["source"] == source]
        if not subset.empty:
            return Path(subset.iloc[0]["path"])
    return Path(checkpoints.iloc[0]["path"])


def build_inference_configs(
    model_name: str = "protenix_base_default_v1.0.0",
    *,
    protenix_repo: str | Path | None = None,
    dtype: str = "bf16",
    use_msa: bool = True,
    use_rna_msa: bool = True,
    n_diffusion_samples: int = 5,
    n_diffusion_steps: int = 20,
    mc_dropout_apply_rate: float = 0.0,
    output_dir: str | Path | None = None,
) -> Any:
    configure_official_repo(protenix_repo)

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    from runner.inference import update_gpu_compatible_configs

    base_configs = deepcopy(configs_base)
    base_configs["data"] = deepcopy(data_configs)
    _deep_update(base_configs, deepcopy(inference_configs))
    _deep_update(base_configs, deepcopy(model_configs[model_name]))

    configs = parse_configs(configs=base_configs, fill_required_with_null=True)
    configs.model_name = model_name
    configs.dtype = dtype
    configs.num_workers = 0
    configs.use_msa = bool(use_msa)
    configs.use_rna_msa = bool(use_rna_msa)
    configs.use_template = False
    configs.use_seeds_in_json = False
    configs.need_atom_confidence = False
    configs.sorted_by_ranking_score = True
    configs.load_strict = True
    configs.sample_diffusion.N_sample = int(n_diffusion_samples)
    configs.sample_diffusion.N_step = int(n_diffusion_steps)
    configs.mc_dropout_apply_rate = float(mc_dropout_apply_rate)
    configs.dump_dir = str(
        Path(output_dir or DEFAULT_NOTEBOOK_OUTPUT_DIR).expanduser().resolve()
    )
    return update_gpu_compatible_configs(configs)


def load_runner(
    checkpoint_path: str | Path,
    *,
    model_name: str = "protenix_base_default_v1.0.0",
    protenix_repo: str | Path | None = None,
    dtype: str = "bf16",
    use_msa: bool = True,
    use_rna_msa: bool = True,
    n_diffusion_samples: int = 5,
    n_diffusion_steps: int = 20,
    mc_dropout_apply_rate: float = 0.0,
    output_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    configure_official_repo(protenix_repo)

    from runner.inference import InferenceRunner

    class LocalCheckpointInferenceRunner(InferenceRunner):
        def __init__(self, configs: Any, user_checkpoint_path: str | Path) -> None:
            self._user_checkpoint_path = str(
                Path(user_checkpoint_path).expanduser().resolve()
            )
            super().__init__(configs)

        def load_checkpoint(self) -> None:
            checkpoint_path_resolved = Path(self._user_checkpoint_path)
            if not checkpoint_path_resolved.exists():
                raise FileNotFoundError(
                    f"Given checkpoint path does not exist [{checkpoint_path_resolved}]"
                )
            checkpoint = torch.load(
                checkpoint_path_resolved,
                map_location=self.device,
                weights_only=False,
            )
            sample_key = list(checkpoint["model"].keys())[0]
            if sample_key.startswith("module."):
                checkpoint["model"] = {
                    key[len("module.") :]: value
                    for key, value in checkpoint["model"].items()
                }
            self.model.load_state_dict(
                state_dict=checkpoint["model"],
                strict=self.configs.load_strict,
            )
            self.model.eval()

    configs = build_inference_configs(
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
    runner = LocalCheckpointInferenceRunner(configs, checkpoint_path)
    return runner, configs


def prepare_validation_example(
    target_id: str,
    *,
    cache_root: str | Path | None = None,
    protenix_repo: str | Path | None = None,
    use_msa: bool = True,
    crop_size: int = 420,
    msa_pair_as_unpair: bool = True,
    use_rna_msa: bool = True,
) -> dict[str, Any]:
    configure_official_repo(protenix_repo)
    resolved_cache_root = resolve_cache_root(cache_root)

    import kaggle_rna_official_runtime as runtime_module

    dataset = runtime_module.KaggleRNADatasetOfficial(
        cache_root=str(resolved_cache_root),
        split="validation",
        use_msa=bool(use_msa),
        crop_size=int(crop_size),
        msa_pair_as_unpair=bool(msa_pair_as_unpair),
        use_rna_msa=bool(use_rna_msa),
    )
    sample_dict = next(
        (item for item in dataset.inputs if item["name"] == target_id),
        None,
    )
    if sample_dict is None:
        raise KeyError(f"Target {target_id!r} was not found in validation_samples.json")

    label = dataset._load_label(target_id)
    (
        prepared_sample_dict,
        residue_coordinate_multi,
        residue_mask_multi,
        was_cropped,
    ) = dataset._prepare_cropped_sample(sample_dict, label)

    sample2feat = runtime_module.SampleDictToFeatures(prepared_sample_dict)
    features_dict, atom_array, _ = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.as_tensor(
        atom_array.distogram_rep_atom_mask
    ).long()

    use_sample_msa = bool(use_msa) and not was_cropped
    msa_features = (
        runtime_module.InferenceMSAFeaturizer.make_msa_feature(
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
        features_dict.update(runtime_module.dict_to_tensor(msa_features))

    input_feature_dict = runtime_module.data_type_transform(
        feat_or_label_dict=runtime_module.make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )
    )
    input_feature_dict["is_distillation"] = torch.tensor([0])
    input_feature_dict["resolution"] = torch.tensor([-1.0], dtype=torch.float32)

    return {
        "target_id": target_id,
        "sequence": prepared_sample_dict["sequences"][0]["rnaSequence"]["sequence"],
        "sample_dict": prepared_sample_dict,
        "entity_poly_type": sample2feat.entity_poly_type,
        "atom_array": atom_array,
        "data": {"input_feature_dict": input_feature_dict},
        "n_token": int(input_feature_dict["token_index"].shape[0]),
        "residue_coordinate_multi": np.asarray(residue_coordinate_multi, dtype=np.float32),
        "residue_mask_multi": np.asarray(residue_mask_multi, dtype=bool),
        "was_cropped": bool(was_cropped),
        "used_msa": bool(use_sample_msa),
    }


def prepare_validation_examples(
    target_ids: Iterable[str],
    *,
    cache_root: str | Path | None = None,
    protenix_repo: str | Path | None = None,
    use_msa: bool = True,
    crop_size: int = 420,
    msa_pair_as_unpair: bool = True,
    use_rna_msa: bool = True,
) -> list[dict[str, Any]]:
    return [
        prepare_validation_example(
            target_id=target_id,
            cache_root=cache_root,
            protenix_repo=protenix_repo,
            use_msa=use_msa,
            crop_size=crop_size,
            msa_pair_as_unpair=msa_pair_as_unpair,
            use_rna_msa=use_rna_msa,
        )
        for target_id in target_ids
    ]


def extract_c1_coordinates(atom_array: Any, coordinates: Any) -> np.ndarray:
    atom_coordinates = _to_numpy(coordinates)
    c1_mask = np.asarray([atom.atom_name == "C1'" for atom in atom_array], dtype=bool)
    return atom_coordinates[..., c1_mask, :]


def kabsch_align(pred_coords: np.ndarray, true_coords: np.ndarray) -> np.ndarray:
    pred_center = pred_coords.mean(axis=0, keepdims=True)
    true_center = true_coords.mean(axis=0, keepdims=True)
    pred_shifted = pred_coords - pred_center
    true_shifted = true_coords - true_center
    covariance = pred_shifted.T @ true_shifted
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1
        rotation = right_t.T @ left.T
    return pred_shifted @ rotation + true_center


def aligned_rmsd(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if mask is not None:
        pred_coords = pred_coords[mask]
        true_coords = true_coords[mask]
    if pred_coords.shape[0] < 3:
        return float("nan"), pred_coords
    aligned_pred = kabsch_align(pred_coords, true_coords)
    diff = aligned_pred - true_coords
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
    return rmsd, aligned_pred


def drmsd(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is not None:
        pred_coords = pred_coords[mask]
        true_coords = true_coords[mask]
    if pred_coords.shape[0] < 3:
        return float("nan")
    pred_dist = np.linalg.norm(pred_coords[:, None, :] - pred_coords[None, :, :], axis=-1)
    true_dist = np.linalg.norm(true_coords[:, None, :] - true_coords[None, :, :], axis=-1)
    upper = np.triu_indices(pred_coords.shape[0], k=1)
    return float(np.sqrt(np.mean((pred_dist[upper] - true_dist[upper]) ** 2)))


def score_prediction(bundle: dict[str, Any], prediction: dict[str, Any]) -> pd.DataFrame:
    pred_c1 = extract_c1_coordinates(bundle["atom_array"], prediction["coordinate"])
    gt_c1 = np.asarray(bundle["residue_coordinate_multi"], dtype=np.float32)
    gt_mask = np.asarray(bundle["residue_mask_multi"], dtype=bool)

    rows: list[dict[str, Any]] = []
    for pred_idx in range(pred_c1.shape[0]):
        best_gt_index = None
        best_rmsd = float("inf")
        best_aligned = None
        best_drmsd = float("nan")
        for gt_idx in range(gt_c1.shape[0]):
            mask = gt_mask[min(gt_idx, gt_mask.shape[0] - 1)]
            rmsd, aligned_pred = aligned_rmsd(pred_c1[pred_idx], gt_c1[gt_idx], mask=mask)
            if np.isnan(rmsd) or rmsd >= best_rmsd:
                continue
            best_rmsd = rmsd
            best_gt_index = gt_idx
            best_aligned = aligned_pred
            best_drmsd = drmsd(pred_c1[pred_idx], gt_c1[gt_idx], mask=mask)

        summary_confidence = prediction["summary_confidence"][pred_idx]
        rows.append(
            {
                "prediction_index": pred_idx,
                "ranking_score": _tensor_to_float(summary_confidence["ranking_score"]),
                "plddt": _tensor_to_float(summary_confidence["plddt"]),
                "best_gt_index": best_gt_index,
                "aligned_rmsd": best_rmsd if best_gt_index is not None else float("nan"),
                "drmsd": best_drmsd,
                "aligned_prediction": best_aligned,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["ranking_score", "aligned_rmsd"],
        ascending=[False, True],
    ).reset_index(drop=True)


def predict_validation_bundle(runner: Any, bundle: dict[str, Any]) -> dict[str, Any]:
    from runner.inference import update_inference_configs

    updated_configs = update_inference_configs(runner.configs, bundle["n_token"])
    runner.configs = updated_configs
    runner.update_model_configs(updated_configs)
    with torch.no_grad():
        return runner.predict(bundle["data"])


def run_validation_inference(
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

            iterator = tqdm(bundle_list, desc="Validation inference")
        except Exception:
            iterator = bundle_list

    results = []
    for bundle in iterator:
        prediction = predict_validation_bundle(runner, bundle)
        results.append(
            {
                "target_id": bundle["target_id"],
                "bundle": bundle,
                "prediction": prediction,
                "metrics": score_prediction(bundle, prediction),
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def summarize_results(results: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for result in results:
        frame = result["metrics"].copy()
        frame.insert(0, "target_id", result["target_id"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_result(results: Iterable[dict[str, Any]], target_id: str) -> dict[str, Any]:
    for result in results:
        if result["target_id"] == target_id:
            return result
    raise KeyError(f"Could not find prediction result for {target_id!r}")


def plot_rna_3d_interactive(
    coords: np.ndarray | torch.Tensor,
    *,
    title: str = "RNA 3D trace",
    color: str = "#1f77b4",
) -> Any:
    import plotly.graph_objects as go

    xyz = _to_numpy(coords)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="lines+markers",
            name=title,
            marker=dict(size=4, color=color),
            line=dict(width=6, color=color),
        )
    )
    figure.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=36),
        height=640,
    )
    return figure


def plot_prediction_vs_truth(
    pred_coords: np.ndarray | torch.Tensor,
    true_coords: np.ndarray | torch.Tensor,
    *,
    title: str = "Prediction vs truth",
    pred_name: str = "Prediction",
    true_name: str = "Ground truth",
) -> Any:
    import plotly.graph_objects as go

    pred_xyz = _to_numpy(pred_coords)
    true_xyz = _to_numpy(true_coords)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=true_xyz[:, 0],
            y=true_xyz[:, 1],
            z=true_xyz[:, 2],
            mode="lines+markers",
            name=true_name,
            marker=dict(size=4, color="#1f77b4"),
            line=dict(width=6, color="#1f77b4"),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=pred_xyz[:, 0],
            y=pred_xyz[:, 1],
            z=pred_xyz[:, 2],
            mode="lines+markers",
            name=pred_name,
            marker=dict(size=4, color="#d62728"),
            line=dict(width=6, color="#d62728"),
        )
    )
    figure.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=36),
        height=700,
    )
    return figure


def save_prediction_artifacts(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int = 101,
    need_atom_confidence: bool = False,
) -> Path:
    configure_official_repo()

    from runner.dumper import DataDumper

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    dumper = DataDumper(
        base_dir=str(output_path),
        need_atom_confidence=need_atom_confidence,
        sorted_by_ranking_score=True,
    )
    dumper.dump(
        dataset_name="",
        pdb_id=result["target_id"],
        seed=seed,
        pred_dict=result["prediction"],
        atom_array=result["bundle"]["atom_array"],
        entity_poly_type=result["bundle"]["entity_poly_type"],
    )
    return output_path / result["target_id"] / f"seed_{seed}" / "predictions"
