#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

SOLUTION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTENIX_REPO = SOLUTION_ROOT.parent / "external" / "Protenix-v1-official"
DEFAULT_RAW_STRUCTURE_ROOT = Path("/scratch/phys/sin/rna-dataset/PDB_RNA")
DEFAULT_MSA_ROOT = Path("/scratch/phys/sin/rna-dataset/MSA")
DEFAULT_CCD_ROOT = Path("/scratch/phys/sin/rna-dataset/protenix_ccd_cache")
DEFAULT_SPLIT_MANIFEST = Path("/scratch/phys/sin/rna-dataset")
DEFAULT_OUT_ROOT = Path("/scratch/phys/sin/rna-dataset/protenix_native_kaggle_subset")
DEFAULT_OFFICIAL_TEMPLATE = (
    DEFAULT_PROTENIX_REPO / "examples" / "examples_with_rna_msa" / "example_9gmw_2.json"
)
VALID_SPLITS = ("train", "validation")
WORKER_CONTEXT: dict[str, Any] = {}


@dataclass
class RequestedSample:
    target_id: str
    split: str
    sequence: str


@dataclass
class ManifestRow:
    target_id: str
    split: str
    sequence: str
    structure_path: str
    msa_path: str
    canonical_msa_id: str
    conversion_status: str
    failure_reason: str
    bioassembly_path: str
    indices_rows: int
    matched_entity_ids: str


@dataclass
class ConversionResult:
    order: int
    requested: RequestedSample
    manifest: ManifestRow
    sample_indices_list: list[dict[str, Any]]
    ordered_rna_sequences: list[str]


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def default_num_workers() -> int:
    candidate = (
        os.environ.get("NATIVE_PREP_NUM_WORKERS")
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or "1"
    )
    try:
        return max(1, int(candidate))
    except ValueError:
        return 1


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


def resolve_official_imports(repo_dir: Path):
    if not repo_dir.exists():
        raise FileNotFoundError(f"Missing official Protenix repo: {repo_dir}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    data_pipeline_module = importlib.import_module(
        "protenix.data.pipeline.data_pipeline"
    )
    file_io_module = importlib.import_module("protenix.utils.file_io")
    return data_pipeline_module.DataPipeline, file_io_module.dump_gzip_pickle


def read_split_rows_from_csv(csv_path: Path, split: str) -> list[RequestedSample]:
    rows: list[RequestedSample] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"target_id", "sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
        for row in reader:
            rows.append(
                RequestedSample(
                    target_id=str(row["target_id"]).strip().upper(),
                    split=split,
                    sequence=normalize_sequence(row["sequence"]),
                )
            )
    return rows


def read_split_manifest(split_manifest: Path) -> list[RequestedSample]:
    if split_manifest.is_dir():
        rows: list[RequestedSample] = []
        for split in VALID_SPLITS:
            csv_path = split_manifest / f"{split}_sequences.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing split CSV: {csv_path}")
            rows.extend(read_split_rows_from_csv(csv_path, split))
        return rows

    if split_manifest.suffix.lower() == ".csv":
        rows: list[RequestedSample] = []
        with split_manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"target_id", "split", "sequence"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Missing columns in {split_manifest}: {sorted(missing)}"
                )
            for row in reader:
                split = str(row["split"]).strip().lower()
                if split not in VALID_SPLITS:
                    raise ValueError(f"Unsupported split {split!r} in {split_manifest}")
                rows.append(
                    RequestedSample(
                        target_id=str(row["target_id"]).strip().upper(),
                        split=split,
                        sequence=normalize_sequence(row["sequence"]),
                    )
                )
        return rows

    if split_manifest.suffix.lower() == ".json":
        payload = json.loads(split_manifest.read_text(encoding="utf-8"))
        rows = []
        for row in payload:
            split = str(row["split"]).strip().lower()
            if split not in VALID_SPLITS:
                raise ValueError(f"Unsupported split {split!r} in {split_manifest}")
            rows.append(
                RequestedSample(
                    target_id=str(row["target_id"]).strip().upper(),
                    split=split,
                    sequence=normalize_sequence(row["sequence"]),
                )
            )
        return rows

    raise ValueError(f"Unsupported split manifest path: {split_manifest}")


def apply_split_limits(
    rows: Iterable[RequestedSample],
    limit_per_split: int,
) -> list[RequestedSample]:
    if limit_per_split <= 0:
        return list(rows)
    counts = defaultdict(int)
    limited_rows: list[RequestedSample] = []
    for row in rows:
        if counts[row.split] >= limit_per_split:
            continue
        limited_rows.append(row)
        counts[row.split] += 1
    return limited_rows


def resolve_structure_path(raw_structure_root: Path, target_id: str) -> Path | None:
    stems = [target_id.lower(), target_id.upper(), target_id]
    suffixes = [".cif", ".cif.gz", ".mmcif", ".mmcif.gz"]
    for stem in stems:
        for suffix in suffixes:
            candidate = raw_structure_root / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def resolve_msa_path(msa_root: Path, target_id: str) -> Path | None:
    candidates = [
        msa_root / f"{target_id}.MSA.fasta",
        msa_root / f"{target_id}.a3m",
        msa_root / f"{target_id}.fasta",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)


def read_alignment_records(msa_path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw_line in msa_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line
            chunks = []
            continue
        chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def write_alignment_records(output_path: Path, records: Sequence[tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f"{header}\n")
            handle.write(f"{sequence}\n")


def strip_alignment_gaps(sequence: str) -> str:
    return "".join(char for char in sequence if char not in "-.")


def alignment_depth_score(records: Sequence[tuple[str, str]]) -> tuple[int, int]:
    non_query = 0
    non_gap = 0
    for index, (_, sequence) in enumerate(records):
        if index == 0:
            continue
        if any(char not in "-." for char in sequence):
            non_query += 1
            non_gap += sum(char not in "-." for char in sequence)
    return non_query, non_gap


def extract_rna_chains(bioassembly_dict: dict[str, Any]) -> list[tuple[str, str]]:
    atom_array = bioassembly_dict["atom_array"]
    token_array = bioassembly_dict["token_array"]
    centre_atoms = atom_array[token_array.get_annotation("centre_atom_index")]

    seen_asym_ids: set[int] = set()
    rna_chains: list[tuple[str, str]] = []
    entity_poly_type = bioassembly_dict.get("entity_poly_type", {})
    sample_sequences = bioassembly_dict.get("sequences", {})

    for asym_id in centre_atoms.asym_id_int:
        asym_id_int = int(asym_id)
        if asym_id_int in seen_asym_ids:
            continue
        seen_asym_ids.add(asym_id_int)

        mask = centre_atoms.asym_id_int == asym_id
        entity_id = str(centre_atoms.label_entity_id[mask][0])
        poly_type = str(entity_poly_type.get(entity_id, ""))
        if "ribonucleotide" not in poly_type:
            continue

        sequence = normalize_sequence(sample_sequences.get(entity_id, ""))
        if not sequence:
            continue
        rna_chains.append((entity_id, sequence))

    return rna_chains


def match_target_sequence_to_rna_chains(
    target_sequence: str,
    rna_chains: Sequence[tuple[str, str]],
) -> list[str] | None:
    if not rna_chains:
        return None

    seq_counter = Counter(sequence for _, sequence in rna_chains if sequence)
    if not seq_counter:
        return None

    total_length = sum(len(sequence) * count for sequence, count in seq_counter.items())
    if total_length != len(target_sequence):
        return None

    seq_keys = tuple(sorted(seq_counter, key=len, reverse=True))
    initial_counts = tuple(seq_counter[key] for key in seq_keys)

    @lru_cache(maxsize=None)
    def _solve(position: int, counts: tuple[int, ...]) -> tuple[str, ...] | None:
        if position == len(target_sequence):
            return () if all(count == 0 for count in counts) else None

        for index, sequence in enumerate(seq_keys):
            if counts[index] <= 0:
                continue
            if not target_sequence.startswith(sequence, position):
                continue
            next_counts = list(counts)
            next_counts[index] -= 1
            tail = _solve(position + len(sequence), tuple(next_counts))
            if tail is not None:
                return (sequence,) + tail
        return None

    match = _solve(0, initial_counts)
    return list(match) if match is not None else None


def compute_query_spans(
    query_alignment: str,
    requested_sequence: str,
    ordered_rna_sequences: Sequence[str],
) -> list[tuple[int, int]]:
    normalized_query = normalize_sequence(strip_alignment_gaps(query_alignment))
    if normalized_query != requested_sequence:
        raise ValueError(
            f"MSA query mismatch: expected {len(requested_sequence)} residues but got {len(normalized_query)}"
        )

    spans: list[tuple[int, int]] = []
    cursor = 0
    for sequence in ordered_rna_sequences:
        sequence_length = len(sequence)
        if query_alignment[cursor : cursor + sequence_length] != sequence:
            raise ValueError(
                f"Target sequence is not aligned contiguously for segment of length {sequence_length}"
            )
        spans.append((cursor, cursor + sequence_length))
        cursor += sequence_length

    if cursor != len(query_alignment):
        raise ValueError(
            f"Unused query suffix remains after RNA segmentation: consumed {cursor}, total {len(query_alignment)}"
        )
    return spans


def split_full_target_rna_msa(
    msa_path: Path,
    requested_sequence: str,
    ordered_rna_sequences: Sequence[str],
) -> dict[str, list[tuple[str, str]]]:
    records = read_alignment_records(msa_path)
    if not records:
        raise ValueError(f"MSA file is empty: {msa_path}")

    query_alignment = records[0][1]
    spans = compute_query_spans(query_alignment, requested_sequence, ordered_rna_sequences)
    segments_by_sequence: dict[str, list[tuple[str, str]]] = {}

    for sequence, (start, end) in zip(ordered_rna_sequences, spans):
        segment_records: list[tuple[str, str]] = []
        for index, (header, aligned_sequence) in enumerate(records):
            if len(aligned_sequence) < end:
                raise ValueError(
                    f"Alignment row shorter than expected segment end {end}: {header}"
                )
            segment = aligned_sequence[start:end]
            if index == 0 or any(char not in "-." for char in segment):
                segment_records.append((header, segment))

        existing = segments_by_sequence.get(sequence)
        if existing is None or alignment_depth_score(segment_records) > alignment_depth_score(existing):
            segments_by_sequence[sequence] = segment_records

    return segments_by_sequence


def canonical_msa_id_for_sequence(sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]
    return f"rna_{digest}"


def format_failure_reason(prefix: str, exc: Exception | None = None) -> str:
    if exc is None:
        return prefix
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 200:
        message = message[:200]
    return f"{prefix}:{type(exc).__name__}:{message}"


def make_manifest(
    requested: RequestedSample,
    structure_path: Path | None,
    msa_path: Path | None,
) -> ManifestRow:
    return ManifestRow(
        target_id=requested.target_id,
        split=requested.split,
        sequence=requested.sequence,
        structure_path=str(structure_path) if structure_path else "",
        msa_path=str(msa_path) if msa_path else "",
        canonical_msa_id="",
        conversion_status="pending",
        failure_reason="",
        bioassembly_path="",
        indices_rows=0,
        matched_entity_ids="",
    )


def init_worker(
    protenix_repo: str,
    ccd_root: str,
    bioassembly_dir: str,
    raw_structure_root: str,
    msa_root: str,
) -> None:
    DataPipeline, dump_gzip_pickle = resolve_official_imports(Path(protenix_repo))
    configure_ccd_paths(Path(ccd_root))
    WORKER_CONTEXT["DataPipeline"] = DataPipeline
    WORKER_CONTEXT["dump_gzip_pickle"] = dump_gzip_pickle
    WORKER_CONTEXT["bioassembly_dir"] = Path(bioassembly_dir)
    WORKER_CONTEXT["raw_structure_root"] = Path(raw_structure_root)
    WORKER_CONTEXT["msa_root"] = Path(msa_root)


def convert_requested_sample(task: tuple[int, RequestedSample]) -> ConversionResult:
    order, requested = task
    raw_structure_root = WORKER_CONTEXT["raw_structure_root"]
    msa_root = WORKER_CONTEXT["msa_root"]
    DataPipeline = WORKER_CONTEXT["DataPipeline"]
    dump_gzip_pickle = WORKER_CONTEXT["dump_gzip_pickle"]
    bioassembly_dir = WORKER_CONTEXT["bioassembly_dir"]

    structure_path = resolve_structure_path(raw_structure_root, requested.target_id)
    msa_path = resolve_msa_path(msa_root, requested.target_id)
    manifest = make_manifest(requested, structure_path, msa_path)

    if structure_path is None:
        manifest.conversion_status = "failed"
        manifest.failure_reason = "missing_structure"
        return ConversionResult(order, requested, manifest, [], [])
    if msa_path is None:
        manifest.conversion_status = "failed"
        manifest.failure_reason = "missing_msa"
        return ConversionResult(order, requested, manifest, [], [])

    try:
        sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            structure_path,
            pdb_cluster_file=None,
            dataset="WeightedPDB",
        )
    except Exception as exc:
        manifest.conversion_status = "failed"
        manifest.failure_reason = format_failure_reason("official_prepare_exception", exc)
        return ConversionResult(order, requested, manifest, [], [])

    if not sample_indices_list or not bioassembly_dict:
        manifest.conversion_status = "failed"
        manifest.failure_reason = "official_prepare_failed"
        return ConversionResult(order, requested, manifest, [], [])

    rna_chains = extract_rna_chains(bioassembly_dict)
    ordered_rna_sequences = match_target_sequence_to_rna_chains(
        requested.sequence,
        rna_chains,
    )
    if ordered_rna_sequences is None:
        manifest.conversion_status = "failed"
        manifest.failure_reason = "sequence_mismatch"
        manifest.matched_entity_ids = ";".join(dict.fromkeys(entity_id for entity_id, _ in rna_chains))
        return ConversionResult(order, requested, manifest, [], [])

    pdb_id = str(
        sample_indices_list[0].get("pdb_id", bioassembly_dict.get("pdb_id", requested.target_id))
    ).lower()
    bioassembly_path = bioassembly_dir / f"{pdb_id}.pkl.gz"
    try:
        dump_gzip_pickle(bioassembly_dict, bioassembly_path)
    except Exception as exc:
        manifest.conversion_status = "failed"
        manifest.failure_reason = format_failure_reason("bioassembly_dump_exception", exc)
        return ConversionResult(order, requested, manifest, [], [])

    for sample_indices in sample_indices_list:
        sample_indices["target_id"] = requested.target_id
        sample_indices["source_split"] = requested.split

    manifest.conversion_status = "success"
    manifest.bioassembly_path = str(bioassembly_path)
    manifest.indices_rows = len(sample_indices_list)
    manifest.matched_entity_ids = ";".join(dict.fromkeys(entity_id for entity_id, _ in rna_chains))
    return ConversionResult(
        order=order,
        requested=requested,
        manifest=manifest,
        sample_indices_list=sample_indices_list,
        ordered_rna_sequences=ordered_rna_sequences,
    )


def run_conversions(
    requested_rows: list[RequestedSample],
    num_workers: int,
    protenix_repo: Path,
    ccd_root: Path,
    bioassembly_dir: Path,
    raw_structure_root: Path,
    msa_root: Path,
) -> list[ConversionResult]:
    initargs = (
        str(protenix_repo),
        str(ccd_root),
        str(bioassembly_dir),
        str(raw_structure_root),
        str(msa_root),
    )
    tasks = list(enumerate(requested_rows))
    total = len(tasks)

    if num_workers <= 1:
        init_worker(*initargs)
        results: list[ConversionResult] = []
        for completed, task in enumerate(tasks, start=1):
            results.append(convert_requested_sample(task))
            if completed % 25 == 0 or completed == total:
                print(f"Processed {completed}/{total} targets")
        return results

    results: list[ConversionResult] = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=initargs,
    ) as executor:
        futures = [executor.submit(convert_requested_sample, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == total:
                print(f"Processed {completed}/{total} targets")
    results.sort(key=lambda result: result.order)
    return results


def register_rna_msa_assets(
    requested: RequestedSample,
    manifest: ManifestRow,
    ordered_rna_sequences: Sequence[str],
    canonical_msa_by_sequence: dict[str, str],
    rna_msa_dir: Path,
) -> list[str]:
    segments_by_sequence = split_full_target_rna_msa(
        Path(manifest.msa_path),
        requested.sequence,
        ordered_rna_sequences,
    )
    canonical_ids: list[str] = []
    for sequence in dict.fromkeys(ordered_rna_sequences):
        if sequence not in segments_by_sequence:
            raise ValueError(f"No split MSA segment generated for RNA sequence of length {len(sequence)}")
        canonical_id = canonical_msa_by_sequence.get(sequence)
        if canonical_id is None:
            canonical_id = canonical_msa_id_for_sequence(sequence)
            canonical_msa_by_sequence[sequence] = canonical_id
        output_dir = rna_msa_dir / canonical_id
        output_path = output_dir / f"{canonical_id}_all.a3m"
        if not output_path.exists():
            write_alignment_records(output_path, segments_by_sequence[sequence])
        canonical_ids.append(canonical_id)
    return canonical_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a native-format Protenix dataset from the Kaggle RNA raw mmCIF and MSA assets."
    )
    parser.add_argument(
        "--raw-structure-root",
        type=Path,
        default=DEFAULT_RAW_STRUCTURE_ROOT,
    )
    parser.add_argument(
        "--msa-root",
        type=Path,
        default=DEFAULT_MSA_ROOT,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--ccd-root",
        type=Path,
        default=DEFAULT_CCD_ROOT,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
    )
    parser.add_argument(
        "--official-template",
        type=Path,
        default=DEFAULT_OFFICIAL_TEMPLATE,
    )
    parser.add_argument(
        "--protenix-repo",
        type=Path,
        default=Path(os.environ.get("PROTENIX_REPO_DIR", str(DEFAULT_PROTENIX_REPO))),
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="If > 0, only convert the first N targets per split for an audited subset build.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_num_workers(),
        help="Number of CPU worker processes to use for per-structure conversion.",
    )
    args = parser.parse_args()

    raw_structure_root = args.raw_structure_root.expanduser().resolve()
    msa_root = args.msa_root.expanduser().resolve()
    split_manifest = args.split_manifest.expanduser().resolve()
    ccd_root = args.ccd_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    official_template = args.official_template.expanduser().resolve()
    protenix_repo = args.protenix_repo.expanduser().resolve()
    num_workers = max(1, args.num_workers)

    if not raw_structure_root.exists():
        raise FileNotFoundError(f"Missing raw structure root: {raw_structure_root}")
    if not msa_root.exists():
        raise FileNotFoundError(f"Missing MSA root: {msa_root}")
    if not split_manifest.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest}")
    if not ccd_root.exists():
        raise FileNotFoundError(f"Missing CCD root: {ccd_root}")
    if not official_template.exists():
        raise FileNotFoundError(f"Missing official template reference: {official_template}")

    requested_rows = read_split_manifest(split_manifest)
    requested_rows = apply_split_limits(requested_rows, args.limit_per_split)

    if not requested_rows:
        raise ValueError("No requested samples were found in the split manifest.")

    seen_target_ids = set()
    for row in requested_rows:
        if row.target_id in seen_target_ids:
            raise ValueError(f"Duplicate target ID in requested splits: {row.target_id}")
        seen_target_ids.add(row.target_id)

    bioassembly_dir = out_root / "bioassembly"
    indices_dir = out_root / "indices"
    mappings_dir = out_root / "mappings"
    manifests_dir = out_root / "manifests"
    rna_msa_dir = out_root / "rna_msa"
    for directory in [bioassembly_dir, indices_dir, mappings_dir, manifests_dir, rna_msa_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Requested samples to convert: {len(requested_rows)}")
    print(f"Parallel worker processes: {num_workers}")

    conversion_results = run_conversions(
        requested_rows=requested_rows,
        num_workers=num_workers,
        protenix_repo=protenix_repo,
        ccd_root=ccd_root,
        bioassembly_dir=bioassembly_dir,
        raw_structure_root=raw_structure_root,
        msa_root=msa_root,
    )

    manifest_rows: list[ManifestRow] = []
    split_indices: dict[str, list[dict[str, Any]]] = {split: [] for split in VALID_SPLITS}
    canonical_msa_by_sequence: dict[str, str] = {}

    for result in conversion_results:
        manifest = result.manifest
        if manifest.conversion_status == "success":
            try:
                canonical_ids = register_rna_msa_assets(
                    requested=result.requested,
                    manifest=manifest,
                    ordered_rna_sequences=result.ordered_rna_sequences,
                    canonical_msa_by_sequence=canonical_msa_by_sequence,
                    rna_msa_dir=rna_msa_dir,
                )
                manifest.canonical_msa_id = ";".join(canonical_ids)
            except Exception as exc:
                manifest.conversion_status = "failed"
                manifest.failure_reason = format_failure_reason("rna_msa_split_failed", exc)

        manifest_rows.append(manifest)
        if manifest.conversion_status != "success":
            continue
        split_indices[result.requested.split].extend(result.sample_indices_list)

    for split in VALID_SPLITS:
        write_csv(split_indices[split], indices_dir / f"{split}_indices.csv")

    write_csv([asdict(row) for row in manifest_rows], manifests_dir / "conversion_manifest.csv")

    metadata = {
        "native_data_root": str(out_root),
        "raw_structure_root": str(raw_structure_root),
        "msa_root": str(msa_root),
        "split_manifest": str(split_manifest),
        "ccd_root": str(ccd_root),
        "official_template": str(official_template),
        "official_repo": str(protenix_repo),
        "limit_per_split": args.limit_per_split,
        "num_workers": num_workers,
        "requested_samples": len(requested_rows),
        "successful_samples": sum(row.conversion_status == "success" for row in manifest_rows),
        "failed_samples": sum(row.conversion_status != "success" for row in manifest_rows),
    }
    (out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (mappings_dir / "rna_seq_to_msadir.json").write_text(
        json.dumps(
            {sequence: [canonical_id] for sequence, canonical_id in canonical_msa_by_sequence.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (mappings_dir / "empty_lookup.json").write_text("{}\n", encoding="utf-8")

    summary = {
        "requested": len(requested_rows),
        "successful": metadata["successful_samples"],
        "failed": metadata["failed_samples"],
        "train_indices_rows": len(split_indices["train"]),
        "validation_indices_rows": len(split_indices["validation"]),
        "unique_rna_msa_sequences": len(canonical_msa_by_sequence),
    }
    (manifests_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Native dataset root: {out_root}")
    print(f"Requested samples: {summary['requested']}")
    print(f"Successful samples: {summary['successful']}")
    print(f"Failed samples: {summary['failed']}")
    print(f"Train indices rows: {summary['train_indices_rows']}")
    print(f"Validation indices rows: {summary['validation_indices_rows']}")
    print(f"Unique RNA MSA sequences: {summary['unique_rna_msa_sequences']}")


if __name__ == "__main__":
    main()
