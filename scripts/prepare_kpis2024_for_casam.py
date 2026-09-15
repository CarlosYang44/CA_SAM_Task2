#!/usr/bin/env python3
"""Prepare KPIs2024 56Nx and DN patch data for the CA-SAM loader.

The source ZIP archives are left untouched. The generated layout is:

    <output-root>/56Nx/{dataset.json,image,label,imask}
    <output-root>/DN/{dataset.json,image,label,imask}

The official CA-SAM training loader requires sparse NPZ labels and an NPY
pseudo-mask for every training sample. The released training code currently
uses ground-truth prompts, but still loads and validates the pseudo-mask. This
script therefore creates a deterministic, single-instance pseudo-mask from the
binary KPIs ground truth: background=-1 and foreground=1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import numpy as np
from PIL import Image


SCRIPT_VERSION = "1.0.0"
DATASETS = ("56Nx", "DN")
EXPECTED_COUNTS = {
    "56Nx": {"training": 558, "test": 463},
    "DN": {"training": 724, "test": 391},
}
DEFAULT_ARCHIVES = {
    "training": "training_data_task1_patch_level.zip",
    "test": "test_data_task1_patch_level.zip",
}


@dataclass(frozen=True)
class Pair:
    dataset: str
    slide: str
    stem: str
    image_member: str
    mask_member: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KPIs2024 patch ZIPs into CA-SAM Med_datasets format."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="KPIs2024 root containing Task1_patch_level/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="CA-SAM data root; 56Nx/ and DN/ are created below it.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to prepare (default: 56Nx DN).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate archives and report the plan without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace generated files that already exist.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Proceed when archive counts differ from the paper's released split.",
    )
    return parser.parse_args()


def find_archive(raw_root: Path, filename: str) -> Path:
    archive_dir = raw_root / "Task1_patch_level"
    direct = archive_dir / filename
    if direct.is_file():
        return direct
    matches = [p for p in archive_dir.glob("*.zip") if p.name.lower() == filename.lower()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {filename!r} under {archive_dir}")


def member_key(member: str, archive_split: str) -> tuple[str, str, str, str] | None:
    path = PurePosixPath(member)
    parts = path.parts
    if len(parts) != 5 or parts[0] != archive_split:
        return None
    _, dataset, slide, kind, filename = parts
    if dataset not in DATASETS or kind not in {"img", "mask"}:
        return None
    if filename.startswith("._") or not filename.lower().endswith(".jpg"):
        return None
    suffix = f"_{kind}.jpg"
    if not filename.lower().endswith(suffix):
        return None
    stem = filename[: -len(suffix)]
    return dataset, slide, stem, kind


def collect_pairs(
    archive: zipfile.ZipFile, archive_split: str, selected: set[str]
) -> dict[str, list[Pair]]:
    members: dict[tuple[str, str, str], dict[str, str]] = {}
    for info in archive.infolist():
        parsed = member_key(info.filename, archive_split)
        if parsed is None:
            continue
        dataset, slide, stem, kind = parsed
        if dataset not in selected:
            continue
        members.setdefault((dataset, slide, stem), {})[kind] = info.filename

    pairs: dict[str, list[Pair]] = {dataset: [] for dataset in selected}
    incomplete: list[str] = []
    for (dataset, slide, stem), found in sorted(members.items()):
        if set(found) != {"img", "mask"}:
            incomplete.append(f"{dataset}/{slide}/{stem}: {sorted(found)}")
            continue
        pairs[dataset].append(
            Pair(dataset, slide, stem, found["img"], found["mask"])
        )
    if incomplete:
        preview = "\n".join(incomplete[:10])
        raise RuntimeError(f"Found image/mask pairs with missing members:\n{preview}")
    return pairs


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_copy(source: BinaryIO, destination: Path) -> None:
    ensure_parent(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    os.replace(temporary, destination)


def atomic_save_npy(destination: Path, array: np.ndarray) -> None:
    ensure_parent(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as output:
        np.save(output, array, allow_pickle=False)
    os.replace(temporary, destination)


def atomic_save_sparse_binary(destination: Path, mask: np.ndarray) -> None:
    """Write the binary mask in SciPy's CSR NPZ interchange format."""
    ensure_parent(destination)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    indices = np.flatnonzero(mask).astype(np.int32, copy=False)
    np.savez_compressed(
        temporary,
        indices=indices,
        indptr=np.asarray([0, len(indices)], dtype=np.int32),
        format=np.asarray(b"csr"),
        shape=np.asarray([1, mask.size], dtype=np.int64),
        data=np.ones(len(indices), dtype=np.uint8),
    )
    os.replace(temporary, destination)


def read_binary_mask(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member) as source:
        with Image.open(source) as image:
            mask = np.asarray(image.convert("L")) > 0
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask for {member}, got {mask.shape}")
    if not mask.any():
        raise ValueError(f"Refusing empty foreground mask: {member}")
    return mask


def output_paths(output_root: Path, pair: Pair) -> tuple[Path, Path, Path]:
    dataset_root = output_root / pair.dataset
    image = dataset_root / "image" / pair.slide / f"{pair.stem}.jpg"
    return image, dataset_root / "label" / pair.slide, dataset_root / "imask" / pair.slide


def prepare_pair(
    archive: zipfile.ZipFile,
    output_root: Path,
    pair: Pair,
    split: str,
    overwrite: bool,
) -> dict[str, str]:
    image_path, label_dir, imask_dir = output_paths(output_root, pair)
    mask = read_binary_mask(archive, pair.mask_member)
    height, width = mask.shape
    shape = (1, height, width, 1)
    label_path = label_dir / f"{pair.stem}.{shape}.npz"
    imask_path = imask_dir / f"{pair.stem}.npy"

    if overwrite or not image_path.is_file():
        with archive.open(pair.image_member) as source:
            atomic_copy(source, image_path)
    if overwrite or not label_path.is_file():
        atomic_save_sparse_binary(label_path, mask)

    record = {
        "image": image_path.relative_to(output_root / pair.dataset).as_posix(),
        "label": label_path.relative_to(output_root / pair.dataset).as_posix(),
    }
    if split == "training":
        if overwrite or not imask_path.is_file():
            pseudo = np.full(mask.shape, -1, dtype=np.int8)
            pseudo[mask] = 1
            atomic_save_npy(imask_path, pseudo)
        record["imask"] = imask_path.relative_to(output_root / pair.dataset).as_posix()
    return record


def atomic_write_json(path: Path, payload: object) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")
    os.replace(temporary, path)


def validate_counts(
    pairs_by_split: dict[str, dict[str, list[Pair]]],
    selected: list[str],
    allow_mismatch: bool,
) -> None:
    errors: list[str] = []
    for dataset in selected:
        for split in ("training", "test"):
            actual = len(pairs_by_split[split][dataset])
            expected = EXPECTED_COUNTS[dataset][split]
            print(f"{dataset:4s} {split:8s}: {actual} pairs (expected {expected})")
            if actual != expected:
                errors.append(f"{dataset} {split}: {actual} != {expected}")
    if errors and not allow_mismatch:
        raise RuntimeError(
            "Archive counts do not match the CA-SAM paper split. "
            "Use --allow-count-mismatch only after auditing the source.\n"
            + "\n".join(errors)
        )


def validate_output(output_root: Path, selected: list[str]) -> None:
    for dataset in selected:
        dataset_root = output_root / dataset
        with (dataset_root / "dataset.json").open(encoding="utf-8") as source:
            metadata = json.load(source)
        if metadata["numTraining"] != len(metadata["training"]):
            raise RuntimeError(f"numTraining mismatch in {dataset}/dataset.json")
        for split in ("training", "test"):
            for record in metadata[split]:
                required = ("image", "label", "imask") if split == "training" else ("image", "label")
                for key in required:
                    path = dataset_root / record[key]
                    if not path.is_file() or path.stat().st_size == 0:
                        raise RuntimeError(f"Missing or empty {key}: {path}")


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    selected = list(dict.fromkeys(args.datasets))
    archives = {
        split: find_archive(raw_root, filename)
        for split, filename in DEFAULT_ARCHIVES.items()
    }

    pairs_by_split: dict[str, dict[str, list[Pair]]] = {}
    for split, archive_path in archives.items():
        archive_split = "train" if split == "training" else "test"
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"CRC failure in {archive_path}: {bad_member}")
            pairs_by_split[split] = collect_pairs(archive, archive_split, set(selected))

    validate_counts(pairs_by_split, selected, args.allow_count_mismatch)
    print(f"raw root   : {raw_root}")
    print(f"output root: {output_root}")
    print("source WSI files and validation split are intentionally left untouched and unused")
    if args.dry_run:
        print("dry run complete; no files written")
        return 0

    records: dict[str, dict[str, list[dict[str, str]]]] = {
        dataset: {"training": [], "test": []} for dataset in selected
    }
    total = sum(
        len(pairs_by_split[split][dataset])
        for split in ("training", "test")
        for dataset in selected
    )
    completed = 0
    for split in ("training", "test"):
        with zipfile.ZipFile(archives[split]) as archive:
            for dataset in selected:
                for pair in pairs_by_split[split][dataset]:
                    records[dataset][split].append(
                        prepare_pair(archive, output_root, pair, split, args.overwrite)
                    )
                    completed += 1
                    if completed % 100 == 0 or completed == total:
                        print(f"prepared {completed}/{total}", flush=True)

    for dataset in selected:
        metadata = {
            "name": dataset,
            "description": "KPIs2024 glomerulus patch segmentation prepared for CA-SAM",
            "dimension": "2D",
            "modality": {"0": "pathology"},
            "labels": {"0": "background", "1": "glomerulus"},
            "numTraining": len(records[dataset]["training"]),
            "training": records[dataset]["training"],
            "test": records[dataset]["test"],
        }
        atomic_write_json(output_root / dataset / "dataset.json", metadata)

    manifest = {
        "script": "prepare_kpis2024_for_casam.py",
        "script_version": SCRIPT_VERSION,
        "source": {
            split: {
                "archive": path.name,
                "size_bytes": path.stat().st_size,
            }
            for split, path in archives.items()
        },
        "datasets": {
            dataset: {
                split: len(records[dataset][split])
                for split in ("training", "test")
            }
            for dataset in selected
        },
        "imask_semantics": {
            "dtype": "int8",
            "background": -1,
            "foreground": 1,
            "source": "binary KPIs2024 ground-truth mask",
            "reason": "required by the released loader; current training collate uses GT prompts",
        },
    }
    atomic_write_json(output_root / "kpis2024_casam_manifest.json", manifest)
    validate_output(output_root, selected)
    print("output validation passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
