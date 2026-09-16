#!/usr/bin/env python3
"""Convert MSD Task09_Spleen volumes into the 2-D CA-SAM data format.

The public MSD test volumes do not include labels.  This script therefore makes
a deterministic, patient-level split of the labelled ``imagesTr`` volumes.  It
keeps only axial slices containing spleen, applies a fixed CT window, and then
selects the paper-reported 876/146 train/test slice counts by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_VERSION = "1.0.0"
DEFAULT_TRAIN_SLICES = 876
DEFAULT_TEST_SLICES = 146


@dataclass(frozen=True)
class SliceSpec:
    case: str
    index: int
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MSD Task09_Spleen for the CA-SAM 2-D loader."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Task09_Spleen directory, or its parent directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="CA-SAM data root; MSD_Spleen/ is created below it.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--test-cases",
        type=int,
        default=6,
        help="Number of labelled volumes held out before slice selection (default: 6).",
    )
    parser.add_argument("--train-slices", type=int, default=DEFAULT_TRAIN_SLICES)
    parser.add_argument("--test-slices", type=int, default=DEFAULT_TEST_SLICES)
    parser.add_argument("--window-min", type=float, default=-57.0)
    parser.add_argument("--window-max", type=float, default=164.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_nibabel():
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError(
            "nibabel is required for MSD preprocessing; install it with: pip install nibabel"
        ) from error
    return nib


def resolve_task_root(raw_root: Path) -> Path:
    raw_root = raw_root.expanduser().resolve()
    candidates = (raw_root, raw_root / "Task09_Spleen")
    for candidate in candidates:
        if (candidate / "imagesTr").is_dir() and (candidate / "labelsTr").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find Task09_Spleen/imagesTr and labelsTr under {raw_root}"
    )


def case_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Unsupported NIfTI filename: {path}")


def labelled_cases(task_root: Path) -> list[tuple[str, Path, Path]]:
    images = [
        path for path in sorted((task_root / "imagesTr").glob("*.nii*"))
        if not path.name.startswith("._")
    ]
    cases: list[tuple[str, Path, Path]] = []
    for image_path in images:
        name = case_name(image_path)
        label_path = task_root / "labelsTr" / image_path.name
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        cases.append((name, image_path, label_path))
    if not cases:
        raise RuntimeError(f"No labelled NIfTI volumes found under {task_root}")
    return cases


def split_cases(
    cases: list[tuple[str, Path, Path]], test_cases: int, seed: int
) -> tuple[list[tuple[str, Path, Path]], list[tuple[str, Path, Path]]]:
    if test_cases < 1 or test_cases >= len(cases):
        raise ValueError(f"--test-cases must be between 1 and {len(cases) - 1}")
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    test_names = {case[0] for case in shuffled[:test_cases]}
    training = [case for case in cases if case[0] not in test_names]
    test = [case for case in cases if case[0] in test_names]
    return training, test


def positive_slices(cases: list[tuple[str, Path, Path]]) -> list[SliceSpec]:
    nib = import_nibabel()
    result: list[SliceSpec] = []
    for name, image_path, label_path in cases:
        image = nib.as_closest_canonical(nib.load(str(image_path)))
        label = nib.as_closest_canonical(nib.load(str(label_path)))
        if image.shape != label.shape or len(image.shape) != 3:
            raise ValueError(
                f"Image/label shape mismatch for {name}: {image.shape} vs {label.shape}"
            )
        label_data = np.asarray(label.dataobj)
        for index in np.flatnonzero(np.any(label_data > 0, axis=(0, 1))):
            result.append(SliceSpec(name, int(index), image_path, label_path))
    return result


def stable_select(slices: list[SliceSpec], count: int, seed: int, split: str) -> list[SliceSpec]:
    if count < 1:
        raise ValueError(f"{split} slice count must be positive")
    if len(slices) < count:
        raise RuntimeError(
            f"Only {len(slices)} positive {split} slices are available, but {count} were requested"
        )

    def key(spec: SliceSpec) -> bytes:
        value = f"{seed}:{split}:{spec.case}:{spec.index}".encode("utf-8")
        return hashlib.sha256(value).digest()

    return sorted(sorted(slices, key=key)[:count], key=lambda item: (item.case, item.index))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_save_png(path: Path, array: np.ndarray) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.save(output, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_save_sparse_binary(path: Path, mask: np.ndarray) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    indices = np.flatnonzero(mask).astype(np.int32, copy=False)
    np.savez_compressed(
        temporary,
        indices=indices,
        indptr=np.asarray([0, len(indices)], dtype=np.int32),
        format=np.asarray(b"csr"),
        shape=np.asarray([1, mask.size], dtype=np.int64),
        data=np.ones(len(indices), dtype=np.uint8),
    )
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")
    os.replace(temporary, path)


def load_case_arrays(
    image_path: Path, label_path: Path, cache: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    name = str(image_path)
    if name not in cache:
        cache.clear()
        nib = import_nibabel()
        image = nib.as_closest_canonical(nib.load(str(image_path)))
        label = nib.as_closest_canonical(nib.load(str(label_path)))
        cache[name] = (
            np.asarray(image.dataobj, dtype=np.float32),
            np.asarray(label.dataobj),
        )
    return cache[name]


def prepare_split(
    specs: list[SliceSpec],
    split: str,
    dataset_root: Path,
    window_min: float,
    window_max: float,
    overwrite: bool,
) -> list[dict[str, str]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records: list[dict[str, str]] = []
    for number, spec in enumerate(specs, start=1):
        image_volume, label_volume = load_case_arrays(
            spec.image_path, spec.label_path, cache
        )
        image_slice = image_volume[:, :, spec.index]
        mask = label_volume[:, :, spec.index] > 0
        if not mask.any():
            raise ValueError(f"Selected empty mask: {spec.case} slice {spec.index}")
        scaled = np.clip((image_slice - window_min) / (window_max - window_min), 0, 1)
        grayscale = np.rint(scaled * 255).astype(np.uint8)
        rgb = np.repeat(grayscale[:, :, None], 3, axis=2)

        stem = f"{spec.case}_z{spec.index:04d}"
        image_path = dataset_root / "image" / split / f"{stem}.png"
        shape = (1, mask.shape[0], mask.shape[1], 1)
        label_path = dataset_root / "label" / split / f"{stem}.{shape}.npz"
        imask_path = dataset_root / "imask" / split / f"{stem}.npy"
        if overwrite or not image_path.is_file():
            atomic_save_png(image_path, rgb)
        if overwrite or not label_path.is_file():
            atomic_save_sparse_binary(label_path, mask)

        record = {
            "image": image_path.relative_to(dataset_root).as_posix(),
            "label": label_path.relative_to(dataset_root).as_posix(),
        }
        if split == "training":
            if overwrite or not imask_path.is_file():
                pseudo = np.full(mask.shape, -1, dtype=np.int8)
                pseudo[mask] = 1
                atomic_save_npy(imask_path, pseudo)
            record["imask"] = imask_path.relative_to(dataset_root).as_posix()
        records.append(record)
        if number % 100 == 0 or number == len(specs):
            print(f"prepared {split}: {number}/{len(specs)}", flush=True)
    return records


def main() -> int:
    args = parse_args()
    if args.window_max <= args.window_min:
        raise ValueError("--window-max must be greater than --window-min")
    task_root = resolve_task_root(args.raw_root)
    cases = labelled_cases(task_root)
    training_cases, test_cases = split_cases(cases, args.test_cases, args.seed)
    training_available = positive_slices(training_cases)
    test_available = positive_slices(test_cases)
    training_specs = stable_select(training_available, args.train_slices, args.seed, "training")
    test_specs = stable_select(test_available, args.test_slices, args.seed, "test")

    print(f"task root: {task_root}")
    print(f"patient split: {len(training_cases)} training, {len(test_cases)} test")
    print(
        f"positive slices: {len(training_available)} training, {len(test_available)} test; "
        f"selected {len(training_specs)}/{len(test_specs)}"
    )
    if args.dry_run:
        print("dry run complete; no files written")
        return 0

    dataset_root = args.output_root.expanduser().resolve() / "MSD_Spleen"
    training = prepare_split(
        training_specs, "training", dataset_root,
        args.window_min, args.window_max, args.overwrite,
    )
    test = prepare_split(
        test_specs, "test", dataset_root,
        args.window_min, args.window_max, args.overwrite,
    )
    metadata = {
        "name": "MSD_Spleen",
        "description": "MSD Task09 spleen CT, patient-level split prepared for CA-SAM",
        "dimension": "2D",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "spleen"},
        "numTraining": len(training),
        "training": training,
        "test": test,
    }
    atomic_json(dataset_root / "dataset.json", metadata)
    manifest = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "source": str(task_root),
        "seed": args.seed,
        "ct_window_hu": [args.window_min, args.window_max],
        "split_unit": "labelled volume/patient",
        "training_cases": [case[0] for case in training_cases],
        "test_cases": [case[0] for case in test_cases],
        "available_positive_slices": {
            "training": len(training_available), "test": len(test_available)
        },
        "selected_slices": {"training": len(training), "test": len(test)},
        "selection": "stable SHA-256 ranking within each patient-level split",
        "imask_semantics": {
            "dtype": "int8", "background": -1, "foreground": 1,
            "source": "MSD ground-truth spleen mask",
        },
        "note": (
            "The public MSD test set has no labels. This is a reproducible local split, "
            "not a claim to reproduce an unpublished CA-SAM patient split."
        ),
    }
    atomic_json(dataset_root / "preparation_manifest.json", manifest)
    print(f"prepared dataset: {dataset_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
