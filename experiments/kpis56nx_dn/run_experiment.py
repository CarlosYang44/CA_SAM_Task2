#!/usr/bin/env python3
"""Run a shared-Alignment-Layer 56Nx -> target forgetting pilot."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


FULL_EPOCHS = 24
EXPECTED = {
    "56Nx": {"training": 558, "test": 463},
    "DN": {"training": 724, "test": 391},
    "MSD_Spleen": {"training": 876, "test": 146},
}
REQUIRED_PACKAGES = (
    "torch", "torchvision", "monai", "numpy", "cv2", "PIL", "scipy", "skimage",
    "tqdm",
)


def tasks(args: argparse.Namespace) -> tuple[str, str]:
    return "56Nx", args.second_dataset


def model_names(args: argparse.Namespace) -> tuple[str, str, str]:
    target = args.second_dataset
    return "M_56Nx", f"M_56Nx_{target}", f"M_{target}_only"


def evaluations(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    target = args.second_dataset
    return {
        "56Nx_before": ("56Nx", "M_56Nx"),
        f"{target}_sequential": (target, f"M_56Nx_{target}"),
        f"56Nx_after_{target}": ("56Nx", f"M_56Nx_{target}"),
        f"{target}_only": (target, f"M_{target}_only"),
    }


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shared-AL 56Nx -> target forgetting/plasticity pilot."
    )
    parser.add_argument(
        "command", choices=("preflight", "smoke", "train", "eval", "summarize", "all")
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=env_path("CASAM_DATA_DIR", "/mnt/ufs/Med_datasets"),
    )
    parser.add_argument(
        "--sam-checkpoint", type=Path,
        default=env_path("CASAM_SAM_CKPT", "./pretrain_model/sam_vit_b_01ec64.pth"),
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=env_path("CASAM_RUN_ROOT", "./outputs/kpis56nx_dn_shared"),
    )
    parser.add_argument(
        "--second-dataset", choices=("DN", "MSD_Spleen"), default="DN",
        help="Task learned after 56Nx (default: DN).",
    )
    parser.add_argument(
        "--initial-56nx-checkpoint", type=Path, default=None,
        help="Reuse an existing M_56Nx checkpoint instead of retraining Run A.",
    )
    parser.add_argument("--device", default=os.environ.get("CASAM_DEVICE", "cuda:0"))
    parser.add_argument(
        "--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--mask-num", type=int, default=5)
    parser.add_argument("--num-cnn", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument(
        "--continual-method", choices=("naive", "sr2"), default="naive",
        help="Run B update rule. sr2 adds singular-value inter-layer relation alignment.",
    )
    parser.add_argument("--sr2-lambda", type=float, default=1.0)
    parser.add_argument("--smoke-dataset-scale", type=float, default=0.02)
    parser.add_argument("--smoke-image-size", type=int, default=256)
    parser.add_argument("--smoke-eval-samples", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.write("started_utc: " + utc_now() + "\n\n")
        process = subprocess.Popen(
            command, cwd=repository_root(), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
        log.write("\nfinished_utc: " + utc_now() + "\n")
        log.write(f"return_code: {return_code}\n")
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}")


def read_dataset_metadata(data_dir: Path, dataset: str) -> dict:
    path = data_dir / dataset / "dataset.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def validate_record_paths(data_dir: Path, dataset: str, metadata: dict) -> None:
    dataset_root = data_dir / dataset
    for split in ("training", "test"):
        records = metadata.get(split)
        if not isinstance(records, list):
            raise ValueError(f"{dataset}/dataset.json: {split!r} must be a list")
        required = ("image", "label", "imask") if split == "training" else ("image", "label")
        for index, record in enumerate(records):
            for key in required:
                value = record.get(key)
                if not value:
                    raise ValueError(f"{dataset} {split}[{index}] lacks {key}")
                path = dataset_root / value
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)


def validate_sample_payload(data_dir: Path, dataset: str, metadata: dict) -> dict:
    import numpy as np
    from PIL import Image
    from scipy import sparse

    record = metadata["training"][0]
    dataset_root = data_dir / dataset
    image_path = dataset_root / record["image"]
    label_path = dataset_root / record["label"]
    imask_path = dataset_root / record["imask"]
    with Image.open(image_path) as image:
        image_size, image_mode = list(image.size), image.mode
    encoded_shape = ast.literal_eval(label_path.name.split(".")[-2])
    label = sparse.load_npz(label_path).toarray().reshape(encoded_shape)
    imask = np.load(imask_path, mmap_mode="r")
    if label.shape[1:3] != imask.shape[-2:]:
        raise ValueError(f"Label/imask shape mismatch for {dataset}: {label.shape} vs {imask.shape}")
    if not np.any(label):
        raise ValueError(f"Empty sample label: {label_path}")
    return {
        "image": record["image"], "image_size": image_size, "image_mode": image_mode,
        "label_shape": list(label.shape), "imask_shape": list(imask.shape),
    }


def git_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def preflight(args: argparse.Namespace) -> dict:
    repo = repository_root()
    data_dir = args.data_dir.resolve()
    checkpoint = args.sam_checkpoint.resolve()
    run_root = args.run_root.resolve()
    required_packages = REQUIRED_PACKAGES + (
        ("nibabel",) if args.second_dataset == "MSD_Spleen" else ()
    )
    missing_packages = [p for p in required_packages if importlib.util.find_spec(p) is None]
    if missing_packages:
        raise RuntimeError("Missing Python packages: " + ", ".join(missing_packages))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    dataset_report: dict[str, object] = {}
    for dataset in tasks(args):
        metadata = read_dataset_metadata(data_dir, dataset)
        counts = {split: len(metadata.get(split, [])) for split in ("training", "test")}
        if counts != EXPECTED[dataset]:
            raise ValueError(f"{dataset} counts {counts} do not match expected {EXPECTED[dataset]}")
        validate_record_paths(data_dir, dataset, metadata)
        dataset_report[dataset] = {
            "counts": counts, "sample": validate_sample_payload(data_dir, dataset, metadata)
        }

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    run_root.mkdir(parents=True, exist_ok=True)
    report = {
        "checked_utc": utc_now(), "repository": str(repo), "git_revision": git_revision(repo),
        "python": sys.version, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_dir": str(data_dir), "sam_checkpoint": str(checkpoint), "run_root": str(run_root),
        "run_root_free_bytes": shutil.disk_usage(run_root).free, "datasets": dataset_report,
    }
    atomic_json(run_root / "preflight.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("preflight passed")
    return report


def driver_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["PYTHONHASHSEED"] = str(args.seed)
    return environment


def experiment_root(args: argparse.Namespace, smoke_run: bool = False) -> Path:
    if args.continual_method == "sr2":
        prefix = "smoke_sr2_shared_al" if smoke_run else f"sr2_shared_al_cnn{args.num_cnn}"
        suffix = f"{prefix}_lambda{args.sr2_lambda:g}"
    else:
        suffix = "smoke_shared_al" if smoke_run else f"shared_al_cnn{args.num_cnn}"
    return args.run_root.resolve() / suffix


def checkpoint_path(root: Path, model_name: str) -> Path:
    return root / "checkpoints" / f"{model_name}.pth"


def stable_config(args: argparse.Namespace, smoke_run: bool) -> dict:
    return {
        "design": f"shared_alignment_layer_56Nx_then_{args.second_dataset}",
        "task_order": list(tasks(args)), "router": "disabled", "method": "cnn",
        "num_cnn": args.num_cnn, "epochs_per_stage": 1 if smoke_run else FULL_EPOCHS,
        "lr": args.lr, "train_batch_size": 1 if smoke_run else args.train_batch_size,
        "eval_batch_size": 1 if smoke_run else args.eval_batch_size,
        "num_workers": args.num_workers,
        "image_size": args.smoke_image_size if smoke_run else args.image_size,
        "mask_num": 1 if smoke_run else args.mask_num,
        "seed": args.seed, "continual_method": args.continual_method,
        "sr2_lambda": args.sr2_lambda if args.continual_method == "sr2" else None,
        "dataset_scale": args.smoke_dataset_scale if smoke_run else 1.0,
        "data_dir": str(args.data_dir.resolve()),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "initial_56Nx_checkpoint": (
            str(args.initial_56nx_checkpoint.resolve())
            if args.initial_56nx_checkpoint is not None else None
        ),
        "git_revision": git_revision(repository_root()),
    }


def ensure_config(args: argparse.Namespace, root: Path, smoke_run: bool) -> None:
    stable = stable_config(args, smoke_run)
    path = root / "experiment_config.json"
    if path.is_file() and not args.force:
        with path.open(encoding="utf-8") as source:
            previous = json.load(source)
        previous.pop("created_utc", None)
        if previous != stable:
            raise RuntimeError(
                f"Existing run configuration differs: {path}. Use another --run-root or --force."
            )
    atomic_json(path, {"created_utc": utc_now(), **stable})


def train_command(
    args: argparse.Namespace, dataset: str, stage_dir: Path, smoke_run: bool,
    initialize_from: Path | None = None,
) -> list[str]:
    command = [
        sys.executable, str(repository_root() / "train_align_CL_VAE.py"),
        "--work_dir", str(stage_dir), "--save_root", str(stage_dir),
        "--ckpt_dir", str(stage_dir / "checkpoint"),
        "--run_name", stage_dir.name, "--dataset_name", dataset,
        "--all_datasets", ",".join(tasks(args)), "--device", args.device,
        "--model_type", "vit_b", "--sam_checkpoint", str(args.sam_checkpoint.resolve()),
        "--data_dir", str(args.data_dir.resolve()), "--method", "cnn",
        "--num_cnn", str(args.num_cnn), "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--epochs", str(1 if smoke_run else FULL_EPOCHS),
        "--train_batch_size", str(1 if smoke_run else args.train_batch_size),
        "--eval_batch_size", str(1 if smoke_run else args.eval_batch_size),
        "--image_size", str(args.smoke_image_size if smoke_run else args.image_size),
        "--mask_num", str(1 if smoke_run else args.mask_num),
        "--dataset_scale", str(args.smoke_dataset_scale if smoke_run else 1.0),
        "--router_type", "none", "--seed", str(args.seed),
        "--fold_seed", str(args.fold_seed),
    ]
    if initialize_from is not None:
        command += ["--align_checkpoint", str(initialize_from)]
        if args.continual_method == "sr2":
            command += [
                "--sr2_teacher_checkpoint", str(initialize_from),
                "--sr2_lambda", str(args.sr2_lambda),
            ]
    return command


def train_stage(
    args: argparse.Namespace, root: Path, stage_name: str, dataset: str,
    model_name: str, smoke_run: bool, initialize_from: Path | None = None,
) -> Path:
    destination = checkpoint_path(root, model_name)
    if destination.is_file() and not args.force:
        if initialize_from is not None:
            metadata_path = root / "checkpoints" / f"{model_name}.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Cannot verify shared-AL lineage; missing {metadata_path}"
                )
            with metadata_path.open(encoding="utf-8") as source:
                metadata = json.load(source)
            if metadata.get("initialized_from_sha256") != sha256(initialize_from):
                raise RuntimeError(
                    f"{model_name} was not initialized from the current {initialize_from}; "
                    "use --force to rebuild the sequential stage"
                )
        print(f"{model_name} already exists; skipping: {destination}")
        return destination
    if initialize_from is not None and not initialize_from.is_file():
        raise FileNotFoundError(f"Required initialization checkpoint is missing: {initialize_from}")
    stage_dir = root / "stages" / stage_name
    command = train_command(args, dataset, stage_dir, smoke_run, initialize_from)
    run_logged(command, root / "driver_logs" / f"train_{stage_name}.log", driver_environment(args))
    produced = stage_dir / "checkpoint" / f"align_cnn_{args.num_cnn}.pth"
    if not produced.is_file():
        raise FileNotFoundError(f"Training completed without expected checkpoint: {produced}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, destination)
    atomic_json(root / "checkpoints" / f"{model_name}.json", {
        "created_utc": utc_now(), "model": model_name, "dataset": dataset,
        "initialized_from": str(initialize_from) if initialize_from else None,
        "initialized_from_sha256": sha256(initialize_from) if initialize_from else None,
        "checkpoint": str(destination), "sha256": sha256(destination),
        "router": "disabled",
    })
    return destination


def obtain_m_56nx(args: argparse.Namespace, root: Path, smoke_run: bool) -> Path:
    """Train Run A, or import a previously trained Run-A checkpoint."""
    if args.initial_56nx_checkpoint is None:
        return train_stage(
            args, root, "run_a_56Nx", "56Nx", "M_56Nx", smoke_run
        )

    source = args.initial_56nx_checkpoint.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Initial 56Nx checkpoint not found: {source}")
    destination = checkpoint_path(root, "M_56Nx")
    if destination.is_file() and not args.force:
        if sha256(destination) != sha256(source):
            raise RuntimeError(
                f"Existing {destination} differs from --initial-56nx-checkpoint; "
                "use another --run-root or --force"
            )
        print(f"M_56Nx already imported; skipping: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    atomic_json(root / "checkpoints" / "M_56Nx.json", {
        "created_utc": utc_now(), "model": "M_56Nx", "dataset": "56Nx",
        "imported_from": str(source), "source_sha256": sha256(source),
        "checkpoint": str(destination), "sha256": sha256(destination),
        "router": "disabled",
    })
    print(f"imported M_56Nx: {source} -> {destination}")
    return destination


def train(args: argparse.Namespace, smoke_run: bool = False) -> Path:
    root = experiment_root(args, smoke_run)
    root.mkdir(parents=True, exist_ok=True)
    ensure_config(args, root, smoke_run)
    target = args.second_dataset
    m_56nx = obtain_m_56nx(args, root, smoke_run)
    train_stage(
        args, root, f"run_b_56Nx_to_{target}", target, f"M_56Nx_{target}", smoke_run,
        initialize_from=m_56nx,
    )
    train_stage(
        args, root, f"run_c_{target}_only", target, f"M_{target}_only", smoke_run
    )
    return root


def eval_command(
    args: argparse.Namespace, dataset: str, checkpoint: Path, output: Path,
    smoke_run: bool,
) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).with_name("eval_shared_al.py")),
        "--dataset-name", dataset, "--align-checkpoint", str(checkpoint),
        "--output", str(output), "--device", args.device,
        "--sam-checkpoint", str(args.sam_checkpoint.resolve()),
        "--data-dir", str(args.data_dir.resolve()), "--model-type", "vit_b",
        "--method", "cnn", "--num-cnn", str(args.num_cnn),
        "--batch-size", str(1 if smoke_run else args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--image-size", str(args.smoke_image_size if smoke_run else args.image_size),
        "--mask-num", str(1 if smoke_run else args.mask_num),
        "--seed", str(args.seed),
    ]
    if smoke_run:
        command += ["--max-samples", str(args.smoke_eval_samples)]
    return command


def evaluate(args: argparse.Namespace, smoke_run: bool = False) -> Path:
    root = experiment_root(args, smoke_run)
    for model_name in model_names(args):
        if not checkpoint_path(root, model_name).is_file():
            raise FileNotFoundError(f"Cannot evaluate; missing {checkpoint_path(root, model_name)}")
    for result_name, (dataset, model_name) in evaluations(args).items():
        evaluate_one(args, root, result_name, dataset, model_name, smoke_run)
    return root


def evaluate_one(
    args: argparse.Namespace, root: Path, result_name: str, dataset: str,
    model_name: str, smoke_run: bool,
) -> Path:
    checkpoint = checkpoint_path(root, model_name)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cannot evaluate; missing {checkpoint}")
    output = root / "results" / "raw" / f"{result_name}.json"
    if output.is_file() and not args.force:
        print(f"evaluation already exists; skipping: {output}")
        return output
    command = eval_command(args, dataset, checkpoint, output, smoke_run)
    run_logged(
        command, root / "driver_logs" / f"eval_{result_name}.log", driver_environment(args)
    )
    if not output.is_file():
        raise FileNotFoundError(f"Evaluation completed without output: {output}")
    return output


def summarize(root: Path, target: str = "DN") -> Path:
    result_map = {
        "before": "56Nx_before",
        "after": f"56Nx_after_{target}",
        "sequential": f"{target}_sequential",
        "only": f"{target}_only",
    }
    raw: dict[str, dict] = {}
    for name in result_map.values():
        path = root / "results" / "raw" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot summarize; missing {path}")
        with path.open(encoding="utf-8") as source:
            raw[name] = json.load(source)

    metric_names = list(raw[result_map["before"]]["metrics"])
    by_metric = {}
    for metric in metric_names:
        before = float(raw[result_map["before"]]["metrics"][metric])
        after = float(raw[result_map["after"]]["metrics"][metric])
        sequential = float(raw[result_map["sequential"]]["metrics"][metric])
        only = float(raw[result_map["only"]]["metrics"][metric])
        by_metric[metric] = {
            "56Nx_before": before, f"56Nx_after_{target}": after,
            f"{target}_sequential": sequential, f"{target}_only": only,
            "forgetting": round(before - after, 10),
            "plasticity_gap": round(only - sequential, 10),
        }
    summary = {
        "created_utc": utc_now(), "primary_metric": "iou", **by_metric["iou"],
        "by_metric": by_metric,
    }
    results = root / "results"
    atomic_json(results / "summary.json", summary)
    columns = (
        "metric", "56Nx_before", f"56Nx_after_{target}",
        f"{target}_sequential", f"{target}_only",
        "forgetting", "plasticity_gap",
    )
    with (results / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for metric, values in by_metric.items():
            writer.writerow({"metric": metric, **values})
    with (results / "summary.md").open("w", encoding="utf-8", newline="\n") as output:
        output.write(
            f"| Metric | 56Nx before | 56Nx after {target} | {target} sequential | "
            f"{target} only | Forgetting | Plasticity gap |\n"
        )
        output.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for metric, values in by_metric.items():
            output.write(
                f"| {metric} | {values['56Nx_before']:.4f} | "
                f"{values[f'56Nx_after_{target}']:.4f} | "
                f"{values[f'{target}_sequential']:.4f} | "
                f"{values[f'{target}_only']:.4f} | "
                f"{values['forgetting']:.4f} | {values['plasticity_gap']:.4f} |\n"
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary written to: {results}")
    return results


def run_pipeline(args: argparse.Namespace, smoke_run: bool = False) -> Path:
    """Execute the scientific stages in order, including stage-boundary evaluation."""
    root = experiment_root(args, smoke_run)
    root.mkdir(parents=True, exist_ok=True)
    ensure_config(args, root, smoke_run)

    target = args.second_dataset
    m_56nx = obtain_m_56nx(args, root, smoke_run)
    evaluate_one(args, root, "56Nx_before", "56Nx", "M_56Nx", smoke_run)

    train_stage(
        args, root, f"run_b_56Nx_to_{target}", target, f"M_56Nx_{target}", smoke_run,
        initialize_from=m_56nx,
    )
    evaluate_one(
        args, root, f"{target}_sequential", target, f"M_56Nx_{target}", smoke_run
    )
    evaluate_one(
        args, root, f"56Nx_after_{target}", "56Nx", f"M_56Nx_{target}", smoke_run
    )

    train_stage(
        args, root, f"run_c_{target}_only", target, f"M_{target}_only", smoke_run
    )
    evaluate_one(args, root, f"{target}_only", target, f"M_{target}_only", smoke_run)
    summarize(root, target)
    return root


def smoke(args: argparse.Namespace) -> Path:
    return run_pipeline(args, smoke_run=True)


def main() -> int:
    args = parse_args()
    if args.continual_method == "sr2" and args.num_cnn < 2:
        raise ValueError("SR2 requires --num-cnn of at least 2")
    if args.sr2_lambda < 0:
        raise ValueError("--sr2-lambda must be non-negative")
    args.data_dir = args.data_dir.expanduser()
    args.sam_checkpoint = args.sam_checkpoint.expanduser()
    args.run_root = args.run_root.expanduser()
    if args.initial_56nx_checkpoint is not None:
        args.initial_56nx_checkpoint = args.initial_56nx_checkpoint.expanduser()
    if args.command == "preflight":
        preflight(args)
        return 0
    if not args.skip_preflight:
        preflight(args)
    if args.command == "smoke":
        smoke(args)
    elif args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "summarize":
        summarize(experiment_root(args), args.second_dataset)
    elif args.command == "all":
        run_pipeline(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
