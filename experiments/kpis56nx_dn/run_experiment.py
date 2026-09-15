#!/usr/bin/env python3
"""Reproducible CA-SAM baseline runner for the 56Nx -> DN task sequence."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TASKS = ("56Nx", "DN")
EXPECTED = {
    "56Nx": {"training": 558, "test": 463},
    "DN": {"training": 724, "test": 391},
}
REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "monai",
    "numpy",
    "cv2",
    "PIL",
    "scipy",
    "skimage",
    "tqdm",
)


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official CA-SAM baseline on 56Nx -> DN."
    )
    parser.add_argument(
        "command", choices=("preflight", "smoke", "train", "eval", "all")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=env_path("CASAM_DATA_DIR", "./Med_datasets"),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=env_path(
            "CASAM_SAM_CKPT", "./pretrain_model/sam_vit_b_01ec64.pth"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=env_path("CASAM_RUN_ROOT", "./outputs/kpis56nx_dn"),
    )
    parser.add_argument("--device", default=os.environ.get("CASAM_DEVICE", "cuda:0"))
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--mask-num", type=int, default=5)
    parser.add_argument("--num-cnn", type=int, default=3)
    parser.add_argument("--vae-epochs", type=int, default=10)
    parser.add_argument("--vae-latent-dim", type=int, default=64)
    parser.add_argument("--vae-beta", type=float, default=16.5)
    parser.add_argument("--vae-lr", type=float, default=5e-4)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=2025)
    parser.add_argument("--smoke-dataset-scale", type=float, default=0.02)
    parser.add_argument("--smoke-image-size", type=int, default=256)
    parser.add_argument(
        "--eval-tag",
        default=None,
        help="Evaluation output tag. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun stages whose expected artifacts already exist.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip validation before smoke/train/eval (not recommended).",
    )
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


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.write("started_utc: " + utc_now() + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=repository_root(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
    try:
        import numpy as np
        from PIL import Image
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError(f"Cannot validate sample payload: {error}") from error

    record = metadata["training"][0]
    dataset_root = data_dir / dataset
    image_path = dataset_root / record["image"]
    label_path = dataset_root / record["label"]
    imask_path = dataset_root / record["imask"]
    with Image.open(image_path) as image:
        image_size = list(image.size)
        image_mode = image.mode
    encoded_shape = ast.literal_eval(label_path.name.split(".")[-2])
    label = sparse.load_npz(label_path).toarray().reshape(encoded_shape)
    imask = np.load(imask_path, mmap_mode="r")
    if label.shape[1:3] != imask.shape[-2:]:
        raise ValueError(
            f"Label/imask shape mismatch for {dataset}: {label.shape} vs {imask.shape}"
        )
    if not np.any(label):
        raise ValueError(f"Empty sample label: {label_path}")
    return {
        "image": record["image"],
        "image_size": image_size,
        "image_mode": image_mode,
        "label_shape": list(label.shape),
        "imask_shape": list(imask.shape),
    }


def git_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def preflight(args: argparse.Namespace) -> dict:
    repo = repository_root()
    data_dir = args.data_dir.resolve()
    checkpoint = args.sam_checkpoint.resolve()
    run_root = args.run_root.resolve()

    missing_packages = [
        package for package in REQUIRED_PACKAGES if importlib.util.find_spec(package) is None
    ]
    if missing_packages:
        raise RuntimeError("Missing Python packages: " + ", ".join(missing_packages))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    dataset_report: dict[str, object] = {}
    for dataset in TASKS:
        metadata = read_dataset_metadata(data_dir, dataset)
        counts = {split: len(metadata.get(split, [])) for split in ("training", "test")}
        if counts != EXPECTED[dataset]:
            raise ValueError(
                f"{dataset} counts {counts} do not match expected {EXPECTED[dataset]}"
            )
        validate_record_paths(data_dir, dataset, metadata)
        dataset_report[dataset] = {
            "counts": counts,
            "sample": validate_sample_payload(data_dir, dataset, metadata),
        }

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    run_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(run_root)
    report = {
        "checked_utc": utc_now(),
        "repository": str(repo),
        "git_revision": git_revision(repo),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "data_dir": str(data_dir),
        "sam_checkpoint": str(checkpoint),
        "run_root": str(run_root),
        "run_root_free_bytes": usage.free,
        "datasets": dataset_report,
    }
    atomic_json(run_root / "preflight.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("preflight passed")
    return report


def driver_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment.setdefault("PYTHONHASHSEED", str(args.fold_seed))
    return environment


def train_root(args: argparse.Namespace) -> Path:
    return args.run_root.resolve() / "train" / f"vae_router_cnn_cnn{args.num_cnn}"


def common_train_args(args: argparse.Namespace, dataset: str, work_dir: Path) -> list[str]:
    index = TASKS.index(dataset)
    return [
        sys.executable,
        str(repository_root() / "train_align_CL_VAE.py"),
        "--work_dir", str(work_dir),
        "--save_root", str(work_dir),
        "--run_name", f"T{index + 1:02d}_{dataset}_cnn{args.num_cnn}",
        "--dataset_name", dataset,
        "--all_datasets", ",".join(TASKS),
        "--device", args.device,
        "--model_type", "vit_b",
        "--sam_checkpoint", str(args.sam_checkpoint.resolve()),
        "--data_dir", str(args.data_dir.resolve()),
        "--method", "cnn",
        "--num_cnn", str(args.num_cnn),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--adapters_ckpt_dir", str(work_dir / "adapters_ckpt"),
        "--router_ckpt_dir", str(work_dir / "vaes_ckpt"),
    ]


def smoke(args: argparse.Namespace) -> None:
    root = args.run_root.resolve() / "smoke"
    expected = root / "adapters_ckpt" / "56Nx" / f"align_cnn_{args.num_cnn}.pth"
    if expected.is_file() and not args.force:
        print(f"smoke artifact already exists; skipping: {expected}")
        return
    command = common_train_args(args, "56Nx", root)
    command += [
        "--epochs", "1",
        "--train_batch_size", "1",
        "--eval_batch_size", "1",
        "--image_size", str(args.smoke_image_size),
        "--mask_num", "1",
        "--dataset_scale", str(args.smoke_dataset_scale),
        "--router_type", "none",
    ]
    run_logged(command, root / "driver_logs" / "smoke_56Nx.log", driver_environment(args))
    if not expected.is_file():
        raise FileNotFoundError(f"Smoke run completed without expected artifact: {expected}")


def train(args: argparse.Namespace) -> None:
    root = train_root(args)
    root.mkdir(parents=True, exist_ok=True)
    stable_config = {
        "task_order": list(TASKS),
        "method": "cnn",
        "num_cnn": args.num_cnn,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "mask_num": args.mask_num,
        "router": {
            "type": "vae",
            "feature": "attn_pool",
            "input_dim": 256,
            "latent_dim": args.vae_latent_dim,
            "beta": args.vae_beta,
            "epochs": args.vae_epochs,
            "lr": args.vae_lr,
            "k_folds": args.k_folds,
            "fold_seed": args.fold_seed,
            "tau": "p97",
        },
        "data_dir": str(args.data_dir.resolve()),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "git_revision": git_revision(repository_root()),
    }
    config_path = root / "experiment_config.json"
    if config_path.is_file() and not args.force:
        with config_path.open(encoding="utf-8") as source:
            previous_config = json.load(source)
        previous_config.pop("created_utc", None)
        if previous_config != stable_config:
            raise RuntimeError(
                f"Existing run configuration differs from this invocation: {config_path}. "
                "Choose another --run-root or use --force intentionally."
            )
    config = {"created_utc": utc_now(), **stable_config}
    atomic_json(config_path, config)

    for dataset in TASKS:
        adapter = root / "adapters_ckpt" / dataset / f"align_cnn_{args.num_cnn}.pth"
        vae = root / "vaes_ckpt" / dataset / "vae.pth"
        tau = root / "vaes_ckpt" / dataset / "tau.json"
        if all(path.is_file() for path in (adapter, vae, tau)) and not args.force:
            print(f"all artifacts already exist; skipping training task {dataset}")
            continue
        command = common_train_args(args, dataset, root)
        command += [
            "--epochs", str(args.epochs),
            "--train_batch_size", str(args.train_batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--image_size", str(args.image_size),
            "--mask_num", str(args.mask_num),
            "--dataset_scale", "1.0",
            "--router_type", "vae",
            "--router_threshold", "1.0",
            "--zero_adapter_mode", "identity",
            "--vae_feat", "attn_pool",
            "--vae_in_dim", "256",
            "--vae_latent_dim", str(args.vae_latent_dim),
            "--vae_beta", str(args.vae_beta),
            "--vae_epochs", str(args.vae_epochs),
            "--vae_lr", str(args.vae_lr),
            "--k_folds", str(args.k_folds),
            "--fold_seed", str(args.fold_seed),
        ]
        run_logged(
            command,
            root / "driver_logs" / f"train_T{TASKS.index(dataset) + 1:02d}_{dataset}.log",
            driver_environment(args),
        )
        missing = [str(path) for path in (adapter, vae, tau) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Training artifacts missing:\n" + "\n".join(missing))


def eval_stage_command(
    args: argparse.Namespace, dataset: str, evaluation_root: Path
) -> list[str]:
    root = train_root(args)
    metrics_dir = evaluation_root / "cl_metrics"
    return [
        sys.executable,
        str(repository_root() / "eval_vae_router_load_adapter.py"),
        "--work_dir", str(evaluation_root),
        "--dataset_name", dataset,
        "--all_datasets", ",".join(TASKS),
        "--device", args.device,
        "--model_type", "vit_b",
        "--sam_checkpoint", str(args.sam_checkpoint.resolve()),
        "--data_dir", str(args.data_dir.resolve()),
        "--method", "cnn",
        "--num_cnn", str(args.num_cnn),
        "--train_batch_size", str(args.train_batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--num_workers", str(args.num_workers),
        "--image_size", str(args.image_size),
        "--mask_num", str(args.mask_num),
        "--adapters_ckpt_dir", str(root / "adapters_ckpt"),
        "--router_ckpt_dir", str(root / "vaes_ckpt"),
        "--vae_feat", "attn_pool",
        "--vae_in_dim", "256",
        "--vae_latent_dim", str(args.vae_latent_dim),
        "--vae_beta", str(args.vae_beta),
        "--vae_epochs", str(args.vae_epochs),
        "--vae_lr", str(args.vae_lr),
        "--router_threshold", "1.0",
        "--zero_adapter_mode", "identity",
        "--cl_matrix_csv", str(metrics_dir / "casam_56nx_dn_iou.csv"),
        "--cl_matrix_biou_csv", str(metrics_dir / "casam_56nx_dn_biou.csv"),
        "--skip_train_vae",
    ]


def evaluate(args: argparse.Namespace) -> Path:
    root = train_root(args)
    missing = []
    for dataset in TASKS:
        for path in (
            root / "adapters_ckpt" / dataset / f"align_cnn_{args.num_cnn}.pth",
            root / "vaes_ckpt" / dataset / "vae.pth",
            root / "vaes_ckpt" / dataset / "tau.json",
        ):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Cannot evaluate; missing artifacts:\n" + "\n".join(missing))

    tag = args.eval_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evaluation_root = args.run_root.resolve() / "eval" / tag
    if evaluation_root.exists() and any(evaluation_root.iterdir()) and not args.force:
        raise FileExistsError(
            f"Evaluation directory is not empty: {evaluation_root}; choose another --eval-tag"
        )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    for dataset in TASKS:
        run_logged(
            eval_stage_command(args, dataset, evaluation_root),
            evaluation_root / "driver_logs" / f"eval_T{TASKS.index(dataset) + 1:02d}_{dataset}.log",
            driver_environment(args),
        )

    expected_rows = len(TASKS)
    for metric in ("iou", "biou"):
        csv_path = evaluation_root / "cl_metrics" / f"casam_56nx_dn_{metric}.csv"
        with csv_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
        if len(rows) != expected_rows + 1:
            raise RuntimeError(
                f"Unexpected {metric} CSV row count in {csv_path}: {len(rows)}"
            )
    print(f"evaluation complete: {evaluation_root}")
    return evaluation_root


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser()
    args.sam_checkpoint = args.sam_checkpoint.expanduser()
    args.run_root = args.run_root.expanduser()

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
    elif args.command == "all":
        smoke(args)
        train(args)
        evaluate(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
