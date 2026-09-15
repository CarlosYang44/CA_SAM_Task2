#!/usr/bin/env python3
"""Evaluate one fixed Alignment Layer without task routing or adapter selection."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import get_loader  # noqa: E402
from segment_anything import sam_model_registry  # noqa: E402
from train_align_CL_VAE import build_align_module, evaluate  # noqa: E402
from utils import FocalDiceloss_IoULoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed shared Alignment Layer; no VAE/router is constructed."
    )
    parser.add_argument("--dataset-name", required=True, choices=("56Nx", "DN"))
    parser.add_argument("--align-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--method", default="cnn", choices=("cnn", "mlp", "transformer", "transplus"))
    parser.add_argument("--num-cnn", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--mask-num", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--multimask", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    args.test_mode = True
    args.dataset_scale = 1.0
    args.dist = False
    args.metrics = ["iou", "dice", "biou"]
    args.sam_checkpoint = str(args.sam_checkpoint.resolve())
    args.data_dir = str(args.data_dir.resolve())

    if not Path(args.sam_checkpoint).is_file():
        raise FileNotFoundError(args.sam_checkpoint)
    align_checkpoint = args.align_checkpoint.resolve()
    if not align_checkpoint.is_file():
        raise FileNotFoundError(align_checkpoint)

    loader = get_loader(args, all_training_sets=[args.dataset_name])
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        subset = Subset(loader.dataset, range(min(args.max_samples, len(loader.dataset))))
        loader = DataLoader(
            subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=loader.collate_fn,
        )

    sam = sam_model_registry[args.model_type](args).to(args.device)
    for parameter in sam.parameters():
        parameter.requires_grad = False
    align_module = build_align_module(args).to(args.device)
    state = torch.load(align_checkpoint, map_location=args.device)
    align_module.load_state_dict(state, strict=True)
    for parameter in align_module.parameters():
        parameter.requires_grad = False

    loss, values = evaluate(
        args, sam, align_module, loader, FocalDiceloss_IoULoss(), epoch=0
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset_name, "align_checkpoint": str(align_checkpoint),
        "router": "disabled", "loss": loss,
        "metrics": dict(zip(args.metrics, values)), "evaluated_samples": len(loader.dataset),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
