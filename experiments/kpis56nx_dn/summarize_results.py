#!/usr/bin/env python3
"""Render CA-SAM 56Nx -> DN IoU/BIoU CSV files as a Markdown table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", type=Path)
    return parser.parse_args()


def read_metric(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {row["Task"]: row for row in rows}


def show(value: str | None) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.4f}"


def main() -> int:
    args = parse_args()
    metrics = args.evaluation_dir / "cl_metrics"
    iou = read_metric(metrics / "casam_56nx_dn_iou.csv")
    biou = read_metric(metrics / "casam_56nx_dn_biou.csv")
    print("| Stage | 56Nx IoU | DN IoU | Avg IoU | 56Nx BIoU | DN BIoU | Avg BIoU |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for stage, name in (("0", "T1 56Nx"), ("1", "T2 DN")):
        if stage not in iou or stage not in biou:
            continue
        print(
            f"| {name} | {show(iou[stage].get('56Nx'))} | "
            f"{show(iou[stage].get('DN'))} | {show(iou[stage].get('Avg'))} | "
            f"{show(biou[stage].get('56Nx'))} | {show(biou[stage].get('DN'))} | "
            f"{show(biou[stage].get('Avg'))} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
