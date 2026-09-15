#!/usr/bin/env python3
"""Print an existing shared-AL summary as a compact Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    path = args.experiment_dir / "results" / "summary.json"
    with path.open(encoding="utf-8") as source:
        summary = json.load(source)
    print("| Metric | 56Nx before | 56Nx after DN | DN sequential | DN only | Forgetting | Plasticity gap |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for metric, values in summary["by_metric"].items():
        print(
            f"| {metric} | {values['56Nx_before']:.4f} | {values['56Nx_after_DN']:.4f} | "
            f"{values['DN_sequential']:.4f} | {values['DN_only']:.4f} | "
            f"{values['forgetting']:.4f} | {values['plasticity_gap']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
