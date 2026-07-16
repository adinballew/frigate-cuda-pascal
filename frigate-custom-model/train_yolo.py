#!/usr/bin/env python3
"""Dry-run friendly wrapper for local Ultralytics YOLO training.

This script does not install dependencies or download datasets. It prints the
training command by default and only executes it when --execute is supplied.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_LABELS = [
    "person",
    "package",
    "car",
    "truck",
    "van",
    "dog",
    "cat",
    "bird",
    "bicycle",
    "motorcycle",
    "backpack",
    "suitcase",
    "waste_bin",
]


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "yolo",
        "detect",
        "train",
        f"model={args.base_model}",
        f"data={args.dataset_yaml}",
        "imgsz=640",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"project={args.project}",
        f"name={args.name}",
        "exist_ok=False",
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    if args.patience is not None:
        cmd.append(f"patience={args.patience}")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-yaml", required=True, type=Path, help="YOLO dataset.yaml from prepare_yolo_dataset.py")
    parser.add_argument("--base-model", default="yolo11n.pt", help="Local/pretrained YOLO model path or Ultralytics model name")
    parser.add_argument("--project", required=True, type=Path, help="Training output directory")
    parser.add_argument("--name", required=True, help="Run name, e.g. frontdoor_backyard_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="Optional Ultralytics device, e.g. cpu, 0, 0,1")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--execute", action="store_true", help="Run the command; default only prints it")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if not args.dataset_yaml.exists():
        raise SystemExit(f"dataset yaml does not exist: {args.dataset_yaml}")

    cmd = build_command(args)
    print("Training target: YOLO detect model, imgsz=640, classes=13")
    print("Class order:")
    for idx, label in enumerate(DEFAULT_LABELS):
        print(f"  {idx}: {label}")
    print("\nCommand:")
    print(" ".join(shlex.quote(part) for part in cmd))

    if not args.execute:
        print("\nDry run only. Add --execute to start local training.")
        return 0

    args.project.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
