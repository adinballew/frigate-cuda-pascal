#!/usr/bin/env python3
"""Run the v0 smoke train/export sequence if the training stack is available.

This is a coordinator around existing dry-run friendly wrappers. It does not
install dependencies. It fails cleanly if `yolo` is unavailable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(str(part) for part in cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-yaml", default="/mnt/user/media/frigate_custom_model/dataset/dataset.yaml")
    parser.add_argument("--project", default="/mnt/user/media/frigate_custom_model/runs")
    parser.add_argument("--models-root", default="/mnt/user/media/frigate_custom_model/models")
    parser.add_argument("--name", default="smoke_v0")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--execute", action="store_true", help="Actually train/export. Default only prints wrapper dry runs.")
    args = parser.parse_args()

    yolo = shutil.which("yolo")
    if args.execute and not yolo:
        raise SystemExit("Cannot execute: `yolo` command not found. Set up training environment first.")
    if not Path(args.dataset_yaml).exists():
        raise SystemExit(f"dataset yaml missing: {args.dataset_yaml}")

    train_cmd = [
        "python3", "docker/frigate/frigate-custom-model/train_yolo.py",
        "--dataset-yaml", args.dataset_yaml,
        "--base-model", args.base_model,
        "--project", args.project,
        "--name", args.name,
        "--epochs", "1",
        "--batch", "2",
    ]
    if args.execute:
        train_cmd.append("--execute")
    rc = run(train_cmd)
    if rc != 0:
        return rc

    weights = Path(args.project) / args.name / "weights" / "best.pt"
    output = Path(args.models_root) / f"{args.name}.onnx"
    export_cmd = [
        "python3", "docker/frigate/frigate-custom-model/export_onnx.py",
        "--weights", str(weights),
        "--output", str(output),
    ]
    if args.execute:
        export_cmd.append("--execute")
    return run(export_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
