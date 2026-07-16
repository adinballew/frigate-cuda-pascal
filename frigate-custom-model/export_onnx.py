#!/usr/bin/env python3
"""Dry-run friendly YOLO -> ONNX export wrapper for Frigate yolo-generic.

Target Frigate model assumptions:
- ONNX
- yolo-generic
- input shape [1, 3, 640, 640]
- NCHW
- RGB
- float

The script prints the export command by default. Use --execute to run export.
If onnx is installed and the output exists, it can inspect input metadata.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


def build_command(weights: Path) -> list[str]:
    return [
        "yolo",
        "export",
        f"model={weights}",
        "format=onnx",
        "imgsz=640",
        "batch=1",
        "dynamic=False",
        "simplify=True",
        "opset=12",
    ]


def inspect_onnx(path: Path) -> None:
    try:
        import onnx  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        print(f"ONNX metadata check skipped; import onnx failed: {exc}")
        return
    model = onnx.load(str(path))
    if not model.graph.input:
        print("ONNX metadata warning: model has no graph inputs")
        return
    first = model.graph.input[0]
    dims: list[str] = []
    tensor_type = first.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(str(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    print(f"ONNX first input: name={first.name} shape=[{', '.join(dims)}] elem_type={tensor_type.elem_type}")
    if dims != ["1", "3", "640", "640"]:
        print("WARNING: expected static shape [1, 3, 640, 640] for Frigate yolo-generic")
    else:
        print("Shape check OK for [1, 3, 640, 640]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Trained YOLO .pt weights")
    parser.add_argument("--output", required=True, type=Path, help="Desired ONNX output path")
    parser.add_argument("--execute", action="store_true", help="Run export; default only prints command")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect an existing --output ONNX file")
    args = parser.parse_args()

    if args.inspect_only:
        if not args.output.exists():
            raise SystemExit(f"output does not exist for inspect-only: {args.output}")
        inspect_onnx(args.output)
        return 0

    cmd = build_command(args.weights)
    print("Export target: ONNX yolo-generic, 640, NCHW, RGB, float, batch=1")
    print("Command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    if not args.weights.exists():
        message = f"weights do not exist yet: {args.weights}"
        if not args.execute:
            print(f"\nDry run note: {message}")
            print("Dry run only. Add --execute after training has produced weights.")
            return 0
        raise SystemExit(message)

    if not args.execute:
        print("\nDry run only. Add --execute to export locally.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        return completed.returncode

    exported_default = args.weights.with_suffix(".onnx")
    if exported_default.exists() and exported_default.resolve() != args.output.resolve():
        shutil.copy2(exported_default, args.output)
        print(f"Copied {exported_default} -> {args.output}")
    elif args.output.exists():
        print(f"ONNX exists at {args.output}")
    else:
        print(f"Export completed but expected output was not found: {args.output}")
        print(f"Check Ultralytics export location near: {args.weights}")
        return 2

    inspect_onnx(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
