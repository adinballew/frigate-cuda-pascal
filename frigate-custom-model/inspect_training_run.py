#!/usr/bin/env python3
"""Summarize a YOLO training run results.csv and exported ONNX metadata."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--onnx", type=Path)
    args = p.parse_args()
    out = {"run": str(args.run)}
    csv_path = args.run / "results.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
        if rows:
            out["last_metrics"] = {k.strip(): v.strip() for k, v in rows[-1].items()}
    weights = args.run / "weights" / "best.pt"
    out["best_pt_exists"] = weights.exists()
    if weights.exists():
        out["best_pt_bytes"] = weights.stat().st_size
    if args.onnx:
        out["onnx_exists"] = args.onnx.exists()
        if args.onnx.exists():
            out["onnx_bytes"] = args.onnx.stat().st_size
            try:
                import onnx  # type: ignore
                m = onnx.load(str(args.onnx))
                i = m.graph.input[0]
                out["onnx_input"] = {
                    "name": i.name,
                    "shape": [str(d.dim_value or d.dim_param or "?") for d in i.type.tensor_type.shape.dim],
                    "elem_type": i.type.tensor_type.elem_type,
                }
                out["onnx_outputs"] = [o.name for o in m.graph.output]
            except Exception as exc:
                out["onnx_inspect_error"] = str(exc)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
