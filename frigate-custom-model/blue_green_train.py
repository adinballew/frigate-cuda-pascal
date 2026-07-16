#!/usr/bin/env python3
"""Blue/green Frigate YOLO candidate trainer.

Creates versioned datasets from human-decisioned labels, trains a *blue*
candidate in the existing `frigate-model-trainer` container, exports ONNX, and
writes a manifest. It never changes Frigate's active/green detector config.

Stdlib-only coordinator; training dependencies live in the trainer container.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_REVIEW_ROOT = Path("/mnt/user/media/frigate_custom_model/review")
DEFAULT_MEDIA_ROOT = Path("/mnt/user/media/frigate_custom_model")
DEFAULT_WORKSPACE = Path("/mnt/user/appdata/openclaw/workspace")
DEFAULT_LABELS = DEFAULT_WORKSPACE / "docs/frigate-custom-model/labels.txt"
DEFAULT_TRAINER_DATA_ROOT = Path("/workspace/frigate_custom_model")
DEFAULT_TRAINER = "frigate-model-trainer"
DOCKER_SOCK = "/var/run/docker.sock"


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


def docker_request(method: str, path: str, body: dict | None = None) -> bytes:
    conn = UnixHTTPConnection(DOCKER_SOCK)
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    payload = resp.read()
    if resp.status >= 300:
        raise RuntimeError(f"Docker {method} {path} -> {resp.status}: {payload[:500]!r}")
    return payload


def docker_exec(container: str, cmd: list[str]) -> int:
    created = json.loads(docker_request("POST", f"/containers/{container}/exec", {
        "AttachStdout": True,
        "AttachStderr": True,
        "Cmd": cmd,
    }))
    output = docker_request("POST", f"/exec/{created['Id']}/start", {"Detach": False, "Tty": False})
    sys.stdout.buffer.write(output)
    sys.stdout.flush()
    info = json.loads(docker_request("GET", f"/exec/{created['Id']}/json"))
    return int(info.get("ExitCode") or 0)


def run(cmd: list[str], *, cwd: Path | None = None, execute: bool = True) -> int:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    if not execute:
        return 0
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    return completed.returncode


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_decisioned(review_root: Path, labels: list[str]) -> dict:
    root = review_root / "_decisioned"
    by_camera: dict[str, int] = {}
    boxes_by_class = {name: 0 for name in labels}
    image_count = 0
    negative_count = 0
    box_count = 0
    missing_labels: list[str] = []
    invalid: list[str] = []
    for image in sorted(root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        if not image.name.startswith("frigate_plus_"):
            continue
        label = image.with_suffix(".txt")
        if not label.exists():
            missing_labels.append(str(image))
            continue
        camera = "FrontDoor" if "FrontDoor" in image.parts else "Backyard" if "Backyard" in image.parts else "unknown"
        image_count += 1
        by_camera[camera] = by_camera.get(camera, 0) + 1
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            negative_count += 1
        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            try:
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except Exception as exc:
                invalid.append(f"{label}:{line_no}: {exc}")
                continue
            if len(parts) != 5 or cid < 0 or cid >= len(labels) or any(x < 0 or x > 1 for x in coords):
                invalid.append(f"{label}:{line_no}: invalid YOLO line {line!r}")
                continue
            boxes_by_class[labels[cid]] += 1
            box_count += 1
    return {
        "image_count": image_count,
        "negative_count": negative_count,
        "box_count": box_count,
        "by_camera": by_camera,
        "boxes_by_class": {k: v for k, v in boxes_by_class.items() if v},
        "missing_labels": missing_labels,
        "invalid": invalid,
    }


def rewrite_dataset_yaml_for_trainer(dataset_yaml: Path, host_dataset_root: Path, trainer_dataset_root: Path) -> None:
    text = dataset_yaml.read_text(encoding="utf-8")
    text = text.replace(f"path: {host_dataset_root}", f"path: {trainer_dataset_root}")
    dataset_yaml.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--trainer-data-root", type=Path, default=DEFAULT_TRAINER_DATA_ROOT)
    parser.add_argument("--min-labels", type=int, default=25, help="Minimum human-decisioned images needed unless --force")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", help="Candidate run name; default blue_candidate_<timestamp>")
    parser.add_argument("--force", action="store_true", help="Train even when below --min-labels")
    parser.add_argument("--execute", action="store_true", help="Actually write dataset, train, and export. Default is plan only.")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    stats = collect_decisioned(args.review_root, labels)
    print(json.dumps({"decisioned": stats}, indent=2, sort_keys=True))
    if stats["invalid"]:
        raise SystemExit("invalid labels present; refusing to train")
    if stats["image_count"] < args.min_labels and not args.force:
        print(f"Below threshold: {stats['image_count']} < {args.min_labels}. Use --force for smoke/candidate training.")
        return 3

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.name or f"blue_candidate_{stamp}"
    dataset_root = args.media_root / "datasets" / name
    trainer_dataset_root = args.trainer_data_root / "datasets" / name
    runs_root = args.trainer_data_root / "runs"
    models_root = args.trainer_data_root / "models" / "blue"
    manifest_path = args.media_root / "models" / "blue" / f"{name}.manifest.json"

    prepare_cmd = [
        sys.executable,
        str(args.workspace / "docker/frigate/frigate-custom-model/prepare_yolo_dataset.py"),
        "--review-root", str(args.review_root),
        "--output-root", str(dataset_root),
        "--labels", str(args.labels),
        "--yaml-path-root", str(trainer_dataset_root),
        "--write",
    ]
    if dataset_root.exists() and args.execute:
        shutil.rmtree(dataset_root)
    rc = run(prepare_cmd, cwd=args.workspace, execute=args.execute)
    if rc != 0:
        return rc

    train_cmd = [
        "bash", "-lc",
        "cd /workspace/frigate_custom_model && "
        f"yolo detect train model={args.base_model} data={trainer_dataset_root / 'dataset.yaml'} "
        f"imgsz=640 epochs={args.epochs} batch={args.batch} project={runs_root} name={name} "
        f"exist_ok=False device={args.device} patience=5 workers=0",
    ]
    print("$ docker exec", args.trainer, train_cmd)
    if args.execute:
        rc = docker_exec(args.trainer, train_cmd)
        if rc != 0:
            return rc

    weights = runs_root / name / "weights" / "best.pt"
    exported = runs_root / name / "weights" / "best.onnx"
    blue_onnx = models_root / f"{name}.onnx"
    export_cmd = [
        "bash", "-lc",
        f"yolo export model={weights} format=onnx imgsz=640 batch=1 dynamic=False simplify=True opset=12 && "
        f"mkdir -p {models_root} && cp {exported} {blue_onnx} && "
        "python3 - <<'PY'\n"
        "import onnx\n"
        f"p='{blue_onnx}'\n"
        "m=onnx.load(p)\n"
        "i=m.graph.input[0]\n"
        "dims=[str(d.dim_value or d.dim_param or '?') for d in i.type.tensor_type.shape.dim]\n"
        "print('onnx', p)\nprint('input', i.name, dims, 'elem_type', i.type.tensor_type.elem_type)\n"
        "PY",
    ]
    print("$ docker exec", args.trainer, export_cmd)
    if args.execute:
        rc = docker_exec(args.trainer, export_cmd)
        if rc != 0:
            return rc

    manifest = {
        "name": name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "state": "blue_candidate",
        "green_deployed": False,
        "dataset_root_host": str(dataset_root),
        "dataset_root_trainer": str(trainer_dataset_root),
        "run_root_trainer": str(runs_root / name),
        "weights_trainer": str(weights),
        "onnx_host": str(args.media_root / "models" / "blue" / f"{name}.onnx"),
        "labels": labels,
        "stats": stats,
        "train": {"base_model": args.base_model, "epochs": args.epochs, "batch": args.batch, "device": args.device},
        "promotion_policy": "Manual only: compare metrics/predictions, then explicitly approve Frigate config switch. This script never promotes blue to green.",
    }
    print(json.dumps({"manifest": manifest}, indent=2, sort_keys=True))
    if args.execute:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        latest = manifest_path.parent / "latest-blue-manifest.json"
        latest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"manifest: {manifest_path}")
        print(f"latest: {latest}")
    else:
        print("Plan only. Add --execute to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
