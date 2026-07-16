#!/usr/bin/env python3
"""Auto-train a Frigate custom-model blue candidate when enough new labels exist.

This wrapper is safe for Cronicle scheduling:
  * It never edits Frigate config and never promotes blue to green.
  * It no-ops with exit 0 below --threshold.
  * It counts only images in review/_decisioned newer than latest blue manifest
    timestamp and only when a paired YOLO .txt label exists.
  * It runs media/training work inside the existing frigate-model-trainer
    container, so the OpenClaw container does not need the media mount.
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import io
import json
import os
import shlex
import socket
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import quote

SOCK = "/var/run/docker.sock"
DEFAULT_TRAINER = "frigate-model-trainer"
DEFAULT_WORKSPACE = Path("/opt/frigate")
DEFAULT_LABELS = Path("/opt/frigate/labels.txt")
DEFAULT_TRAINER_ROOT = Path("/workspace/frigate_custom_model")
DEFAULT_LABEL_LIST = [
    "person", "package", "car", "truck", "van", "dog", "cat", "bird",
    "bicycle", "motorcycle", "backpack", "suitcase", "waste_bin",
]

CHECK_DECISIONED_CODE = r'''
import datetime as dt
import json
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
root = Path(sys.argv[1])
labels = json.loads(sys.argv[2])
manifest_path = root / "models" / "blue" / "latest-blue-manifest.json"
review_decisioned = root / "review" / "_decisioned"
manifest = None
manifest_ts = 0.0
manifest_time_source = "missing"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    created = manifest.get("created_at")
    if created:
        try:
            manifest_ts = dt.datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            manifest_time_source = "created_at"
        except Exception:
            pass
    if not manifest_ts:
        manifest_ts = manifest_path.stat().st_mtime
        manifest_time_source = "manifest_mtime"

stats = {
    "root": str(root),
    "review_decisioned": str(review_decisioned),
    "latest_manifest": str(manifest_path),
    "manifest_exists": manifest_path.exists(),
    "manifest_name": manifest.get("name") if isinstance(manifest, dict) else None,
    "manifest_created_at": manifest.get("created_at") if isinstance(manifest, dict) else None,
    "manifest_timestamp": manifest_ts,
    "manifest_time_source": manifest_time_source,
    "new_image_count": 0,
    "new_negative_count": 0,
    "new_box_count": 0,
    "new_by_camera": {},
    "new_boxes_by_class": {},
    "total_paired_image_count": 0,
    "missing_label_count": 0,
    "missing_labels_sample": [],
    "invalid_count": 0,
    "invalid_sample": [],
}
if not review_decisioned.is_dir():
    print(json.dumps(stats, sort_keys=True))
    raise SystemExit(0)

for image in sorted(review_decisioned.rglob("*")):
    if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
        continue
    label = image.with_suffix(".txt")
    if not label.exists():
        stats["missing_label_count"] += 1
        if len(stats["missing_labels_sample"]) < 10:
            stats["missing_labels_sample"].append(str(image))
        continue
    stats["total_paired_image_count"] += 1
    is_new = image.stat().st_mtime > manifest_ts
    lines = [line.strip() for line in label.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    parsed = []
    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        try:
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except Exception as exc:
            stats["invalid_count"] += 1
            if len(stats["invalid_sample"]) < 20:
                stats["invalid_sample"].append(f"{label}:{line_no}: {exc}")
            continue
        if len(parts) != 5 or cid < 0 or cid >= len(labels) or any(x < 0 or x > 1 for x in coords):
            stats["invalid_count"] += 1
            if len(stats["invalid_sample"]) < 20:
                stats["invalid_sample"].append(f"{label}:{line_no}: invalid YOLO line {line!r}")
            continue
        parsed.append(cid)
    if not is_new:
        continue
    stats["new_image_count"] += 1
    if not lines:
        stats["new_negative_count"] += 1
    camera = "unknown"
    for part in image.parts:
        if part in ("FrontDoor", "Backyard"):
            camera = part
            break
    stats["new_by_camera"][camera] = stats["new_by_camera"].get(camera, 0) + 1
    for cid in parsed:
        name = labels[cid]
        stats["new_boxes_by_class"][name] = stats["new_boxes_by_class"].get(name, 0) + 1
        stats["new_box_count"] += 1
print(json.dumps(stats, sort_keys=True))
'''

WRITE_MANIFEST_CODE = r'''
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
name = sys.argv[2]
manifest = json.loads(sys.argv[3])
out = root / "models" / "blue"
out.mkdir(parents=True, exist_ok=True)
versioned = out / f"{name}.manifest.json"
latest = out / "latest-blue-manifest.json"
payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
versioned.write_text(payload, encoding="utf-8")
latest.write_text(payload, encoding="utf-8")
print(json.dumps({"manifest": str(versioned), "latest": str(latest)}, sort_keys=True))
'''

class UnixHTTP(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(SOCK)


def docker(method: str, path: str, body: bytes | dict | None = None, headers: dict | None = None, timeout: int = 60) -> bytes:
    conn = UnixHTTP("localhost", timeout=timeout)
    data = body
    hdrs = dict(headers or {})
    if isinstance(body, dict):
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
        hdrs["Content-Length"] = str(len(data))
    elif isinstance(body, bytes):
        hdrs["Content-Length"] = str(len(body))
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    if resp.status >= 300:
        raise RuntimeError(f"Docker {method} {path} -> {resp.status}: {raw[:1000]!r}")
    return raw


def demux(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i + 8 <= len(raw) and raw[i] in (0, 1, 2):
        size = int.from_bytes(raw[i + 4:i + 8], "big")
        i += 8
        out.extend(raw[i:i + size])
        i += size
    data = bytes(out) if out and i == len(raw) else raw
    return data.decode("utf-8", "replace")


def exec_container(container: str, cmd: list[str], *, timeout: int = 86400) -> tuple[int, str]:
    create = json.loads(docker("POST", f"/containers/{quote(container)}/exec", {
        "Cmd": cmd,
        "AttachStdout": True,
        "AttachStderr": True,
        "Tty": False,
    }, timeout=60))
    out = docker("POST", f"/exec/{create['Id']}/start", {"Detach": False, "Tty": False}, timeout=timeout)
    info = json.loads(docker("GET", f"/exec/{create['Id']}/json", timeout=60))
    return int(info.get("ExitCode") or 0), demux(out)


def put_archive(container: str, dest_path: str, files: list[tuple[bytes, str, int]]) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for data, arcname, mode in files:
            ti = tarfile.TarInfo(arcname)
            ti.size = len(data)
            ti.mtime = time.time()
            ti.mode = mode
            tf.addfile(ti, io.BytesIO(data))
    docker("PUT", f"/containers/{quote(container)}/archive?path={quote(dest_path)}", buf.getvalue(), {"Content-Type": "application/x-tar"}, timeout=60)


def load_labels(path: Path) -> list[str]:
    if path.exists():
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if labels:
            return labels
    return list(DEFAULT_LABEL_LIST)


def shell(cmd: str) -> list[str]:
    return ["bash", "-lc", cmd]


def run_or_raise(container: str, cmd: list[str], label: str, *, timeout: int = 86400) -> str:
    rc, out = exec_container(container, cmd, timeout=timeout)
    print(out, end="" if out.endswith("\n") else "\n")
    if rc:
        raise RuntimeError(f"{label} failed with exit {rc}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=int, default=20)
    ap.add_argument("--trainer", default=DEFAULT_TRAINER)
    ap.add_argument("--trainer-root", type=Path, default=DEFAULT_TRAINER_ROOT)
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--base-model", default="yolo11n.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="suggestion confidence threshold")
    ap.add_argument("--name", help="run/model name; default blue_auto_<UTC timestamp>")
    ap.add_argument("--dry-run", action="store_true", help="plan only; never train or write suggestions")
    ap.add_argument("--force", action="store_true", help="train even below threshold")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    tmp = f"/tmp/openclaw_auto_blue_{int(time.time())}"
    try:
        rc, _ = exec_container(args.trainer, ["bash", "-lc", f"rm -rf {shlex.quote(tmp)} && mkdir -p {shlex.quote(tmp)}"], timeout=60)
        if rc:
            raise RuntimeError("failed to prepare trainer tmp dir")
        put_archive(args.trainer, tmp, [
            (CHECK_DECISIONED_CODE.encode("utf-8"), "check_decisioned_since_manifest.py", 0o755),
            (("\n".join(labels) + "\n").encode("utf-8"), "labels.txt", 0o644),
        ])
        check_cmd = ["python3", f"{tmp}/check_decisioned_since_manifest.py", str(args.trainer_root), json.dumps(labels)]
        rc, check_out = exec_container(args.trainer, check_cmd, timeout=120)
        if rc:
            print(check_out, file=sys.stderr)
            return rc
        stats = json.loads(check_out.strip().splitlines()[-1])

        ready = stats["new_image_count"] >= args.threshold or args.force
        name = args.name or "blue_auto_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        plan = {
            "status": "ready" if ready else "below_threshold",
            "dry_run": bool(args.dry_run),
            "threshold": args.threshold,
            "force": bool(args.force),
            "trainer": args.trainer,
            "trainer_root": str(args.trainer_root),
            "name": name,
            "new_decisioned": stats,
            "will_train": bool(ready and not args.dry_run),
            "will_generate_suggestions": bool(ready and not args.dry_run),
            "safety": {
                "frigate_config_change": False,
                "green_promotion": False,
                "blue_only": True,
            },
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        if not ready or args.dry_run:
            return 0
        if stats.get("invalid_count"):
            raise RuntimeError("invalid YOLO labels present; refusing to train")

        prepare = args.workspace / "docker/frigate/frigate-custom-model/prepare_yolo_dataset.py"
        suggest = args.workspace / "docker/frigate/frigate-custom-model/suggest_yolo_drafts.py"
        if not prepare.exists() or not suggest.exists():
            raise FileNotFoundError(f"missing helper script(s): {prepare}, {suggest}")
        put_archive(args.trainer, tmp, [
            (prepare.read_bytes(), "prepare_yolo_dataset.py", 0o755),
            (suggest.read_bytes(), "suggest_yolo_drafts.py", 0o755),
        ])

        dataset_root = args.trainer_root / "datasets" / name
        runs_root = args.trainer_root / "runs"
        models_root = args.trainer_root / "models" / "blue"
        prepare_cmd = [
            "python3", f"{tmp}/prepare_yolo_dataset.py",
            "--review-root", str(args.trainer_root / "review"),
            "--output-root", str(dataset_root),
            "--labels", f"{tmp}/labels.txt",
            "--yaml-path-root", str(dataset_root),
            "--write",
        ]
        run_or_raise(args.trainer, prepare_cmd, "prepare dataset", timeout=1800)

        train_line = (
            f"cd {shlex.quote(str(args.trainer_root))} && "
            f"yolo detect train model={shlex.quote(args.base_model)} data={shlex.quote(str(dataset_root / 'dataset.yaml'))} "
            f"imgsz=640 epochs={args.epochs} batch={args.batch} project={shlex.quote(str(runs_root))} "
            f"name={shlex.quote(name)} exist_ok=False device={shlex.quote(args.device)} patience=5 workers=0"
        )
        run_or_raise(args.trainer, shell(train_line), "train blue candidate", timeout=86400)

        weights = runs_root / name / "weights" / "best.pt"
        exported = runs_root / name / "weights" / "best.onnx"
        blue_onnx = models_root / f"{name}.onnx"
        export_line = (
            f"yolo export model={shlex.quote(str(weights))} format=onnx imgsz=640 batch=1 dynamic=False simplify=True opset=12 && "
            f"mkdir -p {shlex.quote(str(models_root))} && cp {shlex.quote(str(exported))} {shlex.quote(str(blue_onnx))}"
        )
        run_or_raise(args.trainer, shell(export_line), "export blue ONNX", timeout=3600)

        manifest = {
            "name": name,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "state": "blue_candidate",
            "green_deployed": False,
            "dataset_root_trainer": str(dataset_root),
            "run_root_trainer": str(runs_root / name),
            "weights_trainer": str(weights),
            "onnx_trainer": str(blue_onnx),
            "labels": labels,
            "new_since_previous_blue": stats,
            "train": {"base_model": args.base_model, "epochs": args.epochs, "batch": args.batch, "device": args.device},
            "promotion_policy": "Manual only. This script never changes Frigate config and never promotes green.",
        }
        put_archive(args.trainer, tmp, [(WRITE_MANIFEST_CODE.encode("utf-8"), "write_manifest.py", 0o755)])
        run_or_raise(args.trainer, ["python3", f"{tmp}/write_manifest.py", str(args.trainer_root), name, json.dumps(manifest)], "write blue manifest", timeout=120)

        suggest_cmd = (
            f"CUDA_VISIBLE_DEVICES='' python3 {shlex.quote(tmp + '/suggest_yolo_drafts.py')} "
            f"--model {shlex.quote(str(blue_onnx))} --review-root {shlex.quote(str(args.trainer_root / 'review'))} "
            f"--conf {args.conf} --overwrite --respect-labeled"
        )
        suggestions_out = run_or_raise(args.trainer, shell(suggest_cmd), "generate suggestions", timeout=14400)
        result = {"status": "trained_blue", "name": name, "onnx_trainer": str(blue_onnx), "suggestions_output_tail": suggestions_out[-4000:]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        try:
            exec_container(args.trainer, ["bash", "-lc", f"rm -rf {shlex.quote(tmp)}"], timeout=60)
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(main())
