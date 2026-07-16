#!/usr/bin/env python3
"""Generate `.suggest.txt` draft labels using the existing trainer container.

This bridges the current path split: the labeler serves host-visible review files,
while the model/deps live in `frigate-model-trainer`. The script copies only
unlabeled review images into a temp dir inside the trainer, runs ONNX inference,
and copies `.suggest.txt` sidecars back next to the original images.
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import shutil
import shlex
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import quote

SOCK = "/var/run/docker.sock"
CONTAINER = "frigate-model-trainer"
ALLOWED_CAMERAS = ("FrontDoor", "Backyard")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SCRIPT = Path(__file__).with_name("generate_draft_labels.py")
DEFAULT_SEED_MODEL = "/workspace/frigate_custom_model/models/v0_package_seed.onnx"
LATEST_BLUE_MANIFEST = "/workspace/frigate_custom_model/models/blue/latest-blue-manifest.json"


def default_model() -> str:
    """Return latest blue ONNX from the trainer container, falling back to seed."""
    snippet = (
        "import json, pathlib; "
        f"p=pathlib.Path({LATEST_BLUE_MANIFEST!r}); "
        f"print(json.loads(p.read_text()).get('onnx_trainer') if p.exists() else {DEFAULT_SEED_MODEL!r})"
    )
    try:
        model = exec_container("python3 -c " + shlex.quote(snippet)).strip().splitlines()[-1]
        return model or DEFAULT_SEED_MODEL
    except Exception:
        return DEFAULT_SEED_MODEL


class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(SOCK)


def docker(method: str, path: str, body: bytes | dict | None = None, headers: dict | None = None) -> bytes:
    conn = UnixHTTP("localhost")
    data = body
    hdrs = dict(headers or {})
    if isinstance(body, dict):
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    if resp.status >= 300:
        raise RuntimeError(f"Docker {method} {path} -> {resp.status}: {raw[:500]!r}")
    return raw


def demux(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i + 8 <= len(raw) and raw[i] in (0, 1, 2):
        size = int.from_bytes(raw[i + 4:i + 8], "big")
        i += 8
        out.extend(raw[i:i + size])
        i += size
    return (out if out and i == len(raw) else raw).decode("utf-8", "replace")


def exec_container(cmd: str) -> str:
    raw = docker("POST", f"/containers/{CONTAINER}/exec", {
        "Cmd": ["bash", "-lc", cmd],
        "AttachStdout": True,
        "AttachStderr": True,
        "Tty": False,
    })
    exec_id = json.loads(raw)["Id"]
    out = docker("POST", f"/exec/{exec_id}/start", {"Detach": False, "Tty": False})
    info = json.loads(docker("GET", f"/exec/{exec_id}/json"))
    text = demux(out)
    if info.get("ExitCode"):
        raise RuntimeError(f"container command failed exit={info.get('ExitCode')}\n{text}")
    return text


def put_archive(dest_path: str, files: list[tuple[Path, str]], extra: list[tuple[bytes, str]] = []) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        dirs = set()
        for src, arcname in files:
            for parent in Path(arcname).parents:
                if str(parent) == ".":
                    continue
                dirs.add(str(parent))
            ti = tf.gettarinfo(str(src), arcname=arcname)
            with src.open("rb") as f:
                tf.addfile(ti, f)
        for data, arcname in extra:
            for parent in Path(arcname).parents:
                if str(parent) == ".":
                    continue
                dirs.add(str(parent))
            ti = tarfile.TarInfo(arcname)
            ti.size = len(data)
            ti.mtime = time.time()
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    docker("PUT", f"/containers/{CONTAINER}/archive?path={quote(dest_path)}", buf.getvalue(), {"Content-Type": "application/x-tar"})


def get_archive(src_path: str) -> bytes:
    return docker("GET", f"/containers/{CONTAINER}/archive?path={quote(src_path)}")


def candidate_images(review_root: Path, cameras: list[str], image: Path | None = None) -> list[Path]:
    if image:
        return [image]
    out: list[Path] = []
    for camera in cameras:
        img_dir = review_root / camera / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            label = img.with_suffix(".txt")
            if label.exists() and label.read_text(encoding="utf-8").strip():
                continue
            out.append(img)
    return out


def _is_blocked_legacy_suggestion(path: Path) -> bool:
    """True for importer sidecars made from Frigate detector metadata.

    The labeler refuses these as model suggestions; remove them before copying
    fresh model output back so stale Frigate boxes cannot keep winning.
    """
    try:
        return "source=frigate" in path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-root", default="/mnt/user/media/frigate_custom_model/review")
    ap.add_argument("--trainer-review-root", default="/workspace/frigate_custom_model/review")
    ap.add_argument(
        "--model",
        default=None,
        help="Trainer-container model path. Default: latest blue ONNX, falling back to v0 seed.",
    )
    ap.add_argument("--camera", action="append", choices=ALLOWED_CAMERAS)
    ap.add_argument("--image", type=Path, help="Single host image path to suggest")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--class-id", action="append", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    review_root = Path(args.review_root)
    cameras = args.camera or list(ALLOWED_CAMERAS)
    if not review_root.is_dir():
        if not SCRIPT.is_file():
            raise SystemExit(f"missing helper script: {SCRIPT}")
        run_id = f"frigate_draft_direct_{int(time.time())}"
        container_root = f"/tmp/{run_id}"
        exec_container(f"rm -rf {container_root!r} && mkdir -p {container_root!r}")
        put_archive(container_root, [], [(SCRIPT.read_bytes(), "generate_draft_labels.py")])
        model = args.model or default_model()
        cmd = [
            "python3", f"{container_root}/generate_draft_labels.py",
            "--review-root", args.trainer_review_root,
            "--model", model,
            "--conf", str(args.conf),
            "--iou", str(args.iou),
            "--overwrite",
            "--write-empty",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        for cid in args.class_id or []:
            cmd += ["--class-id", str(cid)]
        for cam in cameras:
            cmd += ["--camera", cam]
        print(exec_container(" ".join(shlex.quote(x) for x in cmd)), end="")
        exec_container(f"rm -rf {container_root!r}")
        print("direct_trainer_review_root=true")
        return 0
    images = candidate_images(review_root, cameras, args.image)
    if args.limit:
        images = images[:args.limit]
    if not images:
        print("no unlabeled images to suggest")
        return 0
    if not SCRIPT.is_file():
        raise SystemExit(f"missing helper script: {SCRIPT}")

    run_id = f"frigate_draft_{int(time.time())}"
    container_root = f"/tmp/{run_id}"
    exec_container(f"rm -rf {container_root!r} && mkdir -p {container_root!r}")

    files: list[tuple[Path, str]] = []
    for img in images:
        rel = img.relative_to(review_root)
        files.append((img, f"review/{rel.as_posix()}"))
        label = img.with_suffix(".txt")
        if label.exists():
            files.append((label, f"review/{label.relative_to(review_root).as_posix()}"))
    extra = [(SCRIPT.read_bytes(), "generate_draft_labels.py")]
    put_archive(container_root, files, extra)

    model = args.model or default_model()
    cmd = [
        "python3", f"{container_root}/generate_draft_labels.py",
        "--review-root", f"{container_root}/review",
        "--model", model,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--overwrite",
        "--write-empty",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    for cid in args.class_id or []:
        cmd += ["--class-id", str(cid)]
    for cam in cameras:
        cmd += ["--camera", cam]
    quoted = " ".join(shlex.quote(x) for x in cmd)
    print(exec_container(quoted), end="")

    if args.dry_run:
        return 0
    raw = get_archive(f"{container_root}/review")
    wrote = 0
    removed_legacy = 0
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".suggest.txt"):
                continue
            # Docker archive names may be review/FrontDoor/... or just FrontDoor/...
            parts = Path(m.name).parts
            try:
                idx = parts.index("review")
                rel = Path(*parts[idx + 1:])
            except ValueError:
                rel = Path(*parts)
            if len(rel.parts) < 3 or rel.parts[0] not in ALLOWED_CAMERAS or rel.parts[1] != "images":
                continue
            target = review_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(m)
            data = extracted.read() if extracted else b""
            if data.strip():
                target.write_bytes(data)
                wrote += 1
    # If the model produced no boxes for an image, no .suggest.txt is copied
    # back. In that case remove legacy Frigate detector sidecars for the images
    # we processed, otherwise the UI would keep showing those stale boxes.
    wrote_targets = set()
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".suggest.txt"):
                continue
            parts = Path(m.name).parts
            try:
                idx = parts.index("review")
                rel = Path(*parts[idx + 1:])
            except ValueError:
                rel = Path(*parts)
            if len(rel.parts) >= 3 and rel.parts[0] in ALLOWED_CAMERAS and rel.parts[1] == "images":
                extracted = tf.extractfile(m)
                if extracted and extracted.read().strip():
                    wrote_targets.add(review_root / rel)
    for img in images:
        suggest = img.with_suffix(".suggest.txt")
        if suggest in wrote_targets:
            continue
        if suggest.exists() and _is_blocked_legacy_suggestion(suggest):
            suggest.unlink()
            removed_legacy += 1
    print(f"copied_suggestions={wrote} removed_legacy_frigate_suggestions={removed_legacy}")
    exec_container(f"rm -rf {container_root!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
