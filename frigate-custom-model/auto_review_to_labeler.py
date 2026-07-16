#!/usr/bin/env python3
"""Import unreviewed Frigate review items into the labeler queue, generate
suggestions, and mark them as reviewed in Frigate.

This automates the manual flow of clicking Frigate+ on each review item in the
Frigate Review UI. For each unreviewed FrontDoor/Backyard review item:

  1. Fetch the full-resolution recording frame snapshot at data.thumb_time
  2. Write it to the labeler queue with .meta.json sidecar
  3. Optionally generate .suggest.txt via the trainer ONNX model
  4. Mark the Frigate review as reviewed via POST /api/reviews/viewed

Usage:
  # Dry run (default — shows what would happen)
  python3 auto_review_to_labeler.py

  # Execute (writes files + marks reviewed)
  python3 auto_review_to_labeler.py --execute

  # Limit to 5 items
  python3 auto_review_to_labeler.py --execute --limit 5

  # Skip suggestion generation (just import + mark reviewed)
  python3 auto_review_to_labeler.py --execute --no-suggestions

  # Only process one camera
  python3 auto_review_to_labeler.py --camera Backyard --execute
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import shlex
import socket
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CAMERAS = ("FrontDoor", "Backyard")
LABELS_PATH = Path("/opt/frigate/labels.txt")
DEFAULT_FRIGATE_URL = "http://192.168.0.40:5000"
DEFAULT_REVIEW_ROOT = "/mnt/user/media/frigate_custom_model/review"
DOCKER_SOCK = "/var/run/docker.sock"
TRAINER_CONTAINER = "frigate-model-trainer"
SUGGEST_SCRIPT = Path("/opt/frigate/custom-model/generate_draft_labels_via_trainer.py")

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
LABELER_CONTAINER = "frigate-labeler"


# ---------------------------------------------------------------------------
# Docker helpers (for suggestion generation via trainer container + file I/O via labeler container)
# ---------------------------------------------------------------------------

class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(DOCKER_SOCK)


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
    raw = docker("POST", f"/containers/{TRAINER_CONTAINER}/exec", {
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


def put_archive(dest_path: str, files: list[tuple[Path, str]], extra: list[tuple[bytes, str]] = None,
                 container: str = TRAINER_CONTAINER) -> None:
    extra = extra or []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for src, arcname in files:
            ti = tf.gettarinfo(str(src), arcname=arcname)
            with src.open("rb") as f:
                tf.addfile(ti, f)
        for data, arcname in extra:
            ti = tarfile.TarInfo(arcname)
            ti.size = len(data)
            ti.mtime = time.time()
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    docker("PUT", f"/containers/{container}/archive?path={urllib.parse.quote(dest_path)}",
           buf.getvalue(), {"Content-Type": "application/x-tar"})


def put_archive_bytes(dest_path: str, files: list[tuple[bytes, str]],
                      container: str = LABELER_CONTAINER) -> None:
    """Write raw bytes as files into a container via Docker archive API."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for data, arcname in files:
            ti = tarfile.TarInfo(arcname)
            ti.size = len(data)
            ti.mtime = time.time()
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    docker("PUT", f"/containers/{container}/archive?path={urllib.parse.quote(dest_path)}",
           buf.getvalue(), {"Content-Type": "application/x-tar"})


def get_archive(src_path: str, container: str = TRAINER_CONTAINER) -> bytes:
    return docker("GET", f"/containers/{container}/archive?path={urllib.parse.quote(src_path)}")


def labeler_exec(cmd: str) -> str:
    """Execute a command inside the frigate-labeler container."""
    raw = docker("POST", f"/containers/{LABELER_CONTAINER}/exec", {
        "Cmd": ["sh", "-c", cmd],
        "AttachStdout": True, "AttachStderr": True, "Tty": False,
    })
    exec_id = json.loads(raw)["Id"]
    out = docker("POST", f"/exec/{exec_id}/start", {"Detach": False, "Tty": False})
    info = json.loads(docker("GET", f"/exec/{exec_id}/json"))
    text = demux(out)
    if info.get("ExitCode"):
        raise RuntimeError(f"labeler cmd failed exit={info.get('ExitCode')}\n{text}")
    return text


# ---------------------------------------------------------------------------
# Frigate API helpers
# ---------------------------------------------------------------------------

def frigate_get(frigate_url: str, path: str) -> bytes:
    url = f"{frigate_url.rstrip('/')}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def frigate_get_json(frigate_url: str, path: str) -> dict | list:
    return json.loads(frigate_get(frigate_url, path))


def frigate_post_json(frigate_url: str, path: str, body: dict) -> dict:
    url = f"{frigate_url.rstrip('/')}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def mark_reviewed(frigate_url: str, review_ids: list[str]) -> dict:
    """Mark review items as reviewed in Frigate."""
    return frigate_post_json(frigate_url, "/api/reviews/viewed", {
        "ids": review_ids,
        "reviewed": True,
    })


def fetch_review_items(frigate_url: str, cameras: list[str]) -> list[dict]:
    """Fetch unreviewed review items for the given cameras."""
    all_reviews = frigate_get_json(frigate_url, "/api/review")
    if not isinstance(all_reviews, list):
        all_reviews = all_reviews.get("items", []) if isinstance(all_reviews, dict) else []
    camera_set = set(cameras)
    return [
        r for r in all_reviews
        if r.get("camera") in camera_set and not r.get("has_been_reviewed", False)
    ]


def fetch_recording_snapshot(frigate_url: str, camera: str, frame_time: str) -> bytes | None:
    """Fetch full-resolution recording frame snapshot."""
    url = f"/api/{urllib.parse.quote(camera, safe='')}/recordings/{urllib.parse.quote(frame_time, safe='')}/snapshot.png"
    try:
        return frigate_get(frigate_url, url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_event_snapshot(frigate_url: str, event_id: str) -> bytes | None:
    """Fetch event snapshot as fallback."""
    url = f"/api/{urllib.parse.quote(event_id, safe='')}/snapshot.jpg?crop=0&bbox=0&timestamp=0"
    try:
        return frigate_get(frigate_url, url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


# ---------------------------------------------------------------------------
# Image suffix detection
# ---------------------------------------------------------------------------

def image_suffix(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("unknown image format")


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

def existing_queue_ids(review_root: str) -> set[str]:
    """Return set of review IDs already in the queue or _decisioned (via labeler container)."""
    try:
        out = labeler_exec(
            f"find {review_root}/ -maxdepth 4 -name 'frigate_review_*' -type f "
            f"2>/dev/null"
        )
    except Exception:
        return set()
    ids = set()
    for line in out.strip().splitlines():
        name = Path(line.strip()).stem
        if name.startswith("frigate_review_"):
            review_id = name[len("frigate_review_"):]
            ids.add(review_id)
    return ids


def write_to_labeler(dest_dir: str, filename: str, data: bytes) -> None:
    """Write a file into the frigate-labeler container via Docker archive API."""
    put_archive_bytes(dest_dir, [(data, filename)])


# ---------------------------------------------------------------------------
# Suggestion generation via trainer container
# ---------------------------------------------------------------------------

LATEST_BLUE_MANIFEST = "/workspace/frigate_custom_model/models/blue/latest-blue-manifest.json"
DEFAULT_SEED_MODEL = "/workspace/frigate_custom_model/models/v0_package_seed.onnx"
GEN_SCRIPT = Path("/opt/frigate/custom-model/generate_draft_labels.py")


def get_model_path() -> str:
    """Get the latest blue model path from the trainer container."""
    snippet = (
        "import json, pathlib; "
        f"p=pathlib.Path({LATEST_BLUE_MANIFEST!r}); "
        f"print(json.loads(p.read_text()).get('onnx_trainer') if p.exists() else {DEFAULT_SEED_MODEL!r})"
    )
    try:
        import shlex
        model = exec_container("python3 -c " + shlex.quote(snippet)).strip().splitlines()[-1]
        return model or DEFAULT_SEED_MODEL
    except Exception:
        return DEFAULT_SEED_MODEL


def generate_suggestions(review_root: Path, images: list[Path], camera: str) -> dict:
    """Generate .suggest.txt sidecars for the given images via the trainer ONNX model."""
    if not GEN_SCRIPT.is_file():
        return {"ok": False, "error": f"missing helper script: {GEN_SCRIPT}"}
    if not images:
        return {"ok": True, "wrote": 0, "msg": "no images to suggest"}

    run_id = f"auto_review_{int(time.time())}"
    container_root = f"/tmp/{run_id}"
    exec_container(f"rm -rf {container_root!r} && mkdir -p {container_root!r}")

    # Copy images into trainer container
    files: list[tuple[Path, str]] = []
    for img in images:
        rel = img.relative_to(review_root)
        files.append((img, f"review/{rel.as_posix()}"))
    extra = [(GEN_SCRIPT.read_bytes(), "generate_draft_labels.py")]
    put_archive(container_root, files, extra)

    model = get_model_path()
    import shlex
    cmd = [
        "python3", f"{container_root}/generate_draft_labels.py",
        "--review-root", f"{container_root}/review",
        "--model", model,
        "--conf", "0.25",
        "--iou", "0.45",
        "--overwrite",
        "--write-empty",
        "--camera", camera,
    ]
    quoted = " ".join(shlex.quote(x) for x in cmd)
    output = exec_container(quoted)
    print(f"  [suggest] {output.strip()}")

    # Copy .suggest.txt files back
    raw = get_archive(f"{container_root}/review")
    wrote = 0
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
            if len(rel.parts) < 3 or rel.parts[0] not in ALLOWED_CAMERAS or rel.parts[1] != "images":
                continue
            target = review_root / rel
            extracted = tf.extractfile(m)
            data = extracted.read() if extracted else b""
            if data.strip():
                atomic_write_bytes(target, data)
                wrote += 1

    exec_container(f"rm -rf {container_root!r}")
    return {"ok": True, "wrote": wrote}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frigate-url", default=DEFAULT_FRIGATE_URL)
    ap.add_argument("--review-root", default=DEFAULT_REVIEW_ROOT)
    ap.add_argument("--camera", action="append", choices=ALLOWED_CAMERAS)
    ap.add_argument("--limit", type=int, default=0, help="Max items to process (0 = all)")
    ap.add_argument("--execute", action="store_true", help="Actually write files + mark reviewed (default: dry-run)")
    ap.add_argument("--no-suggestions", action="store_true", help="Skip ONNX suggestion generation")
    ap.add_argument("--mark-reviewed", action="store_true", default=True,
                    help="Mark Frigate reviews as reviewed after import (default: True)")
    ap.add_argument("--no-mark-reviewed", dest="mark_reviewed", action="store_false",
                    help="Don't mark Frigate reviews as reviewed")
    args = ap.parse_args()

    frigate_url = args.frigate_url
    review_root = Path(args.review_root)
    cameras = args.camera or list(ALLOWED_CAMERAS)
    dry_run = not args.execute

    if dry_run:
        print("=== DRY RUN (use --execute to actually process) ===")
    print(f"Frigate: {frigate_url}")
    print(f"Review root: {review_root}")
    print(f"Cameras: {', '.join(cameras)}")
    print()

    # 1. Fetch unreviewed items
    items = fetch_review_items(frigate_url, cameras)
    print(f"Unreviewed items: {len(items)}")
    if args.limit:
        items = items[:args.limit]
        print(f"Processing first {len(items)} (limit={args.limit})")
    if not items:
        print("Nothing to do.")
        return 0
    print()

    # 2. Check which are already in the queue
    existing = existing_queue_ids(str(review_root)) if review_root.is_dir() else set()

    imported = 0
    skipped = 0
    failed = 0
    imported_images: dict[str, list[Path]] = {cam: [] for cam in cameras}
    review_ids_to_mark: list[str] = []

    for item in items:
        review_id = item["id"]
        camera = item["camera"]
        data = item.get("data", {}) if isinstance(item.get("data"), dict) else {}
        thumb_time = data.get("thumb_time")
        detections = data.get("detections", [])
        objects = data.get("objects", [])

        if review_id in existing:
            print(f"  SKIP {review_id} ({camera}) — already in queue")
            skipped += 1
            continue

        # Try recording frame snapshot first
        image_bytes = None
        snapshot_source = None
        if thumb_time:
            image_bytes = fetch_recording_snapshot(frigate_url, camera, str(thumb_time))
            if image_bytes:
                snapshot_source = f"recording:{thumb_time}"

        # Fallback: event snapshot from first detection
        if image_bytes is None and detections:
            for det_id in detections:
                image_bytes = fetch_event_snapshot(frigate_url, det_id)
                if image_bytes:
                    snapshot_source = f"event:{det_id}"
                    break

        if image_bytes is None:
            print(f"  FAIL {review_id} ({camera}) — no snapshot available")
            failed += 1
            continue

        suffix = image_suffix(image_bytes)
        stem = f"frigate_review_{review_id}"
        img_dest = review_root / camera / "images" / f"{stem}{suffix}"
        meta_dest = img_dest.with_suffix(".meta.json")

        meta = {
            "source": "auto_review_to_labeler",
            "review_id": review_id,
            "camera": camera,
            "thumb_time": thumb_time,
            "detections": detections,
            "objects": objects,
            "snapshot_source": snapshot_source,
            "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if dry_run:
            print(f"  WOULD IMPORT {review_id} ({camera}) objects={objects} snapshot={snapshot_source} -> {img_dest.name}")
        else:
            dest_dir = str(review_root / camera / "images")
            write_to_labeler(dest_dir, f"{stem}{suffix}", image_bytes)
            write_to_labeler(dest_dir, f"{stem}.meta.json", json.dumps(meta, indent=2, sort_keys=True).encode() + b"\n")
            print(f"  IMPORTED {review_id} ({camera}) objects={objects} snapshot={snapshot_source} -> {img_dest.name}")

        imported += 1
        imported_images[camera].append(img_dest)
        review_ids_to_mark.append(review_id)

    print(f"\nSummary: imported={imported} skipped={skipped} failed={failed}")

    # 3. Generate suggestions
    if not dry_run and not args.no_suggestions and imported > 0:
        print("\n=== Generating suggestions ===")
        for cam, imgs in imported_images.items():
            if not imgs:
                continue
            # Pull images from labeler container into local temp for suggestion generation
            import tempfile
            tmpdir = Path(tempfile.mkdtemp(prefix="auto_review_"))
            local_images = []
            for img_path in imgs:
                rel = img_path.relative_to(review_root)
                local_dest = tmpdir / rel
                local_dest.parent.mkdir(parents=True, exist_ok=True)
                # Pull file from labeler container
                try:
                    raw = get_archive(str(img_path), container=LABELER_CONTAINER)
                    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                        for m in tf.getmembers():
                            if m.isfile():
                                extracted = tf.extractfile(m)
                                if extracted:
                                    local_dest.write_bytes(extracted.read())
                                    local_images.append(local_dest)
                                    break
                except Exception as e:
                    print(f"  [{cam}] WARN: could not pull {img_path.name}: {e}")
            if not local_images:
                print(f"  [{cam}] No images pulled for suggestion generation")
                continue
            print(f"  [{cam}] Generating suggestions for {len(local_images)} images...")
            result = generate_suggestions(tmpdir, local_images, cam)
            print(f"  [{cam}] Result: {result}")
            # Copy .suggest.txt files back to labeler container
            for li in local_images:
                suggest_file = li.with_suffix(".suggest.txt")
                if suggest_file.exists() and suggest_file.read_bytes().strip():
                    rel = suggest_file.relative_to(tmpdir)
                    dest_dir = str(review_root / rel.parent)
                    write_to_labeler(dest_dir, suggest_file.name, suggest_file.read_bytes())
            # Cleanup
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    # 4. Mark reviews as reviewed in Frigate
    if not dry_run and args.mark_reviewed and review_ids_to_mark:
        print(f"\n=== Marking {len(review_ids_to_mark)} reviews as reviewed ===")
        result = mark_reviewed(frigate_url, review_ids_to_mark)
        print(f"  Result: {result}")
    elif dry_run and review_ids_to_mark:
        print(f"\nWould mark {len(review_ids_to_mark)} reviews as reviewed in Frigate")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
