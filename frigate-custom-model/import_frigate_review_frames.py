#!/usr/bin/env python3
"""Bounded full-resolution Frigate Review frame importer.

This is intentionally *not* a recording scanner. It imports frames only for
explicit Review IDs and only from media files named by Review/Event metadata (or
by an explicit fixture JSON for testing). Default mode is dry-run; pass
``--execute`` to write images/sidecars.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ALLOWED_CAMERAS = {"FrontDoor", "Backyard"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv"}
IMPORTER_VERSION = "oc162-review-frame-importer-v1"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{3,160}$")


@dataclasses.dataclass(frozen=True)
class Source:
    review_id: str
    camera: str
    path: Path
    kind: str
    event_id: str | None = None
    frame_time: float | None = None
    box: list[float] | None = None
    label: str | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_json(base_url: str, api_path: str, timeout: float) -> Any:
    url = base_url.rstrip("/") + api_path
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": IMPORTER_VERSION})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-supplied local Frigate URL
        return json.loads(resp.read().decode("utf-8"))


def normalize_review_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("reviews"), list):
            return [x for x in payload["reviews"] if isinstance(x, dict)]
        if isinstance(payload.get("review"), dict):
            return [payload["review"]]
        return [payload]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise SystemExit("review JSON must be an object, {'reviews': [...]}, or a list")


def validate_review_id(review_id: str) -> None:
    if not SAFE_ID_RE.match(review_id):
        raise SystemExit(f"unsafe review id {review_id!r}; pass exact Frigate Review IDs only")
    if any(x in review_id for x in ("*", "?", "[", "]", "/", "\\")):
        raise SystemExit(f"unsafe review id {review_id!r}; broad scans/globs are refused")


def require_read_only_media_root(media_root: Path, allow_writable: bool) -> Path:
    root = media_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"media root not found or not a directory: {root}")
    probe = root / f".oc162-write-probe-{os.getpid()}"
    writable = False
    try:
        with probe.open("xb") as f:
            f.write(b"probe")
        writable = True
    except OSError:
        writable = False
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()
    if writable and not allow_writable:
        raise SystemExit(
            f"media root is writable by this process: {root}. Mount/pass explicit read-only Frigate media, "
            "or use --allow-writable-media-root only for disposable tests."
        )
    return root


def under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def candidate_strings(obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    keys = ("media_path", "recording_path", "recording", "clip_path", "path", "file", "filename")
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out.extend(x for x in v if isinstance(x, str))
    data = obj.get("data")
    if isinstance(data, dict):
        for k in keys + ("recordings", "recording_paths", "source_path"):
            v = data.get(k)
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, list):
                out.extend(x for x in v if isinstance(x, str))
    return out


def event_box(event: dict[str, Any]) -> tuple[list[float] | None, str | None]:
    label = event.get("label") if isinstance(event.get("label"), str) else None
    data = event.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("label"), str):
            label = data["label"]
        box = data.get("box") or data.get("region")
        if isinstance(box, list) and len(box) == 4:
            vals = [as_float(x) for x in box]
            if all(v is not None for v in vals):
                return [float(v) for v in vals if v is not None], label
        objects = data.get("objects")
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                box = item.get("box")
                if isinstance(box, list) and len(box) == 4:
                    vals = [as_float(x) for x in box]
                    if all(v is not None for v in vals):
                        item_label = item.get("label") if isinstance(item.get("label"), str) else label
                        return [float(v) for v in vals if v is not None], item_label
    return None, label


def resolve_media_path(raw: str, media_root: Path) -> Path | None:
    if not raw or "://" in raw:
        return None
    p = Path(raw)
    candidates = [p]
    if p.is_absolute():
        parts = p.parts
        for marker in ("media", "frigate"):
            if marker in parts:
                idx = parts.index(marker)
                candidates.append(media_root.joinpath(*parts[idx + 1 :]))
    else:
        candidates.append(media_root / p)
    for c in candidates:
        try:
            r = c.expanduser().resolve()
        except OSError:
            continue
        if under_root(r, media_root) and r.is_file():
            return r
    return None


def likely_event_clip_paths(event: dict[str, Any], camera: str, media_root: Path) -> list[Path]:
    event_id = str(event.get("id") or event.get("event_id") or "")
    if not event_id:
        return []
    names = [f"{camera}-{event_id}.mp4", f"{event_id}.mp4", f"{camera}-{event_id}.jpg", f"{event_id}.jpg"]
    dirs = [media_root / "clips", media_root / "exports", media_root]
    return [d / n for d in dirs for n in names]


def collect_sources(
    reviews: list[dict[str, Any]],
    review_ids: list[str],
    media_root: Path,
    events_by_id: dict[str, dict[str, Any]],
) -> list[Source]:
    wanted = set(review_ids)
    by_id = {str(r.get("id") or r.get("review_id") or ""): r for r in reviews}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise SystemExit(f"requested review ids not found in metadata: {', '.join(missing)}")
    sources: list[Source] = []
    for review_id in review_ids:
        review = by_id[review_id]
        camera = str(review.get("camera") or review.get("camera_name") or "")
        if camera not in ALLOWED_CAMERAS:
            raise SystemExit(f"review {review_id}: camera {camera!r} not in allowlist {sorted(ALLOWED_CAMERAS)}")
        frame_time = as_float(review.get("start_time")) or as_float(review.get("frame_time"))
        box, label = event_box(review)
        for raw in candidate_strings(review):
            p = resolve_media_path(raw, media_root)
            if p:
                sources.append(Source(review_id, camera, p, "review_metadata", frame_time=frame_time, box=box, label=label))
        data = review.get("data") if isinstance(review.get("data"), dict) else {}
        event_ids: list[str] = []
        for key in ("detections", "objects", "events", "event_ids"):
            val = data.get(key) if isinstance(data, dict) else None
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        event_ids.append(item)
                    elif isinstance(item, dict):
                        eid = item.get("id") or item.get("event_id")
                        if eid:
                            event_ids.append(str(eid))
        for event_id in dict.fromkeys(event_ids):
            event = events_by_id.get(event_id, {})
            ev_box, ev_label = event_box(event)
            ev_frame_time = as_float(event.get("start_time")) or frame_time
            for raw in candidate_strings(event):
                p = resolve_media_path(raw, media_root)
                if p:
                    sources.append(Source(review_id, camera, p, "event_metadata", event_id, ev_frame_time, ev_box or box, ev_label or label))
            for p in likely_event_clip_paths(event or {"id": event_id}, camera, media_root):
                try:
                    r = p.resolve()
                except OSError:
                    continue
                if under_root(r, media_root) and r.is_file():
                    sources.append(Source(review_id, camera, r, "event_clip_candidate", event_id, ev_frame_time, ev_box or box, ev_label or label))
        # Deduplicate while preserving order.
    unique: list[Source] = []
    seen: set[tuple[str, str]] = set()
    for s in sources:
        key = (s.review_id, str(s.path))
        if key not in seen:
            unique.append(s)
            seen.add(key)
    return unique


def run_limited(cmd: list[str], timeout: float, cpu_seconds: int, mem_mb: int) -> subprocess.CompletedProcess[str]:
    def limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, preexec_fn=limits)


def ffprobe_dimensions(path: Path, ffprobe: str, timeout: float) -> tuple[int | None, int | None]:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)]
    try:
        cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if cp.returncode != 0:
            return None, None
        streams = json.loads(cp.stdout).get("streams") or []
        if not streams:
            return None, None
        return int(streams[0].get("width")), int(streams[0].get("height"))
    except Exception:
        return None, None


def load_labels(labels_file: Path | None) -> dict[str, int]:
    if not labels_file or not labels_file.is_file():
        return {}
    out: dict[str, int] = {}
    for idx, line in enumerate(labels_file.read_text(encoding="utf-8").splitlines()):
        label = line.strip()
        if label and not label.startswith("#"):
            out[label] = idx
    return out


def write_suggest(path: Path, box: list[float], label: str | None, labels: dict[str, int], width: int | None, height: int | None) -> bool:
    if not width or not height or not box:
        return False
    if path.with_suffix(".txt").exists() and path.with_suffix(".txt").read_text(encoding="utf-8").strip():
        return False
    x1, y1, x2, y2 = box
    # Frigate boxes are usually pixel xyxy. If all values are <= 1, assume normalized xyxy.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)
    else:
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = abs(x2 - x1) / width
        bh = abs(y2 - y1) / height
    vals = [max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh)]
    class_id = labels.get(label or "", 0)
    path.with_suffix(".suggest.txt").write_text(f"{class_id} {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f} # source=frigate\n", encoding="utf-8")
    return True


def output_stem(source: Source, index: int) -> str:
    safe_review = re.sub(r"[^A-Za-z0-9_.:-]", "_", source.review_id)
    safe_event = re.sub(r"[^A-Za-z0-9_.:-]", "_", source.event_id or "noevent")
    return f"frigate_review_frame_{safe_review}_{safe_event}_{index:02d}"


def import_source(source: Source, index: int, args: argparse.Namespace, labels: dict[str, int]) -> dict[str, Any]:
    ext = source.path.suffix.lower()
    out_dir = Path(args.review_root).expanduser().resolve() / source.camera / "images"
    stem = output_stem(source, index)
    out_img = out_dir / f"{stem}.jpg"
    action = "extract_frame" if ext in VIDEO_EXTS else "copy_image"
    plan: dict[str, Any] = {
        "review_id": source.review_id,
        "event_id": source.event_id,
        "camera": source.camera,
        "source": str(source.path),
        "source_kind": source.kind,
        "action": action,
        "output": str(out_img),
        "dry_run": not args.execute,
    }
    if not args.execute:
        return plan
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_img.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output without --overwrite: {out_img}")
    if ext in IMAGE_EXTS:
        shutil.copy2(source.path, out_img)
    elif ext in VIDEO_EXTS:
        seek = max(0.0, float(args.seek_seconds))
        cmd = [
            args.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(args.ffmpeg_threads),
            "-ss",
            f"{seek:.3f}",
            "-i",
            str(source.path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(out_img),
        ]
        cp = run_limited(cmd, args.ffmpeg_timeout, args.ffmpeg_cpu_seconds, args.ffmpeg_mem_mb)
        plan["ffmpeg_returncode"] = cp.returncode
        if cp.returncode != 0:
            raise SystemExit(f"ffmpeg failed for {source.path}: {cp.stderr[-1000:]}")
    else:
        raise SystemExit(f"unsupported media extension for {source.path}; only images/videos are allowed")
    width, height = ffprobe_dimensions(out_img, args.ffprobe, min(args.ffmpeg_timeout, 10))
    suggest_written = False
    if args.write_suggestions and source.box:
        suggest_written = write_suggest(out_img, source.box, source.label, labels, width, height)
    meta = {
        "importer": IMPORTER_VERSION,
        "imported_at": utc_now(),
        "review_id": source.review_id,
        "event_id": source.event_id,
        "camera": source.camera,
        "source_path": str(source.path),
        "source_kind": source.kind,
        "source_sha256": sha256_file(source.path),
        "output_image": str(out_img),
        "output_sha256": sha256_file(out_img),
        "width": width,
        "height": height,
        "frame_time": source.frame_time,
        "ffmpeg_limits": {
            "timeout_seconds": args.ffmpeg_timeout,
            "threads": args.ffmpeg_threads,
            "cpu_seconds": args.ffmpeg_cpu_seconds,
            "mem_mb": args.ffmpeg_mem_mb,
        },
        "suggestion_written": suggest_written,
        "dry_run_first_required": True,
    }
    out_img.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan.update({"width": width, "height": height, "meta": str(out_img.with_suffix(".meta.json")), "suggestion_written": suggest_written})
    return plan


def selected_reviews(args: argparse.Namespace) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    if args.review_json:
        for path in args.review_json:
            reviews.extend(normalize_review_payload(load_json(Path(path))))
    if args.frigate_url:
        for rid in args.review_id:
            reviews.extend(normalize_review_payload(fetch_json(args.frigate_url, f"/api/review/{rid}", args.http_timeout)))
    return reviews


def load_events(args: argparse.Namespace, reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    if args.event_json:
        for path in args.event_json:
            payload = load_json(Path(path))
            items = payload if isinstance(payload, list) else payload.get("events", [payload]) if isinstance(payload, dict) else []
            for item in items:
                if isinstance(item, dict):
                    eid = item.get("id") or item.get("event_id")
                    if eid:
                        events[str(eid)] = item
    if args.frigate_url:
        event_ids: set[str] = set()
        for review in reviews:
            data = review.get("data") if isinstance(review.get("data"), dict) else {}
            for key in ("detections", "objects", "events", "event_ids"):
                val = data.get(key) if isinstance(data, dict) else None
                if isinstance(val, list):
                    for item in val:
                        eid = item if isinstance(item, str) else item.get("id") if isinstance(item, dict) else None
                        if eid:
                            event_ids.add(str(eid))
        for eid in event_ids - set(events):
            try:
                events[eid] = fetch_json(args.frigate_url, f"/api/events/{eid}", args.http_timeout)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
    return events


def run_import(args: argparse.Namespace) -> int:
    if not args.review_id:
        raise SystemExit("refusing broad scan: pass at least one --review-id")
    if len(args.review_id) > args.max_review_ids:
        raise SystemExit(f"too many review ids ({len(args.review_id)}); max is {args.max_review_ids}")
    for rid in args.review_id:
        validate_review_id(rid)
    if args.frames_per_review < 1 or args.frames_per_review > 5:
        raise SystemExit("--frames-per-review must be 1..5")
    media_root = require_read_only_media_root(Path(args.media_root), args.allow_writable_media_root)
    reviews = selected_reviews(args)
    if not reviews:
        raise SystemExit("no review metadata loaded; pass --review-json and/or --frigate-url")
    events = load_events(args, reviews)
    sources = collect_sources(reviews, args.review_id, media_root, events)
    if not sources:
        raise SystemExit("no bounded media sources found in requested review/event metadata")
    labels = load_labels(Path(args.labels) if args.labels else None)
    results: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for source in sources:
        n = counts.get(source.review_id, 0)
        if n >= args.frames_per_review:
            continue
        results.append(import_source(source, n + 1, args, labels))
        counts[source.review_id] = n + 1
    missing = [rid for rid in args.review_id if counts.get(rid, 0) < 1]
    if missing:
        raise SystemExit(f"no frame imported/planned for review ids: {', '.join(missing)}")
    print(json.dumps({"importer": IMPORTER_VERSION, "dry_run": not args.execute, "results": results}, indent=2, sort_keys=True))
    return 0


def self_test() -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(json.dumps({"self_test": "skipped", "reason": "ffmpeg not found"}))
        return 0
    with tempfile.TemporaryDirectory(prefix="oc162-review-import-") as td:
        base = Path(td)
        media = base / "media"
        review_root = base / "review"
        clips = media / "clips"
        clips.mkdir(parents=True)
        fixtures = []
        for camera, color in (("FrontDoor", "red"), ("Backyard", "blue")):
            rid = f"review-{camera}"
            eid = f"event-{camera}"
            clip = clips / f"{camera}-{eid}.mp4"
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d=1", "-frames:v", "8", "-y", str(clip)], check=True)
            fixtures.append({"id": rid, "camera": camera, "start_time": 1, "data": {"detections": [eid], "box": [32, 18, 160, 120]}})
        review_json = base / "reviews.json"
        review_json.write_text(json.dumps({"reviews": fixtures}), encoding="utf-8")
        os.chmod(media, 0o555)
        old_argv = sys.argv[:]
        try:
            common = ["--media-root", str(media), "--allow-writable-media-root", "--review-root", str(review_root), "--review-json", str(review_json), "--review-id", "review-FrontDoor", "--review-id", "review-Backyard", "--ffmpeg", ffmpeg, "--write-suggestions"]
            sys.argv = [old_argv[0], *common]
            run_import(parse_args(sys.argv[1:]))
            sys.argv = [old_argv[0], *common, "--execute"]
            run_import(parse_args(sys.argv[1:]))
        finally:
            sys.argv = old_argv
            os.chmod(media, 0o755)
        outputs = sorted(str(p.relative_to(review_root)) for p in review_root.rglob("*"))
        assert any("FrontDoor/images" in x and x.endswith(".jpg") for x in outputs)
        assert any("Backyard/images" in x and x.endswith(".jpg") for x in outputs)
        assert sum(1 for x in outputs if x.endswith(".meta.json")) == 2
        assert sum(1 for x in outputs if x.endswith(".suggest.txt")) == 2
        print(json.dumps({"self_test": "passed", "outputs": outputs}, indent=2))
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-id", action="append", default=[], help="Exact Frigate Review ID to import; repeat for one/few IDs")
    ap.add_argument("--review-json", action="append", help="Review metadata JSON fixture/export; repeatable")
    ap.add_argument("--event-json", action="append", help="Optional event metadata JSON fixture/export; repeatable")
    ap.add_argument("--frigate-url", help="Optional Frigate base URL for fetching /api/review/<id> and events")
    ap.add_argument("--http-timeout", type=float, default=10.0)
    ap.add_argument("--media-root", required=True, help="Explicit read-only Frigate media root")
    ap.add_argument("--review-root", default="/mnt/user/media/frigate_custom_model/review")
    ap.add_argument("--labels", default="/opt/frigate/labels.txt")
    ap.add_argument("--frames-per-review", type=int, default=1, help="Bounded output count per review, 1..5")
    ap.add_argument("--max-review-ids", type=int, default=20)
    ap.add_argument("--seek-seconds", type=float, default=0.0, help="Seek offset within source video")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--ffmpeg-timeout", type=float, default=20.0)
    ap.add_argument("--ffmpeg-threads", type=int, default=1)
    ap.add_argument("--ffmpeg-cpu-seconds", type=int, default=15)
    ap.add_argument("--ffmpeg-mem-mb", type=int, default=512)
    ap.add_argument("--write-suggestions", action="store_true", help="Write .suggest.txt from Frigate boxes when dimensions are known")
    ap.add_argument("--execute", action="store_true", help="Actually write outputs. Default is dry-run only.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-writable-media-root", action="store_true", help="Testing only; production imports should use read-only media")
    ap.add_argument("command", nargs="?", choices=["self-test"], help="Run built-in FrontDoor/Backyard proof fixture")
    args = ap.parse_args(argv)
    if args.command == "self-test":
        return args
    if args.ffmpeg_threads != 1:
        raise SystemExit("--ffmpeg-threads is fixed to 1 for bounded imports")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "self-test":
        return self_test()
    return run_import(args)


if __name__ == "__main__":
    raise SystemExit(main())
