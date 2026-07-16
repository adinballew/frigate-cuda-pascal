#!/usr/bin/env python3
"""Minimal local-only YOLO bounding-box labeler for the Frigate custom model.

Serves images from the review staging area and writes YOLO-format label
.txt files next to each image. No external deps; Python stdlib only.

Full runbook (shadow proxy, API, feedback, blue/green training): docs/frigate-custom-model/frigate-labeler.md

Data layout (unchanged from collect_frigate_candidates.py convention):

    review/<Camera>/images/<image>.jpg
    review/<Camera>/images/<image>.txt   # YOLO: "<class_id> cx cy w h" per box
                                          #   (normalized 0..1). Empty file = negative.

Class order comes from --labels file (default: docs/frigate-custom-model/labels.txt).
YOLO ids are positional; do NOT reorder that file after labeling starts.

Usage (host):
    python3 /opt/frigate/labeler/labeler.py \\
        --review-root /mnt/user/media/frigate_custom_model/review \\
        --labels     /opt/frigate/labels.txt \\
        --host 127.0.0.1 --port 8781

Then open http://127.0.0.1:8781/ (or http://<host-ip>:8781/ on LAN if you
bind 0.0.0.0). The server is intended to be started interactively and
killed with Ctrl-C when you're done labeling; it is not a long-lived daemon.

Cameras allowlist: FrontDoor, Backyard (Patio excluded per project policy).
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import posixpath
import re
import sys
import threading
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

ALLOWED_CAMERAS: tuple[str, ...] = ("FrontDoor", "Backyard")
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
# Suggestion sidecars in preference order. First usable one that exists wins.
# NEVER read by the training pipeline; only the labeler UI reads them
# and only when the human-owned `<image>.txt` is missing/empty.
SUGGEST_SUFFIXES: tuple[str, ...] = (".suggest.txt", ".draft.txt")
# Frigate event/review metadata boxes are not custom-model suggestions. They are
# often the bad person/dog/package boxes this UI is meant to correct, so loading
# them as "AI suggestion" creates a stale feedback loop. Model-generated drafts
# currently have no source marker or use non-frigate comments; reject legacy
# sidecars that explicitly came from Frigate.
BLOCKED_SUGGESTION_MARKERS: tuple[str, ...] = ("source=frigate",)
# Feedback audit log relative to review_root.
_FEEDBACK_SUBDIR = "_feedback"
_FEEDBACK_JSONL = "model_feedback.jsonl"

_WRITE_LOCK = threading.Lock()


# ---------- data helpers ---------------------------------------------------


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"labels file not found: {path}")
    labels: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            raise SystemExit(f"invalid label {name!r} in {path}")
        labels.append(name)
    if not labels:
        raise SystemExit(f"no labels parsed from {path}")
    return labels


def _suggestion_paths(image_path: Path) -> list[Path]:
    """All sidecar suggestion paths for one image, in preference order."""
    return [image_path.with_suffix(suffix) for suffix in SUGGEST_SUFFIXES]


def _first_existing_suggestion(image_path: Path) -> Path | None:
    for candidate in _suggestion_paths(image_path):
        if candidate.exists() and _is_usable_suggestion(candidate):
            return candidate
    return None


def _is_usable_suggestion(path: Path) -> bool:
    """Return True when a sidecar is safe to load as a model suggestion.

    We intentionally hide sidecars generated from Frigate's own detector boxes
    (`# source=frigate`). Those are importer hints, not learned model output,
    and showing them as suggestions was the recurring broken behavior.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not text.strip():
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in BLOCKED_SUGGESTION_MARKERS)


def list_images(review_root: Path) -> list[dict]:
    entries: list[dict] = []
    for camera in ALLOWED_CAMERAS:
        img_dir = review_root / camera / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = img.with_suffix(".txt")
            has_label_file = label_path.exists()
            n_boxes = 0
            if has_label_file:
                try:
                    n_boxes = sum(
                        1
                        for line in label_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                except OSError:
                    n_boxes = 0
            suggest_path = _first_existing_suggestion(img) if (not has_label_file or n_boxes == 0) else None
            n_suggestions = 0
            suggest_name = None
            if suggest_path is not None:
                suggest_name = suggest_path.name
                try:
                    n_suggestions = sum(
                        1
                        for line in suggest_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                except OSError:
                    n_suggestions = 0
            entries.append(
                {
                    "camera": camera,
                    "name": img.name,
                    "path": f"{camera}/{img.name}",
                    "has_label_file": has_label_file,
                    "n_boxes": n_boxes,
                    "has_suggestions": n_suggestions > 0,
                    "n_suggestions": n_suggestions,
                    "suggest_name": suggest_name,
                }
            )
    return entries


def read_label_file(review_root: Path, camera: str, name: str) -> dict:
    """Return the current label state for one image.

    Contract:
      * If the human-owned `<image>.txt` exists and has any non-blank line,
        return that text with source='label'. AI suggestions are ignored so
        saved human work is never silently overwritten with model output.
      * Otherwise (missing or empty `.txt`), if a sidecar suggestion file
        exists in SUGGEST_SUFFIXES preference order, return its text with
        source='suggest' and suggest_name pointing at the loaded file.
      * Otherwise, return empty text with source='label'.

    Suggestion lines may include an optional 6th confidence column; the UI
    parses only the first 5 whitespace-separated tokens as YOLO.
    """
    label_path = _resolve_label_path(review_root, camera, name)
    label_text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
    if any(line.strip() for line in label_text.splitlines()):
        return {"text": label_text, "source": "label", "suggest_name": None}
    image_path = label_path.parent / name
    suggest_path = _first_existing_suggestion(image_path)
    if suggest_path is not None:
        try:
            return {
                "text": suggest_path.read_text(encoding="utf-8"),
                "source": "suggest",
                "suggest_name": suggest_path.name,
            }
        except OSError:
            pass
    return {"text": label_text, "source": "label", "suggest_name": None}


# Default archive sub-directory under review_root for decisioned pairs.
_DECISIONED_SUBDIR = "_decisioned"


def append_feedback(
    review_root: Path,
    camera: str,
    image_name: str,
    suggest_name: str | None,
    action: str,
    n_boxes: int,
) -> Path:
    """Append one feedback record to <review_root>/_feedback/model_feedback.jsonl.

    Parameters
    ----------
    review_root:
        The root review directory.
    camera:
        Camera name (must be in ALLOWED_CAMERAS).
    image_name:
        Image filename (e.g. ``foo.jpg``).
    suggest_name:
        Sidecar filename that carried the AI suggestion (may be None).
    action:
        One of ``correct`` | ``incorrect`` | ``corrected`` | ``negative``.
        * ``correct``   — human accepted the suggestion boxes as-is.
        * ``incorrect`` — human rejected the suggestion; boxes were cleared
                          or no boxes remain after correction.
        * ``corrected`` — human modified the suggestion boxes before submit.
        * ``negative``  — suggestion existed but the human decided the image
                          is a negative (no objects).
    n_boxes:
        Number of boxes in the *final* human label (0 for negative).

    Returns the path to the JSONL file that was appended.
    """
    if camera not in ALLOWED_CAMERAS:
        raise ValueError(f"camera not allowed: {camera}")
    if action not in ("correct", "incorrect", "corrected", "negative"):
        raise ValueError(f"invalid feedback action: {action!r}")
    record: dict = {
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "camera": camera,
        "image": image_name,
        "suggest_file": suggest_name,
        "action": action,
        "n_boxes": n_boxes,
    }
    feedback_dir = review_root / _FEEDBACK_SUBDIR
    feedback_path = feedback_dir / _FEEDBACK_JSONL
    with _WRITE_LOCK:
        feedback_dir.mkdir(parents=True, exist_ok=True)
        with feedback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    return feedback_path


def submit_image(
    review_root: Path,
    camera: str,
    name: str,
    boxes: list[dict],
    n_classes: int,
) -> dict:
    """Finalize one image after labeling is complete.

    Steps (atomic-ish via lock):
      1. Write the label file (same logic as write_label_file, including backup
         and sidecar cleanup).
      2. Remove any remaining suggestion sidecars for this image.
      3. Move the image and its .txt label into the archive tree:
             <review_root>/_decisioned/<Camera>/images/<name>
             <review_root>/_decisioned/<Camera>/images/<stem>.txt
         If files already exist at the destination they are overwritten so a
         retry is safe.

    Returns a dict with: ok, image_dest, label_dest, n_boxes.
    Never deletes data — the image and label are *moved*, not removed.
    """
    # 1. Save label (this also removes sidecars via write_label_file).
    label_path = write_label_file(review_root, camera, name, boxes, n_classes)
    img_path = resolve_image_path(review_root, camera, name)

    # 2. Destination under _decisioned, preserving camera/images layout.
    dest_dir = review_root / _DECISIONED_SUBDIR / camera / "images"
    with _WRITE_LOCK:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_img = dest_dir / img_path.name
        dest_label = dest_dir / label_path.name
        # Move image then label.  Use rename; if cross-device, fall back to
        # copy+unlink so we never lose data even on separate filesystems.
        _safe_move(img_path, dest_img)
        _safe_move(label_path, dest_label)

    return {
        "ok": True,
        "image_dest": str(dest_img),
        "label_dest": str(dest_label),
        "n_boxes": len(boxes),
    }


def _safe_move(src: Path, dst: Path) -> None:
    """Move *src* to *dst*, falling back to copy+unlink on cross-device."""
    try:
        src.rename(dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)
        src.unlink()


def write_label_file(
    review_root: Path,
    camera: str,
    name: str,
    boxes: list[dict],
    n_classes: int,
) -> Path:
    label_path = _resolve_label_path(review_root, camera, name)
    lines: list[str] = []
    for box in boxes:
        cid = int(box["class_id"])
        if cid < 0 or cid >= n_classes:
            raise ValueError(f"class_id {cid} out of range 0..{n_classes - 1}")
        cx = float(box["cx"])
        cy = float(box["cy"])
        w = float(box["w"])
        h = float(box["h"])
        for value, tag in ((cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{tag}={value} outside 0..1")
        if w <= 0.0 or h <= 0.0:
            raise ValueError("width and height must be > 0")
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with _WRITE_LOCK:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        # Backup any prior non-empty label once per save for undo/audit.
        if label_path.exists() and label_path.stat().st_size > 0:
            bak_dir = label_path.parent / ".backup"
            bak_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            (bak_dir / f"{label_path.name}.{stamp}.bak").write_text(
                label_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        payload = "\n".join(lines)
        if payload:
            payload += "\n"
        label_path.write_text(payload, encoding="utf-8")
        # Consume-once: remove all suggestion sidecars for this image so the
        # UI stops showing them and the same suggestion never leaks back in.
        for suffix in SUGGEST_SUFFIXES:
            candidate = label_path.with_suffix(suffix)
            if candidate.exists():
                try:
                    candidate.unlink()
                except OSError:
                    pass
    return label_path


def _resolve_label_path(review_root: Path, camera: str, name: str) -> Path:
    if camera not in ALLOWED_CAMERAS:
        raise ValueError(f"camera not allowed: {camera}")
    safe_name = posixpath.basename(name)
    if safe_name != name or safe_name in ("", ".", ".."):
        raise ValueError(f"bad image name: {name!r}")
    img_path = review_root / camera / "images" / safe_name
    if img_path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"unexpected extension: {safe_name!r}")
    return img_path.with_suffix(".txt")


def resolve_image_path(review_root: Path, camera: str, name: str) -> Path:
    if camera not in ALLOWED_CAMERAS:
        raise ValueError(f"camera not allowed: {camera}")
    safe_name = posixpath.basename(name)
    if safe_name != name or safe_name in ("", ".", ".."):
        raise ValueError(f"bad image name: {name!r}")
    img_path = (review_root / camera / "images" / safe_name).resolve()
    root_resolved = review_root.resolve()
    if root_resolved not in img_path.parents:
        raise ValueError("image path escapes review root")
    if not img_path.is_file():
        raise FileNotFoundError(str(img_path))
    if img_path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"unexpected extension: {safe_name!r}")
    return img_path


# ---------- HTTP handler --------------------------------------------------


class LabelerHandler(BaseHTTPRequestHandler):
    server_version = "FrigateLabeler/0.1"

    # Wired in main() via subclass factory.
    review_root: Path
    labels: list[str]

    def log_message(self, fmt: str, *args) -> None:  # quieter than default
        sys.stderr.write(
            "[%s] %s\n" % (self.log_date_time_string(), fmt % args)
        )

    # ---- GET ----

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            return self._send_html(_INDEX_HTML)
        if route == "/api/labels":
            return self._send_json({"labels": self.labels})
        if route == "/api/images":
            return self._send_json({"images": list_images(self.review_root)})
        if route == "/api/label":
            qs = urllib.parse.parse_qs(parsed.query)
            camera = (qs.get("camera") or [""])[0]
            name = (qs.get("name") or [""])[0]
            try:
                info = read_label_file(self.review_root, camera, name)
            except Exception as exc:
                return self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return self._send_json({
                "camera": camera,
                "name": name,
                "text": info["text"],
                "source": info["source"],
                "suggest_name": info["suggest_name"],
            })
        if route == "/image":
            qs = urllib.parse.parse_qs(parsed.query)
            camera = (qs.get("camera") or [""])[0]
            name = (qs.get("name") or [""])[0]
            try:
                img_path = resolve_image_path(self.review_root, camera, name)
            except FileNotFoundError:
                return self._send_error(HTTPStatus.NOT_FOUND, "image not found")
            except Exception as exc:
                return self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return self._send_file(img_path)
        return self._send_error(HTTPStatus.NOT_FOUND, "unknown route")

    # ---- POST ----

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/submit":
            return self._handle_submit()
        if parsed.path == "/api/feedback":
            return self._handle_feedback()
        if parsed.path != "/api/label":
            return self._send_error(HTTPStatus.NOT_FOUND, "unknown route")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            return self._send_error(HTTPStatus.BAD_REQUEST, "bad content length")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, f"bad json: {exc}")
        camera = payload.get("camera") or ""
        name = payload.get("name") or ""
        boxes = payload.get("boxes")
        if not isinstance(boxes, list):
            return self._send_error(HTTPStatus.BAD_REQUEST, "boxes must be a list")
        try:
            path = write_label_file(
                self.review_root, camera, name, boxes, len(self.labels)
            )
        except (ValueError, KeyError) as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        return self._send_json(
            {"ok": True, "path": str(path), "n_boxes": len(boxes)}
        )

    def _handle_feedback(self) -> None:
        """POST /api/feedback — record model-suggestion feedback audit entry.

        Body JSON: {camera, name, suggest_name, action, n_boxes}
          action: correct | incorrect | corrected | negative
        Does NOT submit/archive the image.  Caller is responsible for
        subsequent save/submit calls as needed.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            return self._send_error(HTTPStatus.BAD_REQUEST, "bad content length")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, f"bad json: {exc}")
        camera = payload.get("camera") or ""
        name = payload.get("name") or ""
        suggest_name = payload.get("suggest_name") or None
        action = payload.get("action") or ""
        try:
            n_boxes = int(payload.get("n_boxes", 0))
        except (TypeError, ValueError):
            n_boxes = 0
        try:
            feedback_path = append_feedback(
                self.review_root, camera, name, suggest_name, action, n_boxes
            )
        except ValueError as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        return self._send_json(
            {"ok": True, "feedback_path": str(feedback_path), "action": action}
        )

    def _handle_submit(self) -> None:
        """POST /api/submit — finalize an image: save labels, archive pair."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            return self._send_error(HTTPStatus.BAD_REQUEST, "bad content length")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, f"bad json: {exc}")
        camera = payload.get("camera") or ""
        name = payload.get("name") or ""
        boxes = payload.get("boxes")
        if not isinstance(boxes, list):
            return self._send_error(HTTPStatus.BAD_REQUEST, "boxes must be a list")
        try:
            result = submit_image(
                self.review_root, camera, name, boxes, len(self.labels)
            )
        except FileNotFoundError as exc:
            return self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValueError, KeyError) as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        return self._send_json(result)

    # ---- helpers ----

    def _send_json(self, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        ctype, _ = mimetypes.guess_type(path.name)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------- UI -----------------------------------------------------------


_INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Frigate Labeler</title>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 system-ui,sans-serif; background:#111; color:#eee; height:100vh; display:flex; flex-direction:column; }
  header { padding:8px 12px; background:#1c1c1c; border-bottom:1px solid #333; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:14px; margin:0; font-weight:600; }
  header .spacer { flex:1; }
  main { flex:1; display:flex; min-height:0; }
  #sidebar { width:280px; border-right:1px solid #333; overflow:auto; background:#181818; }
  #sidebar ul { list-style:none; margin:0; padding:0; }
  #sidebar li { padding:6px 10px; border-bottom:1px solid #222; cursor:pointer; display:flex; justify-content:space-between; gap:6px; font-size:12px; }
  #sidebar li:hover { background:#242424; }
  #sidebar li.active { background:#2b3a55; }
  #sidebar li .badge { color:#8fd; font-size:11px; }
  #sidebar li.empty .badge { color:#888; }
  #stage { flex:1; display:flex; flex-direction:column; min-width:0; }
  #canvasWrap { flex:1; position:relative; overflow:auto; background:#0a0a0a; display:flex; align-items:center; justify-content:center; }
  #canvas { display:block; cursor:crosshair; background:#000; max-width:100%; max-height:100%; }
  #tools { padding:6px 10px; background:#1c1c1c; border-top:1px solid #333; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  select, button { background:#222; color:#eee; border:1px solid #444; padding:4px 8px; border-radius:3px; font:inherit; }
  button:hover { background:#2b2b2b; }
  button.primary { background:#2b7a3b; border-color:#3a9950; }
  button.primary:hover { background:#348a44; }
  button.done { background:#1a4a7a; border-color:#2a6aaa; }
  button.done:hover { background:#235890; }
  button.danger { background:#7a2b2b; border-color:#993a3a; }
  button.accept { background:#1a6e3a; border-color:#28a052; }
  button.accept:hover { background:#22884a; }
  button.reject { background:#7a4a10; border-color:#b06018; }
  button.reject:hover { background:#8f5718; }
  #feedbackBar { display:none; padding:5px 10px; background:#252010; border-top:1px solid #7a6010; font-size:12px; align-items:center; gap:8px; flex-wrap:wrap; }
  #feedbackBar.visible { display:flex; }
  #feedbackBar .fb-label { color:#ffd070; font-weight:600; }
  #feedbackBar .fb-desc { color:#bba060; flex:1; min-width:120px; }
  #boxList { border-top:1px solid #333; max-height:160px; overflow:auto; background:#181818; padding:4px 8px; font-size:12px; }
  #boxList table { width:100%; border-collapse:collapse; }
  #boxList th, #boxList td { padding:2px 6px; border-bottom:1px solid #222; text-align:left; }
  #status { font-size:12px; color:#9c9; margin-left:auto; }
  #status.err { color:#f88; }
  .kbd { font-family: ui-monospace, monospace; background:#333; padding:1px 5px; border-radius:3px; font-size:11px; }
</style>
</head>
<body>
<header>
  <h1>Frigate Labeler</h1>
  <span style=\"color:#888; font-size:12px;\">FrontDoor + Backyard only</span>
  <span class=\"spacer\"></span>
  <span style=\"color:#888; font-size:12px;\">
    <span class=\"kbd\">drag</span> box &nbsp;
    <span class=\"kbd\">S</span> save &nbsp;
    <span class=\"kbd\">D</span> submit/done &nbsp;
    <span class=\"kbd\">N</span> next &nbsp;
    <span class=\"kbd\">P</span> prev &nbsp;
    <span class=\"kbd\">Del</span> selected &nbsp;
    <span class=\"kbd\">Esc</span> clear sel
  </span>
</header>
<main>
  <aside id=\"sidebar\"><ul id=\"imgList\"></ul></aside>
  <section id=\"stage\">
    <div id=\"canvasWrap\"><canvas id=\"canvas\" width=\"640\" height=\"480\"></canvas></div>
    <div id=\"feedbackBar\">
      <span class=\"fb-label\">&#x1F916; AI suggestion loaded</span>
      <span class=\"fb-desc\" id=\"fbSuggestDesc\"></span>
      <button id=\"fbCorrectBtn\" class=\"accept\" title=\"Model suggestion correct &#x2014; record feedback &amp; archive\">&#9989; Model Correct (accept as-is)</button>
      <button id=\"fbWrongBtn\" class=\"reject\" title=\"Model wrong &#x2014; clears boxes for re-labeling\">&#10060; Model Wrong (clear &amp; re-label)</button>
    </div>
    <div id=\"tools\">
      <label>Camera:
        <select id=\"cameraFilter\">
          <option value=\"\">All</option>
          <option value=\"FrontDoor\">FrontDoor</option>
          <option value=\"Backyard\">Backyard</option>
        </select>
      </label>
      <label>Class:
        <select id=\"classSelect\"></select>
      </label>
      <button id=\"saveBtn\" class=\"primary\">Save (S)</button>
      <button id=\"submitBtn\" class=\"done\" title=\"Save labels, remove suggestions, move image+label to _decisioned archive (D)\">&#10003; Submit / Done (D)</button>
      <button id=\"emptyBtn\">Mark negative (empty)</button>
      <button id=\"clearBtn\" class=\"danger\">Clear boxes</button>
      <button id=\"prevBtn\">Prev</button>
      <button id=\"nextBtn\">Next</button>
      <span id=\"status\">ready</span>
    </div>
    <div id=\"boxList\"><table><thead><tr><th>#</th><th>class</th><th>cx</th><th>cy</th><th>w</th><th>h</th><th></th></tr></thead><tbody id=\"boxRows\"></tbody></table></div>
  </section>
</main>
<script>
(() => {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const classSelect = document.getElementById('classSelect');
  const cameraFilter = document.getElementById('cameraFilter');
  const imgListEl = document.getElementById('imgList');
  const boxRowsEl = document.getElementById('boxRows');
  const statusEl = document.getElementById('status');
  let labels = [];
  let images = [];
  let currentIdx = -1;
  let img = null;
  let boxes = []; // {class_id, cx, cy, w, h} normalized
  let selectedBox = -1;
  let dragStart = null;
  let boxesFromSuggest = false; // true when current boxes came from AI sidecar
  let feedbackState = null; // null | {suggestName, originalBoxes, decided}
  let feedbackRejected = false; // true after 'Model Wrong' clicked

  const colors = ['#ff5555','#55ff88','#5599ff','#ffcc55','#cc66ff','#66eecc','#ff88cc','#88ff55','#ffaa66','#88ccff','#cc9966','#aaff99','#ff9955'];

  function visibleIndexes() {
    const cam = cameraFilter.value;
    const out = [];
    images.forEach((entry, idx) => { if (!cam || entry.camera === cam) out.push(idx); });
    return out;
  }

  function currentVisiblePosition() {
    return visibleIndexes().indexOf(currentIdx);
  }

  function selectNext(delta) {
    const vis = visibleIndexes();
    if (!vis.length) { setStatus('no images for selected camera', true); return; }
    let pos = currentVisiblePosition();
    if (pos < 0) pos = delta > 0 ? -1 : vis.length;
    const next = Math.max(0, Math.min(vis.length - 1, pos + delta));
    selectImage(vis[next]);
  }

  function setStatus(msg, isErr=false) {
    statusEl.textContent = msg;
    statusEl.classList.toggle('err', !!isErr);
  }

  async function boot() {
    try {
      const [lRes, iRes] = await Promise.all([
        fetch('/api/labels').then(r => r.json()),
        fetch('/api/images').then(r => r.json())
      ]);
      labels = lRes.labels;
      images = iRes.images;
      renderClassSelect();
      renderImgList();
      const vis = visibleIndexes();
      if (vis.length) selectImage(vis[0]); else setStatus('no images in review dirs', true);
    } catch (e) { setStatus('boot failed: '+e, true); }
  }

  function renderClassSelect() {
    classSelect.innerHTML = '';
    labels.forEach((name, i) => {
      const opt = document.createElement('option');
      opt.value = String(i); opt.textContent = i+': '+name;
      classSelect.appendChild(opt);
    });
  }

  function renderImgList() {
    imgListEl.innerHTML = '';
    const vis = visibleIndexes();
    vis.forEach((idx) => {
      const entry = images[idx];
      const li = document.createElement('li');
      li.textContent = entry.path;
      const badge = document.createElement('span');
      badge.className = 'badge';
      if (!entry.has_label_file) { badge.textContent = 'new'; li.classList.add('empty'); }
      else if (entry.n_boxes === 0) { badge.textContent = 'neg'; li.classList.add('empty'); }
      else { badge.textContent = String(entry.n_boxes); }
      if (entry.has_suggestions && entry.n_boxes === 0) {
        badge.textContent = 'suggest '+entry.n_suggestions;
        li.classList.remove('empty');
      }
      li.appendChild(badge);
      li.addEventListener('click', () => selectImage(idx));
      if (idx === currentIdx) li.classList.add('active');
      imgListEl.appendChild(li);
    });
    if (!vis.length) {
      const li = document.createElement('li');
      li.textContent = 'No images for selected camera';
      li.classList.add('empty');
      imgListEl.appendChild(li);
    }
  }

  async function selectImage(idx) {
    currentIdx = idx;
    const entry = images[idx];
    if (!entry) return;
    renderImgList();
    setStatus('loading '+entry.path);
    try {
      const [labelRes, image] = await Promise.all([
        fetch('/api/label?camera='+encodeURIComponent(entry.camera)+'&name='+encodeURIComponent(entry.name)).then(r => r.json()),
        loadImage('/image?camera='+encodeURIComponent(entry.camera)+'&name='+encodeURIComponent(entry.name))
      ]);
      img = image;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      fitCanvasDisplay();
      boxes = parseYolo(labelRes.text || '');
      boxesFromSuggest = (labelRes.source === 'suggest');
      feedbackRejected = false;
      feedbackState = boxesFromSuggest
        ? { suggestName: labelRes.suggest_name || '', originalBoxes: JSON.parse(JSON.stringify(boxes)), decided: false }
        : null;
      selectedBox = -1;
      redraw();
      renderBoxList();
      updateFeedbackBar();
      const srcNote = boxesFromSuggest ? (' AI suggestion (' + (labelRes.suggest_name || '') + ') — use Model Correct/Wrong buttons') : '';
      setStatus(entry.path+' '+img.naturalWidth+'x'+img.naturalHeight+' shown '+canvas.style.width+'×'+canvas.style.height + srcNote, boxesFromSuggest && !feedbackRejected);
    } catch (e) { setStatus('load failed: '+e, true); }
  }

  function loadImage(url) {
    return new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => rej(new Error('img load failed'));
      im.src = url;
    });
  }

  function parseYolo(text) {
    const out = [];
    text.split(/\\n/).forEach(line => {
      const s = line.trim(); if (!s) return;
      const parts = s.split(/\\s+/);
      if (parts.length < 5) return;
      let confidence = null;
      for (const tok of parts.slice(5)) {
        const m = tok.match(/^(?:#)?(?:conf|confidence|score)=([0-9.]+)$/i);
        if (m) {
          const v = parseFloat(m[1]);
          if (!Number.isNaN(v)) confidence = v;
        }
      }
      out.push({
        class_id: parseInt(parts[0], 10) || 0,
        cx: parseFloat(parts[1]), cy: parseFloat(parts[2]),
        w: parseFloat(parts[3]), h: parseFloat(parts[4]),
        confidence: confidence
      });
    });
    return out;
  }

  function fitCanvasDisplay() {
    // Keep the canvas backing store at true image resolution so YOLO math stays
    // exact, but upscale low-res camera feeds so boxes are easier to draw.
    const wrap = document.getElementById('canvasWrap');
    const pad = 24;
    const maxW = Math.max(320, wrap.clientWidth - pad);
    const maxH = Math.max(240, wrap.clientHeight - pad);
    const minDisplayW = 960;
    const minDisplayH = 540;
    const sx = Math.min(maxW / canvas.width, Math.max(1, minDisplayW / canvas.width));
    const sy = Math.min(maxH / canvas.height, Math.max(1, minDisplayH / canvas.height));
    const scale = Math.max(1, Math.min(sx, sy));
    canvas.style.width = Math.round(canvas.width * scale) + 'px';
    canvas.style.height = Math.round(canvas.height * scale) + 'px';
  }

  window.addEventListener('resize', () => { if (img) fitCanvasDisplay(); });

  function redraw() {
    if (!img) return;
    ctx.drawImage(img, 0, 0);
    boxes.forEach((b, i) => {
      const x = (b.cx - b.w/2) * canvas.width;
      const y = (b.cy - b.h/2) * canvas.height;
      const w = b.w * canvas.width;
      const h = b.h * canvas.height;
      ctx.lineWidth = i === selectedBox ? 3 : 2;
      ctx.strokeStyle = colors[b.class_id % colors.length];
      ctx.strokeRect(x, y, w, h);
      const confLabel = (b.confidence !== null && b.confidence !== undefined) ? (' '+Math.round(b.confidence*100)+'%') : '';
      const label = (labels[b.class_id] || '?')+' #'+i+confLabel;
      ctx.font = '14px system-ui, sans-serif';
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
      ctx.fillStyle = colors[b.class_id % colors.length];
      ctx.fillText(label, x + 4, Math.max(12, y - 4));
    });
    if (dragStart && dragStart.current) {
      const [x0, y0] = dragStart.start;
      const [x1, y1] = dragStart.current;
      ctx.strokeStyle = '#ffffff';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      ctx.setLineDash([]);
    }
  }

  function renderBoxList() {
    boxRowsEl.innerHTML = '';
    boxes.forEach((b, i) => {
      const tr = document.createElement('tr');
      const confCell = (b.confidence !== null && b.confidence !== undefined) ? ' '+Math.round(b.confidence*100)+'%' : '';
      tr.innerHTML = '<td>'+i+'</td><td>'+(labels[b.class_id]||'?')+confCell+'</td>'
        +'<td>'+b.cx.toFixed(3)+'</td><td>'+b.cy.toFixed(3)+'</td>'
        +'<td>'+b.w.toFixed(3)+'</td><td>'+b.h.toFixed(3)+'</td>'
        +'<td><button data-idx="'+i+'" class="delBtn danger">del</button></td>';
      if (i === selectedBox) tr.style.background = '#2b3a55';
      tr.addEventListener('click', () => { selectedBox = i; redraw(); renderBoxList(); });
      boxRowsEl.appendChild(tr);
    });
    boxRowsEl.querySelectorAll('.delBtn').forEach(btn => {
      btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const i = parseInt(btn.dataset.idx, 10);
        boxes.splice(i, 1);
        if (selectedBox === i) selectedBox = -1;
        else if (selectedBox > i) selectedBox -= 1;
        redraw(); renderBoxList();
      });
    });
  }

  function canvasCoords(ev) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [(ev.clientX - rect.left) * scaleX, (ev.clientY - rect.top) * scaleY];
  }

  canvas.addEventListener('mousedown', (ev) => {
    if (!img) return;
    const [x, y] = canvasCoords(ev);
    dragStart = { start: [x, y], current: [x, y] };
  });
  canvas.addEventListener('mousemove', (ev) => {
    if (!dragStart) return;
    dragStart.current = canvasCoords(ev);
    redraw();
  });
  window.addEventListener('mouseup', (ev) => {
    if (!dragStart || !img) { dragStart = null; return; }
    const [x0, y0] = dragStart.start;
    const [x1, y1] = dragStart.current || dragStart.start;
    dragStart = null;
    const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
    if (dx < 4 || dy < 4) { redraw(); return; }
    const xMin = Math.max(0, Math.min(x0, x1));
    const yMin = Math.max(0, Math.min(y0, y1));
    const xMax = Math.min(canvas.width, Math.max(x0, x1));
    const yMax = Math.min(canvas.height, Math.max(y0, y1));
    const cx = ((xMin + xMax) / 2) / canvas.width;
    const cy = ((yMin + yMax) / 2) / canvas.height;
    const w  = (xMax - xMin) / canvas.width;
    const h  = (yMax - yMin) / canvas.height;
    const cid = parseInt(classSelect.value, 10) || 0;
    boxes.push({ class_id: cid, cx, cy, w, h });
    selectedBox = boxes.length - 1;
    redraw(); renderBoxList();
  });

  function updateFeedbackBar() {
    const bar = document.getElementById('feedbackBar');
    const desc = document.getElementById('fbSuggestDesc');
    if (feedbackState && !feedbackState.decided) {
      bar.classList.add('visible');
      desc.textContent = feedbackState.suggestName ? '(' + feedbackState.suggestName + ')' : '';
    } else {
      bar.classList.remove('visible');
    }
  }

  async function recordFeedback(action, nBoxes, suggestName) {
    const entry = images[currentIdx];
    if (!entry) return;
    try {
      const r = await fetch('/api/feedback', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          camera: entry.camera,
          name: entry.name,
          suggest_name: suggestName || null,
          action: action,
          n_boxes: nBoxes
        })
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || ('http '+r.status));
      return j;
    } catch (e) {
      setStatus('feedback record failed: '+e, true);
      return null;
    }
  }

  document.getElementById('fbCorrectBtn').addEventListener('click', async () => {
    if (!feedbackState || feedbackState.decided) return;
    const sn = feedbackState.suggestName;
    feedbackState.decided = true;
    boxesFromSuggest = false;
    updateFeedbackBar();
    setStatus('Recording feedback (correct) and submitting…');
    await recordFeedback('correct', boxes.length, sn);
    await submitImageCore();
  });

  document.getElementById('fbWrongBtn').addEventListener('click', () => {
    if (!feedbackState || feedbackState.decided) return;
    feedbackRejected = true;
    feedbackState.decided = true;
    boxesFromSuggest = false;
    updateFeedbackBar();
    boxes = []; selectedBox = -1;
    redraw(); renderBoxList();
    setStatus('Model suggestion rejected — draw correct boxes then Submit, or Mark negative', true);
  });

  async function saveBoxes(overrideBoxes) {
    const entry = images[currentIdx];
    if (!entry) return;
    const payload = { camera: entry.camera, name: entry.name, boxes: overrideBoxes || boxes };
    try {
      const r = await fetch('/api/label', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || ('http '+r.status));
      setStatus('saved '+entry.path+' ('+j.n_boxes+' boxes)');
      entry.has_label_file = true;
      entry.n_boxes = j.n_boxes;
      entry.has_suggestions = false;
      entry.n_suggestions = 0;
      entry.suggest_name = null;
      boxesFromSuggest = false;
      renderImgList();
    } catch (e) { setStatus('save failed: '+e, true); }
  }

  // Core archive call; always invoke AFTER feedback is recorded.
  async function submitImageCore() {
    const entry = images[currentIdx];
    if (!entry) return;
    const payload = { camera: entry.camera, name: entry.name, boxes: boxes };
    try {
      setStatus('submitting ' + entry.path + '…');
      const r = await fetch('/api/submit', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || ('http '+r.status));
      setStatus('\u2713 decisioned ' + entry.path + ' (' + j.n_boxes + ' boxes) \u2192 ' + j.image_dest);
      feedbackState = null; feedbackRejected = false; boxesFromSuggest = false;
      images.splice(currentIdx, 1);
      const vis = visibleIndexes();
      if (vis.length) {
        const nextPos = Math.min(currentIdx, vis[vis.length - 1]);
        currentIdx = -1;
        selectImage(nextPos >= 0 ? nextPos : vis[0]);
      } else {
        currentIdx = -1; img = null;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        boxes = []; selectedBox = -1;
        updateFeedbackBar(); renderBoxList(); renderImgList();
        setStatus('\u2713 All images decisioned — queue is empty!');
      }
    } catch (e) { setStatus('submit failed: '+e, true); }
  }

  async function submitImage() {
    // If suggestion loaded and no feedback button clicked: implicit accept.
    if (feedbackState && !feedbackState.decided) {
      feedbackState.decided = true;
      await recordFeedback('correct', boxes.length, feedbackState.suggestName);
      boxesFromSuggest = false; updateFeedbackBar();
    } else if (feedbackRejected) {
      // User clicked 'Model Wrong', drew new boxes; record before archive.
      const action = boxes.length === 0 ? 'negative' : 'corrected';
      const sn = feedbackState ? feedbackState.suggestName : null;
      await recordFeedback(action, boxes.length, sn);
    }
    await submitImageCore();
  }

  document.getElementById('saveBtn').addEventListener('click', () => saveBoxes());
  document.getElementById('submitBtn').addEventListener('click', () => submitImage());
  document.getElementById('emptyBtn').addEventListener('click', async () => {
    boxes = []; selectedBox = -1; redraw(); renderBoxList();
    if (feedbackState && !feedbackState.decided) {
      feedbackState.decided = true;
      await recordFeedback('negative', 0, feedbackState.suggestName);
      boxesFromSuggest = false; updateFeedbackBar();
    }
    saveBoxes([]);
  });
  document.getElementById('clearBtn').addEventListener('click', () => { boxes = []; selectedBox = -1; redraw(); renderBoxList(); });
  document.getElementById('prevBtn').addEventListener('click', () => selectNext(-1));
  document.getElementById('nextBtn').addEventListener('click', () => selectNext(1));
  cameraFilter.addEventListener('change', () => {
    renderImgList();
    const vis = visibleIndexes();
    if (vis.length) selectImage(vis[0]);
    else setStatus('no images for selected camera', true);
  });

  window.addEventListener('keydown', (ev) => {
    if (ev.target && (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT' || ev.target.tagName === 'TEXTAREA')) return;
    if (ev.key === 's' || ev.key === 'S') { ev.preventDefault(); saveBoxes(); }
    else if (ev.key === 'd' || ev.key === 'D') { ev.preventDefault(); submitImage(); }
    else if (ev.key === 'n' || ev.key === 'N') { selectNext(1); }
    else if (ev.key === 'p' || ev.key === 'P') { selectNext(-1); }
    else if (ev.key === 'Delete' || ev.key === 'Backspace') {
      if (selectedBox >= 0) { boxes.splice(selectedBox, 1); selectedBox = -1; redraw(); renderBoxList(); }
    } else if (ev.key === 'Escape') { selectedBox = -1; redraw(); renderBoxList(); }
    else if (/^[0-9]$/.test(ev.key)) {
      const n = parseInt(ev.key, 10);
      if (n < labels.length) classSelect.value = String(n);
    }
  });

  boot();
})();
</script>
</body></html>
"""


# ---------- main ---------------------------------------------------------


def _make_handler_class(review_root: Path, labels: list[str]) -> type:
    return type(
        "BoundLabelerHandler",
        (LabelerHandler,),
        {"review_root": review_root, "labels": labels},
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review-root", default="/mnt/user/media/frigate_custom_model/review", help="Root that contains <Camera>/images/*.jpg")
    p.add_argument("--labels", default="/opt/frigate/labels.txt", help="Path to labels.txt (one class per line)")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1; use 0.0.0.0 for LAN)")
    p.add_argument("--port", type=int, default=8781, help="Bind port (default 8781)")
    p.add_argument("--check", action="store_true", help="Load labels + list images and exit; no server")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    review_root = Path(args.review_root)
    labels_path = Path(args.labels)
    if not review_root.is_dir():
        print(f"review root not found: {review_root}", file=sys.stderr)
        return 2
    labels = load_labels(labels_path)
    images = list_images(review_root)
    print(f"labels={len(labels)} images={len(images)} review_root={review_root}")
    if args.check:
        for entry in images:
            print(f"  {entry['path']}  boxes={entry['n_boxes']}  labeled={entry['has_label_file']}")
        return 0
    handler = _make_handler_class(review_root, labels)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving on http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
