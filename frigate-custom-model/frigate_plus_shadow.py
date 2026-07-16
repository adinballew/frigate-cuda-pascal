#!/usr/bin/env python3
"""Shadow Frigate+ submissions into the local Frigate Labeler queue.

This stdlib-only helper can run as a tiny reverse proxy in front of Frigate.
It forwards requests unchanged, but after a successful:

    POST /api/events/<event_id>/plus

it fetches the clean local Frigate event snapshot and writes it into the DIY
YOLO review staging tree used by docker/frigate/frigate-labeler/labeler.py.

No Frigate config is changed. No outbound Frigate+ traffic is intercepted.
"""

from __future__ import annotations

import argparse
import http.client
import json
import posixpath
import re
import select
import shutil
import socket
import ssl
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

DEFAULT_ALLOWED_CAMERAS = ("FrontDoor", "Backyard")
DEFAULT_LABELS = Path("/opt/frigate/labels.txt")
DEFAULT_REVIEW_ROOT = Path("/mnt/user/media/frigate_custom_model/review")
DEFAULT_FRIGATE_URL = "http://192.168.0.40:5000"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
# Legacy Frigate <0.14: POST /api/events/<event_id>/plus
PLUS_PATH_RE = re.compile(r"^/(?:api/)?events/([^/]+)/plus/?$")
# Frigate 0.14+: POST /api/<camera>/plus/<frame_time>
RECORDING_PLUS_RE = re.compile(r"^/api/([^/]+)/plus/([^/?]+)/?$")
FRAME_TIME_RE = re.compile(r"^\d+(?:\.\d+)?$")
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
SHADOW_SENTINEL_HEADER = "X-Frigate-Shadow-Proxy"
SHADOW_SENTINEL_HEADER_LOWER = SHADOW_SENTINEL_HEADER.lower()
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Settings:
    frigate_url: str
    proxy_upstream_url: str
    review_root: Path
    labels_path: Path
    allowed_cameras: tuple[str, ...]
    overwrite: bool
    with_draft_boxes: bool
    dry_run: bool
    timeout: int
    insecure: bool
    quiet: bool
    log_file: Path | None


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def log(settings: Settings, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
    if not settings.quiet:
        print(message, flush=True)
    if settings.log_file is not None:
        try:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            with settings.log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


def load_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        label = raw.strip()
        if not label or label.startswith("#"):
            continue
        labels.append(label)
    if not labels:
        raise RuntimeError(f"no labels parsed from {path}")
    return labels


def _ssl_context(insecure: bool) -> ssl.SSLContext | None:
    if not insecure:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _request_bytes(
    settings: Settings,
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=settings.timeout, context=_ssl_context(settings.insecure)) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error for {url}: {exc.reason}") from exc


def get_json(settings: Settings, path: str) -> dict:
    url = f"{settings.frigate_url.rstrip('/')}{path}"
    status, _headers, body = _request_bytes(settings, url)
    if status != 200:
        raise RuntimeError(f"HTTP {status} from {path}")
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"bad JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"expected object from {path}, got {type(data).__name__}")
    return data


def get_bytes(settings: Settings, path: str) -> bytes:
    url = f"{settings.frigate_url.rstrip('/')}{path}"
    status, _headers, body = _request_bytes(settings, url)
    if status != 200:
        raise RuntimeError(f"HTTP {status} from {path}")
    return body


def image_suffix_for(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise RuntimeError("snapshot response is not jpg/png/webp image bytes")


def normalize_camera_name(camera: str, allowed: Iterable[str]) -> str | None:
    if camera in allowed:
        return camera

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    camera_norm = norm(camera)
    for allowed_camera in allowed:
        if norm(allowed_camera) == camera_norm:
            return allowed_camera
    return None


def safe_event_id(event_id: str) -> str:
    decoded = urllib.parse.unquote(event_id)
    if not decoded or not EVENT_ID_RE.fullmatch(decoded):
        raise ValueError(f"bad event id: {event_id!r}")
    return decoded


def safe_file_stem(prefix: str, event_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", event_id)
    safe = posixpath.basename(safe)
    if safe in ("", ".", ".."):
        raise ValueError(f"bad event id after sanitizing: {event_id!r}")
    return f"{prefix}_{safe}"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{threading.get_ident()}")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{threading.get_ident()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# YOLO suggestion helpers
# ---------------------------------------------------------------------------


def _confidence_for_event(event: dict, data: dict) -> float | None:
    """Best-effort detector confidence from Frigate event JSON."""
    for raw in (event.get("top_score"), event.get("score"), data.get("top_score"), data.get("score")):
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0:
            return score
    return None


def box_to_yolo_line(
    label_to_id: dict[str, int],
    label: str,
    box: object,
    confidence: float | None = None,
) -> str | None:
    if label not in label_to_id or not isinstance(box, list) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    # Frigate event data.box is normalized xyxy [x1, y1, x2, y2]. Convert to
    # YOLO normalized cx/cy/w/h. Refuse pixel-space boxes instead of guessing
    # from image size in this shadow path.
    if any(v < 0.0 or v > 1.0 for v in vals):
        return None
    x1, y1, x2, y2 = vals
    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0.0 or h <= 0.0:
        return None
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    line = f"{label_to_id[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    if confidence is not None:
        line += f" # source=frigate conf={confidence:.3f}"
    else:
        line += " # source=frigate"
    return line


def draft_lines_for_event(event: dict, labels: list[str]) -> list[str]:
    label_to_id = {label: i for i, label in enumerate(labels)}
    lines: list[str] = []
    label = event.get("label")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    line = box_to_yolo_line(label_to_id, str(label or ""), data.get("box"), _confidence_for_event(event, data))
    if line:
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Recording-frame import (Frigate 0.14+ /api/<camera>/plus/<frame_time>)
# ---------------------------------------------------------------------------


def safe_frame_time(frame_time: str) -> str:
    decoded = urllib.parse.unquote(frame_time)
    if not FRAME_TIME_RE.fullmatch(decoded):
        raise ValueError(f"bad frame time: {frame_time!r}")
    return decoded


def find_review_for_frame(settings: Settings, camera: str, frame_time: str) -> dict | None:
    """Best-effort lookup of the review item that contains a submitted frame."""
    frame = float(frame_time)
    after = max(0, int(frame) - 30)
    before = int(frame) + 30
    encoded_camera = urllib.parse.quote(camera, safe="")
    status, _headers, body = _request_bytes(
        settings,
        f"{settings.frigate_url.rstrip('/')}/api/review?before={before}&after={after}&cameras={encoded_camera}",
    )
    if status != 200:
        return None
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    items = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []
    candidates = [item for item in items if isinstance(item, dict) and item.get("camera") == camera]
    containing = [
        item
        for item in candidates
        if float(item.get("start_time") or 0) <= frame <= float(item.get("end_time") or frame)
    ]
    if containing:
        return min(containing, key=lambda item: abs(float(item.get("start_time") or frame) - frame))
    return candidates[0] if candidates else None


def draft_lines_for_review(settings: Settings, review: dict | None, labels: list[str]) -> list[str]:
    if not review:
        return []
    data = review.get("data") if isinstance(review.get("data"), dict) else {}
    detections = [det for det in data.get("detections", []) if isinstance(det, str) and det]
    lines: list[str] = []
    for det_id in detections:
        try:
            event = get_json(settings, f"/api/events/{urllib.parse.quote(det_id, safe='')}")
            lines.extend(draft_lines_for_event(event, labels))
        except Exception:
            continue
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    return deduped


def import_recording_frame(settings: Settings, camera: str, frame_time: str) -> dict:
    """Import the exact recording frame that Frigate submitted to Frigate+."""
    frame_time = safe_frame_time(frame_time)
    raw_camera = urllib.parse.unquote(camera)
    camera = normalize_camera_name(raw_camera, settings.allowed_cameras) or ""
    if not camera:
        raise RuntimeError(f"camera {raw_camera!r} is not in allowlist {settings.allowed_cameras}")

    snapshot_path = (
        f"/api/{urllib.parse.quote(camera, safe='')}/recordings/"
        f"{urllib.parse.quote(frame_time, safe='')}/snapshot.png"
    )
    image_bytes = get_bytes(settings, snapshot_path)
    suffix = image_suffix_for(image_bytes)

    labels = load_labels(settings.labels_path)
    review = find_review_for_frame(settings, camera, frame_time)
    review_id = review.get("id") if isinstance(review, dict) else None
    stem = safe_file_stem("frigate_plus", f"{camera}_{frame_time}")
    img_dest = settings.review_root / camera / "images" / f"{stem}{suffix}"
    meta_dest = img_dest.with_suffix(".meta.json")
    suggest_dest = img_dest.with_suffix(".suggest.txt")

    exists = img_dest.exists()
    if exists and not settings.overwrite:
        image_status = "already_exists"
    elif settings.dry_run:
        image_status = "would_write"
    else:
        with _WRITE_LOCK:
            atomic_write_bytes(img_dest, image_bytes)
        image_status = "written"

    meta = {
        "source": "frigate_plus_shadow",
        "source_route": f"/api/{camera}/plus/{frame_time}",
        "camera": camera,
        "frame_time": frame_time,
        "review_id": review_id,
        "review_data": review.get("data") if isinstance(review, dict) else None,
        "snapshot_url": snapshot_path,
        "clean_snapshot": True,
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if settings.dry_run:
        meta_status = "would_write"
    else:
        atomic_write_text(meta_dest, json.dumps(meta, indent=2, sort_keys=True) + "\n")
        meta_status = "written"

    draft_lines: list[str] = []
    suggest_status = "skipped"
    if settings.with_draft_boxes:
        draft_lines = draft_lines_for_review(settings, review, labels)
        if suggest_dest.exists() and not settings.overwrite:
            suggest_status = "already_exists"
        elif settings.dry_run:
            suggest_status = "would_write" if draft_lines else "would_write_empty"
        else:
            atomic_write_text(suggest_dest, "\n".join(draft_lines) + ("\n" if draft_lines else ""))
            suggest_status = "written" if draft_lines else "written_empty"

    return {
        "ok": True,
        "camera": camera,
        "frame_time": frame_time,
        "review_id": review_id,
        "image_dest": str(img_dest),
        "meta_dest": str(meta_dest),
        "suggest_dest": str(suggest_dest),
        "image_status": image_status,
        "meta_status": meta_status,
        "suggest_status": suggest_status,
        "n_suggestions": len(draft_lines),
        "bytes": len(image_bytes),
    }


# ---------------------------------------------------------------------------
# Import logic
# ---------------------------------------------------------------------------


def import_event(settings: Settings, event_id: str) -> dict:
    event_id = safe_event_id(event_id)
    labels = load_labels(settings.labels_path)
    encoded_event = urllib.parse.quote(event_id, safe="")
    event = get_json(settings, f"/api/events/{encoded_event}")

    raw_camera = str(event.get("camera") or "")
    camera = normalize_camera_name(raw_camera, settings.allowed_cameras)
    if not camera:
        raise RuntimeError(
            f"event {event_id} camera {raw_camera!r} is not in allowlist {settings.allowed_cameras}"
        )

    snapshot_path = f"/api/events/{encoded_event}/snapshot.jpg?crop=0&bbox=0&timestamp=0"
    image_bytes = get_bytes(settings, snapshot_path)
    suffix = image_suffix_for(image_bytes)

    stem = safe_file_stem("frigate_plus", event_id)
    img_dest = settings.review_root / camera / "images" / f"{stem}{suffix}"
    meta_dest = img_dest.with_suffix(".meta.json")
    suggest_dest = img_dest.with_suffix(".suggest.txt")

    exists = img_dest.exists()
    if exists and not settings.overwrite:
        image_status = "already_exists"
    elif settings.dry_run:
        image_status = "would_write"
    else:
        with _WRITE_LOCK:
            atomic_write_bytes(img_dest, image_bytes)
        image_status = "written"

    meta = {
        "source": "frigate_plus_shadow",
        "event_id": event_id,
        "camera": camera,
        "frigate_camera": raw_camera,
        "label": event.get("label"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "snapshot_url": snapshot_path,
        "clean_snapshot": True,
        "crop": 0,
        "bbox": 0,
        "timestamp": 0,
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if settings.dry_run:
        meta_status = "would_write"
    else:
        atomic_write_text(meta_dest, json.dumps(meta, indent=2, sort_keys=True) + "\n")
        meta_status = "written"

    draft_lines: list[str] = []
    suggest_status = "skipped"
    if settings.with_draft_boxes:
        draft_lines = draft_lines_for_event(event, labels)
        if suggest_dest.exists() and not settings.overwrite:
            suggest_status = "already_exists"
        elif settings.dry_run:
            suggest_status = "would_write" if draft_lines else "would_write_empty"
        else:
            atomic_write_text(suggest_dest, "\n".join(draft_lines) + ("\n" if draft_lines else ""))
            suggest_status = "written" if draft_lines else "written_empty"

    return {
        "ok": True,
        "event_id": event_id,
        "camera": camera,
        "image_dest": str(img_dest),
        "meta_dest": str(meta_dest),
        "suggest_dest": str(suggest_dest),
        "image_status": image_status,
        "meta_status": meta_status,
        "suggest_status": suggest_status,
        "n_suggestions": len(draft_lines),
        "bytes": len(image_bytes),
    }


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------


def response_is_successful_plus(status: int, body: bytes) -> bool:
    if status < 200 or status >= 300:
        return False
    if not body:
        return True
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return True
    if isinstance(payload, dict) and payload.get("success") is False:
        return False
    return True


def make_proxy_handler(settings: Settings):
    class FrigatePlusShadowProxy(BaseHTTPRequestHandler):
        server_version = "FrigatePlusShadow/0.1"

        def log_message(self, fmt: str, *args) -> None:
            if not settings.quiet:
                sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802
            self._proxy()

        def do_PUT(self) -> None:  # noqa: N802
            self._proxy()

        def do_PATCH(self) -> None:  # noqa: N802
            self._proxy()

        def do_DELETE(self) -> None:  # noqa: N802
            self._proxy()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._proxy()

        def _is_websocket_upgrade(self) -> bool:
            """Check if this request is a WebSocket upgrade request."""
            upgrade = self.headers.get("Upgrade", "")
            return upgrade.lower().strip() == "websocket"

        def _websocket_tunnel(self) -> None:
            """Tunnel a WebSocket upgrade request to the upstream Frigate server."""
            parsed = urllib.parse.urlparse(settings.proxy_upstream_url)
            upstream_host = parsed.hostname or "127.0.0.1"
            upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)

            # Build raw request to upstream (preserve all WS headers)
            ws_headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "x-forwarded-for", SHADOW_SENTINEL_HEADER_LOWER}
            }
            ws_headers["Host"] = f"{upstream_host}:{upstream_port}"
            ws_headers[SHADOW_SENTINEL_HEADER] = "1"

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

            raw_request = (
                f"{self.command} {self.path} HTTP/1.1\r\n"
                + "".join(f"{k}: {v}\r\n" for k, v in ws_headers.items())
                + "\r\n"
            ).encode("utf-8") + body

            try:
                upstream_sock = socket.create_connection(
                    (upstream_host, upstream_port), timeout=settings.timeout
                )
                upstream_sock.sendall(raw_request)

                # Read upstream response (status line + headers)
                resp_buf = bytearray()
                while b"\r\n\r\n" not in resp_buf:
                    chunk = upstream_sock.recv(4096)
                    if not chunk:
                        break
                    resp_buf.extend(chunk)
                    if len(resp_buf) > 65536:
                        break

                if b"\r\n\r\n" not in resp_buf:
                    upstream_sock.close()
                    payload = b"upstream did not complete WS handshake"
                    self.send_response(HTTPStatus.BAD_GATEWAY)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                header_end = resp_buf.index(b"\r\n\r\n") + 4
                resp_headers_raw = bytes(resp_buf[:header_end])
                leftover = bytes(resp_buf[header_end:])

                # Parse status line
                status_line = resp_headers_raw.split(b"\r\n", 1)[0].decode("utf-8", "replace")
                parts = status_line.split(" ", 2)
                status_code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 502

                if status_code != 101:
                    # Upgrade failed - forward error response normally
                    upstream_sock.close()
                    self.send_response(status_code)
                    for line in resp_headers_raw.split(b"\r\n")[1:]:
                        if b": " in line:
                            k, v = line.decode("utf-8", "replace").split(": ", 1)
                            if k.lower() not in {"content-length", "server", "date", "transfer-encoding"}:
                                self.send_header(k, v)
                    self.send_header("Content-Length", str(len(leftover)))
                    self.end_headers()
                    self.wfile.write(leftover)
                    return

                # 101 Switching Protocols - forward raw response to client
                client_sock = self.connection
                client_sock.sendall(resp_headers_raw)
                if leftover:
                    client_sock.sendall(leftover)

                # Bidirectional pipe between client and upstream
                socks = [client_sock, upstream_sock]
                try:
                    while True:
                        rlist, _, xlist = select.select(socks, [], socks, 60)
                        if xlist:
                            break
                        if not rlist:
                            continue
                        for ready in rlist:
                            data = ready.recv(65536)
                            if not data:
                                raise ConnectionError("socket closed")
                            if ready is client_sock:
                                upstream_sock.sendall(data)
                            else:
                                client_sock.sendall(data)
                except (ConnectionError, OSError):
                    pass
                finally:
                    upstream_sock.close()
            except Exception as exc:
                payload = json.dumps({"error": f"WS tunnel error: {exc}"}).encode("utf-8")
                try:
                    self.send_response(HTTPStatus.BAD_GATEWAY)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception:
                    pass

        def _proxy(self) -> None:
            # WebSocket upgrade requests need raw TCP tunneling
            if self._is_websocket_upgrade():
                self._websocket_tunnel()
                return

            if self.headers.get(SHADOW_SENTINEL_HEADER) == "1":
                payload = json.dumps({"error": "shadow proxy loop detected"}).encode("utf-8")
                self.send_response(HTTPStatus.LOOP_DETECTED)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                log(settings, f"[proxy] loop detected for {self.command} {self.path}")
                return

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            upstream_url = f"{settings.proxy_upstream_url.rstrip('/')}{self.path}"
            req_headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
                and key.lower() not in {"host", "x-forwarded-for", SHADOW_SENTINEL_HEADER_LOWER}
            }
            req_headers[SHADOW_SENTINEL_HEADER] = "1"
            # Log mutating non-health requests before proxying
            if self.command in MUTATING_METHODS and not self.path.startswith("/api/stats") and self.path not in ("/healthz", "/health", "/api/health"):
                log(settings, f"[proxy] {self.command} {self.path}")
            try:
                status, resp_headers, resp_body = _request_bytes(
                    settings,
                    upstream_url,
                    method=self.command,
                    body=body,
                    headers=req_headers,
                )
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(HTTPStatus.BAD_GATEWAY)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(status)
            for key, value in resp_headers.items():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                if key.lower() in {"content-length", "server", "date"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

            parsed_path = urllib.parse.urlparse(self.path).path

            # Frigate 0.14+: POST /api/<camera>/plus/<frame_time>
            recording_match = RECORDING_PLUS_RE.match(parsed_path)
            if self.command == "POST" and recording_match and response_is_successful_plus(status, resp_body):
                camera = urllib.parse.unquote(recording_match.group(1))
                frame_time = urllib.parse.unquote(recording_match.group(2))
                log(settings, f"[shadow] recording-plus intercepted: camera={camera} frame_time={frame_time} status={status}")
                threading.Thread(
                    target=self._shadow_import_recording,
                    args=(camera, frame_time),
                    daemon=True,
                ).start()
                return

            # Legacy Frigate <0.14: POST /api/events/<event_id>/plus
            event_match = PLUS_PATH_RE.match(parsed_path)
            if self.command == "POST" and event_match and response_is_successful_plus(status, resp_body):
                event_id = urllib.parse.unquote(event_match.group(1))
                log(settings, f"[shadow] event-plus intercepted: event_id={event_id} status={status}")
                threading.Thread(
                    target=self._shadow_import,
                    args=(event_id,),
                    daemon=True,
                ).start()

        def _shadow_import(self, event_id: str) -> None:
            try:
                result = import_event(settings, event_id)
                log(settings, f"[shadow] imported Frigate+ event {event_id}: {json.dumps(result, sort_keys=True)}")
            except Exception as exc:
                log(settings, f"[shadow] failed to import Frigate+ event {event_id}: {exc}")

        def _shadow_import_recording(self, camera: str, frame_time: str) -> None:
            try:
                result = import_recording_frame(settings, camera, frame_time)
                log(settings, f"[shadow] imported Frigate+ recording frame {frame_time} (camera={camera}): {json.dumps(result, sort_keys=True)}")
            except Exception as exc:
                log(settings, f"[shadow] failed to import Frigate+ recording frame {frame_time} (camera={camera}): {exc}")

    return FrigatePlusShadowProxy


def run_proxy(args: argparse.Namespace, settings: Settings) -> int:
    handler = make_proxy_handler(settings)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    scheme = "http"
    if args.tls_cert or args.tls_key:
        if not args.tls_cert or not args.tls_key:
            raise SystemExit("--tls-cert and --tls-key must be provided together")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.tls_cert, args.tls_key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    log(settings, f"Frigate+ shadow proxy listening on {scheme}://{args.host}:{args.port}")
    log(settings, f"  proxy_upstream={settings.proxy_upstream_url}")
    log(settings, f"  import_frigate_url={settings.frigate_url}")
    log(settings, f"  review_root={settings.review_root}")
    log(settings, "  watching POST /api/events/<event_id>/plus  (legacy Frigate <0.14)")
    log(settings, "  watching POST /api/<camera>/plus/<frame_time>  (Frigate 0.14+ Review)")
    log(settings, "  logging all POST/PUT/PATCH/DELETE non-health requests")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log(settings, "stopping")
    finally:
        httpd.server_close()
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


class _FakeFrigate(BaseHTTPRequestHandler):
    event_id = "1700000000.123456-abcDEF"
    camera = "front_door"
    review_camera = "FrontDoor"
    frame_time = "1700000005.250000"
    review_id = "1700000000.000000-review"
    snapshot = b"\xff\xd8\xff\xe0fake-jpeg-for-shadow-test\xff\xd9"
    frame_snapshot = b"\x89PNG\r\n\x1a\nfake-png-for-recording-frame-test"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == f"/api/events/{self.event_id}":
            payload = {
                "id": self.event_id,
                "camera": self.camera,
                "label": "person",
                "start_time": 1700000000.0,
                "end_time": 1700000010.0,
                "top_score": 0.876,
                "data": {"box": [0.4, 0.3, 0.6, 0.7]},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == f"/api/events/{self.event_id}/snapshot.jpg":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(self.snapshot)))
            self.end_headers()
            self.wfile.write(self.snapshot)
            return
        if parsed.path == f"/api/{self.review_camera}/recordings/{self.frame_time}/snapshot.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.frame_snapshot)))
            self.end_headers()
            self.wfile.write(self.frame_snapshot)
            return
        if parsed.path == "/api/review":
            payload = [
                {
                    "id": self.review_id,
                    "camera": self.review_camera,
                    "start_time": 1700000000.0,
                    "end_time": 1700000010.0,
                    "data": {"detections": [self.event_id]},
                }
            ]
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == f"/api/events/{self.event_id}/plus":
            body = b'{"success": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == f"/api/{self.review_camera}/plus/{self.frame_time}":
            body = b'{"success": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


def _start_server(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def run_self_test() -> int:
    fake_server, fake_url = _start_server(_FakeFrigate)
    tmp = Path(tempfile.mkdtemp(prefix="frigate-plus-shadow-test-"))
    labels = tmp / "labels.txt"
    labels.write_text("person\npackage\n", encoding="utf-8")
    try:
        settings = Settings(
            frigate_url=fake_url,
            proxy_upstream_url=fake_url,
            review_root=tmp / "review",
            labels_path=labels,
            allowed_cameras=DEFAULT_ALLOWED_CAMERAS,
            overwrite=False,
            with_draft_boxes=True,
            dry_run=False,
            timeout=5,
            insecure=False,
            quiet=True,
            log_file=None,
        )
        result = import_event(settings, _FakeFrigate.event_id)
        img = Path(result["image_dest"])
        meta = Path(result["meta_dest"])
        suggest = Path(result["suggest_dest"])
        assert result["image_status"] == "written", result
        assert img.read_bytes() == _FakeFrigate.snapshot
        assert json.loads(meta.read_text(encoding="utf-8"))["source"] == "frigate_plus_shadow"
        assert suggest.read_text(encoding="utf-8").strip() == "0 0.500000 0.500000 0.200000 0.400000 # source=frigate conf=0.876"

        frame_result = import_recording_frame(settings, _FakeFrigate.review_camera, _FakeFrigate.frame_time)
        frame_img = Path(frame_result["image_dest"])
        frame_meta = Path(frame_result["meta_dest"])
        frame_suggest = Path(frame_result["suggest_dest"])
        assert frame_result["image_status"] == "written", frame_result
        assert frame_result["review_id"] == _FakeFrigate.review_id, frame_result
        assert frame_img.read_bytes() == _FakeFrigate.frame_snapshot
        assert json.loads(frame_meta.read_text(encoding="utf-8"))["source_route"] == f"/api/{_FakeFrigate.review_camera}/plus/{_FakeFrigate.frame_time}"
        assert frame_suggest.read_text(encoding="utf-8").strip() == "0 0.500000 0.500000 0.200000 0.400000 # source=frigate conf=0.876"

        # Proxy path smoke: upstream succeeds, proxy responds, importer sees existing file.
        proxy_settings = Settings(**{**settings.__dict__, "overwrite": True})
        proxy_server = ThreadingHTTPServer(("127.0.0.1", 0), make_proxy_handler(proxy_settings))
        thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = proxy_server.server_address[:2]
            url = f"http://{host}:{port}/api/events/{_FakeFrigate.event_id}/plus"
            req = urllib.request.Request(url, data=b'{"include_annotation":1}', method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read()
                assert resp.status == 200, resp.status
                assert json.loads(body.decode("utf-8"))["success"] is True
            time.sleep(0.3)
            assert img.exists(), img

            frame_url = f"http://{host}:{port}/api/{_FakeFrigate.review_camera}/plus/{_FakeFrigate.frame_time}"
            frame_req = urllib.request.Request(frame_url, data=b"", method="POST")
            with urllib.request.urlopen(frame_req, timeout=5) as resp:
                body = resp.read()
                assert resp.status == 200, resp.status
                assert json.loads(body.decode("utf-8"))["success"] is True
            time.sleep(0.3)
            assert frame_img.exists(), frame_img
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()

        print("self-test passed")
        return 0
    finally:
        fake_server.shutdown()
        fake_server.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_settings(args: argparse.Namespace) -> Settings:
    return Settings(
        frigate_url=args.frigate_url.rstrip("/"),
        proxy_upstream_url=args.proxy_upstream_url.rstrip("/"),
        review_root=Path(args.review_root),
        labels_path=Path(args.labels),
        allowed_cameras=tuple(args.allowed_camera or DEFAULT_ALLOWED_CAMERAS),
        overwrite=args.overwrite,
        with_draft_boxes=args.draft_boxes and not args.no_draft_boxes,
        dry_run=args.dry_run,
        timeout=args.timeout,
        insecure=args.insecure,
        quiet=args.quiet,
        log_file=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--frigate-url", default=DEFAULT_FRIGATE_URL, help="Frigate upstream base URL (default: %(default)s)")
    common.add_argument("--proxy-upstream-url", default=DEFAULT_FRIGATE_URL, help="Frigate URL used for proxied browser traffic (default: %(default)s)")
    common.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT), help="Frigate Labeler review root (default: %(default)s)")
    common.add_argument("--labels", default=str(DEFAULT_LABELS), help="YOLO class labels file (default: %(default)s)")
    common.add_argument("--camera", dest="allowed_camera", action="append", choices=list(DEFAULT_ALLOWED_CAMERAS), help="Allowed canonical camera; repeatable. Default: FrontDoor and Backyard")
    common.add_argument("--overwrite", action="store_true", help="Overwrite already imported image/meta/suggestion files")
    common.add_argument("--draft-boxes", action="store_true", help="Write Frigate detector boxes as .suggest.txt importer hints (off by default; labeler ignores source=frigate sidecars)")
    common.add_argument("--no-draft-boxes", action="store_true", help="Deprecated compatibility flag; Frigate detector draft boxes are already off by default")
    common.add_argument("--dry-run", action="store_true", help="Fetch and validate but do not write files")
    common.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds (default: %(default)s)")
    common.add_argument("--insecure", action="store_true", help="Allow self-signed HTTPS when --frigate-url is https://")
    common.add_argument("--quiet", action="store_true", help="Reduce logging")

    p_import = sub.add_parser("import-event", parents=[common], help="Import one Frigate event snapshot into Labeler")
    p_import.add_argument("event_id")
    p_import.add_argument("--json", action="store_true", help="Print machine-readable result")

    p_import_frame = sub.add_parser("import-frame", parents=[common], help="Import one Frigate recording frame into Labeler")
    p_import_frame.add_argument("camera")
    p_import_frame.add_argument("frame_time")
    p_import_frame.add_argument("--json", action="store_true", help="Print machine-readable result")

    p_proxy = sub.add_parser("proxy", parents=[common], help="Run reverse proxy that shadows successful Frigate+ submits")
    p_proxy.add_argument("--host", default="127.0.0.1", help="Bind host (default: %(default)s)")
    p_proxy.add_argument("--port", type=int, default=8972, help="Bind port (default: %(default)s)")
    p_proxy.add_argument("--tls-cert", help="Optional PEM certificate path for HTTPS listener")
    p_proxy.add_argument("--tls-key", help="Optional PEM private key path for HTTPS listener")

    sub.add_parser("self-test", help="Run local fake-Frigate self-test")

    args = parser.parse_args(argv)
    if args.cmd == "self-test":
        return run_self_test()

    settings = build_settings(args)
    if args.cmd == "import-event":
        try:
            result = import_event(settings, args.event_id)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Imported {result['event_id']} -> {result['image_dest']} ({result['image_status']})")
            print(f"  meta={result['meta_status']} suggest={result['suggest_status']} n_suggestions={result['n_suggestions']}")
        return 0

    if args.cmd == "import-frame":
        try:
            result = import_recording_frame(settings, args.camera, args.frame_time)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Imported {result['camera']} frame {result['frame_time']} -> {result['image_dest']} ({result['image_status']})")
            print(f"  review_id={result['review_id']} meta={result['meta_status']} suggest={result['suggest_status']} n_suggestions={result['n_suggestions']}")
        return 0

    if args.cmd == "proxy":
        return run_proxy(args, settings)

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
