#!/usr/bin/env python3
"""MQTT-triggered Frigate snapshot collector for the DIY custom model loop.

Subscribes to Frigate MQTT topics and stages review images for manual labeling.
No external Python dependencies; implements the small subset of MQTT v3.1.1 needed
for anonymous subscribe.

Default behavior:
- broker: 192.168.0.40:1883
- topic: frigate/events
- cameras: FrontDoor, Backyard only
- event labels: person, package, car, truck, dog, cat, bird, waste_bin
- on new event trigger: capture immediately, then every 15s for 45s
- writes: /mnt/user/media/frigate_custom_model/review/<Camera>/images/*.jpg + empty .txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import ssl
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

ALLOWED_CAMERAS = {"FrontDoor", "Backyard"}
DEFAULT_LABELS = {"person", "package", "car", "truck", "dog", "cat", "bird", "waste_bin"}
SUGGEST_SCRIPT = Path(__file__).with_name("generate_draft_labels_via_trainer.py")

# Keep Frigate detection on the low-cost substream, but collect full-res stills
# for Backyard/"patio view" active-learning review.
HIGH_RES_SNAPSHOT_URLS = {
    "FrontDoor": os.environ.get("REOLINK_FRONTDOOR_SNAPSHOT_URL", ""),
    "Backyard": os.environ.get("REOLINK_BACKYARD_SNAPSHOT_URL", ""),
}


def enc_remaining_length(n: int) -> bytes:
    out = bytearray()
    while True:
        digit = n % 128
        n //= 128
        if n:
            digit |= 0x80
        out.append(digit)
        if not n:
            return bytes(out)


def enc_utf8(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("!H", len(b)) + b


def read_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        part = sock.recv(n - got)
        if not part:
            raise ConnectionError("MQTT socket closed")
        chunks.append(part)
        got += len(part)
    return b"".join(chunks)


def read_packet(sock: socket.socket) -> tuple[int, bytes] | None:
    try:
        first = sock.recv(1)
    except socket.timeout:
        return None
    if not first:
        raise ConnectionError("MQTT socket closed")
    multiplier = 1
    remaining = 0
    while True:
        b = read_exact(sock, 1)[0]
        remaining += (b & 127) * multiplier
        if not (b & 128):
            break
        multiplier *= 128
        if multiplier > 128 * 128 * 128:
            raise ValueError("malformed MQTT remaining length")
    return first[0], read_exact(sock, remaining)


class MqttClient:
    def __init__(self, host: str, port: int, client_id: str, keepalive: int = 30):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self.sock: socket.socket | None = None
        self.packet_id = 1
        self.last_tx = 0.0

    def connect(self):
        sock = socket.create_connection((self.host, self.port), timeout=10)
        sock.settimeout(1.0)
        vh = enc_utf8("MQTT") + bytes([4, 2]) + struct.pack("!H", self.keepalive)
        payload = enc_utf8(self.client_id)
        self._send_raw(bytes([0x10]) + enc_remaining_length(len(vh) + len(payload)) + vh + payload, sock)
        packet = read_packet(sock)
        if not packet or packet[0] != 0x20 or len(packet[1]) < 2 or packet[1][1] != 0:
            raise ConnectionError(f"MQTT CONNACK failed: {packet!r}")
        self.sock = sock

    def _send_raw(self, data: bytes, sock: socket.socket | None = None):
        (sock or self.sock).sendall(data)  # type: ignore[union-attr]
        self.last_tx = time.time()

    def subscribe(self, topics: list[str]):
        payload = bytearray()
        for t in topics:
            payload += enc_utf8(t) + b"\x00"
        pid = self.packet_id
        self.packet_id += 1
        body = struct.pack("!H", pid) + payload
        self._send_raw(bytes([0x82]) + enc_remaining_length(len(body)) + body)
        deadline = time.time() + 5
        while time.time() < deadline:
            packet = read_packet(self.sock)  # type: ignore[arg-type]
            if not packet:
                continue
            ptype, body = packet
            if ptype == 0x90:
                return
        raise TimeoutError("MQTT SUBACK timed out")

    def maybe_ping(self):
        if time.time() - self.last_tx > self.keepalive / 2:
            self._send_raw(b"\xc0\x00")

    def messages(self):
        while True:
            packet = read_packet(self.sock)  # type: ignore[arg-type]
            if not packet:
                self.maybe_ping()
                yield None
                continue
            ptype, body = packet
            kind = ptype >> 4
            if kind != 3:  # publish only
                continue
            qos = (ptype >> 1) & 0x03
            if len(body) < 2:
                continue
            tlen = struct.unpack("!H", body[:2])[0]
            topic = body[2:2 + tlen].decode("utf-8", "replace")
            pos = 2 + tlen
            if qos:
                pos += 2
            yield topic, body[pos:]


def safe_label_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_url_for(camera: str, frigate_url: str, prefer_high_res: bool = True) -> str:
    if prefer_high_res and HIGH_RES_SNAPSHOT_URLS.get(camera):
        return HIGH_RES_SNAPSHOT_URLS[camera]
    return f"{frigate_url.rstrip('/')}/api/{camera}/latest.jpg"


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(k, "***" if k.lower() in {"password", "pass"} else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def capture_latest(frigate_url: str, review_root: Path, camera: str, reason: str, seen_dir: Path, dry_run: bool = False, prefer_high_res: bool = True) -> Path | None:
    url = snapshot_url_for(camera, frigate_url, prefer_high_res=prefer_high_res)
    ctx = ssl._create_unverified_context() if url.startswith("https://") else None
    with urlopen(url, timeout=10, context=ctx) as r:
        data = r.read()
    if len(data) < 1024:
        raise RuntimeError(f"snapshot too small for {camera}: {len(data)} bytes")
    digest = sha256(data)
    seen_file = seen_dir / f"{camera}.{digest}.seen"
    if seen_file.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reason)[:40]
    dest = review_root / camera / "images" / f"{camera}_mqtt_{stamp}_{safe_reason}.jpg"
    label = dest.with_suffix(".txt")
    if dry_run:
        print(f"DRY capture {camera} reason={reason} url={safe_url(url)} bytes={len(data)} -> {dest}", flush=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    seen_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    label.write_text("", encoding="utf-8")
    seen_file.write_text(str(time.time()), encoding="utf-8")
    print(f"captured camera={camera} reason={reason} bytes={len(data)} file={dest}", flush=True)
    return dest


def generate_suggestions(image_path: Path, conf: float) -> None:
    if not SUGGEST_SCRIPT.exists():
        print(f"suggest_skip missing_script={SUGGEST_SCRIPT}", flush=True)
        return
    cmd = [
        sys.executable,
        str(SUGGEST_SCRIPT),
        "--image",
        str(image_path),
        "--conf",
        str(conf),
        "--overwrite",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
    except Exception as e:
        print(f"suggest_error image={image_path}: {e}", flush=True)
        return
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        print(f"suggest_error image={image_path} exit={result.returncode}: {result.stderr.strip()}", flush=True)


def event_trigger(payload: bytes, cameras: set[str], labels: set[str], event_types: set[str]) -> tuple[str, str] | None:
    try:
        obj = json.loads(payload.decode("utf-8", "replace"))
    except Exception:
        return None
    after = obj.get("after") if isinstance(obj, dict) else None
    before = obj.get("before") if isinstance(obj, dict) else None
    item = after or before or obj
    if not isinstance(item, dict):
        return None
    camera = item.get("camera") or obj.get("camera")
    label = item.get("label") or obj.get("label")
    etype = obj.get("type", "event")
    if event_types and etype not in event_types:
        return None
    if camera in cameras and (not label or label in labels):
        return camera, f"event_{etype}_{label or 'unknown'}"
    return None


def motion_trigger(topic: str, payload: bytes, cameras: set[str]) -> tuple[str, str] | None:
    parts = topic.split("/")
    if len(parts) != 3 or parts[0] != "frigate" or parts[2] != "motion":
        return None
    camera = parts[1]
    text = payload.decode("utf-8", "replace").strip().upper()
    if camera in cameras and text in {"ON", "1", "TRUE"}:
        return camera, "motion_on"
    return None


def run(args) -> int:
    cameras = set(args.camera or sorted(ALLOWED_CAMERAS))
    bad = cameras - ALLOWED_CAMERAS
    if bad:
        raise SystemExit(f"unsupported camera(s): {sorted(bad)}")
    labels = set(args.label or sorted(DEFAULT_LABELS))
    review_root = Path(args.review_root)
    seen_dir = review_root / ".watcher_seen"
    active_until: dict[str, float] = {}
    next_capture: dict[str, float] = {}
    reasons: dict[str, str] = {}
    event_types = set(args.event_type or ["new"])
    topics = [args.topic_prefix.rstrip("/") + "/events"]
    if args.include_motion:
        topics.append(args.topic_prefix.rstrip("/") + "/+/motion")

    while True:
        client = MqttClient(args.broker_host, args.broker_port, args.client_id, keepalive=args.keepalive)
        try:
            print(f"connecting mqtt {args.broker_host}:{args.broker_port} topics={topics}", flush=True)
            client.connect()
            client.subscribe(topics)
            print(f"watching cameras={sorted(cameras)} labels={sorted(labels)} event_types={sorted(event_types)} include_motion={args.include_motion} duration={args.duration}s interval={args.interval}s", flush=True)
            started = time.time()
            for msg in client.messages():
                now = time.time()
                if args.max_seconds and now - started >= args.max_seconds:
                    print("max seconds reached", flush=True)
                    return 0
                if msg is not None:
                    topic, payload = msg
                    trig = None
                    if topic == args.topic_prefix.rstrip("/") + "/events":
                        trig = event_trigger(payload, cameras, labels, event_types)
                    elif args.include_motion:
                        trig = motion_trigger(topic, payload, cameras)
                    if trig:
                        camera, reason = trig
                        active_until[camera] = max(active_until.get(camera, 0), now + args.duration)
                        next_capture[camera] = now
                        reasons[camera] = reason
                        print(f"trigger camera={camera} reason={reason} active_until={datetime.fromtimestamp(active_until[camera]).isoformat(timespec='seconds')}", flush=True)
                for camera, until in list(active_until.items()):
                    if now > until:
                        active_until.pop(camera, None)
                        next_capture.pop(camera, None)
                        reasons.pop(camera, None)
                        print(f"inactive camera={camera}", flush=True)
                        continue
                    if now >= next_capture.get(camera, 0):
                        try:
                            captured = capture_latest(args.frigate_url, review_root, camera, reasons.get(camera, "trigger"), seen_dir, args.dry_run, prefer_high_res=not args.frigate_only)
                            if captured and args.auto_suggest and not args.dry_run:
                                generate_suggestions(captured, args.suggest_conf)
                        except Exception as e:
                            print(f"capture_error camera={camera}: {e}", flush=True)
                        next_capture[camera] = now + args.interval
        except KeyboardInterrupt:
            print("stopped", flush=True)
            return 0
        except Exception as e:
            print(f"mqtt watcher error: {e}; reconnecting in {args.reconnect_delay}s", flush=True)
            time.sleep(args.reconnect_delay)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--broker-host", default="192.168.0.40")
    ap.add_argument("--broker-port", type=int, default=1883)
    ap.add_argument("--frigate-url", default="http://192.168.0.40:5000")
    ap.add_argument("--topic-prefix", default="frigate")
    ap.add_argument("--client-id", default=f"frigate-active-learning-{os.getpid()}")
    ap.add_argument("--review-root", default="/mnt/user/media/frigate_custom_model/review")
    ap.add_argument("--camera", action="append", choices=sorted(ALLOWED_CAMERAS), help="Allowed camera; repeatable. Default: FrontDoor + Backyard")
    ap.add_argument("--label", action="append", help="Event label to trigger on; repeatable. Default: common package-relevant labels")
    ap.add_argument("--event-type", action="append", choices=["new", "update", "end"], help="Frigate event lifecycle type to capture. Repeatable. Default: new only")
    ap.add_argument("--include-motion", action="store_true", help="Also subscribe to raw frigate/+/motion ON topics. Noisy; off by default")
    ap.add_argument("--duration", type=float, default=45.0, help="Seconds to keep capturing after each trigger")
    ap.add_argument("--interval", type=float, default=15.0, help="Seconds between snapshots while active")
    ap.add_argument("--keepalive", type=int, default=30)
    ap.add_argument("--reconnect-delay", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frigate-only", action="store_true", help="Use Frigate latest.jpg for all cameras; disables high-res Reolink snapshot override")
    ap.add_argument("--no-auto-suggest", dest="auto_suggest", action="store_false", help="Do not run draft model suggestions after captures")
    ap.add_argument("--suggest-conf", type=float, default=0.20, help="Confidence threshold for draft suggestions")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="Exit after N seconds; useful for smoke tests")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
