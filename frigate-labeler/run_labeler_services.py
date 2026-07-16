#!/usr/bin/env python3
"""Run Frigate Labeler UI and Frigate+ shadow proxy in one container.

Intended for the persistent Unraid Docker container `frigate-labeler`.
No external dependencies.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

WORKSPACE = os.environ.get("WORKSPACE", "/opt/frigate")
REVIEW_ROOT = os.environ.get("REVIEW_ROOT", "/mnt/user/media/frigate_custom_model/review")
LABELS = os.environ.get("LABELS", "/opt/frigate/labels.txt")
FRIGATE_URL = os.environ.get("FRIGATE_URL", "http://192.168.0.40:5000")
PROXY_UPSTREAM_URL = os.environ.get("PROXY_UPSTREAM_URL", FRIGATE_URL)
LABELER_HOST = os.environ.get("LABELER_HOST", "0.0.0.0")
LABELER_PORT = os.environ.get("LABELER_PORT", "8781")
SHADOW_HOST = os.environ.get("SHADOW_HOST", "0.0.0.0")
SHADOW_PORT = os.environ.get("SHADOW_PORT", "8972")
SHADOW_TLS_CERT = os.environ.get("SHADOW_TLS_CERT", "")
SHADOW_TLS_KEY = os.environ.get("SHADOW_TLS_KEY", "")

children: list[subprocess.Popen] = []
stopping = False


def start(cmd: list[str]) -> subprocess.Popen:
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    children.append(proc)
    return proc


def stop_all(signum: int | None = None, frame=None) -> None:
    global stopping
    if stopping:
        return
    stopping = True
    print("stopping services", flush=True)
    for proc in children:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    while time.time() < deadline:
        if all(proc.poll() is not None for proc in children):
            break
        time.sleep(0.2)
    for proc in children:
        if proc.poll() is None:
            proc.kill()


def main() -> int:
    signal.signal(signal.SIGTERM, stop_all)
    signal.signal(signal.SIGINT, stop_all)

    labeler = start([
        sys.executable,
        "/opt/frigate/labeler/labeler.py",
        "--review-root", REVIEW_ROOT,
        "--labels", LABELS,
        "--host", LABELER_HOST,
        "--port", LABELER_PORT,
    ])
    shadow_cmd = [
        sys.executable,
        "/opt/frigate/custom-model/frigate_plus_shadow.py",
        "proxy",
        "--frigate-url", FRIGATE_URL,
        "--proxy-upstream-url", PROXY_UPSTREAM_URL,
        "--review-root", REVIEW_ROOT,
        "--no-draft-boxes",
        "--host", SHADOW_HOST,
        "--port", SHADOW_PORT,
    ]
    if SHADOW_TLS_CERT or SHADOW_TLS_KEY:
        shadow_cmd.extend(["--tls-cert", SHADOW_TLS_CERT, "--tls-key", SHADOW_TLS_KEY])
    shadow = start(shadow_cmd)

    while not stopping:
        for proc, name in ((labeler, "labeler"), (shadow, "shadow-proxy")):
            code = proc.poll()
            if code is not None:
                print(f"{name} exited with code {code}; stopping container", flush=True)
                stop_all()
                return int(code or 1)
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
