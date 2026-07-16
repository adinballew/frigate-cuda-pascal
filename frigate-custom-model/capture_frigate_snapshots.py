#!/usr/bin/env python3
"""Capture current Frigate camera snapshots into the DIY model review staging area.

Default is dry-run. With --write, downloads latest.jpg for allowed cameras only.
This seeds review with real FrontDoor/Backyard imagery without touching Frigate config.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import ssl
from urllib.request import urlopen
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ALLOWED_CAMERAS = ("FrontDoor", "Backyard")

# Backyard/"patio view" uses a Reolink substream for Frigate detect
# (640x360), but the camera's still-image API returns full-res 3840x2160.
# Use that for review/training captures while leaving Frigate detect cheap.
HIGH_RES_SNAPSHOT_URLS = {
    "FrontDoor": os.environ.get("REOLINK_FRONTDOOR_SNAPSHOT_URL", ""),
    "Backyard": os.environ.get("REOLINK_BACKYARD_SNAPSHOT_URL", ""),
}


def read_snapshot(camera: str, frigate_url: str, prefer_high_res: bool = True) -> tuple[str, bytes]:
    if prefer_high_res and HIGH_RES_SNAPSHOT_URLS.get(camera):
        url = HIGH_RES_SNAPSHOT_URLS[camera]
        ctx = ssl._create_unverified_context() if url.startswith("https://") else None
        with urlopen(url, timeout=10, context=ctx) as response:
            return url, response.read()
    url = f"{frigate_url.rstrip('/')}/api/{camera}/latest.jpg"
    with urlopen(url, timeout=10) as response:
        return url, response.read()


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(k, "***" if k.lower() in {"password", "pass"} else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frigate-url", default="http://192.168.0.40:5000")
    parser.add_argument("--review-root", default="/mnt/user/media/frigate_custom_model/review")
    parser.add_argument("--camera", action="append", choices=ALLOWED_CAMERAS, help="Camera to capture; repeatable. Default: all allowed cameras")
    parser.add_argument("--write", action="store_true", help="Actually download images; default is dry-run")
    parser.add_argument("--frigate-only", action="store_true", help="Use Frigate latest.jpg for all cameras; disables high-res Reolink snapshot override")
    args = parser.parse_args()

    cameras = args.camera or list(ALLOWED_CAMERAS)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.review_root)
    for camera in cameras:
        dest = root / camera / "images" / f"{camera}_{stamp}.jpg"
        label = dest.with_suffix(".txt")
        url, data = read_snapshot(camera, args.frigate_url, prefer_high_res=not args.frigate_only) if args.write else (
            HIGH_RES_SNAPSHOT_URLS.get(camera, f"{args.frigate_url.rstrip('/')}/api/{camera}/latest.jpg") if not args.frigate_only else f"{args.frigate_url.rstrip('/')}/api/{camera}/latest.jpg",
            b"",
        )
        print(f"capture camera={camera} url={safe_url(url)} -> {dest}")
        print(f"empty_negative_label={label}")
        if args.write:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if len(data) < 1024:
                raise SystemExit(f"snapshot too small for {camera}: {len(data)} bytes")
            dest.write_bytes(data)
            # Start as reviewed negative until annotated otherwise.
            label.write_text("", encoding="utf-8")
            print(f"wrote {dest} bytes={len(data)}")
    if not args.write:
        print("Dry run only. Add --write to capture snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
