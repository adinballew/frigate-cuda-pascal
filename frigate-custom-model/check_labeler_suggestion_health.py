#!/usr/bin/env python3
"""Fail if Frigate Labeler exposes Frigate detector boxes as model suggestions.

The labeler UI should only show custom-model generated sidecars as suggestions.
Legacy/importer sidecars containing `source=frigate` are allowed to exist on disk
briefly, but must never be returned by /api/label as source='suggest'.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BLOCKED_MARKER = "source=frigate"


def fetch_json(base: str, path: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://192.168.0.40:8781")
    ap.add_argument("--review-root", default="/mnt/user/media/frigate_custom_model/review")
    ap.add_argument("--scan-disk", action="store_true", help="Also count legacy source=frigate sidecars on disk")
    args = ap.parse_args()

    images = fetch_json(args.base, "/api/images").get("images", [])
    failures: list[str] = []
    suggestions = 0
    for entry in images:
        if not entry.get("has_suggestions"):
            continue
        suggestions += 1
        q = urllib.parse.urlencode({"camera": entry["camera"], "name": entry["name"]})
        label = fetch_json(args.base, "/api/label?" + q)
        text = str(label.get("text", ""))
        if label.get("source") != "suggest":
            failures.append(f"{entry['path']}: listed as suggestion but source={label.get('source')!r}")
        if BLOCKED_MARKER in text.lower():
            failures.append(f"{entry['path']}: exposes blocked Frigate detector sidecar {label.get('suggest_name')}")

    disk_legacy = 0
    if args.scan_disk:
        root = Path(args.review_root)
        if root.exists():
            for path in root.rglob("*.suggest.txt"):
                try:
                    if BLOCKED_MARKER in path.read_text(encoding="utf-8", errors="replace").lower():
                        disk_legacy += 1
                except OSError:
                    pass

    result = {"ok": not failures, "api_images": len(images), "api_suggestions": suggestions, "disk_legacy_frigate_sidecars": disk_legacy, "failures": failures}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
