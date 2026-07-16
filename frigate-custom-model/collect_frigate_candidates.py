#!/usr/bin/env python3
"""Stage local Frigate candidate images for DIY custom detector review.

This script is intentionally conservative and dry-run by default. It copies only
small image files already present on local disk when --write is supplied. It does
not download datasets, connect to Frigate APIs, or run training.

Camera policy:
- include: FrontDoor, Backyard
- exclude: Patio

The script infers camera from the file path/name by token matching. Ambiguous
files are reported and, with --write, copied to needs_camera_review.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

INCLUDE_CAMERAS = ("FrontDoor", "Backyard")
EXCLUDE_CAMERAS = ("Patio",)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Candidate:
    src: Path
    camera: str | None
    action: str
    dest: Path | None
    reason: str


def normalized_tokens(path: Path) -> str:
    return str(path).replace("-", "_").replace(" ", "_").lower()


def infer_camera(path: Path) -> tuple[str | None, str]:
    haystack = normalized_tokens(path)
    for camera in EXCLUDE_CAMERAS:
        if camera.lower() in haystack:
            return camera, "excluded camera"
    matches = [camera for camera in INCLUDE_CAMERAS if camera.lower() in haystack]
    if len(matches) == 1:
        return matches[0], "included camera"
    if len(matches) > 1:
        return None, "ambiguous camera tokens"
    return None, "camera not found"


def iter_images(source: Path) -> Iterable[Path]:
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def unique_dest(dest_dir: Path, src: Path) -> Path:
    candidate = dest_dir / src.name
    if not candidate.exists():
        return candidate
    stem = src.stem
    suffix = src.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i:03d}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def plan_candidates(source: Path, review_root: Path) -> list[Candidate]:
    planned: list[Candidate] = []
    for src in iter_images(source):
        camera, reason = infer_camera(src)
        if camera in INCLUDE_CAMERAS:
            dest_dir = review_root / camera / "images"
            planned.append(Candidate(src, camera, "stage", unique_dest(dest_dir, src), reason))
        elif camera in EXCLUDE_CAMERAS:
            dest_dir = review_root / "rejected" / camera
            planned.append(Candidate(src, camera, "reject", unique_dest(dest_dir, src), reason))
        else:
            dest_dir = review_root / "needs_camera_review"
            planned.append(Candidate(src, None, "needs_review", unique_dest(dest_dir, src), reason))
    return planned


def copy_candidate(candidate: Candidate) -> None:
    if candidate.dest is None:
        return
    candidate.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate.src, candidate.dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local folder containing candidate images")
    parser.add_argument("--review-root", required=True, type=Path, help="Review staging root")
    parser.add_argument("--write", action="store_true", help="Actually copy files; default is dry-run")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images to process after sorting")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source does not exist: {args.source}")
    if not args.source.is_dir():
        raise SystemExit(f"source is not a directory: {args.source}")

    planned = plan_candidates(args.source, args.review_root)
    if args.limit > 0:
        planned = planned[: args.limit]

    counts: dict[str, int] = {"stage": 0, "reject": 0, "needs_review": 0}
    for item in planned:
        counts[item.action] = counts.get(item.action, 0) + 1
        dest = str(item.dest) if item.dest else "-"
        print(f"{item.action:12} camera={item.camera or 'UNKNOWN':10} src={item.src} -> {dest} ({item.reason})")
        if args.write:
            copy_candidate(item)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"\n{mode} summary: total={len(planned)} stage={counts.get('stage', 0)} reject={counts.get('reject', 0)} needs_review={counts.get('needs_review', 0)}")
    if not args.write:
        print("No files copied. Re-run with --write after reviewing the plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
