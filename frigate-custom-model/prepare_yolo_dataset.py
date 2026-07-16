#!/usr/bin/env python3
"""Validate reviewed annotations and prepare a YOLO dataset split.

Expected reviewed layout is flexible but camera-aware. The script searches under
--review-root for images and same-stem .txt YOLO label files. Any path containing
Patio is excluded. Paths containing FrontDoor or Backyard are accepted. Unknown
camera paths are reported and skipped.

By default, only Frigate+ shadow-submit snapshots (`frigate_plus_*`) are accepted.
Legacy `frigate_review*` thumbnail imports are skipped so low-quality junk cannot
leak into new training runs. Pass --include-legacy if you intentionally need old
review/candidate images.

Dry-run is the default. Use --write to copy images/labels and write dataset.yaml.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

INCLUDE_CAMERAS = ("FrontDoor", "Backyard")
EXCLUDE_CAMERAS = ("Patio",)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    camera: str
    split: str


def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise SystemExit(f"labels file is empty: {path}")
    if len(labels) != len(set(labels)):
        raise SystemExit(f"labels file contains duplicates: {path}")
    return labels


def camera_for(path: Path) -> str | None:
    text = str(path).replace("-", "_").replace(" ", "_").lower()
    for camera in EXCLUDE_CAMERAS:
        if camera.lower() in text:
            return camera
    matches = [camera for camera in INCLUDE_CAMERAS if camera.lower() in text]
    return matches[0] if len(matches) == 1 else None


def validate_label_file(path: Path, class_count: int) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return ["missing label file"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            errors.append(f"line {line_no}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            errors.append(f"line {line_no}: parse error: {exc}")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"line {line_no}: class id {class_id} outside 0..{class_count - 1}")
        for coord in coords:
            if not 0.0 <= coord <= 1.0:
                errors.append(f"line {line_no}: coordinate {coord} outside 0..1")
    return errors


def is_shadow_snapshot(image: Path) -> bool:
    return image.name.startswith("frigate_plus_")


def is_legacy_review_import(image: Path) -> bool:
    return image.name.startswith("frigate_review")


def find_samples(
    review_root: Path,
    class_count: int,
    include_legacy: bool = False,
) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    accepted: list[tuple[Path, Path, str]] = []
    issues: list[str] = []
    for image in sorted(review_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        if not include_legacy and not is_shadow_snapshot(image):
            reason = "legacy review import" if is_legacy_review_import(image) else "non-shadow image"
            issues.append(f"skipped {reason}: {image}")
            continue
        camera = camera_for(image)
        if camera in EXCLUDE_CAMERAS:
            issues.append(f"excluded Patio image skipped: {image}")
            continue
        if camera not in INCLUDE_CAMERAS:
            issues.append(f"unknown/ambiguous camera skipped: {image}")
            continue
        label = image.with_suffix(".txt")
        errors = validate_label_file(label, class_count)
        if errors:
            issues.append(f"invalid label for {image}: {'; '.join(errors)}")
            continue
        accepted.append((image, label, camera))
    return accepted, issues


def assign_splits(items: list[tuple[Path, Path, str]], val_fraction: float, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    samples: list[Sample] = []
    for idx, (image, label, camera) in enumerate(shuffled):
        split = "val" if idx < val_count else "train"
        samples.append(Sample(image=image, label=label, camera=camera, split=split))
    return sorted(samples, key=lambda sample: (sample.split, sample.camera, str(sample.image)))


def make_unique(dest_dir: Path, src: Path) -> Path:
    candidate = dest_dir / src.name
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = dest_dir / f"{src.stem}_{i:03d}{src.suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def write_dataset(samples: list[Sample], output_root: Path, labels: list[str], yaml_path_root: Path | None = None) -> None:
    for sample in samples:
        image_dir = output_root / "images" / sample.split
        label_dir = output_root / "labels" / sample.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dest = make_unique(image_dir, sample.image)
        label_dest = label_dir / f"{image_dest.stem}.txt"
        shutil.copy2(sample.image, image_dest)
        shutil.copy2(sample.label, label_dest)

    yaml_root = yaml_path_root or output_root
    yaml_lines = [
        f"path: {yaml_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(labels))
    (output_root / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--yaml-path-root", type=Path, help="Path to write in dataset.yaml when training runs inside a container with different mounts")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include old/non-shadow images. Default accepts only frigate_plus_* snapshots.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 0.5:
        raise SystemExit("--val-fraction must be > 0 and < 0.5")
    labels = load_labels(args.labels)
    accepted, issues = find_samples(args.review_root, len(labels), include_legacy=args.include_legacy)
    samples = assign_splits(accepted, args.val_fraction, args.seed)

    for issue in issues:
        print(f"ISSUE {issue}")
    for sample in samples:
        print(f"{sample.split:5} camera={sample.camera:10} image={sample.image} label={sample.label}")

    print(f"\nSummary: accepted={len(samples)} issues={len(issues)} labels={len(labels)}")
    if args.write:
        write_dataset(samples, args.output_root, labels, args.yaml_path_root)
        print(f"Wrote YOLO dataset to {args.output_root}")
        if args.yaml_path_root:
            print(f"dataset.yaml path root: {args.yaml_path_root}")
    else:
        print("Dry run only. Re-run with --write to copy files and create dataset.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
