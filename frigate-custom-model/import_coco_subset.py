#!/usr/bin/env python3
"""Import a small COCO-format YOLO subset into the DIY Frigate 13-class label order.

This is intentionally conservative. It remaps only classes we explicitly support
and drops all other annotations. Images with no remaining boxes are kept as hard
negatives only when --keep-empty is supplied.

Default is dry-run. Use --write to copy files.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# COCO class id -> our class id
COCO_TO_OURS = {
    0: 0,   # person -> person
    1: 8,   # bicycle -> bicycle
    2: 2,   # car -> car
    3: 9,   # motorcycle -> motorcycle
    5: 3,   # bus -> truck-ish? disabled by default unless --map-bus-to-truck
    7: 3,   # truck -> truck
    15: 7,  # bird -> bird
    16: 6,  # cat -> cat
    17: 5,  # dog -> dog
    24: 10, # backpack -> backpack
    28: 11, # suitcase -> suitcase
}

LABELS = [
    "person", "package", "car", "truck", "van", "dog", "cat", "bird",
    "bicycle", "motorcycle", "backpack", "suitcase", "waste_bin",
]


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def remap_label(src_label: Path, *, map_bus_to_truck: bool) -> list[str]:
    if not src_label.exists():
        return []
    out: list[str] = []
    for raw in src_label.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            continue
        try:
            coco_id = int(float(parts[0]))
        except ValueError:
            continue
        if coco_id == 5 and not map_bus_to_truck:
            continue
        if coco_id not in COCO_TO_OURS:
            continue
        ours = COCO_TO_OURS[coco_id]
        out.append(" ".join([str(ours), *parts[1:]]))
    return out


def split_for(path: Path) -> str | None:
    text = str(path).replace("\\", "/").lower()
    if "/val" in text:
        return "val"
    if "/train" in text:
        return "train"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="COCO YOLO dataset root, e.g. datasets/coco8")
    parser.add_argument("--output-root", required=True, type=Path, help="Output YOLO dataset root in our 13-class order")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fallback val split when source has no explicit val images")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--keep-empty", action="store_true", help="Keep images with no supported labels as negatives")
    parser.add_argument("--map-bus-to-truck", action="store_true", help="Map COCO bus to our truck class; off by default")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if not args.source_root.exists():
        raise SystemExit(f"source root missing: {args.source_root}")

    accepted = 0
    skipped = 0
    class_counts = {name: 0 for name in LABELS}
    images = list(iter_images(args.source_root))
    if args.limit:
        images = images[: args.limit]

    planned: list[tuple[Path, Path, list[str], str | None]] = []
    for image in images:
        source_split = split_for(image)
        # Convert images/train/foo.jpg -> labels/train/foo.txt
        parts = list(image.parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
            label = Path(*parts).with_suffix(".txt")
        except ValueError:
            label = image.with_suffix(".txt")
        lines = remap_label(label, map_bus_to_truck=args.map_bus_to_truck)
        if not lines and not args.keep_empty:
            skipped += 1
            continue
        planned.append((image, label, lines, source_split))

    source_splits = {item[3] for item in planned}
    needs_fallback_split = "val" not in source_splits
    val_images: set[Path] = set()
    if needs_fallback_split and planned:
        if not 0.0 < args.val_fraction < 0.5:
            raise SystemExit("--val-fraction must be > 0 and < 0.5")
        shuffled = [item[0] for item in planned]
        random.Random(args.seed).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * args.val_fraction)) if len(shuffled) > 1 else 0
        val_images = set(shuffled[:val_count])

    for image, label, lines, source_split in planned:
        if needs_fallback_split:
            split = "val" if image in val_images else "train"
        elif source_split in {"train", "val"}:
            split = source_split
        else:
            skipped += 1
            continue
        img_dest = args.output_root / "images" / split / image.name
        label_dest = args.output_root / "labels" / split / f"{image.stem}.txt"
        print(f"{split:5} boxes={len(lines):2d} {image} -> {img_dest}")
        for line in lines:
            class_counts[LABELS[int(line.split()[0])]] += 1
        accepted += 1
        if args.write:
            img_dest.parent.mkdir(parents=True, exist_ok=True)
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, img_dest)
            label_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_text = "\n".join([
        f"path: {args.output_root}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {idx}: {name}" for idx, name in enumerate(LABELS)],
        "",
    ])
    if args.write:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    print("\nSummary:")
    print(f"accepted_images={accepted}")
    print(f"skipped_images={skipped}")
    for name, count in class_counts.items():
        if count:
            print(f"class_count {name}={count}")
    if not args.write:
        print("Dry run only. Add --write to copy images/labels and create dataset.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
