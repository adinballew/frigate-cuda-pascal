#!/usr/bin/env python3
"""Generate YOLO draft suggestions for Frigate review images.

Runs Ultralytics inference against the current custom model (default:
the exported v0 seed ONNX) and writes a sidecar `.suggest.txt` next to
each image. These sidecars are NEVER read by the training pipeline;
only the labeler UI reads them, and only when the human-owned `.txt`
label file is missing or empty.

Design constraints:
  * Do not overwrite `.txt` label files (that is the training ground truth).
  * Do not write suggestions when the human has already produced a
    non-empty `.txt` label file for the same image (opt-in --overwrite).
  * Emit an empty sidecar (0 bytes) when nothing was detected above --conf,
    so we don't re-run the same image next pass.
  * Support Ultralytics-supported weights: `.pt` (native) or exported `.onnx`.
  * Class list must match training/labels.txt (13 classes).

Typical invocation inside the trainer container:

    python /workspace/frigate_custom_model/docker/frigate/frigate-custom-model/suggest_yolo_drafts.py \
        --model /workspace/frigate_custom_model/models/v0_package_seed.onnx \
        --review-root /workspace/frigate_custom_model/review \
        --conf 0.25

CUDA/onnxruntime GPU is unreliable in the trainer image (missing cuDNN 9),
so when running an .onnx model you almost certainly want to hide the GPU:

    CUDA_VISIBLE_DEVICES="" python .../suggest_yolo_drafts.py --model .../*.onnx

Stock COCO models can be used as a placeholder while the custom model is
still weak; pass --coco-mapping to map class ids into the 13-class label set:

    python .../suggest_yolo_drafts.py --model .../yolo11n.pt --coco-mapping

Reports a JSON summary on stdout for easy consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ALLOWED_CAMERAS = ("FrontDoor", "Backyard")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SUGGEST_SUFFIX = ".suggest.txt"
LABEL_SUFFIX = ".txt"


def _iter_images(review_root: Path, cameras):
    for cam in cameras:
        img_dir = review_root / cam / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield cam, p


def _labeled_non_empty(image_path: Path) -> bool:
    """True if `<image>.txt` exists AND has any non-blank line."""
    label = image_path.with_suffix(LABEL_SUFFIX)
    if not label.exists():
        return False
    try:
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return True
    except OSError:
        return False
    return False


def _suggest_path(image_path: Path) -> Path:
    # Use `.suggest.txt` alongside image, sharing the base name (minus ext).
    # e.g. FrontDoor_20260705-164205.jpg -> FrontDoor_20260705-164205.suggest.txt
    return image_path.with_suffix(SUGGEST_SUFFIX)


def _write_suggestions(suggest_path: Path, boxes) -> int:
    lines = []
    for cid, cx, cy, w, h, conf in boxes:
        lines.append(
            f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}"
        )
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    suggest_path.parent.mkdir(parents=True, exist_ok=True)
    suggest_path.write_text(payload, encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="/workspace/frigate_custom_model/models/v0_package_seed.onnx",
        help="Ultralytics-compatible weights (.pt or .onnx).",
    )
    parser.add_argument(
        "--review-root",
        default="/workspace/frigate_custom_model/review",
        help="Root containing <Camera>/images/*.jpg",
    )
    parser.add_argument(
        "--cameras",
        default=",".join(ALLOWED_CAMERAS),
        help=f"Comma-separated cameras. Default {','.join(ALLOWED_CAMERAS)}.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default 640).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=25,
        help="Max detections per image (default 25).",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Torch device string (e.g. 'cuda:0', 'cpu'). Default: Ultralytics auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate suggestions even when a sidecar already exists.",
    )
    parser.add_argument(
        "--respect-labeled",
        action="store_true",
        help="Skip images whose `.txt` label file has any non-blank line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N images per camera (0 = unlimited).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write sidecar files; just print planned actions.",
    )
    parser.add_argument(
        "--coco-mapping",
        action="store_true",
        help=(
            "Interpret predictions as COCO class ids and map to the 13-class "
            "label list (person, package, car, truck, van, dog, cat, bird, "
            "bicycle, motorcycle, backpack, suitcase, waste_bin). Use only "
            "with a stock COCO-pretrained model like yolo11n.pt."
        ),
    )
    args = parser.parse_args()

    review_root = Path(args.review_root)
    if not review_root.is_dir():
        print(f"ERROR: review root not found: {review_root}", file=sys.stderr)
        return 2

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    unknown = [c for c in cameras if c not in ALLOWED_CAMERAS]
    if unknown:
        print(f"ERROR: unknown cameras {unknown}; allowed={list(ALLOWED_CAMERAS)}", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        return 2

    # Lazy import so `--help` works even without ultralytics available.
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(model_path))
    class_names = getattr(model, "names", None) or {}
    if isinstance(class_names, dict):
        class_names_list = [class_names[i] for i in sorted(class_names.keys())]
    else:
        class_names_list = list(class_names)

    # COCO -> 13-class mapping (index = COCO class id, value = custom class id or None).
    # Only used with --coco-mapping. Missing keys are dropped from suggestions.
    COCO_TO_CUSTOM = {
        0: 0,   # person
        1: 8,   # bicycle
        2: 2,   # car
        3: 9,   # motorcycle
        5: None,  # bus (drop)
        7: 3,   # truck (also maps van context imperfectly; kept as truck)
        14: 7,  # bird
        15: 6,  # cat
        16: 5,  # dog
        24: 10, # backpack
        26: 12, # handbag -> waste_bin? No: drop.
        28: 11, # suitcase
    }
    if args.coco_mapping:
        # Correct 26 back to "drop" to avoid mis-mapping. Kept above only for clarity.
        COCO_TO_CUSTOM[26] = None

    # Pre-plan images to process (respect skips + limits) before touching the model.
    todo: list[tuple[str, Path]] = []
    per_camera: dict[str, int] = {c: 0 for c in cameras}
    skipped_existing = 0
    skipped_labeled = 0
    for cam, img in _iter_images(review_root, cameras):
        if args.respect_labeled and _labeled_non_empty(img):
            skipped_labeled += 1
            continue
        suggest_path = _suggest_path(img)
        if suggest_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        if args.limit and per_camera[cam] >= args.limit:
            continue
        per_camera[cam] += 1
        todo.append((cam, img))

    per_image: list[dict] = []
    for cam, img in todo:
        # Ultralytics returns a list of Results objects, one per image
        # (we pass single paths so we always get 1).
        predict_kwargs = dict(
            source=str(img),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=False,
        )
        if args.device:
            predict_kwargs["device"] = args.device
        results = model.predict(**predict_kwargs)
        res = results[0]

        boxes_norm: list[tuple[int, float, float, float, float, float]] = []
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # xywhn is already normalized center-format; safer than converting xyxy.
            xywhn = boxes.xywhn.cpu().numpy() if hasattr(boxes.xywhn, "cpu") else boxes.xywhn
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            for b, c, cf in zip(xywhn, cls, confs):
                cid = int(c)
                if args.coco_mapping:
                    mapped = COCO_TO_CUSTOM.get(cid)
                    if mapped is None:
                        continue
                    cid = mapped
                cx, cy, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                # Clamp to [0, 1] guard-rail; occasional Ultralytics jitter.
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)
                if w <= 0.0 or h <= 0.0:
                    continue
                boxes_norm.append((cid, cx, cy, w, h, float(cf)))

        suggest_path = _suggest_path(img)
        if args.dry_run:
            n_written = len(boxes_norm)
        else:
            n_written = _write_suggestions(suggest_path, boxes_norm)
        per_image.append({
            "camera": cam,
            "image": img.name,
            "suggest_path": str(suggest_path),
            "n_boxes": n_written,
            "labeled_non_empty": _labeled_non_empty(img),
        })

    summary = {
        "model": str(model_path),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "class_names": class_names_list,
        "coco_mapping": bool(args.coco_mapping),
        "cameras": cameras,
        "planned": len(todo),
        "skipped_existing": skipped_existing,
        "skipped_labeled": skipped_labeled,
        "processed": len(per_image),
        "images": per_image,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
