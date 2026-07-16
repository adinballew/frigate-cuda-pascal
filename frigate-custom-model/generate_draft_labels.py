#!/usr/bin/env python3
"""Generate draft YOLO labels from the seed/custom ONNX model.

Writes `.suggest.txt` sidecars next to review images. These are UI drafts only:
training labels remain the normal `.txt` files and are written only when the
human saves corrections in the labeler.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

ALLOWED_CAMERAS = ("FrontDoor", "Backyard")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape[1] - nw, new_shape[0] - nh
    dw /= 2
    dh /= 2
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def predict_image(sess: ort.InferenceSession, image_path: Path, conf: float, iou: float, classes: set[int] | None):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"could not read image: {image_path}")
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded, gain, pad = letterbox(rgb, (640, 640))
    x = padded.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x})[0]
    pred = np.squeeze(out)
    if pred.ndim != 2:
        raise RuntimeError(f"unexpected output shape {out.shape}")
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # [8400, 4+n]
    boxes = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(scores_all.shape[0]), cls_ids]
    mask = scores >= conf
    if classes is not None:
        mask &= np.isin(cls_ids, list(classes))
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    if len(boxes) == 0:
        return []

    # ONNX export gives xywh in 640-letterboxed pixel space.
    cx, cy, bw, bh = boxes.T
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    left, top = pad
    x1 = (x1 - left) / gain
    x2 = (x2 - left) / gain
    y1 = (y1 - top) / gain
    y2 = (y2 - top) / gain
    x1 = np.clip(x1, 0, w0)
    x2 = np.clip(x2, 0, w0)
    y1 = np.clip(y1, 0, h0)
    y2 = np.clip(y2, 0, h0)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep_all: list[int] = []
    for cid in sorted(set(int(c) for c in cls_ids.tolist())):
        inds = np.where(cls_ids == cid)[0]
        keep_all.extend(inds[k] for k in nms(boxes_xyxy[inds], scores[inds], iou))
    keep_all = sorted(keep_all, key=lambda i: float(scores[i]), reverse=True)

    rows = []
    for i in keep_all:
        x1, y1, x2, y2 = boxes_xyxy[i]
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cxn = ((x1 + x2) / 2) / w0
        cyn = ((y1 + y2) / 2) / h0
        wn = (x2 - x1) / w0
        hn = (y2 - y1) / h0
        rows.append((int(cls_ids[i]), float(cxn), float(cyn), float(wn), float(hn), float(scores[i])))
    return rows


def image_paths(root: Path, cameras: Iterable[str]):
    for camera in cameras:
        img_dir = root / camera / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-root", default="/mnt/user/media/frigate_custom_model/review")
    ap.add_argument("--model", default="/mnt/user/media/frigate_custom_model/models/v0_package_seed.onnx")
    ap.add_argument("--camera", action="append", choices=ALLOWED_CAMERAS, help="Camera to scan; repeatable. Default all")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--class-id", action="append", type=int, help="Restrict to class id; repeatable. Default all")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .suggest.txt files")
    ap.add_argument("--write-empty", action="store_true", help="Write empty suggestion files when no detections")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.review_root)
    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    classes = set(args.class_id) if args.class_id else None
    cameras = args.camera or list(ALLOWED_CAMERAS)
    scanned = wrote = skipped_labeled = 0
    for img in image_paths(root, cameras):
        if args.limit and scanned >= args.limit:
            break
        scanned += 1
        label = img.with_suffix(".txt")
        suggest = img.with_suffix(".suggest.txt")
        if label.exists() and label.read_text(encoding="utf-8").strip():
            skipped_labeled += 1
            continue
        if suggest.exists() and not args.overwrite:
            continue
        rows = predict_image(sess, img, args.conf, args.iou, classes)
        lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} # conf={score:.3f}" for cid, cx, cy, w, h, score in rows]
        if lines or args.write_empty:
            print(f"suggest {img} boxes={len(lines)}")
            if not args.dry_run:
                suggest.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
                wrote += 1
    print(f"scanned={scanned} wrote={wrote} skipped_labeled={skipped_labeled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
