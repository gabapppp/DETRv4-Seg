"""Measure masks at original image resolution using supplied (oracle) boxes."""

import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .coco import decode_mask, image_path, read_json, write_json
from .infer import MaskPredictor
from .prepare import preview_grid
from .roi import roi_inputs


def evaluate(
    annotations,
    images_root,
    checkpoint,
    output,
    device="auto",
    split="val",
    batch_size=32,
    threshold=0.5,
    preview=None,
):
    start = time.monotonic()
    coco = read_json(annotations)
    grouped = defaultdict(list)
    for ann in coco["annotations"]:
        if split != "all" and ann.get("split") != split:
            continue
        if ann.get("iscrowd", 0):
            continue
        if not ann.get("segmentation"):
            raise ValueError(
                "Evaluation requires polygon or RLE masks for every selected annotation"
            )
        grouped[ann["image_id"]].append(ann)
    if not grouped:
        raise ValueError(
            "No masks selected. For a COCO file without split fields, use --split all."
        )
    images = {im["id"]: im for im in coco["images"]}
    predictor = MaskPredictor(checkpoint, device)
    records, previews, sources = [], [], Counter()
    for image_id, objects in grouped.items():
        im = images[image_id]
        bgr = cv2.imread(str(image_path(images_root, im["file_name"])))
        if bgr is None or bgr.shape[:2] != (im["height"], im["width"]):
            raise ValueError(f"Image {image_id}: missing file or dimensions mismatch")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = predictor.predict(rgb, [ann["bbox"] for ann in objects], batch_size, threshold)
        for ann, mask in zip(objects, masks):
            truth = decode_mask(ann["segmentation"], im["height"], im["width"])
            intersection = int(np.logical_and(mask, truth).sum())
            union = int(np.logical_or(mask, truth).sum())
            source = ann.get("mask_source", "ground_truth")
            sources[source] += 1
            records.append(
                {
                    "annotation_id": ann["id"],
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "mask_source": source,
                    "iou": intersection / max(1, union),
                    "dice": 2 * intersection / max(1, int(mask.sum()) + int(truth.sum())),
                    "predicted_area_px": int(mask.sum()),
                    "target_area_px": int(truth.sum()),
                }
            )
            if preview and len(previews) < 36:
                crop, _, (x0, y0, x1, y1) = roi_inputs(rgb, ann["bbox"], 96, predictor.padding)
                for label, value in (("target", truth), ("prediction", mask)):
                    target = cv2.resize(
                        value[y0:y1, x0:x1], (96, 96), interpolation=cv2.INTER_NEAREST
                    )
                    previews.append((crop, target, f"id={ann['id']} {label}"))
    report = {
        "images": len(grouped),
        "objects": len(records),
        "split": split,
        "native_mask_iou": float(np.mean([r["iou"] for r in records])),
        "native_mask_dice": float(np.mean([r["dice"] for r in records])),
        "per_category": {
            str(cat["id"]): {
                "name": cat["name"],
                "objects": sum(r["category_id"] == cat["id"] for r in records),
                "native_mask_iou": float(
                    np.mean([r["iou"] for r in records if r["category_id"] == cat["id"]])
                ),
            }
            for cat in coco["categories"]
            if any(r["category_id"] == cat["id"] for r in records)
        },
        "label_sources": dict(sources),
        "elapsed_seconds": time.monotonic() - start,
        "note": "Uses annotation boxes, not detector predictions. Pseudo-mask agreement "
        "is not accuracy against human masks. These are per-instance means, not COCO AP.",
        "instances": records,
    }
    write_json(output, report)
    if preview:
        Path(preview).parent.mkdir(parents=True, exist_ok=True)
        preview_grid(previews, preview)
    return {k: v for k, v in report.items() if k != "instances"}
