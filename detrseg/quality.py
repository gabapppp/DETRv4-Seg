"""Local, label-free mask stability diagnostics. These are not calibrated accuracy probabilities."""

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .coco import image_path, read_json, write_json
from .infer import MaskPredictor
from .pipeline import bundle_paths


def binary_iou(first, second):
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def perturb_box(box, scale=1.0, dx=0.0, dy=0.0):
    x, y, w, h = box
    return [
        x + (1 - scale) * w / 2 + dx * w,
        y + (1 - scale) * h / 2 + dy * h,
        w * scale,
        h * scale,
    ]


def summarize_values(values):
    array = np.asarray(values, dtype=float)
    return (
        {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p10": float(np.quantile(array, 0.1)),
            "minimum": float(array.min()),
        }
        if len(array)
        else None
    )


def pairwise_overlap(masks):
    """Measure shared pixels; an overlap alone does not prove either instance is incorrect."""
    largest = np.zeros(len(masks), dtype=float)
    pairs = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            intersection = int(np.logical_and(masks[first], masks[second]).sum())
            if intersection:
                fraction = intersection / max(
                    1, min(int(masks[first].sum()), int(masks[second].sum()))
                )
                largest[first] = max(largest[first], fraction)
                largest[second] = max(largest[second], fraction)
                pairs.append(
                    {
                        "first_index": first,
                        "second_index": second,
                        "intersection_px": intersection,
                        "intersection_over_smaller": fraction,
                    }
                )
    return largest, pairs


def audit_masks(
    bundle,
    annotations,
    images_root,
    detections,
    output,
    device="auto",
    score_threshold=0.5,
    batch_size=32,
    max_objects=1000,
):
    if max_objects <= 0 or not 0 <= score_threshold <= 1:
        raise ValueError("Positive max_objects and a valid score threshold required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}")
    state, paths = bundle_paths(bundle)
    predictor = MaskPredictor(paths["mask_checkpoint"], device)
    threshold = state.get("mask_threshold", 0.5)
    coco = read_json(annotations)
    images = {im["id"]: im for im in coco["images"]}
    grouped = defaultdict(list)
    selected = [p for p in read_json(detections) if p["score"] >= score_threshold]
    # Deterministic ordering, not selecting just the most confident/easiest detections.
    selected.sort(key=lambda p: (p["image_id"], p["category_id"], *p["bbox"]))
    for index, obj in enumerate(selected[:max_objects]):
        if obj["image_id"] not in images:
            raise ValueError("Detection image_id is absent from annotations")
        grouped[obj["image_id"]].append((index, obj))
    if not grouped:
        raise ValueError("No detections selected for the mask audit")
    output.mkdir(parents=True, exist_ok=True)
    records, overlap_pairs = [], []
    for image_id, entries in grouped.items():
        im = images[image_id]
        bgr = cv2.imread(str(image_path(images_root, im["file_name"])))
        if bgr is None or bgr.shape[:2] != (im["height"], im["width"]):
            raise ValueError(f"Image {image_id}: missing file or dimensions mismatch")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = [obj["bbox"] for _, obj in entries]
        masks, qualities = predictor.predict(rgb, boxes, batch_size, threshold, diagnostics=True)
        flipped_boxes = [[im["width"] - x - w, y, w, h] for x, y, w, h in boxes]
        flipped = predictor.predict(
            np.ascontiguousarray(rgb[:, ::-1]), flipped_boxes, batch_size, threshold
        )
        variants = {"horizontal_flip": [mask[:, ::-1] for mask in flipped]}
        for name, parameters in (
            ("box_scale_090", {"scale": 0.9}),
            ("box_scale_110", {"scale": 1.1}),
            ("shift_left", {"dx": -0.05}),
            ("shift_right", {"dx": 0.05}),
            ("shift_up", {"dy": -0.05}),
            ("shift_down", {"dy": 0.05}),
        ):
            changed = []
            for box in boxes:
                value = perturb_box(box, **parameters)
                # Keep edge boxes intersecting the image without changing their size.
                value[0] = float(np.clip(value[0], -value[2] + 1, im["width"] - 1))
                value[1] = float(np.clip(value[1], -value[3] + 1, im["height"] - 1))
                changed.append(value)
            variants[name] = predictor.predict(rgb, changed, batch_size, threshold)
        largest, pairs = pairwise_overlap(masks)
        for pair in pairs:
            overlap_pairs.append(
                {
                    "image_id": image_id,
                    "first_record": entries[pair["first_index"]][0],
                    "second_record": entries[pair["second_index"]][0],
                    **pair,
                }
            )
        for local_index, ((index, obj), mask, quality) in enumerate(zip(entries, masks, qualities)):
            stability = {
                name: binary_iou(mask, values[local_index]) for name, values in variants.items()
            }
            minimum = min(stability.values())
            flags = []
            if quality["area_px"] == 0:
                flags.append("empty_mask")
            if minimum < 0.85:
                flags.append("sensitive_to_flip_or_bbox")
            if largest[local_index] >= 0.1:
                flags.append("shares_pixels_with_another_instance")
            if quality["support_boundary_touch"]:
                flags.append("touches_output_limit")
            if quality["postprocess_added_fraction"] > 0.2:
                flags.append("shape_prior_adds_many_pixels")
            records.append(
                {
                    "record_id": index,
                    "image_id": image_id,
                    "category_id": obj["category_id"],
                    "bbox": obj["bbox"],
                    "detector_score": obj["score"],
                    **quality,
                    "stability_iou": stability,
                    "minimum_stability_iou": minimum,
                    "mean_stability_iou": float(np.mean(list(stability.values()))),
                    "max_overlap_over_smaller": float(largest[local_index]),
                    "review_flags": flags,
                }
            )
        print(f"audit image={image_id} instances={len(entries)}", flush=True)
    report = {
        "images": len(grouped),
        "instances": len(records),
        "total_eligible_instances": len(selected),
        "truncated": len(records) < len(selected),
        "mask_threshold": threshold,
        "score_threshold": score_threshold,
        "mask_architecture": predictor.architecture,
        "padding": predictor.padding,
        "mask_overflow": predictor.mask_overflow,
        "calibrated_to_human_masks": False,
        "minimum_stability_iou": summarize_values([r["minimum_stability_iou"] for r in records]),
        "mean_stability_iou": summarize_values([r["mean_stability_iou"] for r in records]),
        "per_perturbation": {
            name: summarize_values([r["stability_iou"][name] for r in records])
            for name in records[0]["stability_iou"]
        },
        "empty_masks": sum(r["area_px"] == 0 for r in records),
        "review_flagged_instances": sum(bool(r["review_flags"]) for r in records),
        "instances_below_minimum_stability_085": sum(
            r["minimum_stability_iou"] < 0.85 for r in records
        ),
        "instances_with_overlap_ge_010": sum(r["max_overlap_over_smaller"] >= 0.1 for r in records),
        "instances_touching_output_limit": sum(r["support_boundary_touch"] for r in records),
        "mean_postprocess_added_fraction": float(
            np.mean([r["postprocess_added_fraction"] for r in records])
        ),
        "note": "Label-free consistency diagnostics, not accuracy or the probability a mask is correct. "
        "Review thresholds 0.85/0.10/0.20 are exploratory heuristics, not calibrated acceptance limits.",
        "records": records,
        "overlap_pairs": overlap_pairs,
    }
    write_json(output / "audit.json", report)
    fields = (
        "record_id",
        "image_id",
        "category_id",
        "detector_score",
        "area_px",
        "minimum_stability_iou",
        "mean_stability_iou",
        "max_overlap_over_smaller",
        "postprocess_added_fraction",
        "support_boundary_touch",
        "review_flags",
    )
    with (output / "review.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in sorted(
            records, key=lambda r: (not bool(r["review_flags"]), r["minimum_stability_iou"])
        ):
            writer.writerow(
                {
                    name: ";".join(record[name]) if name == "review_flags" else record[name]
                    for name in fields
                }
            )
    return {k: v for k, v in report.items() if k not in ("records", "overlap_pairs")}
