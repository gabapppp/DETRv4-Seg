"""Create an auditable pseudo-label set and a reusable small ROI cache."""

import hashlib
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .coco import decode_mask, encode_mask, image_path, read_json, validate_coco, write_json
from .roi import pseudo_mask, roi_inputs


def split_images(images, val_fraction, seed, group_map=None):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = {}
    for im in images:
        group = str(im["id"]) if group_map is None else str(group_map[str(im["id"])])
        groups.setdefault(group, []).append(im["id"])
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("At least two independent image groups are required for validation")
    random.Random(seed).shuffle(keys)
    n = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
    val_ids = {image_id for key in keys[:n] for image_id in groups[key]}
    return {im["id"]: "val" if im["id"] in val_ids else "train" for im in images}


def preview_grid(items, output, size=160):
    if not items:
        return
    columns = 6
    grid = np.full(
        (((len(items) + columns - 1) // columns) * (size + 28), columns * size, 3), 30, np.uint8
    )
    for i, (rgb, mask, text) in enumerate(items):
        rgb = cv2.resize(rgb, (size, size))
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
        rgb[mask] = (rgb[mask] * 0.55 + np.array([40, 230, 90]) * 0.45).astype(np.uint8)
        x, y = (i % columns) * size, (i // columns) * (size + 28)
        grid[y : y + size, x : x + size] = rgb
        cv2.putText(
            grid[y + size : y + size + 28, x : x + size],
            text,
            (3, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def prepare(
    annotations,
    images_root,
    output,
    size=96,
    padding=0.2,
    max_objects=3000,
    val_fraction=0.2,
    seed=42,
    grabcut=False,
    budget_seconds=420,
    groups=None,
    fill_mode="holes",
    mask_overflow=0.0,
):
    if size < 16 or padding <= 0 or max_objects < 2 or budget_seconds <= 0:
        raise ValueError("size >= 16, padding > 0, max_objects >= 2, budget_seconds > 0 required")
    if not 0 <= mask_overflow <= padding:
        raise ValueError("mask_overflow must be between zero and crop padding")
    start = time.monotonic()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}. Choose a new cache directory.")
    coco = read_json(annotations)
    audit = validate_coco(coco, images_root)
    mapping = split_images(
        coco["images"], val_fraction, seed, read_json(groups) if groups else None
    )
    output.mkdir(parents=True, exist_ok=True)
    images = {im["id"]: im for im in coco["images"]}
    candidates = list(coco["annotations"])
    random.Random(seed).shuffle(candidates)
    data = {
        split: {key: [] for key in ("rgb", "prior", "masks", "annotation_ids", "image_ids")}
        for split in ("train", "val")
    }
    kept, rejected, previews, loaded = [], [], [], {}
    timed_out = False
    for ann in candidates:
        if len(kept) >= max_objects:
            break
        if time.monotonic() - start >= budget_seconds:
            timed_out = True
            break
        im = images[ann["image_id"]]
        if ann.get("iscrowd", 0):
            rejected.append({"id": ann["id"], "reason": "crowd"})
            continue
        if im["id"] not in loaded:
            loaded[im["id"]] = cv2.cvtColor(
                cv2.imread(str(image_path(images_root, im["file_name"]))), cv2.COLOR_BGR2RGB
            )
        rgb = loaded[im["id"]]
        if ann.get("segmentation"):
            mask = decode_mask(ann["segmentation"], im["height"], im["width"])
            quality = {
                **ann.get("mask_quality", {}),
                "source": ann.get("mask_source", "ground_truth"),
            }
        else:
            # Output allowance is not a foreground label: opening clustering into
            # nearby particles contaminates the targets. Anchor pseudo-labels to
            # the annotated instance; train with perturbed bbox priors instead.
            mask, quality = pseudo_mask(rgb, ann["bbox"], padding, grabcut, seed, fill_mode)
        if mask is None:
            rejected.append({"id": ann["id"], **quality})
            continue
        crop, prior, bounds = roi_inputs(rgb, ann["bbox"], size, padding)
        x0, y0, x1, y1 = bounds
        target = cv2.resize(mask[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)
        if not target.any() or not prior.any():
            rejected.append({"id": ann["id"], "reason": "empty_resized_mask_or_bbox"})
            continue
        split = mapping[im["id"]]
        entry = data[split]
        for key, value in (
            ("rgb", crop),
            ("prior", prior),
            ("masks", target),
            ("annotation_ids", ann["id"]),
            ("image_ids", im["id"]),
        ):
            entry[key].append(value)
        kept.append(
            {
                **ann,
                "segmentation": encode_mask(mask),
                "area": int(mask.sum()),
                "mask_source": quality["source"],
                "mask_quality": quality,
                "split": split,
            }
        )
        if len(previews) < 36:
            label = fill_mode if quality["source"].startswith("pseudo") else "GT"
            previews.append((crop, target, f"id={ann['id']} {split} {label}"))
    settings = {
        "roi_size": size,
        "padding": padding,
        "mask_overflow": mask_overflow,
        "pseudo_labels_clipped_to_annotation_bbox": True,
        "seed": seed,
        "grabcut": grabcut,
        "max_objects": max_objects,
        "val_fraction": val_fraction,
        "annotations": str(Path(annotations).resolve()),
        "annotations_sha256": hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
        "images_root": str(Path(images_root).resolve()),
        "group_file": str(Path(groups).resolve()) if groups else None,
        "fill_mode": fill_mode,
    }
    report = {
        "audit": audit,
        "settings": settings,
        "accepted": len(kept),
        "train_objects": len(data["train"]["rgb"]),
        "val_objects": len(data["val"]["rgb"]),
        "sources": dict(Counter(a["mask_source"] for a in kept)),
        "rejected": len(rejected),
        "rejection_reasons": dict(Counter(a["reason"] for a in rejected)),
        "rejected_annotations": rejected,
        "budget_exhausted": timed_out,
        "unprocessed": len(candidates) - len(kept) - len(rejected),
        "elapsed_seconds": time.monotonic() - start,
        "image_splits": {str(k): v for k, v in mapping.items()},
        "validation_note": "Metrics on pseudo-masks measure agreement with the label generator; "
        "they do not measure true segmentation accuracy.",
    }
    write_json(output / "report.json", report)
    write_json(output / "instances_masks.json", {**coco, "annotations": kept})
    preview_grid(previews, output / "preview.jpg")
    for split, entry in data.items():
        if not entry["rgb"]:
            raise ValueError(
                f"No accepted {split} masks. Increase subset/budget or review {output / 'report.json'}."
            )
    for split, entry in data.items():
        np.savez_compressed(output / f"{split}.npz", **{k: np.asarray(v) for k, v in entry.items()})
    report["elapsed_seconds"] = time.monotonic() - start
    write_json(output / "report.json", report)
    return report
