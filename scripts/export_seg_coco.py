#!/usr/bin/env python3
"""Export a reproducible COCO mask-review subset, without running a detector.

Run this script yourself to read the supplied annotations/images. Nothing is
read from a default dataset location. Existing files are never overwritten.
"""

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api

from detrseg.coco import (
    decode_mask,
    encode_mask,
    image_path,
    read_json,
    single_class_coco,
    write_json,
)
from detrseg.roi import bbox_prior, pseudo_mask, roi_bounds


def mask_segmentation(mask, encoding="polygons"):
    """Trace pixel-cell edges, so polygon rasterization preserves small masks.

    COCO polygon unions cannot subtract holes. Use exact RLE for those cases,
    or whenever the polygon round trip differs from the supplied binary mask.
    """
    mask = np.asarray(mask > 0, dtype=np.uint8)
    height, width = mask.shape
    if not mask.any():
        raise ValueError("Cannot export an empty instance mask")
    if encoding == "rle":
        return encode_mask(mask), "rle_requested"
    yy, xx = np.nonzero(mask)
    left, top = int(xx.min()), int(yy.min())
    crop = mask[top : int(yy.max()) + 1, left : int(xx.max()) + 1]
    padded = np.pad(crop, 1)
    edges = defaultdict(set)
    for y, x in zip(*np.nonzero(crop)):
        x, y = int(x), int(y)
        if not padded[y, x + 1]:
            edges[(x, y)].add((x + 1, y))
        if not padded[y + 1, x + 2]:
            edges[(x + 1, y)].add((x + 1, y + 1))
        if not padded[y + 2, x + 1]:
            edges[(x + 1, y + 1)].add((x, y + 1))
        if not padded[y + 1, x]:
            edges[(x, y + 1)].add((x, y))
    directions = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    polygons = []
    while edges:
        start = min(edges)
        current = start
        direction = None
        points = []
        while True:
            points.append(current)
            choices = edges[current]
            if direction is None:
                target = min(choices)
            else:
                # At a diagonal contact, follow the right turn to keep the two
                # foreground pixel components as separate closed polygons.
                priority = {1: 0, 0: 1, 3: 2, 2: 3}
                target = min(
                    choices,
                    key=lambda p: priority[
                        (directions[(p[0] - current[0], p[1] - current[1])] - direction) % 4
                    ],
                )
            choices.remove(target)
            if not choices:
                del edges[current]
            direction = directions[(target[0] - current[0], target[1] - current[1])]
            current = target
            if current == start:
                break
        area_twice = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
        if area_twice < 0:
            return encode_mask(mask), "holes_preserved_as_rle"
        vertices = []
        for index, point in enumerate(points):
            before, after = points[index - 1], points[(index + 1) % len(points)]
            if (point[0] - before[0]) * (after[1] - point[1]) != (point[1] - before[1]) * (
                after[0] - point[0]
            ):
                vertices.extend([point[0] + left, point[1] + top])
        polygons.append(vertices)
    restored = mask_api.decode(mask_api.merge(mask_api.frPyObjects(polygons, height, width)))
    if not np.array_equal(restored, mask):
        return encode_mask(mask), "polygon_roundtrip_fallback_rle"
    return polygons, "polygons"


def annotation_index(coco):
    if not isinstance(coco, dict):
        raise TypeError("COCO input must be a JSON object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list):
            raise TypeError(f"COCO field must be a list: {key}")
        ids = [item["id"] for item in coco[key]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate {key} IDs")
    image_ids = {im["id"] for im in coco["images"]}
    category_ids = {cat["id"] for cat in coco["categories"]}
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["image_id"] not in image_ids or ann["category_id"] not in category_ids:
            raise ValueError(f"Unknown image/category in annotation {ann['id']}")
        by_image[ann["image_id"]].append(ann)
    return by_image


def instance_key(im, ann):
    """Match reviewed instances across split filename rewrites and COCO ID changes."""
    filename = im.get("source_file_name") or im.get("input_file_name") or im["file_name"]
    bbox = ann.get("source_bbox", ann["bbox"])
    return str(filename).replace("\\", "/"), tuple(round(float(value), 4) for value in bbox)


def export_review(
    annotations,
    images_root,
    output,
    count=100,
    mode="roi",
    seed=42,
    mask_checkpoint=None,
    device="cpu",
    padding=0.25,
    fill_mode="convex",
    encoding="polygons",
    make_zip=False,
    exclude_reviewed=None,
    require_count=False,
    single_class=False,
    class_name="ball",
):
    if count < 1 or padding < 0 or not np.isfinite(padding):
        raise ValueError("count must be positive and padding finite/nonnegative")
    if mode not in ("images", "roi") or encoding not in ("polygons", "rle"):
        raise ValueError("Invalid mode or segmentation encoding")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    zip_path = output.with_name(output.name + ".zip")
    if make_zip and zip_path.exists():
        raise ValueError(f"ZIP already exists: {zip_path}")
    coco = read_json(annotations)
    if single_class:
        coco = single_class_coco(coco, class_name)
    by_image = annotation_index(coco)
    excluded = set()
    for path in exclude_reviewed or []:
        reviewed = read_json(path)
        annotation_index(reviewed)
        reviewed_images = {im["id"]: im for im in reviewed["images"]}
        excluded.update(
            instance_key(reviewed_images[ann["image_id"]], ann) for ann in reviewed["annotations"]
        )
    excluded_count = 0
    for im in coco["images"]:
        old = by_image[im["id"]]
        by_image[im["id"]] = [ann for ann in old if instance_key(im, ann) not in excluded]
        excluded_count += len(old) - len(by_image[im["id"]])
    selected_images = [im for im in coco["images"] if by_image[im["id"]]]
    random.Random(seed).shuffle(selected_images)
    selections = []
    if mode == "images":
        selections = [(im, by_image[im["id"]]) for im in selected_images[:count]]
    else:
        # Round-robin across shuffled images, so crowded images cannot dominate
        # a small ROI review subset. Each chosen object is exported once.
        queues = []
        rng = random.Random(seed)
        for im in selected_images:
            anns = [ann for ann in by_image[im["id"]] if not ann.get("iscrowd", 0)]
            rng.shuffle(anns)
            queues.append((im, anns))
        while len(selections) < count and any(anns for _, anns in queues):
            for im, anns in queues:
                if anns and len(selections) < count:
                    selections.append((im, [anns.pop()]))
    if not selections:
        raise ValueError("No annotated images/instances available for mask review")
    if require_count and len(selections) != count:
        raise ValueError(
            f"Requested {count} new {mode} samples, but only {len(selections)} remain after exclusions"
        )
    predictor = None
    if mask_checkpoint:
        from detrseg.infer import MaskPredictor

        predictor = MaskPredictor(mask_checkpoint, device)
    image_dir = output / "images/default"
    image_dir.mkdir(parents=True, exist_ok=True)
    exported_images, exported_anns, rows = [], [], []
    sources, encodings = Counter(), Counter()
    last_image_id, rgb = None, None
    for index, (im, anns) in enumerate(selections, 1):
        source = image_path(images_root, im["file_name"])
        if last_image_id != im["id"]:
            with Image.open(source) as image:
                if image.size != (im["width"], im["height"]):
                    raise ValueError(f"Image dimensions do not match COCO: {im['id']}")
                rgb = np.asarray(image.convert("RGB")).copy()
            last_image_id = im["id"]
        height, width = rgb.shape[:2]
        exported_id = im["id"] if mode == "images" else index
        name = f"{index:06d}_{source.name}" if mode == "images" else f"roi_{index:06d}.png"
        full_bounds = (0, 0, width, height)
        bounds = full_bounds
        if mode == "images":
            shutil.copy2(source, image_dir / name)
        for ann in anns:
            bbox = list(map(float, ann["bbox"]))
            roi_bounds(bbox, width, height, padding)  # Validate finite, intersecting bbox.
            reason = ""
            if ann.get("segmentation"):
                mask = decode_mask(ann["segmentation"], height, width)
                mask_source = "existing_segmentation"
            elif predictor is not None:
                mask = predictor.predict(rgb, [bbox])[0]
                mask_source = "model_draft"
            else:
                mask, quality = pseudo_mask(
                    rgb, bbox, padding=max(padding, 0.01), seed=seed, fill_mode=fill_mode
                )
                reason = quality.get("reason", "")
                mask_source = "pseudo_draft"
            if mask is None or not mask.any():
                # Preserve the instance as an explicitly marked placeholder,
                # rather than silently dropping boxes that need manual masks.
                mask = bbox_prior(bbox, full_bounds)
                mask_source = "bbox_placeholder"
                reason = reason or "empty_model_mask"
            if not mask.any():
                raise ValueError(f"Annotation {ann['id']} has no foreground pixels inside image")
            if mode == "roi":
                x0, y0, x1, y1 = roi_bounds(bbox, width, height, padding)
                yy, xx = np.nonzero(mask)
                # Retain the whole supplied mask even if it extends beyond its bbox.
                bounds = (
                    min(x0, int(xx.min())),
                    min(y0, int(yy.min())),
                    max(x1, int(xx.max()) + 1),
                    max(y1, int(yy.max()) + 1),
                )
                x0, y0, x1, y1 = bounds
                Image.fromarray(rgb[y0:y1, x0:x1]).save(image_dir / name)
                mask = mask[y0:y1, x0:x1]
            segmentation, encoding_reason = mask_segmentation(
                mask, "rle" if ann.get("iscrowd", 0) else encoding
            )
            rle = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
            attrs = {
                **ann.get("attributes", {}),
                "mask_review_status": "pending",
                "mask_source": mask_source,
            }
            exported_anns.append(
                {
                    **{
                        k: v
                        for k, v in ann.items()
                        if k not in ("keypoints", "num_keypoints", "split")
                    },
                    "id": ann["id"],
                    "image_id": exported_id,
                    "segmentation": segmentation,
                    "bbox": mask_api.toBbox(rle).tolist(),
                    "area": int(mask_api.area(rle)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "attributes": attrs,
                    "mask_source": mask_source,
                    "mask_review_status": "pending",
                    "source_image_id": im["id"],
                    "source_annotation_id": ann["id"],
                    "source_bbox": bbox,
                    "source_crop_xyxy": list(bounds),
                }
            )
            rows.append(
                {
                    "file_name": name,
                    "image_id": exported_id,
                    "annotation_id": ann["id"],
                    "source_image_id": im["id"],
                    "source_annotation_id": ann["id"],
                    "source_file_name": im["file_name"],
                    "source_crop_xyxy": json.dumps(bounds),
                    "source_bbox": json.dumps(bbox),
                    "mask_source": mask_source,
                    "encoding": encoding_reason,
                    "reason": reason,
                    "review_status": "pending",
                }
            )
            sources[mask_source] += 1
            encodings[encoding_reason] += 1
        x0, y0, x1, y1 = bounds
        exported_images.append(
            {
                **im,
                "id": exported_id,
                "file_name": name,
                "width": x1 - x0,
                "height": y1 - y0,
                "source_image_id": im["id"],
                "source_file_name": im["file_name"],
                "source_crop_xyxy": list(bounds),
            }
        )
        print(f"export {index}/{len(selections)}: {len(anns)} instance(s)", flush=True)
    dataset = {
        **{k: v for k, v in coco.items() if k not in ("images", "annotations")},
        "images": exported_images,
        "annotations": exported_anns,
    }
    dataset.setdefault("info", {})
    dataset.setdefault("licenses", [])
    write_json(output / "annotations/instances_default.json", dataset)
    with (output / "review.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "mode": mode,
        "requested_count": count,
        "images": len(exported_images),
        "instances": len(exported_anns),
        "seed": seed,
        "mask_sources": dict(sources),
        "encodings": dict(encodings),
        "all_masks_require_review": True,
        "excluded_instances": excluded_count,
        "single_class": single_class,
        "exclude_reviewed": [str(Path(path).resolve()) for path in exclude_reviewed or []],
        "source_annotations": str(Path(annotations).resolve()),
        "mask_checkpoint": str(Path(mask_checkpoint).resolve()) if mask_checkpoint else None,
        "note": "Only existing annotations are exported; unannotated objects are not inferred. "
        "Images may be incompletely annotated. Bbox placeholders require manual drawing.",
    }
    write_json(output / "report.json", report)
    (output / "README.txt").write_text(
        "MASK REVIEW DRAFT - not verified ground truth.\n"
        "Images: images/default/; COCO instances: annotations/instances_default.json.\n"
        "All known annotations in the selected images are retained (ROI mode: selected instances).\n"
        "Draw/correct every target mask. bbox_placeholder entries are rectangles to replace.\n"
        "review.csv tracks sources and pending review. Holes use RLE to preserve geometry.\n"
        "source_crop_xyxy / review.csv map ROI coordinates back to their original image.\n"
        "Annotation coverage may be partial: do not treat unlabeled objects as background.\n",
        encoding="utf-8",
    )
    if make_zip:
        with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(output.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(output).as_posix())
        report["zip"] = str(zip_path)
    return {**report, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", dest="images_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--count", type=int, default=100, help="Images in images mode; instances in roi mode"
    )
    parser.add_argument("--mode", choices=("images", "roi"), default="roi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mask-checkpoint", help="Optional ROI mask .pt; existing segmentations take priority"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--fill-mode", choices=("holes", "convex"), default="convex")
    parser.add_argument("--encoding", choices=("polygons", "rle"), default="polygons")
    parser.add_argument("--zip", dest="make_zip", action="store_true")
    parser.add_argument(
        "--exclude-reviewed",
        nargs="+",
        help="Previous review COCO exports to exclude using original filename/bbox",
    )
    parser.add_argument(
        "--require-count",
        action="store_true",
        help="Stop before reading images if fewer than --count new samples exist",
    )
    parser.add_argument(
        "--single-class", action="store_true", help="Export one merged label for all instances"
    )
    parser.add_argument("--class-name", default="ball")
    args = vars(parser.parse_args(argv))
    try:
        print(json.dumps(export_review(**args), ensure_ascii=False, indent=2))
    except (TypeError, ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
