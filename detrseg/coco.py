"""COCO validation, CVAT conversion, and lossless mask serialization."""

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def image_path(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes images root: {filename}")
    return path


def single_class_coco(coco, name="ball", target_id=1):
    """Make one semantic class without merging or deleting distinct instances."""
    if not name.strip() or target_id < 0:
        raise ValueError("Single class needs a nonempty name and nonnegative ID")
    categories = coco["categories"]
    ids = {category["id"] for category in categories}
    if not ids or len(ids) != len(categories):
        raise ValueError("COCO categories must have unique IDs and contain at least one class")
    if any(ann["category_id"] not in ids for ann in coco["annotations"]):
        raise ValueError("Annotation category ID is absent from categories")
    return {
        **coco,
        "categories": [{"id": target_id, "name": name.strip(), "supercategory": "object"}],
        "annotations": [{**ann, "category_id": target_id} for ann in coco["annotations"]],
    }


def encode_mask(mask):
    rle = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def decode_mask(segmentation, height, width):
    if not segmentation:
        raise ValueError("Empty segmentation")
    if isinstance(segmentation, list):
        if any(len(p) < 6 or len(p) % 2 or not np.isfinite(p).all() for p in segmentation):
            raise ValueError("Invalid COCO polygon")
        rle = mask_api.merge(mask_api.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        if segmentation.get("size") != [height, width]:
            raise ValueError("RLE size does not match image")
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            if any(not isinstance(n, int) or n < 0 for n in counts):
                raise ValueError("Invalid uncompressed RLE counts")
            if sum(counts) != height * width:
                raise ValueError("RLE counts do not cover image")
            rle = mask_api.frPyObjects(segmentation, height, width)
        elif isinstance(counts, (str, bytes)):
            rle = segmentation
        else:
            raise TypeError("Invalid RLE counts")
    else:
        raise TypeError("Segmentation must be polygons or RLE")
    result = mask_api.decode(rle)
    if result.shape != (height, width) or not result.any():
        raise ValueError("Mask is empty or has incorrect dimensions")
    return result.astype(np.uint8)


def validate_coco(coco, images_root=None):
    """Fail on inconsistent annotations; count decoded masks, not truthy JSON."""
    images, annotations, categories = (coco[key] for key in ("images", "annotations", "categories"))
    for name, entries in (
        ("images", images),
        ("annotations", annotations),
        ("categories", categories),
    ):
        ids = [item["id"] for item in entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {name} IDs")
    by_image = {im["id"]: im for im in images}
    cat_ids = {cat["id"] for cat in categories}
    errors, counts = [], Counter()
    for im in images:
        if im["width"] <= 0 or im["height"] <= 0:
            errors.append(f"Image {im['id']}: invalid dimensions")
        if images_root is not None:
            try:
                with Image.open(image_path(images_root, im["file_name"])) as loaded:
                    if loaded.size != (im["width"], im["height"]):
                        errors.append(f"Image {im['id']}: file dimensions mismatch")
                    loaded.verify()
            except (OSError, ValueError) as exc:
                errors.append(f"Image {im['id']}: {exc}")
    for ann in annotations:
        try:
            im = by_image[ann["image_id"]]
            if ann["category_id"] not in cat_ids:
                raise ValueError("Unknown category_id")
            x, y, w, h = ann["bbox"]
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError("Invalid bbox")
            if x >= im["width"] or y >= im["height"] or x + w <= 0 or y + h <= 0:
                raise ValueError("BBox is outside image")
            if ann.get("segmentation"):
                decode_mask(ann["segmentation"], im["height"], im["width"])
                counts["with_masks"] += 1
            else:
                counts["without_masks"] += 1
            counts["crowd"] += bool(ann.get("iscrowd", 0))
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            errors.append(f"Annotation {ann.get('id')}: {exc}")
    if errors:
        raise ValueError("COCO validation failed:\n" + "\n".join(errors[:30]))
    return {
        "images": len(images),
        "objects": len(annotations),
        "with_masks": counts["with_masks"],
        "without_masks": counts["without_masks"],
        "crowd": counts["crowd"],
        "categories": [
            dict(cat, objects=sum(a["category_id"] == cat["id"] for a in annotations))
            for cat in categories
        ],
    }


def convert_cvat(xml_path, images_root):
    """Convert image-level CVAT boxes/polygons; reject unsupported shapes visibly."""
    root = ET.parse(xml_path).getroot()
    images = root.findall("image")
    if not images:
        raise ValueError("Expected CVAT image annotations, not a video track export")
    shapes = [shape for im in images for shape in im]
    unsupported = {s.tag for s in shapes} - {"box", "polygon"}
    if unsupported:
        raise ValueError(f"Unsupported CVAT shape types: {sorted(unsupported)}")
    names = sorted({shape.attrib["label"] for shape in shapes})
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]
    cat_map = {cat["name"]: cat["id"] for cat in categories}
    coco = {"images": [], "annotations": [], "categories": categories}
    for element in images:
        im = {
            "id": int(element.attrib["id"]),
            "file_name": element.attrib["name"],
            "width": int(element.attrib["width"]),
            "height": int(element.attrib["height"]),
        }
        coco["images"].append(im)
        for shape in element:
            attrs = shape.attrib
            segmentation = []
            if shape.tag == "box":
                if float(attrs.get("rotation", 0)) != 0:
                    raise ValueError("Rotated CVAT boxes require explicit conversion first")
                x, y, right, bottom = (float(attrs[k]) for k in ("xtl", "ytl", "xbr", "ybr"))
            else:
                points = [
                    list(map(float, point.split(","))) for point in attrs["points"].split(";")
                ]
                x, y = np.min(points, axis=0)
                right, bottom = np.max(points, axis=0)
                segmentation = [[v for point in points for v in point]]
            if right <= x or bottom <= y:
                raise ValueError(f"Invalid CVAT box in image {im['id']}")
            x, y = max(0.0, float(x)), max(0.0, float(y))
            right, bottom = min(im["width"], float(right)), min(im["height"], float(bottom))
            area = (right - x) * (bottom - y)
            if segmentation:
                area = int(decode_mask(segmentation, im["height"], im["width"]).sum())
            crowd = any(
                a.attrib.get("name") == "is_crowd" and (a.text or "").lower() in ("true", "1")
                for a in shape
            )
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]) + 1,
                    "image_id": im["id"],
                    "category_id": cat_map[attrs["label"]],
                    "bbox": [x, y, right - x, bottom - y],
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": int(crowd),
                }
            )
    validate_coco(coco, images_root)
    return coco
