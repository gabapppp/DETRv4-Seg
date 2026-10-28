"""COCO AP for actual detector outputs; mask AP requires supplied reference masks."""

import contextlib
import io
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_metrics(coco, predictions, kind="bbox"):
    if kind not in ("bbox", "segm"):
        raise ValueError("kind must be bbox or segm")
    if not coco["images"] or not coco["annotations"]:
        raise ValueError("COCO evaluation needs images and reference annotations")
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0, "predictions": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {"info": {}, **coco}
        ground_truth.createIndex()
        # COCO selects bbox geometry if bbox is present, even for segm; strip it for mask AP.
        values = (
            predictions
            if kind == "bbox"
            else [
                {k: p[k] for k in ("image_id", "category_id", "score", "segmentation")}
                for p in predictions
            ]
        )
        results = ground_truth.loadRes(values)
        evaluator = COCOeval(ground_truth, results, kind)
        evaluator.params.imgIds = [im["id"] for im in coco["images"]]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        **{
            name: float(evaluator.stats[index])
            for name, index in (("AP", 0), ("AP50", 1), ("AP75", 2), ("AR100", 8))
        },
        "predictions": len(predictions),
    }


def detection_at_threshold(coco, predictions, threshold=0.5, iou_threshold=0.5):
    """Class-aware one-to-one matching; expose misses and false alarms at an operating score."""
    if not 0 <= threshold <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("Invalid detection score or IoU threshold")
    truth, groups = defaultdict(list), defaultdict(list)
    for ann in coco["annotations"]:
        if not ann.get("iscrowd", 0):
            truth[(ann["image_id"], ann["category_id"])].append(ann["bbox"])
    for obj in predictions:
        if obj["score"] >= threshold:
            groups[(obj["image_id"], obj["category_id"])].append(obj)
    counts = defaultdict(lambda: np.zeros(3, dtype=int))
    for key in truth.keys() | groups.keys():
        targets = np.asarray(truth[key], dtype=float).reshape(-1, 4)
        matched = set()
        for obj in sorted(groups[key], key=lambda p: -p["score"]):
            box = np.asarray(obj["bbox"], dtype=float)
            low = np.maximum(targets[:, :2], box[:2])
            high = np.minimum(targets[:, :2] + targets[:, 2:], box[:2] + box[2:])
            intersection = np.maximum(0.0, high - low).prod(axis=1)
            ious = intersection / np.maximum(
                1e-9, targets[:, 2:].prod(axis=1) + box[2:].prod() - intersection
            )
            for index in matched:
                ious[index] = -1
            index = int(ious.argmax()) if len(ious) else -1
            if index >= 0 and ious[index] >= iou_threshold:
                matched.add(index)
                counts[key[1]][0] += 1
            else:
                counts[key[1]][1] += 1
        counts[key[1]][2] += len(targets) - len(matched)

    def summarize(values):
        tp, fp, fn = map(int, values)
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        return {
            "true_positives": tp,
            "false_positives": fp,
            "missed_objects": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-9, precision + recall),
        }

    return {
        "score_threshold": threshold,
        "iou_threshold": iou_threshold,
        "overall": summarize(sum(counts.values(), np.zeros(3, dtype=int))),
        "per_category": {
            str(cat["id"]): {"name": cat["name"], **summarize(counts[cat["id"]])}
            for cat in coco["categories"]
        },
    }
