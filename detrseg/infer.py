"""Attach native-resolution masks to COCO-format detector predictions."""

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from .coco import encode_mask, image_path, read_json, write_json
from .model import make_mask_head
from .roi import bbox_prior, clean_component, mask_support, restore_roi, roi_inputs
from .train import select_device


class MaskPredictor:
    def __init__(self, checkpoint, device="auto"):
        self.device = select_device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported mask checkpoint format")
        self.size, self.padding = state["roi_size"], state["padding"]
        self.roi_resize_mode = state.get("roi_resize_mode", "resize")
        if self.roi_resize_mode not in ("resize", "native"):
            raise ValueError("Unsupported checkpoint ROI resize mode")
        self.mask_overflow = state.get("mask_overflow", 0.0)
        if not 0 <= self.mask_overflow <= self.padding:
            raise ValueError("Checkpoint mask overflow exceeds crop padding")
        self.fill_mode = state.get("fill_mode", "holes")
        self.architecture = state.get("architecture", "tiny")
        self.model = make_mask_head(self.architecture, state["width"], pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model = self.model.eval().to(self.device)

    @torch.inference_mode()
    def predict(self, rgb, boxes, batch_size=32, threshold=0.5, diagnostics=False):
        """Return uint8 masks at image resolution, in the original box order."""
        if batch_size <= 0 or not 0 < threshold < 1:
            raise ValueError("batch_size > 0 and 0 < threshold < 1 required")
        masks, qualities = [], []
        for start in range(0, len(boxes), batch_size):
            selected = boxes[start : start + batch_size]
            crops = [
                roi_inputs(rgb, box, self.size, self.padding, self.roi_resize_mode)
                for box in selected
            ]
            if any(not prior.any() for _, prior, _ in crops):
                raise ValueError("A detection bbox is too small to form a pixel mask")
            inputs = np.stack(
                [
                    np.concatenate((crop.astype(np.float32) / 255, prior[..., None]), 2).transpose(
                        2, 0, 1
                    )
                    for crop, prior, _ in crops
                ]
            )
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                probabilities = self.model(torch.from_numpy(inputs).to(self.device)).sigmoid()
            probabilities = probabilities[:, 0].float().cpu().numpy()
            for box, (_, _, bounds), probability in zip(selected, crops, probabilities):
                x0, y0, x1, y1 = bounds
                # Resize probabilities before thresholding to preserve sub-ROI edge information.
                restored = restore_roi(probability, bounds, self.roi_resize_mode)
                prior = bbox_prior(box, bounds)
                support = mask_support(box, bounds, self.mask_overflow)
                component, _ = clean_component(
                    restored >= threshold,
                    prior,
                    0,
                    float(support.sum() / max(1, prior.sum())),
                    fill_mode=self.fill_mode,
                    support=support,
                )
                mask = np.zeros(rgb.shape[:2], np.uint8)
                if component is not None:
                    mask[y0:y1, x0:x1] = component
                masks.append(mask)
                if diagnostics:
                    final = np.zeros_like(prior) if component is None else component
                    raw = (restored >= threshold) & (support > 0)
                    area = int(final.sum())
                    border = (support > 0) & (
                        cv2.erode(
                            support,
                            np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        == 0
                    )
                    qualities.append(
                        {
                            "area_px": area,
                            "mean_pixel_activation_uncalibrated": float(restored[final > 0].mean())
                            if area
                            else None,
                            "postprocess_added_fraction": float(
                                ((final > 0) & ~raw).sum() / max(1, area)
                            ),
                            "foreground_discarded_fraction": float(
                                (raw & (final == 0)).sum() / max(1, int(raw.sum()))
                            ),
                            "support_boundary_touch": bool(np.any((final > 0) & border)),
                            "outside_bbox_fraction": float(
                                ((final > 0) & (prior == 0)).sum() / max(1, area)
                            ),
                            "area_over_bbox_pixels": float(area / max(1, int(prior.sum()))),
                            "threshold_ambiguity_fraction": float(
                                (np.abs(restored[support > 0] - threshold) < 0.1).mean()
                            )
                            if support.any()
                            else 0.0,
                        }
                    )
        return (masks, qualities) if diagnostics else masks


def infer(
    annotations,
    images_root,
    detections,
    checkpoint,
    output,
    device="auto",
    batch_size=32,
    threshold=0.5,
    score_threshold=0.25,
    overlays=None,
):
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1")
    coco, predictions = read_json(annotations), read_json(detections)
    if not isinstance(predictions, list):
        raise TypeError("Detections must be a COCO result list")
    images = {im["id"]: im for im in coco["images"]}
    categories = {cat["id"] for cat in coco["categories"]}
    grouped = defaultdict(list)
    for detection in predictions:
        if detection["image_id"] not in images or detection["category_id"] not in categories:
            raise ValueError(
                "Unknown image/category ID in detections. Check the detector label map."
            )
        if not math.isfinite(detection["score"]) or not 0 <= detection["score"] <= 1:
            raise ValueError("Invalid detection score")
        if detection["score"] >= score_threshold:
            grouped[detection["image_id"]].append(detection)
    predictor = MaskPredictor(checkpoint, device)
    results = []
    if overlays:
        Path(overlays).mkdir(parents=True, exist_ok=True)
    for image_id, objects in grouped.items():
        im = images[image_id]
        bgr = cv2.imread(str(image_path(images_root, im["file_name"])))
        if bgr is None or bgr.shape[:2] != (im["height"], im["width"]):
            raise ValueError(f"Image {image_id}: missing file or dimensions mismatch")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = predictor.predict(rgb, [obj["bbox"] for obj in objects], batch_size, threshold)
        for obj, mask in zip(objects, masks):
            area = int(mask.sum())
            results.append(
                {
                    **obj,
                    "segmentation": encode_mask(mask),
                    "area": area,
                    "equivalent_diameter_px": math.sqrt(4 * area / math.pi),
                    "empty_mask": area == 0,
                }
            )
            if overlays:
                color = np.array([60, 210, 80] if obj["category_id"] % 2 else [235, 100, 60])
                rgb[mask > 0] = (rgb[mask > 0] * 0.5 + color * 0.5).astype(np.uint8)
                x, y, w, h = obj["bbox"]
                cv2.rectangle(
                    rgb,
                    (round(x), round(y)),
                    (round(x + w), round(y + h)),
                    tuple(int(c) for c in color),
                    1,
                )
        if overlays:
            cv2.imwrite(
                str(Path(overlays) / f"{image_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
    write_json(output, results)
    return {
        "images": len(grouped),
        "instances": len(results),
        "empty_masks": sum(obj["empty_mask"] for obj in results),
        "output": str(output),
    }
