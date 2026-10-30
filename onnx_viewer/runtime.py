"""DETR ONNX inference and native-image contour circle measurements."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from detrseg.roi import (
    bbox_prior,
    clean_component,
    mask_support,
    restore_roi,
    roi_bounds,
    roi_inputs,
)


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def circle_from_contour(contour):
    if contour is None or len(contour) < 3:
        return {"center_x": None, "center_y": None, "diameter_px": None}
    points = np.asarray(contour, np.float32).reshape(-1, 1, 2)
    if not np.isfinite(points).all() or cv2.contourArea(points) <= 0:
        return {"center_x": None, "center_y": None, "diameter_px": None}
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    return {"center_x": float(cx), "center_y": float(cy), "diameter_px": float(2 * radius)}


def roi_contour(rgb, box):
    """Detector-only fallback: choose an isolated Otsu component aligned with its box."""
    bounds = roi_bounds(box, rgb.shape[1], rgb.shape[0], padding=0.2)
    x0, y0, x1, y1 = bounds
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if gray.size < 9 or float(gray.std()) < 2:
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x, y, width, height = box
    best, best_rank = None, 0.25
    for candidate in (binary, 255 - binary):
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rx == 0 or ry == 0 or rx + rw >= gray.shape[1] or ry + rh >= gray.shape[0]:
                continue
            area = cv2.contourArea(contour)
            if not max(3, 0.05 * width * height) <= area <= 1.5 * width * height:
                continue
            cx, cy = x0 + rx + rw / 2, y0 + ry + rh / 2
            if not (x <= cx <= x + width and y <= cy <= y + height):
                continue
            ix = max(0, min(x + width, x0 + rx + rw) - max(x, x0 + rx))
            iy = max(0, min(y + height, y0 + ry + rh) - max(y, y0 + ry))
            intersection = ix * iy
            rank = intersection / max(1, width * height + rw * rh - intersection)
            if rank > best_rank:
                best_rank = rank
                best = contour + np.array([[[x0, y0]]], np.int32)
    return best


class OnnxPipeline:
    def __init__(self, model_path, provider="CPUExecutionProvider", mask_path=None, input_size=640):
        if provider not in ort.get_available_providers():
            raise ValueError(f"Provider unavailable: {provider}")
        self.provider = provider
        path = Path(model_path).resolve()
        self.config = {}
        manifest = path if path.suffix.lower() == ".json" else path.parent / "manifest.json"
        if manifest.is_file():
            config = json.loads(manifest.read_text(encoding="utf-8"))
            if path == manifest or (manifest.parent / config.get("detector", "")).resolve() == path:
                self.config = config
                path = (manifest.parent / config["detector"]).resolve()
                if mask_path is None and config.get("mask"):
                    mask_path = (manifest.parent / config["mask"]).resolve()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        providers = (
            [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
        )
        self.detector = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.image_input = next((i for i in self.detector.get_inputs() if len(i.shape) == 4), None)
        if self.image_input is None or self.image_input.shape[1] != 3:
            raise ValueError("Detector must have one NCHW RGB image input with 3 channels")
        self._check_batch(self.image_input)
        shape = self.image_input.shape
        self.height = (
            shape[2] if isinstance(shape[2], int) else int(self.config.get("height", input_size))
        )
        self.width = (
            shape[3] if isinstance(shape[3], int) else int(self.config.get("width", input_size))
        )
        if min(self.height, self.width) < 1:
            raise ValueError("Input dimensions must be positive")
        self.output_names = {i.name for i in self.detector.get_outputs()}
        self.raw = "pred_boxes" in self.output_names and bool(
            self.output_names & {"logits", "pred_logits"}
        )
        if not self.raw and not {"labels", "boxes", "scores"} <= self.output_names:
            raise ValueError("Expected logits/pred_boxes or labels/boxes/scores detector outputs")
        self.resize_mode = self.config.get(
            "resize_mode", "letterbox" if self.config.get("schema") == 2 else "resize"
        )
        if self.resize_mode not in ("resize", "letterbox"):
            raise ValueError("resize_mode must be resize or letterbox")
        self.mask = None
        if mask_path:
            self.mask = ort.InferenceSession(
                str(mask_path), sess_options=options, providers=providers
            )
            inputs = self.mask.get_inputs()
            if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 4:
                raise ValueError(
                    "ROI mask ONNX must have one NCHW input with RGB + bbox prior (4 channels)"
                )
            self.mask_input = inputs[0]
            self._check_batch(self.mask_input)
            self.roi_size = self.mask_input.shape[2]
            if not isinstance(self.roi_size, int) or self.mask_input.shape[3] != self.roi_size:
                raise ValueError("Mask ONNX needs a fixed square ROI input")
            self.mask_output = self.mask.get_outputs()[0].name
        self.padding = float(self.config.get("padding", 0.25))
        self.overflow = float(self.config.get("mask_overflow", 0.0))
        self.fill_mode = self.config.get("fill_mode", "holes")
        self.roi_resize_mode = self.config.get("roi_resize_mode", "resize")
        if self.roi_resize_mode not in ("resize", "native"):
            raise ValueError("Unsupported manifest ROI resize mode")
        if not 0 <= self.overflow <= self.padding:
            raise ValueError("Mask overflow must be between zero and crop padding")
        self.measurement_source = "ONNX mask" if self.mask else "ROI Otsu (detector only)"
        self.model_description = (
            f"{path.name} | {self.width}x{self.height} | {provider} | {self.measurement_source}"
        )

    @staticmethod
    def _check_batch(input_info):
        if isinstance(input_info.shape[0], int) and input_info.shape[0] != 1:
            raise ValueError("Viewer requires dynamic batch or fixed batch 1 ONNX")

    @staticmethod
    def _dtype(input_info):
        kinds = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }
        if input_info.type not in kinds:
            raise ValueError(f"Unsupported ONNX input type: {input_info.type}")
        return kinds[input_info.type]

    def _prepare(self, rgb):
        oh, ow = rgb.shape[:2]
        if self.resize_mode == "letterbox":
            scale = min(self.width / ow, self.height / oh)
            nw, nh = (
                min(self.width, max(1, round(ow * scale))),
                min(self.height, max(1, round(oh * scale))),
            )
            left, top = (self.width - nw) // 2, (self.height - nh) // 2
            canvas = np.full((self.height, self.width, 3), 114, np.uint8)
            canvas[top : top + nh, left : left + nw] = np.asarray(
                Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)
            )
            geometry = (nw / ow, nh / oh, left, top)
        else:
            canvas = np.asarray(
                Image.fromarray(rgb).resize((self.width, self.height), Image.Resampling.BILINEAR)
            )
            geometry = (self.width / ow, self.height / oh, 0, 0)
        pixels = canvas.astype(np.float32)
        if self.config.get("do_rescale", True):
            pixels *= float(self.config.get("rescale_factor", 1 / 255))
        if self.config.get("do_normalize", False):
            pixels = (pixels - np.asarray(self.config["image_mean"], np.float32)) / np.asarray(
                self.config["image_std"], np.float32
            )
        tensor = np.ascontiguousarray(
            pixels.transpose(2, 0, 1)[None], dtype=self._dtype(self.image_input)
        )
        feed = {self.image_input.name: tensor}
        for extra in self.detector.get_inputs():
            if extra.name == self.image_input.name:
                continue
            if extra.name not in ("orig_target_sizes", "orig_sizes"):
                raise ValueError(f"Unsupported detector input: {extra.name}")
            # Upstream RT-DETRv4 scales xyxy by [width,height,width,height].
            target_size = [ow, oh] if self.resize_mode == "resize" else [self.width, self.height]
            feed[extra.name] = np.asarray([target_size], dtype=self._dtype(extra))
        return feed, geometry

    def _detections(self, rgb, threshold):
        feed, (sx, sy, left, top) = self._prepare(rgb)
        outputs = dict(
            zip([o.name for o in self.detector.get_outputs()], self.detector.run(None, feed))
        )
        oh, ow = rgb.shape[:2]
        if self.raw:
            logits = outputs.get("logits", outputs.get("pred_logits"))[0]
            boxes = outputs["pred_boxes"][0]
            if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
                raise ValueError("Expected detector shapes [1,Q,C] and [1,Q,4]")
            probabilities = 1 / (1 + np.exp(-np.clip(logits.astype(np.float32), -80, 80)))
            flat = probabilities.ravel()
            selected = np.argsort(-flat, kind="stable")[: int(self.config.get("num_queries", 300))]
            queries, labels = np.divmod(selected, logits.shape[1])
            scores = flat[selected]
            boxes = np.asarray(boxes[queries], np.float64)
            boxes = np.concatenate(
                (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), axis=1
            )
            boxes *= [self.width, self.height, self.width, self.height]
            boxes = (boxes - [left, top, left, top]) / [sx, sy, sx, sy]
        else:
            labels = np.asarray(outputs["labels"]).reshape(-1)
            scores = np.asarray(outputs["scores"]).reshape(-1)
            boxes = np.asarray(outputs["boxes"], np.float64).reshape(-1, 4)
            if not len(labels) == len(scores) == len(boxes):
                raise ValueError("Detector labels, boxes and scores have different lengths")
            if self.resize_mode == "letterbox":
                boxes = (boxes - [left, top, left, top]) / [sx, sy, sx, sy]
        boxes = np.clip(boxes, 0, [ow, oh, ow, oh])
        result = []
        mapping = self.config.get("id2label", {})
        for label, score, box in zip(labels, scores, boxes):
            x0, y0, x1, y1 = map(float, box)
            if (
                not np.isfinite([score, *box]).all()
                or score < threshold
                or x1 - x0 < 1
                or y1 - y0 < 1
            ):
                continue
            result.append(
                {
                    "label": str(mapping.get(str(int(label)), int(label))),
                    "score": float(score),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                }
            )
        return result

    def _mask_contour(self, rgb, box, threshold):
        crop, prior, bounds = roi_inputs(
            rgb, box, self.roi_size, self.padding, self.roi_resize_mode
        )
        if not prior.any():
            return None
        inputs = np.concatenate((crop.astype(np.float32) / 255, prior[..., None]), axis=2)
        tensor = np.ascontiguousarray(
            inputs.transpose(2, 0, 1)[None], dtype=self._dtype(self.mask_input)
        )
        output = self.mask.run([self.mask_output], {self.mask_input.name: tensor})[0]
        if output.ndim != 4 or output.shape[:2] != (1, 1):
            raise ValueError("Mask output must have shape [1,1,H,W]")
        probability = output[0, 0].astype(np.float32)
        if self.config.get("mask_output", "logits") == "logits":
            probability = 1 / (1 + np.exp(-np.clip(probability, -80, 80)))
        x0, y0 = bounds[:2]
        if self.roi_resize_mode == "native" and probability.shape != (self.roi_size, self.roi_size):
            raise ValueError("Native mask output must match its input canvas resolution")
        probability = restore_roi(probability, bounds, self.roi_resize_mode)
        prior = bbox_prior(box, bounds)
        support = mask_support(box, bounds, self.overflow)
        component, _ = clean_component(
            probability >= threshold,
            prior,
            0,
            float(support.sum() / max(1, prior.sum())),
            fill_mode=self.fill_mode,
            support=support,
        )
        if component is None:
            return None
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return np.concatenate(contours) + np.array([[[x0, y0]]], np.int32) if contours else None

    def predict(self, rgb, score_threshold=0.5, mask_threshold=0.5):
        if not 0 <= score_threshold <= 1 or not 0 < mask_threshold < 1:
            raise ValueError("Score threshold must be in [0,1]; mask threshold must be in (0,1)")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image")
        start = time.perf_counter()
        objects = self._detections(rgb, score_threshold)
        for obj in objects:
            contour = (
                self._mask_contour(rgb, obj["bbox"], mask_threshold)
                if self.mask
                else roi_contour(rgb, obj["bbox"])
            )
            obj.update(
                circle_from_contour(contour),
                contour=contour,
                measurement_source=self.measurement_source,
            )
        return {
            "objects": objects,
            "elapsed_ms": (time.perf_counter() - start) * 1000,
            "provider": self.provider,
            "measurement_source": self.measurement_source,
        }


def draw_overlay(
    rgb,
    result_or_objects,
    *,
    show_boxes=True,
    show_contours=True,
    show_circles=True,
    show_diameters=True,
    hidden_instances=(),
    hidden_boxes=(),
):
    """Draw enabled layers without changing measurements or stable C1-based IDs.

    ``hidden_instances`` and ``hidden_boxes`` contain one-based instance numbers,
    matching the viewer's C1, C2, ... table rows. Hiding a complete instance also
    hides its box; hiding only its box leaves other enabled layers visible.
    """
    objects = (
        result_or_objects["objects"] if isinstance(result_or_objects, dict) else result_or_objects
    )
    overlay = rgb.copy()
    for index, obj in enumerate(objects, 1):
        if index in hidden_instances:
            continue
        x, y, width, height = obj["bbox"]
        if show_boxes and index not in hidden_boxes:
            cv2.rectangle(
                overlay,
                (round(x), round(y)),
                (round(x + width), round(y + height)),
                (60, 180, 255),
                1,
            )
        if show_contours and obj.get("contour") is not None:
            cv2.drawContours(overlay, [np.asarray(obj["contour"], np.int32)], -1, (50, 220, 100), 1)
        diameter = obj.get("diameter_px")
        text = f"C{index} D=N/A"
        if diameter is not None:
            cx, cy = round(obj["center_x"]), round(obj["center_y"])
            if show_circles:
                cv2.circle(overlay, (cx, cy), max(1, round(diameter / 2)), (255, 210, 40), 2)
                cv2.drawMarker(overlay, (cx, cy), (255, 80, 80), cv2.MARKER_CROSS, 7, 1)
            text = f"C{index} D={diameter:.2f} px"
        if show_diameters:
            anchor = (max(0, round(x)), max(14, round(y) - 5))
            cv2.putText(
                overlay, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                overlay,
                text,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 230, 80),
                1,
                cv2.LINE_AA,
            )
    return overlay
