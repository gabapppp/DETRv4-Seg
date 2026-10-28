"""Frozen RT-DETRv4 deployment with explicit external COCO category IDs."""

import numpy as np
import torch
from PIL import Image

from .coco import image_path, read_json, write_json
from .train import select_device
from .upstream import configuration, move_model


class DetectorPredictor:
    def __init__(self, repo, config, checkpoint, label_map, device="auto", size=None):
        cfg = configuration(repo, config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.mapping = {int(k): int(v) for k, v in label_map.items()}
        self.size = size or state.get("input_size", cfg.yaml_cfg.get("eval_spatial_size", [640])[0])
        expected_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
        if expected_size != [self.size, self.size]:
            raise ValueError(
                "Detector input size must match eval_spatial_size in its training config"
            )
        weights = state["ema"]["module"] if "ema" in state else state["model"]
        model = cfg.model
        model.load_state_dict(weights)
        self.device = select_device(device)
        self.model = move_model(model.eval().deploy().eval().requires_grad_(False), self.device)
        self.postprocessor = cfg.postprocessor.deploy().eval().to(self.device)

    @torch.inference_mode()
    def predict(self, rgb, score_threshold=0.25):
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        height, width = rgb.shape[:2]
        resized = Image.fromarray(rgb).resize((self.size, self.size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()[None]
        original = torch.tensor([[width, height]], device=self.device)
        labels, boxes, scores = self.postprocessor(
            self.model(tensor.to(self.device) / 255), original
        )
        results = []
        for label, box, score in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
            if score < score_threshold:
                continue
            if label not in self.mapping:
                raise ValueError(f"Model label {label} is missing from the label map")
            x0, y0, x1, y1 = box
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(width), x1), min(float(height), y1)
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                results.append(
                    {
                        "category_id": self.mapping[label],
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": score,
                    }
                )
        return results


def detect(
    repo,
    config,
    checkpoint,
    annotations,
    images_root,
    label_map,
    output,
    device="auto",
    score_threshold=0.25,
    size=None,
):
    coco = read_json(annotations)
    mapping = read_json(label_map)
    if not set(map(int, mapping.values())) <= {cat["id"] for cat in coco["categories"]}:
        raise ValueError("Label map contains category IDs absent from annotations")
    predictor = DetectorPredictor(repo, config, checkpoint, mapping, device, size)
    results = []
    for im in coco["images"]:
        with Image.open(image_path(images_root, im["file_name"])) as loaded:
            if loaded.size != (im["width"], im["height"]):
                raise ValueError(f"Image {im['id']}: dimensions mismatch")
            rgb = np.asarray(loaded.convert("RGB"))
        results.extend({"image_id": im["id"], **p} for p in predictor.predict(rgb, score_threshold))
    write_json(output, results)
    return {"images": len(coco["images"]), "detections": len(results), "output": str(output)}
