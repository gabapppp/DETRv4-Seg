"""Train and run a complete image → detector → ROI-mask pipeline."""

import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .coco import encode_mask, read_json, write_json
from .detector import DetectorPredictor, detect
from .detector_train import train_detector
from .infer import MaskPredictor, infer
from .metrics import coco_metrics
from .prepare import prepare
from .train import train
from .upstream import DEFAULT_MODEL_SIZE, bootstrap


def bundle_paths(bundle):
    path = Path(bundle).resolve()
    state = read_json(path)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported pipeline bundle")
    paths = {
        name: str((path.parent / state[name]).resolve())
        for name in ("repo", "config", "detector_checkpoint", "mask_checkpoint", "label_map")
    }
    return state, paths


def create_bundle(
    repo,
    config,
    detector_checkpoint,
    mask_checkpoint,
    label_map,
    annotations,
    output,
    mask_threshold=0.5,
):
    """Connect fine-tuned components without overwriting a previous model bundle."""
    import torch

    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Bundle already exists: {output}")
    if not 0 < mask_threshold < 1:
        raise ValueError("Mask threshold must be between zero and one")
    paths = {
        "repo": Path(repo).resolve(),
        "config": Path(config).resolve(),
        "detector_checkpoint": Path(detector_checkpoint).resolve(),
        "mask_checkpoint": Path(mask_checkpoint).resolve(),
        "label_map": Path(label_map).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise ValueError(f"Missing bundle component: {path}")
    categories = read_json(annotations)["categories"]
    mapping = {int(k): int(v) for k, v in read_json(label_map).items()}
    if set(mapping.values()) != {cat["id"] for cat in categories}:
        raise ValueError("Bundle categories differ from detector label map")
    detector = torch.load(detector_checkpoint, map_location="cpu", weights_only=True)
    if "label_map" in detector and detector["label_map"] != mapping:
        raise ValueError("Detector checkpoint label map differs from supplied map")
    mask = torch.load(mask_checkpoint, map_location="cpu", weights_only=True)
    if mask.get("format_version") != 1:
        raise ValueError("Unsupported mask checkpoint")
    state = {
        "format_version": 1,
        **{name: os.path.relpath(path, output.parent) for name, path in paths.items()},
        "categories": categories,
        "mask_label_sources": mask.get("label_sources", {}),
        "mask_overflow": mask.get("mask_overflow", 0.0),
        "padding": mask["padding"],
        "mask_threshold": mask_threshold,
    }
    write_json(output, state)
    return {"bundle": str(output), "mask_overflow": state["mask_overflow"]}


class InstancePredictor:
    def __init__(self, bundle, device="auto"):
        state, paths = bundle_paths(bundle)
        self.categories = state["categories"]
        self.mask_threshold = state.get("mask_threshold", 0.5)
        self.detector = DetectorPredictor(
            paths["repo"],
            paths["config"],
            paths["detector_checkpoint"],
            read_json(paths["label_map"]),
            device,
        )
        self.mask_head = MaskPredictor(paths["mask_checkpoint"], device)

    def predict(self, rgb, score_threshold=0.25, mask_threshold=None, batch_size=32):
        mask_threshold = self.mask_threshold if mask_threshold is None else mask_threshold
        objects = self.detector.predict(rgb, score_threshold)
        masks = self.mask_head.predict(
            rgb, [obj["bbox"] for obj in objects], batch_size, mask_threshold
        )
        return [
            {
                **obj,
                "segmentation": encode_mask(mask),
                "area": int(mask.sum()),
                "equivalent_diameter_px": math.sqrt(4 * int(mask.sum()) / math.pi),
                "empty_mask": not bool(mask.any()),
            }
            for obj, mask in zip(objects, masks)
        ], masks


def predict(
    bundle,
    input,
    output,
    device="auto",
    score_threshold=0.25,
    mask_threshold=None,
    batch_size=32,
    overlays=True,
):
    source, output = Path(input).resolve(), Path(output).resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}")
    root = source if source.is_dir() else source.parent
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = (
        sorted(
            p
            for p in source.rglob("*")
            if p.suffix.lower() in extensions and not p.is_relative_to(output)
        )
        if source.is_dir()
        else [source]
    )
    if not files:
        raise ValueError("No image files found")
    output.mkdir(parents=True, exist_ok=True)
    if overlays:
        (output / "overlays").mkdir()
    predictor = InstancePredictor(bundle, device)
    if mask_threshold is None:
        mask_threshold = getattr(predictor, "mask_threshold", 0.5)
    images, predictions, rows = [], [], []
    names = {cat["id"]: cat["name"] for cat in predictor.categories}
    start = time.monotonic()
    for index, path in enumerate(files):
        with Image.open(path) as loaded:
            rgb = np.asarray(loaded.convert("RGB")).copy()
        objects, masks = predictor.predict(rgb, score_threshold, mask_threshold, batch_size)
        images.append(
            {
                "id": index,
                "file_name": path.relative_to(root).as_posix(),
                "width": rgb.shape[1],
                "height": rgb.shape[0],
            }
        )
        for obj, mask in zip(objects, masks):
            predictions.append({"image_id": index, **obj})
            rows.append(
                {
                    "image": path.relative_to(root).as_posix(),
                    "category_id": obj["category_id"],
                    "category": names[obj["category_id"]],
                    "score": obj["score"],
                    "area_px": obj["area"],
                    "equivalent_diameter_px": obj["equivalent_diameter_px"],
                    "empty_mask": obj["empty_mask"],
                }
            )
            if overlays:
                color = np.array([40, 220, 80] if obj["category_id"] % 2 else [230, 90, 50])
                rgb[mask > 0] = (rgb[mask > 0] * 0.55 + color * 0.45).astype(np.uint8)
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
                str(output / "overlays" / f"{index:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
        print(f"[{index + 1}/{len(files)}] {path.name}: {len(objects)} instances", flush=True)
    write_json(
        output / "images.json",
        {
            "images": images,
            "categories": predictor.categories,
            "annotations": [],
            "images_root": str(root),
        },
    )
    write_json(output / "instances.json", predictions)
    with (output / "measurements.csv").open("w", newline="") as file:
        fields = (
            "image",
            "category_id",
            "category",
            "score",
            "area_px",
            "equivalent_diameter_px",
            "empty_mask",
        )
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images": len(files),
        "instances": len(predictions),
        "empty_masks": sum(p["empty_mask"] for p in predictions),
        "elapsed_seconds": time.monotonic() - start,
        "bundle": str(Path(bundle).resolve()),
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
    }
    write_json(output / "summary.json", summary)
    return summary


def fit(
    annotations,
    images_root,
    output,
    repo="vendor/RT-DETRv4",
    pretrained=None,
    detector_steps=600,
    mask_steps=600,
    budget_seconds=1800,
    device="auto",
    seed=42,
    fill_mode="holes",
    groups=None,
    detector_size=640,
    detector_batch_size=4,
    max_objects=3000,
    padding=0.25,
    mask_overflow=0.25,
    p2=False,
    mask_architecture="tiny",
    model_size=DEFAULT_MODEL_SIZE,
    mlflow_uri=None,
    mlflow_experiment=None,
    mlflow_run_name=None,
    mlflow_log_every=20,
):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}")
    if min(detector_steps, mask_steps, budget_seconds) <= 0:
        raise ValueError("Positive steps and budget required")
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)

    def remaining():
        left = budget_seconds - (time.monotonic() - start)
        if left <= 0:
            raise ValueError(
                "Pipeline budget expired; completed stage artifacts remain in the output"
            )
        return left

    bootstrap(repo, pretrained, model_size=model_size)
    prepare(
        annotations,
        images_root,
        output / "cache",
        seed=seed,
        fill_mode=fill_mode,
        padding=padding,
        mask_overflow=mask_overflow,
        groups=groups,
        max_objects=max_objects,
        budget_seconds=min(420, remaining() * 0.2),
    )
    train(
        output / "cache",
        output / "mask",
        steps=mask_steps,
        architecture=mask_architecture,
        budget_seconds=min(1020, remaining() * 0.2),
        device=device,
        seed=seed,
        mlflow_uri=mlflow_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=f"{mlflow_run_name}-seg" if mlflow_run_name else None,
        mlflow_log_every=mlflow_log_every,
    )
    train_detector(
        annotations,
        images_root,
        output / "detector",
        repo,
        pretrained,
        split_report=output / "cache/report.json",
        steps=detector_steps,
        budget_seconds=remaining() * 0.85,
        device=device,
        seed=seed,
        size=detector_size,
        batch_size=detector_batch_size,
        p2=p2,
        model_size=model_size,
        mlflow_uri=mlflow_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=f"{mlflow_run_name}-det" if mlflow_run_name else None,
        mlflow_log_every=mlflow_log_every,
    )
    bundle = {
        "format_version": 1,
        "repo": os.path.relpath(Path(repo).resolve(), output),
        "config": "detector/config.yml",
        "detector_checkpoint": "detector/best.pt",
        "mask_checkpoint": "mask/best.pt",
        "label_map": "detector/label_map.json",
        "categories": read_json(annotations)["categories"],
        "mask_labels": "pseudo"
        if read_json(output / "cache/report.json")["audit"]["without_masks"]
        else "ground_truth",
    }
    write_json(output / "bundle.json", bundle)
    detect(
        repo,
        output / "detector/config.yml",
        output / "detector/best.pt",
        output / "detector/val.json",
        images_root,
        output / "detector/label_map.json",
        output / "detections.json",
        device=device,
        score_threshold=0.001,
    )
    infer(
        output / "detector/val.json",
        images_root,
        output / "detections.json",
        output / "mask/best.pt",
        output / "instances.json",
        device=device,
        score_threshold=0.25,
        overlays=output / "overlays",
    )
    detections = read_json(output / "detections.json")
    reference = read_json(output / "cache/instances_masks.json")
    reference["images"] = read_json(output / "detector/val.json")["images"]
    reference["annotations"] = [a for a in reference["annotations"] if a["split"] == "val"]
    metrics = {
        "bbox": coco_metrics(read_json(output / "detector/val.json"), detections),
        "segmentation_reference": "accepted cached masks on held-out images",
        "mask_label_sources": read_json(output / "cache/report.json")["sources"],
        "note": "BBox AP uses all held-out boxes. Mask AP uses only accepted cached masks "
        "and score >=0.25 predictions; pseudo-mask AP is not human-mask accuracy.",
        "mask_agreement_AP": coco_metrics(reference, read_json(output / "instances.json"), "segm"),
    }
    write_json(output / "metrics.json", metrics)
    summary = {
        "elapsed_seconds": time.monotonic() - start,
        "bundle": str(output / "bundle.json"),
        "metrics": metrics,
    }
    write_json(output / "summary.json", summary)
    return summary
