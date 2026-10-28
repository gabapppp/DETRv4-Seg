"""Supervised fine-tuning of the pretrained RT-DETRv4 student, without a VFM teacher."""

import copy
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .coco import image_path, read_json, validate_coco, write_json
from .metrics import coco_metrics
from .prepare import split_images
from .pyramid import p2_config, transfer_p2
from .train import select_device
from .upstream import configuration, move_model


def detector_splits(coco, mapping):
    """Map arbitrary COCO category IDs to contiguous training labels, keeping external IDs."""
    categories = sorted(coco["categories"], key=lambda cat: cat["id"])
    label_map = {i: cat["id"] for i, cat in enumerate(categories)}
    result = {}
    if set(map(str, (im["id"] for im in coco["images"]))) != set(mapping):
        raise ValueError("Split report image IDs do not match annotations")
    if set(mapping.values()) != {"train", "val"}:
        raise ValueError("Both train and val images are required")
    for split in ("train", "val"):
        images = [im for im in coco["images"] if mapping[str(im["id"])] == split]
        ids = {im["id"] for im in images}
        result[split] = {
            **coco,
            "info": coco.get("info", {}),
            "images": images,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        }
    return result, label_map


class DetectionDataset(Dataset):
    def __init__(self, coco, root, size, label_map, augment=False):
        self.coco, self.root, self.size, self.augment = coco, root, size, augment
        self.category_to_label = {v: k for k, v in label_map.items()}
        self.annotations = defaultdict(list)
        for ann in coco["annotations"]:
            if not ann.get("iscrowd", 0):
                self.annotations[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.coco["images"])

    def __getitem__(self, index):
        im = self.coco["images"][index]
        with Image.open(image_path(self.root, im["file_name"])) as loaded:
            rgb = np.asarray(
                loaded.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            ).copy()
        boxes, labels = [], []
        for ann in self.annotations[im["id"]]:
            x, y, w, h = ann["bbox"]
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(im["width"], x + w), min(im["height"], y + h)
            boxes.append(
                [
                    (x0 + x1) / 2 / im["width"],
                    (y0 + y1) / 2 / im["height"],
                    (x1 - x0) / im["width"],
                    (y1 - y0) / im["height"],
                ]
            )
            labels.append(self.category_to_label[ann["category_id"]])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment:
            if random.random() < 0.5:
                rgb = rgb[:, ::-1].copy()
                boxes[:, 0] = 1 - boxes[:, 0]
            rgb = np.clip(rgb.astype(np.float32) * random.uniform(0.9, 1.1), 0, 255)
        pixels = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        return pixels, {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(im["id"]),
            "orig_size": torch.tensor([im["width"], im["height"]]),
        }


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


@torch.no_grad()
def update_ema(averaged_model, student_model, decay):
    for averaged, student in zip(averaged_model.parameters(), student_model.parameters()):
        if averaged.is_floating_point():
            averaged.lerp_(student.detach(), 1 - decay)
        else:
            averaged.copy_(student)
    for (name, averaged), (_, student) in zip(
        averaged_model.named_buffers(), student_model.named_buffers()
    ):
        if name.endswith(("running_mean", "running_var")) and averaged.is_floating_point():
            averaged.lerp_(student.detach(), 1 - decay)
        else:
            # Anchor caches contain inf at invalid positions; averaging inf-inf
            # would introduce NaN and corrupt otherwise valid decoder geometry.
            averaged.copy_(student)


@torch.inference_mode()
def validate_detector(model, postprocessor, loader, device, label_map):
    model.eval()
    predictions = []
    for images, targets in loader:
        original_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
        outputs = postprocessor(model(images.to(device)), original_sizes)
        for target, result in zip(targets, outputs):
            for label, box, score in zip(
                result["labels"].tolist(), result["boxes"].tolist(), result["scores"].tolist()
            ):
                if score < 0.001:
                    continue
                x0, y0, x1, y1 = box
                predictions.append(
                    {
                        "image_id": target["image_id"].item(),
                        "category_id": label_map[label],
                        "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                        "score": score,
                    }
                )
    return coco_metrics(loader.dataset.coco, predictions), predictions


def train_detector(
    annotations,
    images_root,
    output,
    repo="vendor/RT-DETRv4",
    pretrained="weights/rtv4_s_coco.pth",
    split_report=None,
    steps=600,
    batch_size=4,
    size=640,
    lr=2e-4,
    budget_seconds=900,
    val_every=100,
    device="auto",
    seed=42,
    threads=4,
    patience=0,
    p2=False,
    ema_decay=0.0,
):
    if (
        min(steps, batch_size, val_every, threads, lr, budget_seconds) <= 0
        or patience < 0
        or not 0 <= ema_decay < 1
        or size < 128
        or size % 32
    ):
        raise ValueError(
            "Positive training options required; detector size must be >=128 and a multiple of 32"
        )
    start = time.monotonic()
    output, repo = Path(output).resolve(), Path(repo).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = select_device(device)
    coco = read_json(annotations)
    validate_coco(coco, images_root)
    mapping = (
        read_json(split_report)["image_splits"]
        if split_report
        else {str(k): v for k, v in split_images(coco["images"], 0.2, seed).items()}
    )
    splits, label_map = detector_splits(coco, mapping)
    output.mkdir(parents=True, exist_ok=True)
    for split, subset in splits.items():
        write_json(output / f"{split}.json", subset)
    write_json(output / "label_map.json", label_map)
    # Same S student architecture as the official checkpoint. Teacher is used only in
    # original pretraining; supervised custom-dataset fine-tuning needs no DINO files.
    import yaml

    config = {
        "__include__": [str(repo / "configs/rtv4/rtv4_hgnetv2_s_coco.yml")],
        "num_classes": len(label_map),
        "remap_mscoco_category": False,
        "eval_spatial_size": [size, size],
        "HGNetv2": {"pretrained": False},
        "HybridEncoder": {"distill_teacher_dim": 0},
        "RTv4Criterion": {"losses": ["mal", "boxes", "local"]},
    }
    config_path = output / "config.yml"
    if p2:
        for key, value in p2_config().items():
            config.setdefault(key, {}).update(value)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    cfg = configuration(repo, config_path)
    # Persist the resolved configuration so inference does not retain an absolute
    # __include__ path into the machine used for training.
    resolved = {key: value for key, value in cfg.yaml_cfg.items() if key != "__include__"}
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    model = cfg.model
    state = torch.load(pretrained, map_location="cpu", weights_only=True)
    if (
        "label_map" in state
        and {int(k): int(v) for k, v in state["label_map"].items()} != label_map
    ):
        raise ValueError("Fine-tuning checkpoint category mapping differs from the current dataset")
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    if state.get("p2", False) and not p2:
        raise ValueError("A P2 checkpoint requires --p2")
    current = model.state_dict()
    adding_p2 = p2 and not state.get("p2", False)
    if adding_p2:
        weights = transfer_p2(weights, current)
    # Evaluation anchors follow the new input resolution, rather than source weights.
    weights = dict(weights)
    for name in ("decoder.anchors", "decoder.valid_mask"):
        if name in current:
            weights[name] = current[name]
    matched = {k: v for k, v in weights.items() if k in current and v.shape == current[k].shape}
    missing = model.load_state_dict(matched, strict=False)
    # Different class heads may be reinitialized, but the architecture must otherwise match.
    allowed = (
        "decoder.enc_score_head.",
        "decoder.dec_score_head.",
        "decoder.denoising_class_embed.",
    )
    if adding_p2:
        allowed += (
            "encoder.input_proj.0.",
            "encoder.lateral_convs.2.",
            "encoder.fpn_blocks.2.",
            "encoder.downsample_convs.0.",
            "encoder.pan_blocks.0.",
        )
    if any(not name.startswith(allowed) for name in missing.missing_keys):
        raise ValueError(f"Pretrained architecture mismatch: {missing.missing_keys}")
    model = move_model(model, device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False) if ema_decay else None
    criterion, postprocessor = cfg.criterion.to(device), cfg.postprocessor.to(device)
    datasets = {
        s: DetectionDataset(c, images_root, size, label_map, s == "train")
        for s, c in splits.items()
    }
    loaders = {
        s: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=s == "train",
            collate_fn=detection_collate,
            num_workers=0,
        )
        for s, ds in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("backbone.")],
                "lr": lr * 0.1,
            },
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")],
                "lr": lr,
            },
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator, completed, history, best_ap = iter(loaders["train"]), 0, [], -1.0
    best_step, stale_validations = 0, 0

    def save(path, metrics):
        torch.save(
            {
                "model": {
                    k: v.detach().cpu()
                    for k, v in (ema_model if ema_model is not None else model).state_dict().items()
                },
                "step": completed,
                "metrics": metrics,
                "label_map": label_map,
                "input_size": size,
                "seed": seed,
                "p2": p2,
                "ema_decay": ema_decay,
            },
            path,
        )

    def validation():
        nonlocal best_ap, best_step, stale_validations
        metrics, predictions = validate_detector(
            ema_model if ema_model is not None else model,
            postprocessor,
            loaders["val"],
            device,
            label_map,
        )
        history.append(
            {"step": completed, "validation": metrics, "elapsed_seconds": time.monotonic() - start}
        )
        if metrics["AP"] > best_ap:
            best_ap = metrics["AP"]
            best_step = completed
            stale_validations = 0
            save(output / "best.pt", metrics)
            write_json(output / "best_predictions.json", predictions)
        else:
            stale_validations += 1
        write_json(output / "history.json", history)
        print(
            f"detector step={completed} AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f}",
            flush=True,
        )
        return metrics

    # Preserve a fine-tuned starting model as the candidate if later updates regress.
    baseline = validation() if "label_map" in state else None
    while completed < steps and time.monotonic() - start < budget_seconds:
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"])
            images, targets = next(iterator)
        model.train()
        criterion.train()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        warmup = min(1.0, (completed + 1) / min(50, steps))
        decay = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * completed / steps))
        for group, factor in zip(optimizer.param_groups, (0.1, 1.0)):
            group["lr"] = lr * factor * warmup * decay
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            outputs = model(images.to(device), targets)

        # Hungarian matching and classification/box losses require float32 numerics.
        def float_outputs(value):
            if isinstance(value, torch.Tensor):
                return value.float() if value.is_floating_point() else value
            if isinstance(value, dict):
                return {k: float_outputs(v) for k, v in value.items()}
            if isinstance(value, list):
                return [float_outputs(v) for v in value]
            return value

        losses = criterion(float_outputs(outputs), targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite detector loss at step {completed + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        if ema_model is not None:
            update_ema(ema_model, model, ema_decay)
        completed += 1
        if completed % 20 == 0:
            print(f"detector step={completed} loss={loss.item():.4f}", flush=True)
        if completed % val_every == 0 or completed == steps:
            validation()
            if patience and stale_validations >= patience:
                break
    if completed == 0:
        raise ValueError("Detector budget expired before the first update")
    metrics = (
        validation()
        if not history or history[-1]["step"] != completed
        else history[-1]["validation"]
    )
    save(output / "last.pt", metrics)
    summary = {
        "steps": completed,
        "requested_steps": steps,
        "best_AP": best_ap,
        "best_step": best_step,
        "initial_validation": baseline,
        "elapsed_seconds": time.monotonic() - start,
        "budget_exhausted": completed < steps and time.monotonic() - start >= budget_seconds,
        "early_stopped": bool(patience and stale_validations >= patience),
        "device": str(device),
        "batch_size": batch_size,
        "size": size,
        "train_images": len(datasets["train"]),
        "val_images": len(datasets["val"]),
        "label_map": label_map,
        "reinitialized": missing.missing_keys,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else 0.0,
        "pretrained": str(Path(pretrained).resolve()),
        "initial_checkpoint_step": state.get("step"),
        "lr": lr,
        "seed": seed,
        "patience": patience,
        "p2": p2,
        "amp": device.type == "cuda",
        "ema_decay": ema_decay,
        "val_every": val_every,
    }
    write_json(output / "summary.json", summary)
    return summary
