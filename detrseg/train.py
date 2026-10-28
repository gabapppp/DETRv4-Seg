"""Budgeted mask-head training. The detector is never loaded or updated here."""

import copy
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .coco import read_json, write_json
from .model import make_mask_head, mask_loss
from .roi import bbox_prior


class CropDataset(Dataset):
    def __init__(self, path, augment=False, bbox_jitter=0.0):
        with np.load(path, allow_pickle=False) as data:
            self.rgb = data["rgb"]
            self.prior = data["prior"]
            self.masks = data["masks"]
            self.image_ids = data["image_ids"]
        self.augment = augment
        self.bbox_jitter = bbox_jitter

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        rgb, prior, mask = self.rgb[index], self.prior[index], self.masks[index]
        if self.augment:
            size = rgb.shape[0]
            # Shift/zoom all three together: labels stay aligned while crop context varies.
            matrix = cv2.getRotationMatrix2D((size / 2, size / 2), 0, np.random.uniform(0.92, 1.08))
            matrix[:, 2] += np.random.uniform(-0.04, 0.04, 2) * size
            rgb = cv2.warpAffine(rgb, matrix, (size, size), borderMode=cv2.BORDER_REFLECT_101)
            prior = cv2.warpAffine(prior, matrix, (size, size), flags=cv2.INTER_NEAREST)
            mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)
            if np.random.random() < 0.5:
                rgb, prior, mask = rgb[:, ::-1], prior[:, ::-1], mask[:, ::-1]
            rgb = np.clip(rgb.astype(np.float32) * np.random.uniform(0.85, 1.15), 0, 255)
            if self.bbox_jitter and prior.any():
                yy, xx = np.nonzero(prior)
                width, height = float(xx.max() - xx.min() + 1), float(yy.max() - yy.min() + 1)
                center_x, center_y = (
                    (xx.max() + xx.min()) / 2 + 0.5,
                    (yy.max() + yy.min()) / 2 + 0.5,
                )
                scales = np.random.uniform(1 - self.bbox_jitter, 1 + self.bbox_jitter / 3, 2)
                offsets = np.random.uniform(-self.bbox_jitter / 4, self.bbox_jitter / 4, 2)
                # Perturb only the detector prior, preserving image and target mask.
                # This teaches recovery beyond a tight/misaligned predicted bbox.
                new_w, new_h = width * scales[0], height * scales[1]
                prior = bbox_prior(
                    [
                        center_x + offsets[0] * width - new_w / 2,
                        center_y + offsets[1] * height - new_h / 2,
                        new_w,
                        new_h,
                    ],
                    (0, 0, size, size),
                )
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
        prior = torch.from_numpy(np.ascontiguousarray(prior)).unsqueeze(0).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        return torch.cat((rgb, prior)), mask


def select_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access a CUDA device")
    return result


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(1)


@torch.inference_mode()
def validate(model, loader, device, amp):
    model.eval()
    totals = np.zeros(3)
    count = 0
    for inputs, target in loader:
        inputs, target = inputs.to(device), target.to(device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(inputs)
            loss = mask_loss(logits, target)
        prediction = logits.float().sigmoid() >= 0.5
        truth = target > 0.5
        intersection = (prediction & truth).sum((1, 2, 3)).float()
        union = (prediction | truth).sum((1, 2, 3)).float()
        dice = (2 * intersection + 1e-6) / (prediction.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1e-6)
        totals += [
            loss.item() * len(inputs),
            ((intersection + 1e-6) / (union + 1e-6)).sum().item(),
            dice.sum().item(),
        ]
        count += len(inputs)
    if count == 0:
        raise ValueError("Empty validation cache")
    return dict(zip(("loss", "iou", "dice"), (totals / count).tolist()))


def train(
    cache,
    output,
    steps=600,
    budget_seconds=1020,
    batch_size=32,
    lr=1e-3,
    device="auto",
    workers=0,
    val_every=100,
    seed=42,
    amp=True,
    threads=4,
    architecture="tiny",
    pretrained_encoder=True,
    initial_checkpoint=None,
):
    if min(steps, budget_seconds, batch_size, val_every, threads) <= 0 or workers < 0 or lr <= 0:
        raise ValueError("Training arguments must be positive (workers may be zero)")
    start = time.monotonic()
    output, cache = Path(output), Path(cache)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}. Choose a new run directory.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    device = select_device(device)
    amp = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    report = read_json(cache / "report.json")
    bbox_jitter = min(0.15, report["settings"].get("mask_overflow", 0.0) * 0.6)
    training = CropDataset(cache / "train.npz", augment=True, bbox_jitter=bbox_jitter)
    validation = CropDataset(cache / "val.npz")
    if set(training.image_ids) & set(validation.image_ids):
        raise ValueError("Image leakage between train and validation cache")
    if len(training) == 0 or len(validation) == 0:
        raise ValueError("Training and validation caches must be non-empty")
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    loader = DataLoader(training, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(validation, shuffle=False, **loader_args)
    model = make_mask_head(architecture, pretrained=pretrained_encoder and not initial_checkpoint)
    if initial_checkpoint:
        starting = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if starting.get("architecture", "tiny") != architecture:
            raise ValueError("Initial checkpoint architecture differs from the requested mask head")
        model.load_state_dict(starting["model"])
    model = model.to(device)
    if architecture == "resnet18_unet":
        groups = [
            {"params": model.encoder.parameters(), "lr": lr * 0.1},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
                "lr": lr,
            },
        ]
    else:
        groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output.mkdir(parents=True, exist_ok=True)
    best_iou, best_step, best_metrics = -1.0, 0, None
    history, completed = [], 0

    def checkpoint(step, metrics):
        return {
            "format_version": 1,
            "architecture": architecture,
            "model": copy.deepcopy(model.state_dict()),
            "width": 16,
            "roi_size": report["settings"]["roi_size"],
            "padding": report["settings"]["padding"],
            "mask_overflow": report["settings"].get("mask_overflow", 0.0),
            "bbox_jitter": bbox_jitter,
            "fill_mode": report["settings"].get("fill_mode", "holes"),
            "step": step,
            "categories": report["audit"]["categories"],
            "validation": metrics,
            "label_sources": report["sources"],
            "validation_note": report["validation_note"],
            "seed": seed,
        }

    iterator = iter(loader)
    if initial_checkpoint:
        initial_metrics = validate(model, val_loader, device, amp)
        best_iou, best_metrics = initial_metrics["iou"], initial_metrics
        history.append(
            {"step": 0, "val": initial_metrics, "elapsed_seconds": time.monotonic() - start}
        )
        torch.save(checkpoint(0, initial_metrics), output / "best.pt")
    while completed < steps:
        if time.monotonic() - start >= budget_seconds:
            break
        try:
            inputs, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, target = next(iterator)
        model.train()
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(inputs)
            loss = mask_loss(logits, target)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {completed + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        completed += 1
        if completed % val_every == 0 or completed == steps:
            metrics = validate(model, val_loader, device, amp)
            history.append(
                {
                    "step": completed,
                    "train_loss": loss.item(),
                    "val": metrics,
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
            print(
                f"step={completed} loss={loss.item():.4f} val_iou={metrics['iou']:.4f} "
                f"val_dice={metrics['dice']:.4f}",
                flush=True,
            )
            if metrics["iou"] > best_iou:
                best_iou, best_step, best_metrics = metrics["iou"], completed, metrics
                torch.save(checkpoint(completed, metrics), output / "best.pt")
            write_json(output / "history.json", history)
    if completed == 0:
        raise ValueError("Time budget expired during setup, before the first training step")
    if not history or history[-1]["step"] != completed:
        metrics = validate(model, val_loader, device, amp)
        history.append(
            {"step": completed, "val": metrics, "elapsed_seconds": time.monotonic() - start}
        )
        if metrics["iou"] > best_iou:
            best_iou, best_step, best_metrics = metrics["iou"], completed, metrics
            torch.save(checkpoint(completed, metrics), output / "best.pt")
    torch.save(checkpoint(completed, history[-1]["val"]), output / "last.pt")
    summary = {
        "steps": completed,
        "requested_steps": steps,
        "best_step": best_step,
        "best_validation": best_metrics,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": amp,
        "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": architecture,
        "pretrained_encoder": pretrained_encoder and architecture == "resnet18_unet",
        "initial_checkpoint": str(Path(initial_checkpoint).resolve())
        if initial_checkpoint
        else None,
        "elapsed_seconds": time.monotonic() - start,
        "budget_seconds": budget_seconds,
        "stopped_by_budget": completed < steps,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else 0,
        "train_objects": len(training),
        "val_objects": len(validation),
        "label_sources": report["sources"],
        "validation_note": report["validation_note"],
        "settings": {
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "workers": workers,
            "val_every": val_every,
            "threads": threads,
            "torch_version": str(torch.__version__),
            "cache": str(cache.resolve()),
        },
    }
    write_json(output / "history.json", history)
    write_json(output / "summary.json", summary)
    return summary
