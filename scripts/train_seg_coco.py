#!/usr/bin/env python3
"""Prepare and train only the mask head from an extracted COCO segmentation export."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detrseg.coco import image_path, read_json, validate_coco, write_json
from detrseg.prepare import fixed_splits, prepare, split_images
from detrseg.tracking import add_tracking_arguments


def find_annotations(dataset, annotations=None):
    if annotations:
        path = Path(annotations)
        return path if path.is_absolute() else dataset / path
    default = dataset / "annotations/instances_default.json"
    if default.is_file():
        return default
    candidates = sorted(dataset.rglob("instances_*.json"))
    if len(candidates) != 1:
        raise ValueError(
            "Expected one COCO instances JSON; specify --annotations relative to dataset"
        )
    return candidates[0]


def find_images(dataset, annotations, coco):
    subset = annotations.stem.removeprefix("instances_")
    candidates = [
        annotations.parent.parent / "images" / subset,
        dataset / "images" / subset,
        dataset / "images",
        dataset,
    ]
    for root in candidates:
        if all(image_path(root, im["file_name"]).is_file() for im in coco["images"]):
            return root
    raise ValueError(
        "Image files do not match COCO file_name; export COCO with Save images enabled"
    )


def source_groups(coco, review_csv=None):
    rows = {}
    if review_csv:
        with Path(review_csv).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                source = row.get("source_file_name") or row.get("source_image_id")
                if not source or not row.get("file_name"):
                    raise ValueError(
                        "review.csv requires file_name and source_file_name/source_image_id"
                    )
                name = Path(row["file_name"]).name
                if name in rows and rows[name] != source:
                    raise ValueError(f"Ambiguous source mapping for {name}")
                rows[name] = source
    result, origins = {}, set()
    for im in coco["images"]:
        if review_csv:
            source = rows.get(Path(im["file_name"]).name)
            if source is None:
                raise ValueError(f"Crop is absent from review.csv: {im['file_name']}")
            origin = "source"
        elif im.get("source_file_name") is not None:
            source, origin = im["source_file_name"], "source"
        elif im.get("source_image_id") is not None:
            source, origin = im["source_image_id"], "source"
        else:
            source, origin = im["id"], "image"
        result[str(im["id"])] = f"{origin}:{source}"
        origins.add(origin)
    if len(origins) != 1:
        raise ValueError("Partial source metadata; supply --review-csv for all images")
    return result, "source" if origins == {"source"} else "image"


def run(args):
    dataset = Path(args.dataset).resolve()
    annotations = find_annotations(dataset, args.annotations)
    coco = read_json(annotations)
    audit = validate_coco(coco)
    if audit["objects"] < 2 or audit["without_masks"] or audit["crowd"]:
        raise ValueError(
            "Need at least two instance annotations, all with segmentation and iscrowd=0"
        )
    images = find_images(dataset, annotations, coco)
    review_csv = args.review_csv
    if not review_csv and (dataset / "review.csv").is_file():
        review_csv = dataset / "review.csv"
    split_report = Path(args.split_report) if args.split_report else dataset / "split_report.json"
    if args.split_report or split_report.is_file():
        split_report = split_report.resolve()
        saved = read_json(split_report)
        groups = saved.get("groups")
        fixed_splits(coco["images"], saved, groups)
        group_basis = "saved_split"
    else:
        split_report = None
        groups, group_basis = source_groups(coco, review_csv)
        split_images(coco["images"], args.val_fraction, args.seed, groups)
    initial = Path(args.initial_checkpoint).resolve() if args.initial_checkpoint else None
    if initial and not initial.is_file():
        raise ValueError(f"Initial checkpoint not found: {initial}")
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    config = {
        **vars(args),
        "annotations": str(annotations),
        "images": str(images),
        "group_basis": group_basis,
        "review_csv": str(review_csv) if review_csv else None,
        "split_report": str(split_report) if split_report else None,
        "objects": audit["objects"],
        "output": str(output),
    }
    print(json.dumps(config, indent=2), flush=True)
    if group_basis == "image":
        print(
            "Split by exported image. For crops from the same source, pass original --review-csv.",
            flush=True,
        )
    write_json(output / "config.json", config)
    group_file = output / "groups.json"
    if groups is not None:
        write_json(group_file, groups)
    report = prepare(
        annotations=annotations,
        images_root=images,
        output=output / "cache",
        size=args.size,
        padding=0.25,
        mask_overflow=0.25,
        max_objects=audit["objects"],
        val_fraction=args.val_fraction,
        seed=args.seed,
        groups=group_file if groups is not None else None,
        split_report=split_report,
        fill_mode="holes",
        budget_seconds=args.prepare_budget_seconds,
        roi_resize_mode=args.roi_resize_mode,
    )
    if report["accepted"] != audit["objects"]:
        raise ValueError(f"Some masks were not cached. Inspect {output / 'cache/report.json'}")
    print(
        f"Cache: {report['train_objects']} train / {report['val_objects']} val objects", flush=True
    )
    if args.prepare_only:
        return {"cache": str(output / "cache"), "output": str(output)}
    from detrseg.train import train

    result = train(
        cache=output / "cache",
        output=output / "mask",
        architecture=args.architecture,
        initial_checkpoint=initial,
        pretrained_encoder=not args.no_pretrained_encoder,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        budget_seconds=args.budget_seconds,
        val_every=args.val_every,
        seed=args.seed,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        mlflow_log_every=args.mlflow_log_every,
        width=args.width,
    )
    return {"output": str(output), "checkpoint": str(output / "mask/best.pt"), "training": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extracted COCO folder, including images")
    parser.add_argument("--annotations", help="Optional COCO JSON path relative to dataset")
    parser.add_argument(
        "--split-report", help="Reuse a fixed split; defaults to dataset/split_report.json"
    )
    parser.add_argument(
        "--review-csv", help="Original ROI exporter CSV; keeps source images in one split"
    )
    parser.add_argument("--output", default=f"runs/seg_gt_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument(
        "--initial-checkpoint", help="Fine-tune this mask head instead of starting fresh"
    )
    parser.add_argument(
        "--architecture", choices=("tiny", "tiny_unet", "resnet18_unet"), default="resnet18_unet"
    )
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--roi-resize-mode", choices=("resize", "native"), default="resize")
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    parser.add_argument("--size", type=int, default=192, help="ROI cache size in pixels")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--prepare-budget-seconds", type=float, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    add_tracking_arguments(parser)
    try:
        print(json.dumps(run(parser.parse_args(argv)), indent=2))
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Seg training failed: {exc}\n")


if __name__ == "__main__":
    main()
