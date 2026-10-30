#!/usr/bin/env python3
"""Copy one extracted COCO dataset into fixed train/val subsets."""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detrseg.coco import image_path, read_json, validate_coco, write_json
from detrseg.prepare import split_images
from scripts.train_seg_coco import find_annotations, find_images, source_groups


def split_dataset(
    dataset, output, task, annotations=None, review_csv=None, groups=None, val_fraction=0.2, seed=42
):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if task not in ("det", "seg"):
        raise ValueError("task must be det or seg")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    annotation_path = find_annotations(dataset, annotations)
    coco = read_json(annotation_path)
    audit = validate_coco(coco)
    if not audit["objects"]:
        raise ValueError("No annotations to split")
    if task == "seg" and (audit["without_masks"] or audit["crowd"]):
        raise ValueError("Seg dataset requires segmentation for every annotation and iscrowd=0")
    images_root = find_images(dataset, annotation_path, coco)
    if groups:
        group_map, group_basis = read_json(groups), "provided_groups"
    else:
        if not review_csv and (dataset / "review.csv").is_file():
            review_csv = dataset / "review.csv"
        group_map, group_basis = source_groups(coco, review_csv)
    # Join known source groups when their image files are exact duplicates, so
    # the copies cannot leak across train/val. Annotations are kept separately.
    parents = {
        str(group_map[str(im["id"])]): str(group_map[str(im["id"])]) for im in coco["images"]
    }

    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    hashes, files = {}, {}
    duplicate_files = 0
    for im in coco["images"]:
        source = image_path(images_root, im["file_name"])
        files[im["id"]] = source
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        group = str(group_map[str(im["id"])])
        if digest in hashes:
            first, second = root(group), root(hashes[digest])
            parents[first] = second
            duplicate_files += 1
        else:
            hashes[digest] = group
    group_map = {str(im["id"]): root(str(group_map[str(im["id"])])) for im in coco["images"]}
    mapping = split_images(coco["images"], val_fraction, seed, group_map)
    counts = Counter(mapping[ann["image_id"]] for ann in coco["annotations"])
    if not counts["train"] or not counts["val"]:
        raise ValueError(
            "Both splits need annotations; change seed or provide independent --groups"
        )
    output_images = []
    for index, im in enumerate(coco["images"], 1):
        source = files[im["id"]]
        split = mapping[im["id"]]
        relative = Path(split) / "images" / f"{index:06d}{source.suffix.lower()}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        output_images.append(
            {**im, "file_name": relative.as_posix(), "input_file_name": im["file_name"]}
        )
    combined = {**coco, "images": output_images}
    write_json(output / "annotations/instances_default.json", combined)
    for split in ("train", "val"):
        selected = [im for im in output_images if mapping[im["id"]] == split]
        ids = {im["id"] for im in selected}
        subset = {
            **coco,
            "images": [{**im, "file_name": Path(im["file_name"]).name} for im in selected],
            "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in ids],
        }
        write_json(output / split / "annotations/instances_default.json", subset)
    report = {
        "task": task,
        "dataset": str(dataset),
        "annotations": str(annotation_path),
        "output": str(output),
        "seed": seed,
        "val_fraction": val_fraction,
        "group_basis": group_basis,
        "groups": group_map,
        "group_count": len(set(group_map.values())),
        "duplicate_image_files": duplicate_files,
        "image_splits": {str(key): value for key, value in mapping.items()},
        "images": dict(Counter(mapping.values())),
        "objects": dict(counts),
        "note": "Split by known sources and exact file duplicates. Without source metadata, "
        "different crops from one original image cannot be identified; supply --review-csv or --groups.",
    }
    write_json(output / "split_report.json", report)
    return {key: value for key, value in report.items() if key not in ("groups", "image_splits")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="One extracted COCO dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True, choices=("det", "seg"))
    parser.add_argument("--annotations", help="COCO JSON path relative to dataset")
    parser.add_argument("--review-csv", help="ROI export mapping to original source images")
    parser.add_argument("--groups", help="Explicit JSON image-ID to independent source-group map")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    try:
        print(json.dumps(split_dataset(**vars(parser.parse_args(argv))), indent=2))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(1, f"Split failed: {exc}\n")


if __name__ == "__main__":
    main()
