#!/usr/bin/env python3
r"""Merge two COCO categories while preserving every image, box and instance mask.

Example, from the project directory:
    python3 scripts/merge_coco_labels.py \
        --input dataset/annotations/instances_default.json \
        --output dataset/annotations/instances_ball.json \
        --source-ids 1 2 --target-id 1 --name ball

Uses only the Python standard library. The input and existing output files are
never overwritten. Without --source-ids, the input must contain exactly two labels.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def merge_labels(data, source_ids=None, target_id=1, name="ball"):
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list) for key in ("images", "annotations", "categories")
    ):
        raise ValueError(
            "Input must be a COCO object with images, annotations and categories lists"
        )
    if target_id < 0 or not name.strip():
        raise ValueError("Target ID must be nonnegative and the label name must not be empty")
    categories = data["categories"]
    by_id = {category["id"]: category for category in categories}
    if len(by_id) != len(categories):
        raise ValueError("Duplicate category IDs in input")
    if source_ids is None:
        if len(categories) != 2:
            raise ValueError("Input must have exactly two labels, or specify --source-ids ID1 ID2")
        source_ids = list(by_id)
    selected = set(source_ids)
    if len(source_ids) != 2 or len(selected) != 2:
        raise ValueError("Select exactly two different source category IDs")
    if not selected <= by_id.keys():
        raise ValueError(f"Unknown source category IDs: {sorted(selected - by_id.keys())}")
    if target_id in by_id and target_id not in selected:
        raise ValueError("Target ID belongs to a label outside the selected pair")
    counts = Counter(annotation["category_id"] for annotation in data["annotations"])
    unknown = counts.keys() - by_id.keys()
    if unknown:
        raise ValueError(f"Annotations refer to unknown category IDs: {sorted(unknown)}")

    # Keep metadata from the target label if it already exists, otherwise from
    # the first source. Other categories and all annotation geometry stay intact.
    merged = {**by_id.get(target_id, by_id[source_ids[0]]), "id": target_id, "name": name.strip()}
    merged_categories = []
    inserted = False
    for category in categories:
        if category["id"] not in selected:
            merged_categories.append(category)
        elif not inserted:
            merged_categories.append(merged)
            inserted = True
    result = {
        **data,
        "categories": merged_categories,
        "annotations": [
            {**annotation, "category_id": target_id}
            if annotation["category_id"] in selected
            else annotation
            for annotation in data["annotations"]
        ],
    }
    report = {
        "source_labels": [
            {"id": key, "name": by_id[key]["name"], "annotations": counts[key]}
            for key in source_ids
        ],
        "target_label": {"id": target_id, "name": merged["name"]},
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "changed_annotations": sum(counts[key] for key in selected if key != target_id),
        "remaining_categories": len(merged_categories),
    }
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="Source COCO annotation JSON")
    parser.add_argument("--output", type=Path, required=True, help="New COCO annotation JSON")
    parser.add_argument("--source-ids", nargs=2, type=int, metavar=("ID1", "ID2"))
    parser.add_argument("--target-id", type=int, default=1, help="Merged category ID (default: 1)")
    parser.add_argument("--name", default="ball", help="Merged label name (default: ball)")
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different files")
        with args.input.open(encoding="utf-8") as file:
            data = json.load(file)
        result, report = merge_labels(data, args.source_ids, args.target_id, args.name)
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(serialized)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({**report, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
