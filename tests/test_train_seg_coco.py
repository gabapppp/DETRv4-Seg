import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from detrseg.coco import encode_mask
from detrseg.prepare import split_images
from scripts.train_seg_coco import find_annotations, find_images, main, source_groups


def synthetic_export(tmp_path, layout="images/default"):
    dataset = tmp_path / "coco"
    images = dataset / layout
    images.mkdir(parents=True)
    mask = np.zeros((40, 48), np.uint8)
    mask[10:28, 12:31] = 1
    rgb = np.zeros((40, 48, 3), np.uint8)
    rgb[mask > 0] = 200
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "ball"}]}
    for i in range(1, 5):
        name = f"roi_{i:06d}.png"
        sample = rgb.copy()
        sample[0, 0] = i
        Image.fromarray(sample).save(images / name)
        coco["images"].append({"id": i, "file_name": name, "height": 40, "width": 48})
        coco["annotations"].append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [12, 10, 19, 18],
                "segmentation": encode_mask(mask),
                "area": int(mask.sum()),
                "iscrowd": 0,
            }
        )
    annotations = dataset / "annotations/instances_default.json"
    annotations.parent.mkdir()
    annotations.write_text(json.dumps(coco))
    return dataset, images, annotations, coco


@pytest.mark.parametrize("layout", ["images/default", "images", "."])
def test_coco_image_layout_resolution(tmp_path, layout):
    dataset, images, annotations, coco = synthetic_export(tmp_path, layout)
    assert find_annotations(dataset) == annotations
    assert find_images(dataset, annotations, coco).resolve() == images.resolve()


def test_source_metadata_keeps_related_crops_together():
    coco = {"images": [{"id": i, "source_image_id": i // 2} for i in range(4)]}
    groups, basis = source_groups(coco)
    mapping = split_images(coco["images"], 0.5, 42, groups)
    assert basis == "source"
    assert mapping[0] == mapping[1]
    assert mapping[2] == mapping[3]
    assert mapping[0] != mapping[2]


def test_missing_segmentation_stops_before_prepare(tmp_path):
    dataset, _, annotations, coco = synthetic_export(tmp_path)
    coco["annotations"][0].pop("segmentation")
    annotations.write_text(json.dumps(coco))
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        main(["--dataset", str(dataset), "--output", str(output), "--prepare-only"])
    assert error.value.code == 1
    assert not output.exists()


def test_cvat_export_trains_only_masks_and_groups_from_review_csv(tmp_path):
    dataset, _, annotations, coco = synthetic_export(tmp_path)
    before = hashlib.sha256(annotations.read_bytes()).hexdigest()
    review_csv = tmp_path / "review.csv"
    with review_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file_name", "source_file_name"])
        writer.writeheader()
        for im in coco["images"]:
            writer.writerow(
                {
                    "file_name": im["file_name"],
                    "source_file_name": f"source_{(im['id'] - 1) // 2}.png",
                }
            )
    output = tmp_path / "run"
    main(
        [
            "--dataset",
            str(dataset),
            "--review-csv",
            str(review_csv),
            "--output",
            str(output),
            "--architecture",
            "tiny",
            "--no-pretrained-encoder",
            "--device",
            "cpu",
            "--size",
            "32",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--val-every",
            "1",
            "--budget-seconds",
            "60",
        ]
    )
    config = json.loads((output / "config.json").read_text())
    report = json.loads((output / "cache/report.json").read_text())
    assert config["group_basis"] == "source"
    assert report["accepted"] == 4 and report["sources"] == {"ground_truth": 4}
    splits = report["image_splits"]
    assert splits["1"] == splits["2"]
    assert splits["3"] == splits["4"]
    assert splits["1"] != splits["3"]
    cached = json.loads((output / "cache/instances_masks.json").read_text())
    expected = {ann["id"]: ann["segmentation"] for ann in coco["annotations"]}
    assert all(ann["segmentation"] == expected[ann["id"]] for ann in cached["annotations"])
    assert (output / "mask/best.pt").is_file()
    assert not (output / "detector").exists()
    assert hashlib.sha256(annotations.read_bytes()).hexdigest() == before
    with pytest.raises(SystemExit):
        main(["--dataset", str(dataset), "--output", str(output), "--prepare-only"])
