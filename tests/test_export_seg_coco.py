import csv
import hashlib
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from detrseg.coco import decode_mask, encode_mask, validate_coco
from scripts.export_seg_coco import export_review, mask_segmentation


@pytest.mark.parametrize("kind", ["pixel", "rectangle", "diagonal", "hole", "edge", "random"])
def test_polygon_export_exact_pixel_roundtrip(kind):
    mask = np.zeros((17, 21), np.uint8)
    if kind == "pixel":
        mask[8, 9] = 1
    elif kind == "rectangle":
        mask[3:12, 5:17] = 1
    elif kind == "diagonal":
        mask[4, 4] = mask[5, 5] = mask[6, 6] = 1
    elif kind == "hole":
        mask[2:14, 2:17] = 1
        mask[5:10, 5:12] = 0
    elif kind == "edge":
        mask[:7, :6] = 1
    else:
        mask = (np.random.default_rng(9).random(mask.shape) > 0.5).astype(np.uint8)
    segmentation, reason = mask_segmentation(mask)
    np.testing.assert_array_equal(decode_mask(segmentation, *mask.shape), mask)
    if kind == "hole":
        assert isinstance(segmentation, dict) and reason == "holes_preserved_as_rle"
    if kind in {"pixel", "rectangle", "diagonal", "edge"}:
        assert isinstance(segmentation, list)


def synthetic_dataset(tmp_path):
    images = tmp_path / "source_images"
    images.mkdir()
    mask = np.zeros((40, 60), np.uint8)
    mask[8:17, 11:21] = 1
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[mask > 0] = 230
    Image.fromarray(rgb).save(images / "first.png")
    Image.fromarray(np.full_like(rgb, 50)).save(images / "second.png")
    coco = {
        "images": [
            {"id": 11, "file_name": "first.png", "width": 60, "height": 40},
            {"id": 22, "file_name": "second.png", "width": 60, "height": 40},
        ],
        "categories": [{"id": 7, "name": "ball"}],
        "annotations": [
            {
                "id": 101,
                "image_id": 11,
                "category_id": 7,
                "bbox": [10, 7, 12, 12],
                "area": 90,
                "segmentation": encode_mask(mask),
                "iscrowd": 0,
            },
            {
                "id": 102,
                "image_id": 11,
                "category_id": 7,
                "bbox": [35, 20, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 201,
                "image_id": 22,
                "category_id": 7,
                "bbox": [20, 10, 15, 12],
                "area": 180,
                "iscrowd": 0,
            },
        ],
    }
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps(coco))
    return annotations, images, coco, mask


def test_roi_review_retains_all_instances_and_source_geometry(tmp_path):
    annotations, images, source, original_mask = synthetic_dataset(tmp_path)
    original_hash = hashlib.sha256(annotations.read_bytes()).hexdigest()
    output = tmp_path / "review"
    report = export_review(annotations, images, output, count=100, make_zip=True)
    assert report["images"] == report["instances"] == 3
    assert report["mask_sources"]["bbox_placeholder"] == 2
    coco = json.loads((output / "annotations/instances_default.json").read_text())
    validate_coco(coco, output / "images/default")
    assert {ann["source_annotation_id"] for ann in coco["annotations"]} == {101, 102, 201}
    assert all(ann["mask_review_status"] == "pending" for ann in coco["annotations"])
    assert all(ann["attributes"]["mask_review_status"] == "pending" for ann in coco["annotations"])
    mapping = {im["id"]: im for im in coco["images"]}
    ann = next(a for a in coco["annotations"] if a["source_annotation_id"] == 101)
    im = mapping[ann["image_id"]]
    x0, y0, x1, y1 = ann["source_crop_xyxy"]
    reconstructed = np.zeros_like(original_mask)
    reconstructed[y0:y1, x0:x1] = decode_mask(ann["segmentation"], im["height"], im["width"])
    np.testing.assert_array_equal(reconstructed, original_mask)
    assert im["source_file_name"] == "first.png"
    assert source["annotations"][0]["bbox"] == ann["source_bbox"]
    with (output / "review.csv").open(encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert hashlib.sha256(annotations.read_bytes()).hexdigest() == original_hash
    with zipfile.ZipFile(report["zip"]) as archive:
        assert archive.testzip() is None
        assert "annotations/instances_default.json" in archive.namelist()
        assert len([name for name in archive.namelist() if name.startswith("images/default/")]) == 3
    with pytest.raises(ValueError, match="new or empty"):
        export_review(annotations, images, output)


def test_full_images_keep_every_existing_box_and_are_reproducible(tmp_path):
    annotations, images, _, _ = synthetic_dataset(tmp_path)
    first, second = tmp_path / "one", tmp_path / "two"
    for output in (first, second):
        report = export_review(annotations, images, output, count=1, mode="images", seed=42)
        coco = json.loads((output / "annotations/instances_default.json").read_text())
        validate_coco(coco, output / "images/default")
        assert report["images"] == 1
        for im in coco["images"]:
            assert (output / "images/default" / im["file_name"]).read_bytes() == (
                images / im["source_file_name"]
            ).read_bytes()
    assert json.loads((first / "annotations/instances_default.json").read_text()) == json.loads(
        (second / "annotations/instances_default.json").read_text()
    )


def test_roi_selection_spreads_first_samples_across_images(tmp_path):
    annotations, images, _, _ = synthetic_dataset(tmp_path)
    output = tmp_path / "balanced"
    export_review(annotations, images, output, count=2)
    coco = json.loads((output / "annotations/instances_default.json").read_text())
    assert {im["source_image_id"] for im in coco["images"]} == {11, 22}


def test_mask_model_is_optional_and_only_proposes_missing_masks(tmp_path):
    import torch

    from detrseg.model import make_mask_head

    annotations, images, _, _ = synthetic_dataset(tmp_path)
    model = make_mask_head("tiny", width=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.out.bias.fill_(10)
    checkpoint = tmp_path / "synthetic_mask.pt"
    torch.save(
        {
            "format_version": 1,
            "architecture": "tiny",
            "width": 4,
            "roi_size": 32,
            "padding": 0.25,
            "mask_overflow": 0.0,
            "fill_mode": "holes",
            "model": model.state_dict(),
        },
        checkpoint,
    )
    report = export_review(
        annotations, images, tmp_path / "model_review", count=3, mask_checkpoint=checkpoint
    )
    assert report["mask_sources"] == {"existing_segmentation": 1, "model_draft": 2}
