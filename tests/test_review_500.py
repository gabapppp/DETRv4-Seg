import json

import numpy as np
import pytest
from PIL import Image

from detrseg.coco import encode_mask
from scripts.export_seg_coco import export_review


def test_500_additional_review_instances_exclude_previous_cohort(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (40, 40)).save(images / "source.png")
    coco = {
        "images": [{"id": 1, "file_name": "source.png", "width": 40, "height": 40}],
        "categories": [{"id": 1, "name": "ball"}, {"id": 2, "name": "calculated_ball"}],
        "annotations": [],
    }
    for i in range(502):
        x, y = i % 38 + 1, i // 38 + 1
        mask = np.zeros((40, 40), np.uint8)
        mask[y, x] = 1
        coco["annotations"].append(
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1 + i % 2,
                "bbox": [x, y, 1, 1],
                "segmentation": encode_mask(mask),
                "area": 1,
                "iscrowd": 0,
            }
        )
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps(coco))
    old = tmp_path / "old_review"
    export_review(annotations, images, old, count=2)
    old_json = old / "annotations/instances_default.json"
    reviewed = json.loads(old_json.read_text())
    # IDs may change on round-trip through an annotation tool; source identity remains.
    for ann in reviewed["annotations"]:
        ann["id"] += 9000
    old_json.write_text(json.dumps(reviewed))
    report = export_review(
        annotations,
        images,
        tmp_path / "review500",
        count=500,
        exclude_reviewed=[old_json],
        require_count=True,
        make_zip=True,
        single_class=True,
    )
    assert report["images"] == report["instances"] == 500
    assert report["excluded_instances"] == 2
    fresh = json.loads((tmp_path / "review500/annotations/instances_default.json").read_text())
    assert {ann["source_annotation_id"] for ann in reviewed["annotations"]}.isdisjoint(
        ann["source_annotation_id"] for ann in fresh["annotations"]
    )
    assert all(ann["mask_review_status"] == "pending" for ann in fresh["annotations"])
    assert len(fresh["categories"]) == 1
    assert {ann["category_id"] for ann in fresh["annotations"]} == {1}
    assert len(json.loads(annotations.read_text())["categories"]) == 2
    with pytest.raises(ValueError, match="only 500 remain"):
        export_review(
            annotations,
            images,
            tmp_path / "too_many",
            count=501,
            exclude_reviewed=[old_json],
            require_count=True,
        )
    assert not (tmp_path / "too_many").exists()
