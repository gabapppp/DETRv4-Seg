import hashlib
import json
import shutil

import pytest
from test_train_seg_coco import synthetic_export

from detrseg.coco import validate_coco
from detrseg.prepare import fixed_splits
from scripts.split_coco import split_dataset
from scripts.train_seg_coco import main as train_main


@pytest.mark.parametrize("task", ["det", "seg"])
def test_materialized_splits_preserve_masks_files_and_sources(tmp_path, task):
    dataset, images, annotations, coco = synthetic_export(tmp_path)
    if task == "det":
        for ann in coco["annotations"]:
            ann.pop("segmentation")
    for im in coco["images"]:
        im["source_image_id"] = (im["id"] - 1) // 2
    annotations.write_text(json.dumps(coco))
    before = hashlib.sha256(annotations.read_bytes()).hexdigest()
    output = tmp_path / "sub_dataset" / task
    split_dataset(dataset, output, task)
    report = json.loads((output / "split_report.json").read_text())
    combined = json.loads((output / "annotations/instances_default.json").read_text())
    validate_coco(combined, output)
    mapping = fixed_splits(combined["images"], report, report["groups"])
    if task == "det":
        from detrseg.detector_train import detector_splits

        detector_sets, _ = detector_splits(combined, report["image_splits"])
        assert all(
            mapping[im["id"]] == split
            for split, subset in detector_sets.items()
            for im in subset["images"]
        )
    assert mapping[1] == mapping[2]
    assert mapping[3] == mapping[4]
    assert mapping[1] != mapping[3]
    assert combined["annotations"] == coco["annotations"]
    assert hashlib.sha256(annotations.read_bytes()).hexdigest() == before
    for im in combined["images"]:
        assert (output / im["file_name"]).read_bytes() == (
            images / im["input_file_name"]
        ).read_bytes()
    ids = []
    for split in ("train", "val"):
        subset = json.loads((output / split / "annotations/instances_default.json").read_text())
        validate_coco(subset, output / split / "images")
        assert all(mapping[im["id"]] == split for im in subset["images"])
        ids.extend(im["id"] for im in subset["images"])
    assert sorted(ids) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="new or empty"):
        split_dataset(dataset, output, task)


def test_exact_image_duplicates_stay_in_one_split(tmp_path):
    dataset, images, _, _ = synthetic_export(tmp_path)
    shutil.copy2(images / "roi_000001.png", images / "roi_000002.png")
    output = tmp_path / "sub_dataset"
    split_dataset(dataset, output, "seg")
    report = json.loads((output / "split_report.json").read_text())
    assert report["duplicate_image_files"] == 1
    assert report["group_count"] == 3
    assert report["image_splits"]["1"] == report["image_splits"]["2"]


def test_seg_rejects_box_only_input(tmp_path):
    dataset, _, annotations, coco = synthetic_export(tmp_path)
    coco["annotations"][0].pop("segmentation")
    annotations.write_text(json.dumps(coco))
    output = tmp_path / "sub_dataset"
    with pytest.raises(ValueError, match="segmentation"):
        split_dataset(dataset, output, "seg")
    assert not output.exists()


def test_seg_training_reuses_materialized_split_despite_different_seed(tmp_path):
    dataset, _, _, _ = synthetic_export(tmp_path)
    split = tmp_path / "split"
    split_dataset(dataset, split, "seg", seed=42)
    report = json.loads((split / "split_report.json").read_text())
    run = tmp_path / "run"
    train_main(
        [
            "--dataset",
            str(split),
            "--output",
            str(run),
            "--prepare-only",
            "--size",
            "32",
            "--seed",
            "1",
            "--val-fraction",
            "0.75",
        ]
    )
    cache = json.loads((run / "cache/report.json").read_text())
    assert cache["image_splits"] == report["image_splits"]
    assert cache["accepted"] == 4


def test_fixed_split_rejects_mismatched_ids_or_group_leakage():
    images = [{"id": 1}, {"id": 2}]
    with pytest.raises(ValueError, match="exactly"):
        fixed_splits(images, {"image_splits": {"1": "train", "3": "val"}})
    with pytest.raises(ValueError, match="nonempty"):
        fixed_splits(images, {"image_splits": {"1": "train", "2": "train"}})
    with pytest.raises(ValueError, match="crosses"):
        fixed_splits(
            images, {"image_splits": {"1": "train", "2": "val"}}, {"1": "same", "2": "same"}
        )
