import json

import numpy as np
import pytest
from PIL import Image

from detrseg.coco import encode_mask, write_json
from detrseg.detector_train import DetectionDataset, detector_splits
from detrseg.metrics import coco_metrics, detection_at_threshold
from detrseg.pipeline import bundle_paths, predict
from detrseg.upstream import move_model


def small_coco():
    return {
        "images": [
            {"id": 2, "width": 40, "height": 30, "file_name": "a.png"},
            {"id": 8, "width": 40, "height": 30, "file_name": "b.png"},
        ],
        "categories": [{"id": 42, "name": "second"}, {"id": 7, "name": "first"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 2,
                "category_id": 42,
                "bbox": [10, 5, 20, 10],
                "area": 200,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 8,
                "category_id": 7,
                "bbox": [5, 5, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
    }


def test_detector_split_and_noncontiguous_labels(tmp_path):
    coco = small_coco()
    splits, mapping = detector_splits(coco, {"2": "train", "8": "val"})
    assert mapping == {0: 7, 1: 42}
    assert splits["train"]["annotations"][0]["category_id"] == 42
    assert len(splits["val"]["annotations"]) == 1
    Image.new("RGB", (40, 30)).save(tmp_path / "a.png")
    dataset = DetectionDataset(splits["train"], tmp_path, 128, mapping)
    image, target = dataset[0]
    assert image.shape == (3, 128, 128)
    assert target["labels"].tolist() == [1]
    np.testing.assert_allclose(target["boxes"].numpy(), [[0.5, 1 / 3, 0.5, 1 / 3]], rtol=1e-6)
    with pytest.raises(ValueError, match="image IDs"):
        detector_splits(coco, {"2": "train"})


def test_coco_detection_and_mask_ap():
    coco = small_coco()
    predictions = []
    for ann in coco["annotations"]:
        mask = np.zeros((30, 40), np.uint8)
        x, y, w, h = ann["bbox"]
        mask[y : y + h, x : x + w] = 1
        ann["segmentation"] = encode_mask(mask)
        predictions.append({**ann, "score": 1.0})
    assert coco_metrics(coco, predictions)["AP"] == pytest.approx(1.0)
    assert coco_metrics(coco, predictions, "segm")["AP"] == pytest.approx(1.0)
    assert coco_metrics(coco, [])["AP"] == 0
    # An accurate box must not conceal an incorrect mask when evaluating segm.
    predictions[0]["segmentation"] = encode_mask(np.zeros((30, 40), np.uint8))
    assert coco_metrics(coco, predictions, "segm")["AP"] < 0.6


def test_detection_threshold_counts_misses_duplicates_and_wrong_classes():
    coco = small_coco()
    first = coco["annotations"][0]
    predictions = [
        {**first, "score": 0.9},
        {**first, "score": 0.8},
        {**coco["annotations"][1], "category_id": 42, "score": 0.95},
    ]
    result = detection_at_threshold(coco, predictions, threshold=0.5)
    assert result["overall"]["true_positives"] == 1
    assert result["overall"]["false_positives"] == 2
    assert result["overall"]["missed_objects"] == 1


def test_predict_exports_without_annotation_boxes(tmp_path, monkeypatch):
    from detrseg import pipeline

    class FakePredictor:
        def __init__(self, bundle, device):
            self.categories = [{"id": 7, "name": "particle"}]

        def predict(self, rgb, *args):
            mask = np.zeros(rgb.shape[:2], np.uint8)
            mask[2:6, 2:6] = 1
            return [
                {
                    "category_id": 7,
                    "score": 0.8,
                    "bbox": [2, 2, 4, 4],
                    "segmentation": encode_mask(mask),
                    "area": 16,
                    "equivalent_diameter_px": 4.51,
                    "empty_mask": False,
                }
            ], [mask]

    monkeypatch.setattr(pipeline, "InstancePredictor", FakePredictor)
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (12, 10)).save(source / "a.png")
    out = tmp_path / "results"
    report = predict("unused.json", source, out, device="cpu")
    assert report["instances"] == 1
    assert (out / "overlays/000000.png").is_file()
    assert "particle" in (out / "measurements.csv").read_text()
    assert json.loads((out / "instances.json").read_text())[0]["image_id"] == 0
    with pytest.raises(ValueError, match="not empty"):
        predict("unused.json", source, out)


def test_bundle_resolves_relative_paths(tmp_path):
    bundle = tmp_path / "bundle.json"
    write_json(
        bundle,
        {
            "format_version": 1,
            **{
                name: name
                for name in (
                    "repo",
                    "config",
                    "detector_checkpoint",
                    "mask_checkpoint",
                    "label_map",
                )
            },
        },
    )
    _, paths = bundle_paths(bundle)
    assert paths["mask_checkpoint"] == str(tmp_path / "mask_checkpoint")


def test_upstream_position_cache_moves_without_checkpoint_keys():
    import torch

    module = torch.nn.Module()
    module.pos_embed2 = torch.ones(1, 4, 8)
    moved = move_model(module, "meta")
    assert moved.pos_embed2.device.type == "meta"
    assert "pos_embed2" not in moved.state_dict()
