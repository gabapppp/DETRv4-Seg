import json

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from detrseg.coco import (
    convert_cvat,
    decode_mask,
    encode_mask,
    image_path,
    validate_coco,
    write_json,
)
from detrseg.evaluate import evaluate
from detrseg.infer import MaskPredictor, infer
from detrseg.model import TinyMaskHead
from detrseg.prepare import prepare, split_images
from detrseg.roi import bbox_prior, clean_component, mask_support, pseudo_mask, roi_bounds
from detrseg.train import CropDataset, train


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    coco = {"images": [], "annotations": [], "categories": [{"id": 7, "name": "ball"}]}
    for index in range(4):
        rgb = np.full((64, 64, 3), 230, np.uint8)
        cv2.circle(rgb, (32, 32), 10, (35, 45, 30), -1)
        Image.fromarray(rgb).save(images / f"{index}.png")
        mask = np.zeros((64, 64), np.uint8)
        cv2.circle(mask, (32, 32), 10, 1, -1)
        coco["images"].append({"id": index, "file_name": f"{index}.png", "width": 64, "height": 64})
        coco["annotations"].append(
            {
                "id": index + 1,
                "image_id": index,
                "category_id": 7,
                "bbox": [21, 21, 22, 22],
                "area": int(mask.sum()),
                "segmentation": encode_mask(mask) if index % 2 else [],
                "iscrowd": 0,
            }
        )
    path = tmp_path / "instances.json"
    write_json(path, coco)
    return path, images, coco


def test_cvat_conversion_and_mask_audit(tmp_path):
    Image.new("RGB", (32, 20)).save(tmp_path / "a.png")
    xml = tmp_path / "annotations.xml"
    xml.write_text("""<annotations><image id="0" name="a.png" width="32" height="20">
      <box label="ball" xtl="1" ytl="2" xbr="12" ybr="15"/>
      <polygon label="other" points="15,2;25,2;25,15;15,15"/>
      </image></annotations>""")
    coco = convert_cvat(xml, tmp_path)
    report = validate_coco(coco, tmp_path)
    assert report["with_masks"] == report["without_masks"] == 1
    assert coco["annotations"][0]["bbox"] == [1, 2, 11, 13]
    assert [cat["id"] for cat in coco["categories"]] == [1, 2]


@pytest.mark.parametrize("kind", ["compressed", "uncompressed", "polygon"])
def test_coco_mask_formats(kind):
    mask = np.zeros((9, 13), np.uint8)
    mask[2:7, 3:10] = 1
    if kind == "compressed":
        segmentation = encode_mask(mask)
    elif kind == "uncompressed":
        counts, previous, length = [], 0, 0
        for pixel in mask.ravel(order="F"):
            if pixel != previous:
                counts.append(length)
                previous, length = pixel, 0
            length += 1
        counts.append(length)
        segmentation = {"size": [9, 13], "counts": counts}
    else:
        segmentation = [[3, 2, 10, 2, 10, 7, 3, 7]]
    np.testing.assert_array_equal(decode_mask(segmentation, 9, 13), mask)


def test_audit_rejects_invalid_masks_and_references(dataset):
    _, images, coco = dataset
    coco["annotations"][0]["segmentation"] = [[1, 2, 3]]
    with pytest.raises(ValueError, match="polygon"):
        validate_coco(coco, images)
    coco["annotations"][0]["segmentation"] = []
    coco["annotations"][0]["category_id"] = 0
    with pytest.raises(ValueError, match="category"):
        validate_coco(coco)


def test_image_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        image_path(tmp_path, "../outside.png")


@pytest.mark.parametrize("box", [[0, 0, 10, 10], [-4.2, 3, 12.2, 9], [54, 54, 10, 10]])
def test_clipped_geometry(box):
    bounds = roi_bounds(box, 64, 64)
    assert 0 <= bounds[0] < bounds[2] <= 64
    assert 0 <= bounds[1] < bounds[3] <= 64
    assert bbox_prior(box, bounds).any()


@pytest.mark.parametrize("grabcut", [False, True])
def test_pseudo_mask_finds_object(dataset, grabcut):
    _, images, _ = dataset
    rgb = np.asarray(Image.open(images / "0.png"))
    mask, quality = pseudo_mask(rgb, [21, 21, 22, 22], grabcut=grabcut)
    expected = np.zeros((64, 64), np.uint8)
    cv2.circle(expected, (32, 32), 10, 1, -1)
    assert mask is not None, quality
    intersection = np.logical_and(mask, expected).sum()
    assert intersection / np.logical_or(mask, expected).sum() > 0.95


def test_pseudo_rejects_uniform_and_outside():
    mask, quality = pseudo_mask(np.full((32, 32, 3), 127, np.uint8), [8, 8, 16, 16])
    assert mask is None and quality["reason"] == "low_contrast"
    with pytest.raises(ValueError, match="intersect"):
        roi_bounds([40, 40, 2, 2], 32, 32)


def test_particle_silhouette_recovers_bright_core_with_open_rim():
    """An open C-shaped rim has no enclosed hole for RETR_EXTERNAL to fill."""
    rim = np.zeros((41, 41), np.uint8)
    cv2.circle(rim, (20, 20), 14, 1, thickness=4)
    rim[15:26, 29:] = 0
    prior = np.ones_like(rim)
    old, _ = clean_component(rim, prior, fill_mode="holes")
    silhouette, _ = clean_component(rim, prior, fill_mode="convex")
    assert old[20, 20] == 0
    assert silhouette[20, 20] == 1
    assert silhouette.sum() > old.sum()
    assert not silhouette[0].any()


def test_particle_fill_does_not_modify_existing_ground_truth(dataset, tmp_path):
    annotations, images, original = dataset
    cache = tmp_path / "convex-cache"
    prepare(annotations, images, cache, size=32, max_objects=4, fill_mode="convex")
    generated = json.loads((cache / "instances_masks.json").read_text())
    by_id = {ann["id"]: ann for ann in generated["annotations"]}
    for ann in original["annotations"]:
        if ann["segmentation"]:
            np.testing.assert_array_equal(
                decode_mask(by_id[ann["id"]]["segmentation"], 64, 64),
                decode_mask(ann["segmentation"], 64, 64),
            )


def test_group_split_preserves_sources():
    images = [{"id": i} for i in range(8)]
    groups = {str(i): str(i // 2) for i in range(8)}
    split = split_images(images, 0.25, 42, groups)
    assert set(split.values()) == {"train", "val"}
    for i in range(0, 8, 2):
        assert split[i] == split[i + 1]


def test_prepare_train_infer_roundtrip(dataset, tmp_path):
    annotations, images, _coco = dataset
    cache, run = tmp_path / "cache", tmp_path / "run"
    report = prepare(annotations, images, cache, size=32, max_objects=4, budget_seconds=30)
    assert report["sources"] == {"ground_truth": 2, "pseudo_kmeans": 2}
    with np.load(cache / "train.npz") as training, np.load(cache / "val.npz") as validation:
        assert not set(training["image_ids"]) & set(validation["image_ids"])
    summary = train(
        cache, run, steps=3, batch_size=2, val_every=2, budget_seconds=60, device="cpu", threads=1
    )
    assert summary["steps"] == 3
    assert summary["parameters"] < 200000
    assert (run / "best.pt").exists()
    detections = tmp_path / "detections.json"
    prediction = {"image_id": 0, "category_id": 7, "bbox": [21, 21, 22, 22], "score": 0.9}
    write_json(detections, [prediction])
    out = tmp_path / "masks.json"
    result = infer(
        annotations,
        images,
        detections,
        run / "best.pt",
        out,
        device="cpu",
        overlays=tmp_path / "overlays",
    )
    assert result["instances"] == 1
    saved = json.loads(out.read_text())[0]
    assert {key: saved[key] for key in prediction} == prediction
    assert saved["segmentation"]["size"] == [64, 64]
    assert (tmp_path / "overlays" / "0.png").exists()
    if saved["area"]:
        mask = decode_mask(saved["segmentation"], 64, 64)
        assert mask.sum() == saved["area"]
        assert not mask[:21].any() and not mask[:, :21].any()
    metrics = evaluate(
        cache / "instances_masks.json",
        images,
        run / "best.pt",
        tmp_path / "evaluation.json",
        device="cpu",
        preview=tmp_path / "compare.jpg",
    )
    assert metrics["objects"] == report["val_objects"]
    assert 0 <= metrics["native_mask_iou"] <= 1
    assert (tmp_path / "compare.jpg").exists()
    with pytest.raises(ValueError, match="not empty"):
        prepare(annotations, images, cache, size=32, max_objects=4)


def test_predictor_native_coordinates(tmp_path):
    """Constant foreground logits must paste to the clipped native bbox, not the resized ROI."""
    model = TinyMaskHead()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.out.bias.data.fill_(10)
    checkpoint = tmp_path / "constant.pt"
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "width": 16,
            "roi_size": 32,
            "padding": 0.2,
        },
        checkpoint,
    )
    predictor = MaskPredictor(checkpoint, "cpu")
    rgb = np.zeros((25, 40, 3), np.uint8)
    boxes = [[-2, 3, 9, 10], [31, 18, 9, 7]]
    masks = predictor.predict(rgb, boxes, batch_size=1)
    for box, mask in zip(boxes, masks):
        expected = bbox_prior(box, (0, 0, 40, 25))
        np.testing.assert_array_equal(mask, expected)
    assert predictor.predict(rgb, []) == []
    diagnostic_masks, quality = predictor.predict(rgb, boxes, diagnostics=True)
    assert len(diagnostic_masks) == len(quality) == len(boxes)
    assert quality[0]["mean_pixel_activation_uncalibrated"] > 0.99
    assert quality[0]["postprocess_added_fraction"] == 0


def test_predictor_allows_exact_25_percent_mask_extension(tmp_path):
    model = TinyMaskHead()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.out.bias.data.fill_(10)
    checkpoint = tmp_path / "extended.pt"
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "width": 16,
            "roi_size": 32,
            "padding": 0.25,
            "mask_overflow": 0.25,
        },
        checkpoint,
    )
    predictor = MaskPredictor(checkpoint, "cpu")
    rgb = np.zeros((60, 70, 3), np.uint8)
    for box in ([20, 20, 20, 20], [0, 0, 12, 16], [61, 52, 9, 8]):
        result = predictor.predict(rgb, [box])[0]
        allowed = mask_support(box, (0, 0, 70, 60), 0.25)
        np.testing.assert_array_equal(result, allowed)
        assert result.sum() > bbox_prior(box, (0, 0, 70, 60)).sum()


def test_output_padding_does_not_expand_pseudo_labels_into_neighbors(dataset, tmp_path):
    annotations, images, _ = dataset
    cache = tmp_path / "safe-padding"
    report = prepare(
        annotations,
        images,
        cache,
        size=32,
        max_objects=4,
        padding=0.25,
        mask_overflow=0.25,
        fill_mode="convex",
    )
    assert report["settings"]["mask_overflow"] == 0.25
    assert report["settings"]["pseudo_labels_clipped_to_annotation_bbox"]
    generated = json.loads((cache / "instances_masks.json").read_text())
    for ann in generated["annotations"]:
        if ann["mask_source"].startswith("pseudo"):
            mask = decode_mask(ann["segmentation"], 64, 64)
            prior = bbox_prior(ann["bbox"], (0, 0, 64, 64))
            assert not np.any(mask & (1 - prior))


def test_pseudo_label_stays_anchored_to_annotated_particle():
    image = np.full((64, 64, 3), 230, np.uint8)
    cv2.circle(image, (32, 32), 12, (25, 35, 20), -1)
    box = [21, 21, 22, 22]
    result, quality = pseudo_mask(image, box, padding=0.25, fill_mode="convex")
    assert result is not None, quality
    inside = bbox_prior(box, (0, 0, 64, 64))
    assert not np.any(result & (1 - inside))


def test_mask_padding_cannot_silently_truncate_overflow(dataset, tmp_path):
    annotations, images, _ = dataset
    with pytest.raises(ValueError, match="crop padding"):
        prepare(annotations, images, tmp_path / "invalid-cache", padding=0.2, mask_overflow=0.25)


def test_bbox_jitter_changes_prior_but_keeps_target_aligned(dataset, tmp_path):
    annotations, images, _ = dataset
    cache = tmp_path / "jitter"
    prepare(annotations, images, cache, size=32, max_objects=4)
    baseline = CropDataset(cache / "train.npz", augment=True, bbox_jitter=0.0)
    perturbed = CropDataset(cache / "train.npz", augment=True, bbox_jitter=0.15)
    np.random.seed(42)
    original_inputs, original_target = baseline[0]
    np.random.seed(42)
    jittered_inputs, jittered_target = perturbed[0]
    assert torch.equal(original_inputs[:3], jittered_inputs[:3])
    assert torch.equal(original_target, jittered_target)
    assert not torch.equal(original_inputs[3], jittered_inputs[3])
