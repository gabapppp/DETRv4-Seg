import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from test_train_seg_coco import synthetic_export

from detrseg.coco import single_class_coco
from detrseg.infer import MaskPredictor
from detrseg.model import make_mask_head, mask_loss
from detrseg.roi import restore_roi, roi_canvas, roi_content_bounds, roi_inputs
from scripts.experiment_s_p2_tiny import main as experiment_main
from scripts.train_seg_coco import main as train_main


@pytest.fixture(autouse=True)
def no_external_tracking(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


@pytest.mark.parametrize("shape", [(20, 20), (21, 29), (30, 30)])
def test_native_canvas_keeps_every_pixel_and_small_edge(shape):
    image = np.random.default_rng(12).integers(0, 256, (*shape, 3), dtype=np.uint8)
    canvas = roi_canvas(image, 64, "native")
    left, top, right, bottom = roi_content_bounds((0, 0, shape[1], shape[0]), 64, "native")
    np.testing.assert_array_equal(canvas[top:bottom, left:right], image)
    plane = image[:, :, 0].astype(np.float32)
    restored = restore_roi(
        roi_canvas(plane, 64, "native"), (10, 20, 10 + shape[1], 20 + shape[0]), "native"
    )
    np.testing.assert_array_equal(restored, plane)
    assert np.count_nonzero(canvas) == np.count_nonzero(image)


def test_native_does_not_silently_shrink_larger_rois():
    with pytest.raises(ValueError, match="Increase --size"):
        roi_canvas(np.zeros((65, 30, 3), np.uint8), 64, "native")


@pytest.mark.parametrize("size", [20, 21, 30, 64])
def test_tiny_unet_output_and_stride_one_detail_gradients(size):
    torch.set_num_threads(1)
    model = make_mask_head("tiny_unet", width=8)
    inputs = torch.rand(2, 4, size, size, requires_grad=True)
    outputs = model(inputs)
    assert outputs.shape == (2, 1, size, size)
    mask_loss(outputs, (inputs[:, :1] > 0.5).float()).backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    assert model.detail[0].stride == (1, 1)
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules())
    assert sum(p.numel() for p in model.parameters()) < sum(
        p.numel() for p in make_mask_head("tiny", width=8).parameters()
    )


def test_native_cache_train_checkpoint_and_predictor_roundtrip(tmp_path):
    dataset, _, _, _ = synthetic_export(tmp_path)
    output = tmp_path / "run"
    train_main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--architecture",
            "tiny_unet",
            "--width",
            "8",
            "--size",
            "64",
            "--roi-resize-mode",
            "native",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--val-every",
            "1",
            "--device",
            "cpu",
            "--no-pretrained-encoder",
            "--budget-seconds",
            "60",
        ]
    )
    checkpoint = output / "mask/best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["architecture"] == "tiny_unet" and state["width"] == 8
    assert state["roi_resize_mode"] == "native"
    predictor = MaskPredictor(checkpoint, "cpu")
    assert predictor.roi_resize_mode == "native"

    class EchoPrior(torch.nn.Module):
        def forward(self, x):
            return (x[:, 3:4] - 0.5) * 20

    predictor.model = EchoPrior()
    predictor.mask_overflow = 0
    rgb = np.full((80, 80, 3), 50, np.uint8)
    mask = predictor.predict(rgb, [[15, 20, 20, 30]])[0]
    expected = np.zeros((80, 80), np.uint8)
    expected[20:50, 15:35] = 1
    np.testing.assert_array_equal(mask, expected)
    with np.load(output / "cache/train.npz") as cache:
        # Original 19×18 mask occupies exactly 342 native pixels, with no upsampling.
        assert all(int(mask.sum()) == 19 * 18 for mask in cache["masks"])


def test_single_class_keeps_distinct_instances_and_input_unchanged():
    coco = {
        "images": [],
        "categories": [{"id": 7, "name": "a"}, {"id": 42, "name": "b"}],
        "annotations": [
            {"id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]},
            {"id": 2, "category_id": 42, "bbox": [5, 6, 3, 4]},
        ],
    }
    before = json.dumps(coco)
    result = single_class_coco(coco)
    assert result["categories"] == [{"id": 1, "name": "ball", "supercategory": "object"}]
    assert [ann["category_id"] for ann in result["annotations"]] == [1, 1]
    assert [ann["id"] for ann in result["annotations"]] == [1, 2]
    assert json.dumps(coco) == before


def test_s_p2_training_config_and_saved_labels_are_single_class(tmp_path, monkeypatch):
    import yaml

    from detrseg import detector_train

    _, images, annotations, coco = synthetic_export(tmp_path)
    coco["categories"] = [{"id": 7, "name": "ball"}, {"id": 42, "name": "calculated_ball"}]
    for ann in coco["annotations"]:
        ann["category_id"] = 7 if ann["id"] % 2 else 42
    annotations.write_text(json.dumps(coco))
    original = annotations.read_bytes()
    pretrained = tmp_path / "fake_s.pt"
    torch.save({"model_size": "s", "model": {}}, pretrained)
    output = tmp_path / "detector"

    class StopBeforeModel(Exception):
        pass

    def check_configuration(repo, config):
        saved = yaml.safe_load(config.read_text())
        assert saved["num_classes"] == 1
        assert saved["DFINETransformer"]["feat_strides"] == [4, 8, 16, 32]
        assert saved["__include__"][0].endswith("rtv4_hgnetv2_s_coco.yml")
        raise StopBeforeModel

    monkeypatch.setattr(detector_train, "configuration", check_configuration)
    with pytest.raises(StopBeforeModel):
        detector_train.train_detector(
            annotations,
            images,
            output,
            repo=tmp_path / "fake_repo",
            pretrained=pretrained,
            model_size="s",
            p2=True,
            single_class=True,
            steps=1,
            device="cpu",
        )
    assert json.loads((output / "label_map.json").read_text()) == {"0": 1}
    for split in ("train", "val"):
        data = json.loads((output / f"{split}.json").read_text())
        assert len(data["categories"]) == 1
        assert {ann["category_id"] for ann in data["annotations"]} == {1}
    assert annotations.read_bytes() == original


def test_python_module_entrypoint_exposes_training_cli():
    result = subprocess.run(
        [sys.executable, "-m", "detrseg", "--help"], check=True, capture_output=True, text=True
    )
    assert "train-detector" in result.stdout


def test_experiment_dry_run_does_not_read_data_or_execute_training(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted to execute training")

    monkeypatch.setattr("scripts.experiment_s_p2_tiny.subprocess.run", forbidden)
    experiment_main(["--dry-run", "--det-dataset", "not_read/det", "--seg-dataset", "not_read/seg"])
    commands = capsys.readouterr().out
    assert "--model-size s --p2 --single-class" in commands
    assert "--architecture tiny_unet" in commands
    assert "--roi-resize-mode native" in commands and "resnet" not in commands.lower()


def test_onnx_native_prior_uses_same_coordinates(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from test_onnx_viewer import fixture_detector

    from onnx_viewer.runtime import OnnxPipeline

    config = fixture_detector(tmp_path)
    config.update(mask="mask.onnx", roi_resize_mode="native", padding=0.25, mask_overflow=0)
    (tmp_path / "manifest.json").write_text(json.dumps(config))

    class EchoPrior(torch.nn.Module):
        def forward(self, x):
            return (x[:, 3:4] - 0.5) * 20

    torch.onnx.export(
        EchoPrior(),
        torch.rand(1, 4, 64, 64),
        str(tmp_path / "mask.onnx"),
        input_names=["roi"],
        output_names=["mask_logits"],
        opset_version=17,
        dynamo=False,
    )
    runtime = OnnxPipeline(tmp_path / "manifest.json")
    rgb = np.zeros((100, 200, 3), np.uint8)
    result = runtime.predict(rgb)
    obj = result["objects"][0]
    # Detector fixture box is 40×40: native contour spans 39 pixels each way.
    assert obj["diameter_px"] == pytest.approx(np.hypot(39, 39), abs=0.01)
    assert obj["center_x"] == pytest.approx(99.5, abs=0.01)
    crop, _, bounds = roi_inputs(rgb, obj["bbox"], 64, 0.25, "native")
    assert crop.shape == (64, 64, 3)
    assert bounds[2] - bounds[0] <= 64
