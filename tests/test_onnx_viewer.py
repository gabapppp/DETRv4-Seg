import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import cv2
from onnx import TensorProto, helper, numpy_helper

from onnx_viewer.runtime import OnnxPipeline, circle_from_contour, draw_overlay, roi_contour


def constant_model(path, inputs, outputs):
    nodes = [
        helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array))
        for name, array in outputs.items()
    ]
    graph = helper.make_graph(
        nodes,
        "fixture",
        inputs,
        [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(array.dtype), array.shape
            )
            for name, array in outputs.items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)


def fixture_detector(tmp_path, postprocessed=False, letterbox=False):
    image_input = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    if postprocessed:
        inputs = [
            image_input,
            helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2]),
        ]
        outputs = {
            "labels": np.array([[0]], np.int64),
            "scores": np.array([[0.99]], np.float32),
            "boxes": np.array([[[80, 30, 120, 70]]], np.float32),
        }
    else:
        inputs = [image_input]
        outputs = {
            "logits": np.array([[[10]]], np.float32),
            "pred_boxes": np.array([[[0.5, 0.5, 0.2, 0.2 if letterbox else 0.4]]], np.float32),
        }
    constant_model(tmp_path / "detector.onnx", inputs, outputs)
    if postprocessed:
        # Compute boxes from the actual sizes input, as upstream does. A non-square
        # source image must catch a width/height swap, even when graph input is square.
        model = onnx.load(tmp_path / "detector.onnx")
        del model.graph.node[-1]
        model.graph.node.extend(
            [
                helper.make_node(
                    "Cast", ["orig_target_sizes"], ["sizes_float"], to=TensorProto.FLOAT
                ),
                helper.make_node("Concat", ["sizes_float", "sizes_float"], ["xyxy_scale"], axis=1),
                helper.make_node(
                    "Constant",
                    [],
                    ["normalized_boxes"],
                    value=numpy_helper.from_array(np.array([[[0.4, 0.3, 0.6, 0.7]]], np.float32)),
                ),
                helper.make_node("Mul", ["normalized_boxes", "xyxy_scale"], ["boxes"]),
            ]
        )
        onnx.save(model, tmp_path / "detector.onnx")
    config = {
        "detector": "detector.onnx",
        "resize_mode": "letterbox" if letterbox else "resize",
        "id2label": {"0": "ball"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(config))
    return config


def ball_image():
    image = np.full((100, 200, 3), 20, np.uint8)
    cv2.circle(image, (100, 50), 20, (220, 220, 220), -1)
    return image


def test_circle_geometry_and_no_fabricated_diameter():
    contour = np.array([[[80, 50]], [[100, 30]], [[120, 50]], [[100, 70]]], np.int32)
    result = circle_from_contour(contour)
    assert result["center_x"] == pytest.approx(100)
    assert result["center_y"] == pytest.approx(50)
    assert result["diameter_px"] == pytest.approx(40, abs=0.001)
    assert circle_from_contour(None)["diameter_px"] is None
    assert roi_contour(np.full((100, 200, 3), 20, np.uint8), [80, 30, 40, 40]) is None


@pytest.mark.parametrize("postprocessed,letterbox", [(False, False), (False, True), (True, False)])
def test_real_ort_detector_geometry_and_contour(tmp_path, postprocessed, letterbox):
    fixture_detector(tmp_path, postprocessed, letterbox)
    runtime = OnnxPipeline(tmp_path / "manifest.json")
    image = ball_image()
    result = runtime.predict(image)
    assert len(result["objects"]) == 1
    obj = result["objects"][0]
    np.testing.assert_allclose(obj["bbox"], [80, 30, 40, 40], atol=0.001)
    assert obj["label"] == "ball"
    assert obj["center_x"] == pytest.approx(100, abs=0.1)
    assert obj["center_y"] == pytest.approx(50, abs=0.1)
    assert obj["diameter_px"] == pytest.approx(40, abs=0.01)
    if postprocessed:
        feed, _ = runtime._prepare(image)
        np.testing.assert_array_equal(feed["orig_target_sizes"], [[200, 100]])
    assert draw_overlay(image, result).shape == image.shape
    assert not np.array_equal(draw_overlay(image, result), image)
    assert runtime.predict(image, score_threshold=1)["objects"] == []


def test_mask_measurement_uses_native_image_coordinates(tmp_path):
    config = fixture_detector(tmp_path)
    logits = np.full((1, 1, 32, 32), -10, np.float32)
    cv2.circle(logits[0, 0], (16, 16), 8, 10, -1)
    constant_model(
        tmp_path / "mask.onnx",
        [helper.make_tensor_value_info("roi", TensorProto.FLOAT, [1, 4, 32, 32])],
        {"mask_logits": logits},
    )
    config.update(mask="mask.onnx", padding=0.25, mask_overflow=0.25)
    (tmp_path / "manifest.json").write_text(json.dumps(config))
    runtime = OnnxPipeline(tmp_path / "manifest.json")
    # A uniform image cannot supply an Otsu circle: measurement must come from ONNX mask.
    result = runtime.predict(np.full((100, 200, 3), 20, np.uint8))
    obj = result["objects"][0]
    assert obj["measurement_source"] == "ONNX mask"
    assert obj["center_x"] == pytest.approx(100, abs=2)
    assert obj["center_y"] == pytest.approx(50, abs=2)
    assert 27 <= obj["diameter_px"] <= 34


def test_unsupported_model_contract_fails_clearly(tmp_path):
    constant_model(
        tmp_path / "unknown.onnx",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        {"unknown": np.zeros((1, 6), np.float32)},
    )
    with pytest.raises(ValueError, match="logits/pred_boxes or labels/boxes/scores"):
        OnnxPipeline(tmp_path / "unknown.onnx")
