"""Visibility and export tests use synthetic pixels only; no trained model is loaded."""

import csv
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("onnxruntime")
import cv2
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog

from onnx_viewer import app as viewer
from onnx_viewer.runtime import draw_overlay


def synthetic_result():
    return {
        "objects": [
            {
                "bbox": [50 + offset, 70, 60, 60],
                "label": "ball",
                "score": 0.9,
                "contour": np.array(
                    [[[60 + offset, 80]], [[100 + offset, 80]], [[80 + offset, 120]]],
                    np.int32,
                ),
                "center_x": 80 + offset,
                "center_y": 100,
                "diameter_px": 50.0,
                "measurement_source": "ONNX mask",
            }
            for offset in (0, 200)
        ],
        "elapsed_ms": 2.0,
        "provider": "CPUExecutionProvider",
        "measurement_source": "ONNX mask",
    }


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    pixels = np.full((800, 1400, 3), [20, 30, 40], np.uint8)
    monkeypatch.setattr(viewer, "load_rgb", lambda _path: pixels.copy())
    widget = viewer.MainWindow()
    widget.show()
    qapp.processEvents()
    widget.paths = [tmp_path / "one.png", tmp_path / "two.png"]
    widget.image_list.addItems([path.name for path in widget.paths])
    widget.image_list.setCurrentRow(0)
    widget._on_result(str(widget.paths[0]), synthetic_result())
    yield widget
    widget.close()
    qapp.processEvents()


@pytest.mark.parametrize("layer", ["show_boxes", "show_contours", "show_circles", "show_diameters"])
def test_each_layer_can_be_displayed_independently(layer):
    pixels = np.zeros((200, 400, 3), np.uint8)
    result = synthetic_result()
    original = deepcopy(result)
    options = dict.fromkeys(
        ("show_boxes", "show_contours", "show_circles", "show_diameters"), False
    )
    np.testing.assert_array_equal(draw_overlay(pixels, result, **options), pixels)
    options[layer] = True
    assert np.any(draw_overlay(pixels, result, **options) != pixels)
    np.testing.assert_array_equal(
        result["objects"][0]["contour"], original["objects"][0]["contour"]
    )
    assert result["objects"][0]["diameter_px"] == original["objects"][0]["diameter_px"]
    assert not pixels.any()


def test_instance_and_box_visibility_keep_stable_ids(monkeypatch):
    pixels = np.zeros((200, 400, 3), np.uint8)
    result = synthetic_result()
    np.testing.assert_array_equal(draw_overlay(pixels, result, hidden_instances={1, 2}), pixels)
    boxes_only = {"show_contours": False, "show_circles": False, "show_diameters": False}
    rendered = draw_overlay(pixels, result, hidden_boxes={1}, **boxes_only)
    assert not rendered[:, :200].any()
    assert rendered[:, 200:].any()
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda _image, text, *_args: texts.append(text))
    draw_overlay(pixels, result, hidden_instances={1})
    assert texts == ["C2 D=50.00 px", "C2 D=50.00 px"]


def test_visibility_redraw_uses_cached_pixels_and_preserves_zoom(window, monkeypatch):
    def no_image_read(_path):
        pytest.fail("Visibility changes must use cached pixels")

    monkeypatch.setattr(viewer, "load_rgb", no_image_read)
    window.actual_size_button.click()
    window.preview.horizontalScrollBar().setValue(150)
    before = window.preview.view_state()
    original = window.original_preview._source.toImage()
    window.table.item(0, 1).setCheckState(Qt.CheckState.Unchecked)
    assert window.hidden_boxes[str(window.paths[0])] == {1}
    window.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert window.hidden_instances[str(window.paths[0])] == {2}
    for checkbox in window.overlay_checks.values():
        checkbox.setChecked(False)
    assert window.preview._source.toImage() == original
    assert window.original_preview._source.toImage() == original
    assert window.preview.view_state()[0] == pytest.approx(before[0])
    assert abs(window.preview.view_state()[1].x() - before[1].x()) < 2
    assert window.original_preview.view_state()[0] == pytest.approx(before[0])
    assert not window.preview.view_state()[2]
    assert window.worker is None
    assert window.results[str(window.paths[0])]["objects"][0]["diameter_px"] == 50.0


def test_visibility_is_per_image_and_resets_on_new_inference(window):
    path = str(window.paths[0])
    window.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    window.table.item(1, 1).setCheckState(Qt.CheckState.Unchecked)
    window.image_list.setCurrentRow(1)
    window._on_result(str(window.paths[1]), synthetic_result())
    assert window.table.item(0, 0).checkState() == Qt.CheckState.Checked
    window.image_list.setCurrentRow(0)
    assert window.table.item(0, 0).checkState() == Qt.CheckState.Unchecked
    assert window.table.item(1, 1).checkState() == Qt.CheckState.Unchecked
    window.hide_all_button.click()
    assert window.hidden_instances[path] == {1, 2}
    window.show_all_button.click()
    assert not window.hidden_instances[path]
    assert not window.hidden_boxes[path]
    window.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    window._on_result(path, synthetic_result())
    assert path not in window.hidden_instances
    assert window.table.item(0, 0).checkState() == Qt.CheckState.Checked


def test_checkbox_accepts_keyboard_toggle(window):
    window.table.setCurrentCell(0, 1)
    QTest.keyClick(window.table, Qt.Key.Key_Space)
    assert window.table.item(0, 1).checkState() == Qt.CheckState.Unchecked
    assert window.hidden_boxes[str(window.paths[0])] == {1}


def test_hold_to_peek_restores_without_rereading_or_changing_measurements(window, monkeypatch):
    monkeypatch.setattr(viewer, "load_rgb", lambda _path: pytest.fail("Must use cached pixels"))
    window.preview.set_view_state(1.5, QPointF(140, 130), False)
    window.preview._notify()
    position = window.preview.mapFromScene(QPointF(80, 100))
    before = window.preview._source.toImage()
    QTest.mousePress(window.preview.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert window._peek_instance == 1
    assert window.preview._source.toImage() != before
    assert not window.hidden_instances
    assert window.table.item(0, 0).checkState() == Qt.CheckState.Checked
    QTest.mouseRelease(window.preview.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert window._peek_instance is None
    assert window.preview._source.toImage() == before
    assert window.preview.view_state()[0] == pytest.approx(1.5)
    assert window.worker is None


def test_h_hides_hovered_instance_until_explicit_restore(window, qapp):
    position = window.preview.mapFromScene(QPointF(80, 100))
    window.preview.setFocus()
    QTest.mouseMove(window.preview.viewport(), position)
    qapp.processEvents()
    QTest.keyClick(window.preview, Qt.Key.Key_H)
    path = str(window.paths[0])
    assert window.hidden_instances[path] == {1}
    assert window.table.item(0, 0).checkState() == Qt.CheckState.Unchecked
    assert window._instance_at(QPointF(80, 100)) is None
    window.image_list.setCurrentRow(1)
    window.image_list.setCurrentRow(0)
    assert window.hidden_instances[path] == {1}
    window.show_all_button.click()
    assert not window.hidden_instances[path]


def test_drag_cancels_peek_and_empty_space_does_not_hide(window):
    position = window.preview.mapFromScene(QPointF(80, 100))
    QTest.mousePress(window.preview.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert window._peek_instance == 1
    QTest.mouseMove(window.preview.viewport(), position + QPoint(30, 20))
    assert window._peek_instance is None
    QTest.mouseRelease(window.preview.viewport(), Qt.MouseButton.LeftButton)
    window._hide_at(QPointF(600, 600))
    assert not window.hidden_instances


def test_diameter_is_a_compact_screen_widget_instead_of_image_text(window, qapp):
    for key, checkbox in window.overlay_checks.items():
        checkbox.setChecked(key == "show_diameters")
    original = window.original_preview._source.toImage()
    assert window.preview._source.toImage() == original
    qapp.processEvents()
    QTest.mouseMove(window.preview.viewport(), QPoint(1, 1))
    position = window.preview.mapFromScene(QPointF(80, 100))
    QTest.mouseMove(window.preview.viewport(), position)
    qapp.processEvents()
    assert window.diameter_detail.isVisible()
    assert window.diameter_detail.text() == "C1 · Ø 50.00 px"
    height = window.diameter_detail.height()
    window.preview.zoom(1.2, position)
    window._show_hover_detail(QPointF(80, 100))
    assert window.diameter_detail.height() == height
    assert window.preview._source.toImage() == original
    window.overlay_checks["show_diameters"].setChecked(False)
    assert not window.diameter_detail.isVisible()


def test_pick_prefers_mask_and_ignores_hidden_instance(window):
    objects = window.results[str(window.paths[0])]["objects"]
    objects[1]["bbox"] = [40, 60, 280, 100]
    assert window._instance_at(QPointF(80, 100)) == 1
    window._hide_at(QPointF(80, 100))
    assert window._instance_at(QPointF(80, 100)) == 2
    window._hide_at(QPointF(-1, 100))
    assert window.hidden_instances[str(window.paths[0])] == {1}


def test_h_during_peek_stays_hidden_after_release(window, qapp):
    position = window.preview.mapFromScene(QPointF(80, 100))
    QTest.mouseMove(window.preview.viewport(), position)
    QTest.mousePress(window.preview.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qapp.processEvents()
    QTest.keyClick(window.preview, Qt.Key.Key_H)
    QTest.mouseRelease(window.preview.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert window.hidden_instances[str(window.paths[0])] == {1}
    assert window._peek_instance is None
    assert window.table.item(0, 0).checkState() == Qt.CheckState.Unchecked


def test_h_shortcut_does_not_intercept_typing_in_model_field(window, qapp):
    position = window.preview.mapFromScene(QPointF(80, 100))
    QTest.mouseMove(window.preview.viewport(), position)
    window.model_path.setFocus()
    qapp.processEvents()
    QTest.keyClicks(window.model_path, "h")
    assert window.model_path.text().endswith("h")
    assert not window.hidden_instances


def test_export_matches_visibility_and_csv_keeps_all_instances(window, tmp_path, monkeypatch):
    destination = tmp_path / "saved"
    destination.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(destination))
    window.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    window.table.item(1, 1).setCheckState(Qt.CheckState.Unchecked)
    window.overlay_checks["show_diameters"].setChecked(False)
    expected = viewer.load_rgb(window.paths[0])
    expected = draw_overlay(
        expected,
        window.results[str(window.paths[0])],
        **window._overlay_options(str(window.paths[0])),
    )
    window._save_results()
    encoded = np.fromfile(destination / "one.png.overlay.png", np.uint8)
    saved = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(saved, expected)
    with (destination / "measurements.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["circle"] for row in rows] == ["C1", "C2"]
    assert [row["instance_visible"] for row in rows] == ["False", "True"]
    assert [row["box_visible"] for row in rows] == ["False", "False"]
    assert [row["diameter_px"] for row in rows] == ["50.0", "50.0"]
