import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("onnxruntime")
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog

from onnx_viewer import app as viewer


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    image = np.full((1200, 1600, 3), [20, 30, 40], np.uint8)
    monkeypatch.setattr(viewer, "load_rgb", lambda _path: image.copy())
    widget = viewer.MainWindow()
    widget.show()
    qapp.processEvents()
    widget.paths = [tmp_path / "one.png", tmp_path / "two.png"]
    widget.image_list.addItems([path.name for path in widget.paths])
    widget.image_list.setCurrentRow(0)
    qapp.processEvents()
    yield widget
    widget.close()
    qapp.processEvents()


def assert_synced(window):
    scale, center, fit = window.preview.view_state()
    other_scale, other_center, other_fit = window.original_preview.view_state()
    assert scale == pytest.approx(other_scale)
    assert abs(center.x() - other_center.x()) < 2 / scale
    assert abs(center.y() - other_center.y()) < 2 / scale
    assert fit == other_fit


def test_zoom_wheel_pan_and_fit_are_synchronized(window, qapp):
    assert_synced(window)
    window.actual_size_button.click()
    assert window.preview.transform().m11() == pytest.approx(1.0)
    assert window.zoom_label.text() == "100%"
    assert_synced(window)
    anchor = QPointF(window.preview.viewport().rect().center() + QPoint(30, 40))
    before = window.preview.mapToScene(anchor.toPoint())
    event = QWheelEvent(
        anchor,
        anchor,
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(window.preview.viewport(), event)
    assert window.preview.transform().m11() == pytest.approx(1.2)
    after = window.preview.mapToScene(anchor.toPoint())
    assert abs(before.x() - after.x()) < 2
    assert abs(before.y() - after.y()) < 2
    assert_synced(window)
    view = window.original_preview
    old = view.view_state()[1]
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + 100)
    view.verticalScrollBar().setValue(view.verticalScrollBar().value() + 80)
    assert view.view_state()[1] != old
    assert_synced(window)
    window.zoom_out_button.click()
    assert window.preview.transform().m11() == pytest.approx(1.0)
    QTest.mouseDClick(window.original_preview.viewport(), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert window.preview._fit_mode
    assert window.preview.transform().m11() < 1.0
    assert_synced(window)


def test_inference_keeps_original_and_preserves_inspection_view(window):
    window.actual_size_button.click()
    window.preview.horizontalScrollBar().setValue(100)
    scale, center, _ = window.preview.view_state()
    result = {
        "objects": [
            {
                "bbox": [200, 100, 80, 80],
                "label": "ball",
                "score": 0.9,
                "center_x": 240,
                "center_y": 140,
                "diameter_px": 80.0,
                "measurement_source": "ONNX mask",
            }
        ],
        "elapsed_ms": 2.0,
        "provider": "CPUExecutionProvider",
        "measurement_source": "ONNX mask",
    }
    window._on_result(str(window.paths[0]), result)
    assert window.table.rowCount() == 1
    assert window.table.item(0, 7).text() == "80.00"
    original = window.original_preview._source.toImage()
    overlay = window.preview._source.toImage()
    assert original.pixelColor(200, 100).getRgb()[:3] == (20, 30, 40)
    assert overlay.pixelColor(200, 100) != original.pixelColor(200, 100)
    assert window.preview.transform().m11() == pytest.approx(scale)
    assert abs(window.preview.view_state()[1].x() - center.x()) < 2
    assert_synced(window)
    window.image_list.setCurrentRow(1)
    assert window.preview._fit_mode and window.original_preview._fit_mode
    assert window.table.rowCount() == 0
    assert window.preview._source.toImage() == window.original_preview._source.toImage()


def test_invalid_image_and_empty_folder_clear_both_panes(window, tmp_path, monkeypatch):
    def invalid(_path):
        raise OSError("invalid synthetic image")

    monkeypatch.setattr(viewer, "load_rgb", invalid)
    window._show_current()
    assert window.preview._source is window.original_preview._source is None
    assert "invalid synthetic image" in window.image_info.text()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    window._pick_folder()
    assert not window.paths
    assert window.preview._source is window.original_preview._source is None
