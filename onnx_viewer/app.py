"""Small folder-based PyQt6 viewer for ONNX instance inference."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QSignalBlocker, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QCursor, QImage, QPainter, QPixmap, QShortcut, QTransform
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .runtime import OnnxPipeline, draw_overlay, load_rgb

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class InferenceWorker(QThread):
    """Own the ORT session and process images away from the GUI thread."""

    ready = pyqtSignal(str)
    result_ready = pyqtSignal(str, object)
    image_failed = pyqtSignal(str, str)
    fatal_error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, paths: list[Path], settings: dict[str, Any], parent=None):
        super().__init__(parent)
        self.paths = paths
        self.settings = settings

    def run(self) -> None:
        try:
            pipeline = OnnxPipeline(
                self.settings["model_path"],
                provider=self.settings["provider"],
                mask_path=self.settings["mask_path"],
                input_size=self.settings["input_size"],
            )
            self.ready.emit(pipeline.model_description)
            for index, path in enumerate(self.paths, 1):
                if self.isInterruptionRequested():
                    break
                try:
                    result = pipeline.predict(
                        load_rgb(path),
                        score_threshold=self.settings["score_threshold"],
                        mask_threshold=self.settings["mask_threshold"],
                    )
                    self.result_ready.emit(str(path), result)
                except Exception as exc:  # noqa: BLE001 - report per-image backend errors in the GUI
                    self.image_failed.emit(str(path), str(exc))
                self.progress.emit(index, len(self.paths))
        except Exception as exc:  # noqa: BLE001 - propagate backend errors across the thread boundary
            self.fatal_error.emit(str(exc))


class ImagePreview(QGraphicsView):
    """Zoom/pan in image coordinates; a peer can follow the same viewport."""

    view_changed = pyqtSignal(float, object, bool)
    pointer_moved = pyqtSignal(object)
    peek_requested = pyqtSignal(object)
    hide_requested = pyqtSignal(object)

    def __init__(self, interactive: bool = False):
        super().__init__()
        self._interactive = interactive
        self._pointer_inside = False
        self._source: QPixmap | None = None
        self._fit_mode = True
        self._changing_view = False
        self._press_position = None
        self.setMouseTracking(True)
        self._hide_shortcut = QShortcut("H", self)
        self._hide_shortcut.setAutoRepeat(False)
        self._hide_shortcut.setEnabled(False)
        self._hide_shortcut.activated.connect(self._request_hide)
        QApplication.instance().focusChanged.connect(self._update_hide_shortcut)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self.scene().addItem(self._item)
        self._message = QLabel("Choose an image folder to begin", self.viewport())
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._message.setStyleSheet("color: #cdd6e4; background: transparent;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #171b23; color: #cdd6e4; border-radius: 6px;")
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setToolTip("Scroll to zoom · drag to pan · double-click to fit")

    def set_rgb(self, rgb: np.ndarray, reset: bool = True) -> None:
        state = self.view_state()
        had_image = self._source is not None
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        self._changing_view = True
        self._source = QPixmap.fromImage(image.copy())
        self._item.setPixmap(self._source)
        self.setSceneRect(self._item.boundingRect())
        self._message.hide()
        self._changing_view = False
        if reset or not had_image:
            self.fit_image()
        else:
            self.set_view_state(*state)
            self._notify()

    def clear(self, message: str = "No image") -> None:
        self._changing_view = True
        self._source = None
        self._item.setPixmap(QPixmap())
        self.setSceneRect(0, 0, 0, 0)
        self.resetTransform()
        self._fit_mode = True
        self._message.setText(message)
        self._message.setGeometry(self.viewport().rect())
        self._message.show()
        self._changing_view = False

    def view_state(self):
        return (
            self.transform().m11(),
            self.mapToScene(self.viewport().rect().center()),
            self._fit_mode,
        )

    def set_view_state(self, scale: float, center: QPointF, fit_mode: bool) -> None:
        if self._source is None:
            return
        self._changing_view = True
        self._fit_mode = fit_mode
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(center)
        self._changing_view = False

    def _notify(self):
        if self._source is not None and not self._changing_view:
            self.view_changed.emit(*self.view_state())

    def fit_image(self) -> None:
        if self._source is None:
            return
        scale = min(
            max(1, self.viewport().width() - 4) / self._source.width(),
            max(1, self.viewport().height() - 4) / self._source.height(),
        )
        self.set_view_state(scale, self._item.boundingRect().center(), True)
        self._notify()

    def actual_size(self) -> None:
        self.set_view_state(1.0, self.view_state()[1], False)
        self._notify()

    def zoom(self, factor: float, anchor=None) -> None:
        if self._source is None:
            return
        anchor = anchor or self.viewport().rect().center()
        before = self.mapToScene(anchor)
        scale = min(32.0, max(0.0001, self.transform().m11() * factor))
        offset = anchor - self.viewport().rect().center()
        center = before - QPointF(offset.x() / scale, offset.y() / scale)
        self.set_view_state(scale, center, False)
        self._notify()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.zoom(1.2 ** max(-8.0, min(8.0, delta / 120)), event.position().toPoint())
        event.accept()

    def _request_hide(self) -> None:
        point = self.viewport().mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(point):
            self.hide_requested.emit(self.mapToScene(point))

    def _update_hide_shortcut(self, *_args) -> None:
        editing = isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox, QDoubleSpinBox))
        self._hide_shortcut.setEnabled(self._interactive and self._pointer_inside and not editing)

    def enterEvent(self, event) -> None:
        self._pointer_inside = True
        self._update_hide_shortcut()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._pointer_inside = False
        self._update_hide_shortcut()
        self.pointer_moved.emit(None)
        self.peek_requested.emit(None)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self._source is not None:
            self._press_position = event.position().toPoint()
            self.peek_requested.emit(self.mapToScene(self._press_position))

    def mouseMoveEvent(self, event) -> None:
        if self._press_position is not None:
            distance = (event.position().toPoint() - self._press_position).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._press_position = None
                self.peek_requested.emit(None)
        super().mouseMoveEvent(event)
        self.pointer_moved.emit(self.mapToScene(event.position().toPoint()))

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_position = None
            self.peek_requested.emit(None)

    def focusOutEvent(self, event) -> None:
        self._press_position = None
        self.peek_requested.emit(None)
        super().focusOutEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_image()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def scrollContentsBy(self, dx, dy) -> None:
        super().scrollContentsBy(dx, dy)
        self._notify()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._message.setGeometry(self.viewport().rect())
        if self._fit_mode:
            self.fit_image()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DETRv4-Seg · ONNX Runtime · Circle & Diameter")
        self.resize(1250, 850)
        self.paths: list[Path] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}
        self.worker: InferenceWorker | None = None
        self._displayed_path: str | None = None
        self._displayed_rgb: np.ndarray | None = None
        self.hidden_instances: dict[str, set[int]] = {}
        self.hidden_boxes: dict[str, set[int]] = {}
        self._peek_instance: int | None = None
        self._hover_point: QPointF | None = None
        self._closing = False
        self._setup_ui()
        bundled_model = Path(__file__).resolve().parents[1] / "models" / "manifest.json"
        if bundled_model.is_file():
            self.model_path.setText(str(bundled_model))

    def _setup_ui(self) -> None:
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        form = QFormLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Detector ONNX or exported manifest.json")
        self.mask_path = QLineEdit()
        self.mask_path.setPlaceholderText("Override mask ONNX; manifest may supply its own mask")
        self.model_button = QPushButton("Browse…")
        self.model_button.clicked.connect(self._pick_model)
        self.mask_button = QPushButton("Browse…")
        self.mask_button.clicked.connect(self._pick_mask)
        form.addRow("Model / manifest", self._path_row(self.model_path, self.model_button))
        form.addRow("Mask model (optional)", self._path_row(self.mask_path, self.mask_button))
        layout.addLayout(form)

        controls = QHBoxLayout()
        self.folder_button = QPushButton("Open image folder…")
        self.folder_button.clicked.connect(self._pick_folder)
        controls.addWidget(self.folder_button)
        self.provider = QComboBox()
        self.provider.addItem("CPU", "CPUExecutionProvider")
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                self.provider.addItem("CUDA", "CUDAExecutionProvider")
        except ImportError:
            pass
        controls.addWidget(QLabel("Device"))
        controls.addWidget(self.provider)
        self.input_size = QSpinBox()
        self.input_size.setRange(32, 8192)
        self.input_size.setSingleStep(32)
        self.input_size.setValue(640)
        self.input_size.setToolTip(
            "Input size for dynamic ONNX graphs; fixed graphs use their own size"
        )
        controls.addWidget(QLabel("Input"))
        controls.addWidget(self.input_size)
        self.score = self._threshold(0.5)
        self.mask_threshold = self._threshold(0.5)
        self.mask_threshold.setMaximum(0.99)
        controls.addWidget(QLabel("Score ≥"))
        controls.addWidget(self.score)
        controls.addWidget(QLabel("Mask ≥"))
        controls.addWidget(self.mask_threshold)
        controls.addStretch()
        layout.addLayout(controls)
        self.model_info = QLabel("Circle C and diameter D are measured in original-image pixels.")
        self.model_info.setWordWrap(True)
        self.model_info.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.model_info)

        splitter = QSplitter()
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(180)
        self.image_list.currentRowChanged.connect(self._show_current)
        splitter.addWidget(self.image_list)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        zoom_controls = QHBoxLayout()
        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("+")
        self.fit_button = QPushButton("Fit")
        self.actual_size_button = QPushButton("1:1")
        self.zoom_label = QLabel("100%")
        for button in (
            self.zoom_out_button,
            self.zoom_in_button,
            self.fit_button,
            self.actual_size_button,
        ):
            zoom_controls.addWidget(button)
        zoom_controls.addWidget(self.zoom_label)
        zoom_controls.addWidget(QLabel("Scroll: zoom · Drag: pan · Double-click: fit"))
        zoom_controls.addStretch()
        right_layout.addLayout(zoom_controls)
        overlay_controls = QHBoxLayout()
        overlay_controls.addWidget(QLabel("Show overlays:"))
        self.overlay_checks: dict[str, QCheckBox] = {}
        for key, label in (
            ("show_boxes", "Bounding boxes"),
            ("show_contours", "Mask contours"),
            ("show_circles", "Circles"),
            ("show_diameters", "Diameter (hover)"),
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            if key == "show_diameters":
                checkbox.setToolTip(
                    "Show a compact diameter popup on hover; fixed size at any zoom"
                )
            checkbox.toggled.connect(self._redraw_overlay)
            overlay_controls.addWidget(checkbox)
            self.overlay_checks[key] = checkbox
        overlay_controls.addStretch()
        self.show_all_button = QPushButton("Show all instances")
        self.hide_all_button = QPushButton("Hide all instances")
        self.show_all_button.clicked.connect(lambda: self._set_all_instances_visible(True))
        self.hide_all_button.clicked.connect(lambda: self._set_all_instances_visible(False))
        overlay_controls.addWidget(self.show_all_button)
        overlay_controls.addWidget(self.hide_all_button)
        right_layout.addLayout(overlay_controls)
        self.original_preview = ImagePreview()
        self.preview = ImagePreview(interactive=True)
        # Only the inference pane handles instance actions; original stays a pan/zoom view.
        self.preview.pointer_moved.connect(self._show_hover_detail)
        self.preview.peek_requested.connect(self._peek_at)
        self.preview.hide_requested.connect(self._hide_at)
        self.diameter_detail = QLabel(self.preview.viewport())
        self.diameter_detail.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.diameter_detail.setStyleSheet(
            "background: rgba(20, 24, 32, 220); color: #ffe650; "
            "border-radius: 3px; padding: 3px 5px; font-size: 11px;"
        )
        self.diameter_detail.hide()
        self.preview.setToolTip(
            "Hover: diameter · Hold left click: peek · H: hide instance · "
            "Show all: restore · Scroll: zoom · Drag: pan"
        )
        self.preview.view_changed.connect(lambda *_args: self._show_hover_detail(None))
        self.original_preview.view_changed.connect(self.preview.set_view_state)
        self.preview.view_changed.connect(self.original_preview.set_view_state)
        for view in (self.original_preview, self.preview):
            view.view_changed.connect(
                lambda scale, _center, _fit: self.zoom_label.setText(f"{scale * 100:.0f}%")
            )
        self.zoom_out_button.clicked.connect(lambda: self.preview.zoom(1 / 1.2))
        self.zoom_in_button.clicked.connect(lambda: self.preview.zoom(1.2))
        self.fit_button.clicked.connect(self.preview.fit_image)
        self.actual_size_button.clicked.connect(self.preview.actual_size)
        self.comparison_splitter = QSplitter(Qt.Orientation.Horizontal)
        for title, view in (
            ("Original", self.original_preview),
            ("Inference · Circle & Diameter", self.preview),
        ):
            pane = QWidget()
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            pane_layout.addWidget(QLabel(title))
            pane_layout.addWidget(view, 1)
            self.comparison_splitter.addWidget(pane)
        self.comparison_splitter.setChildrenCollapsible(False)
        self.comparison_splitter.setStretchFactor(0, 1)
        self.comparison_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(self.comparison_splitter, 1)
        self.image_info = QLabel("No folder loaded")
        self.image_info.setTextFormat(Qt.TextFormat.PlainText)
        self.image_info.setWordWrap(True)
        right_layout.addWidget(self.image_info)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "Show",
                "Box",
                "Circle",
                "Label",
                "Score",
                "Center X",
                "Center Y",
                "Diameter (px)",
                "Source",
            ]
        )
        self.table.setToolTip(
            "Show: all overlays for this instance. Box: only its bounding box. "
            "Global layer checkboxes also apply. Measurements remain unchanged."
        )
        self.table.itemChanged.connect(self._instance_visibility_changed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(210)
        right_layout.addWidget(self.table)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 960])
        layout.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.previous_button = QPushButton("← Previous")
        self.previous_button.clicked.connect(lambda: self._move(-1))
        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(lambda: self._move(1))
        self.run_button = QPushButton("Infer current")
        self.run_button.clicked.connect(lambda: self._run(False))
        self.run_folder_button = QPushButton("Infer folder")
        self.run_folder_button.clicked.connect(lambda: self._run(True))
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        self.save_button = QPushButton("Save results…")
        self.save_button.setToolTip(
            "Save overlays using current visibility settings. CSV keeps all measurements "
            "and includes instance_visible and box_visible flags."
        )
        self.save_button.clicked.connect(self._save_results)
        for widget in (
            self.previous_button,
            self.next_button,
            self.run_button,
            self.run_folder_button,
            self.cancel_button,
            self.save_button,
        ):
            actions.addWidget(widget)
        layout.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.statusBar().showMessage("Ready. Select an ONNX model and an image folder.")
        self._settings_widgets = (
            self.model_path,
            self.mask_path,
            self.model_button,
            self.mask_button,
            self.folder_button,
            self.provider,
            self.input_size,
            self.score,
            self.mask_threshold,
            self.run_button,
            self.run_folder_button,
            self.save_button,
        )
        for widget in (
            self.model_path,
            self.mask_path,
        ):
            widget.textChanged.connect(self._invalidate_results)
        self.provider.currentIndexChanged.connect(self._invalidate_results)
        self.input_size.valueChanged.connect(self._invalidate_results)
        self.score.valueChanged.connect(self._invalidate_results)
        self.mask_threshold.valueChanged.connect(self._invalidate_results)

    @staticmethod
    def _path_row(edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    @staticmethod
    def _threshold(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select detector model or manifest", "", "ONNX / manifest (*.onnx *.json)"
        )
        if path:
            self.model_path.setText(path)

    def _pick_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select mask model", "", "ONNX (*.onnx)")
        if path:
            self.mask_path.setText(path)

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self.paths = sorted(
            (
                path
                for path in Path(folder).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
        self.results.clear()
        self.errors.clear()
        self.hidden_instances.clear()
        self.hidden_boxes.clear()
        self._displayed_path = None
        self._displayed_rgb = None
        self.image_list.clear()
        self.image_list.addItems([path.name for path in self.paths])
        self.progress.setValue(0)
        if self.paths:
            self.image_list.setCurrentRow(0)
        else:
            self.image_info.setText("No supported images in this folder")
            self.original_preview.clear("No images found")
            self.preview.clear("No images found")
            self.table.setRowCount(0)
        self.statusBar().showMessage(f"{len(self.paths)} images · {folder}")

    def _invalidate_results(self, *_args) -> None:
        if self.results or self.errors:
            self.results.clear()
            self.errors.clear()
            self.hidden_instances.clear()
            self.hidden_boxes.clear()
            for index, path in enumerate(self.paths):
                self.image_list.item(index).setText(path.name)
            self._show_current()
            self.statusBar().showMessage("Settings changed. Run inference to refresh measurements.")

    def _move(self, direction: int) -> None:
        row = self.image_list.currentRow() + direction
        if 0 <= row < len(self.paths):
            self.image_list.setCurrentRow(row)

    def _show_current(self, *_args) -> None:
        self._peek_instance = None
        self._show_hover_detail(None)
        row = self.image_list.currentRow()
        if not 0 <= row < len(self.paths):
            return
        path = self.paths[row]
        with QSignalBlocker(self.table):
            self.table.setRowCount(0)
        try:
            rgb = load_rgb(path)
            overlay = rgb
            result = self.results.get(str(path))
            if result is not None:
                overlay = draw_overlay(rgb, result, **self._screen_overlay_options(str(path)))
                self._fill_table(result["objects"], str(path))
                detail = (
                    f"{len(result['objects'])} objects · {result['elapsed_ms']:.1f} ms · "
                    f"{result['provider']} · source: {result['measurement_source']}"
                )
            else:
                detail = self.errors.get(str(path), "Not inferred")
            reset = self._displayed_path != str(path)
            with QSignalBlocker(self.original_preview), QSignalBlocker(self.preview):
                self.original_preview.set_rgb(rgb, reset=reset)
                self.preview.set_rgb(overlay, reset=reset)
            self.original_preview.set_view_state(*self.preview.view_state())
            self.preview._notify()
            self._displayed_path = str(path)
            self._displayed_rgb = rgb
            self.image_info.setText(
                f"{row + 1}/{len(self.paths)} · {path.name} · "
                f"{rgb.shape[1]}×{rgb.shape[0]} px\n{detail}"
            )
        except Exception as exc:  # noqa: BLE001 - keep the viewer open on invalid images or overlays
            self.original_preview.clear("Cannot display image")
            self.preview.clear("Cannot display image")
            self._displayed_path = None
            self._displayed_rgb = None
            self.image_info.setText(f"{path.name}: {exc}")

    def _fill_table(self, objects: list[dict[str, Any]], path: str | None = None) -> None:
        path = path or self._displayed_path
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(objects))
            for row, obj in enumerate(objects):
                for column, hidden in (
                    (0, self.hidden_instances.get(path, set())),
                    (1, self.hidden_boxes.get(path, set())),
                ):
                    item = QTableWidgetItem()
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    item.setCheckState(
                        Qt.CheckState.Unchecked if row + 1 in hidden else Qt.CheckState.Checked
                    )
                    self.table.setItem(row, column, item)
                cells = (
                    f"C{row + 1}",
                    str(obj.get("label", "")),
                    f"{obj['score']:.3f}",
                    self._number(obj.get("center_x")),
                    self._number(obj.get("center_y")),
                    self._number(obj.get("diameter_px")),
                    str(obj.get("measurement_source", "")),
                )
                for column, value in enumerate(cells, 2):
                    self.table.setItem(row, column, QTableWidgetItem(value))

    def _overlay_options(self, path: str) -> dict[str, Any]:
        return {
            **{key: checkbox.isChecked() for key, checkbox in self.overlay_checks.items()},
            "hidden_instances": self.hidden_instances.get(path, set()),
            "hidden_boxes": self.hidden_boxes.get(path, set()),
        }

    def _screen_overlay_options(self, path: str) -> dict[str, Any]:
        options = self._overlay_options(path)
        # Diameter is a screen-space hover widget, never rasterized into the preview.
        options["show_diameters"] = False
        options["hidden_instances"] = set(options["hidden_instances"])
        if self._peek_instance is not None and path == self._displayed_path:
            options["hidden_instances"].add(self._peek_instance)
        return options

    def _instance_at(self, point: QPointF | None) -> int | None:
        """Pick visible instances in original pixels, preferring masks over loose boxes."""
        if point is None or self._displayed_rgb is None:
            return None
        x, y = point.x(), point.y()
        height, width = self._displayed_rgb.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return None
        candidates = []
        for index, obj in enumerate(
            self.results.get(self._displayed_path, {}).get("objects", []), 1
        ):
            if index in self.hidden_instances.get(self._displayed_path, set()):
                continue
            bx, by, bw, bh = obj["bbox"]
            contour = obj.get("contour")
            if (
                contour is not None
                and cv2.pointPolygonTest(
                    np.asarray(contour, np.float32).reshape(-1, 1, 2), (x, y), False
                )
                >= 0
            ):
                candidates.append((0, abs(cv2.contourArea(np.asarray(contour, np.float32))), index))
            elif obj.get("diameter_px") is not None and (
                (x - obj["center_x"]) ** 2 + (y - obj["center_y"]) ** 2
                <= (obj["diameter_px"] / 2) ** 2
            ):
                candidates.append((1, obj["diameter_px"] ** 2, index))
            elif bx <= x <= bx + bw and by <= y <= by + bh:
                candidates.append((2, bw * bh, index))
        return min(candidates)[2] if candidates else None

    def _show_hover_detail(self, point: QPointF | None) -> None:
        self._hover_point = point
        index = self._instance_at(point)
        if (
            index is None
            or self._peek_instance is not None
            or not self.overlay_checks["show_diameters"].isChecked()
        ):
            self.diameter_detail.hide()
            return
        obj = self.results[self._displayed_path]["objects"][index - 1]
        self.diameter_detail.setText(f"C{index} · Ø {self._number(obj.get('diameter_px'))} px")
        self.diameter_detail.adjustSize()
        position = self.preview.mapFromScene(point)
        viewport = self.preview.viewport()
        self.diameter_detail.move(
            max(0, min(position.x() + 12, viewport.width() - self.diameter_detail.width())),
            max(0, min(position.y() + 16, viewport.height() - self.diameter_detail.height())),
        )
        self.diameter_detail.show()
        self.diameter_detail.raise_()

    def _peek_at(self, point: QPointF | None) -> None:
        index = self._instance_at(point)
        if index != self._peek_instance:
            self._peek_instance = index
            self._redraw_overlay()

    def _hide_at(self, point: QPointF) -> None:
        index = self._instance_at(point)
        path = self._displayed_path
        if index is None or path is None:
            return
        self.hidden_instances.setdefault(path, set()).add(index)
        self._peek_instance = None
        self._fill_table(self.results[path]["objects"], path)
        self._redraw_overlay()
        self.statusBar().showMessage(
            f"C{index} hidden for this image · Show all instances to restore"
        )

    def _redraw_overlay(self, *_args) -> None:
        """Render cached pixels only; visibility changes never run inference or reload files."""
        path = self._displayed_path
        if path is None or self._displayed_rgb is None:
            return
        result = self.results.get(path)
        if result is None:
            return
        point = self._hover_point
        overlay = draw_overlay(self._displayed_rgb, result, **self._screen_overlay_options(path))
        self.preview.set_rgb(overlay, reset=False)
        self._show_hover_detail(point)

    def _instance_visibility_changed(self, item: QTableWidgetItem) -> None:
        path = self._displayed_path
        if path is None or item.column() not in (0, 1):
            return
        states = self.hidden_instances if item.column() == 0 else self.hidden_boxes
        hidden = states.setdefault(path, set())
        index = item.row() + 1
        if item.checkState() == Qt.CheckState.Checked:
            hidden.discard(index)
        else:
            hidden.add(index)
        self._redraw_overlay()

    def _set_all_instances_visible(self, visible: bool) -> None:
        path = self._displayed_path
        if path is None or path not in self.results:
            return
        count = len(self.results[path]["objects"])
        self._peek_instance = None
        self.hidden_instances[path] = set() if visible else set(range(1, count + 1))
        if visible:
            self.hidden_boxes[path] = set()
        self._fill_table(self.results[path]["objects"], path)
        self._redraw_overlay()

    @staticmethod
    def _number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f}"

    def _run(self, whole_folder: bool) -> None:
        if self.worker is not None:
            return
        model = Path(self.model_path.text().strip()).expanduser()
        mask = self.mask_path.text().strip()
        if not model.is_file():
            QMessageBox.warning(self, "Select model", "Choose an existing ONNX model or manifest.")
            return
        if mask and not Path(mask).expanduser().is_file():
            QMessageBox.warning(self, "Mask model", "The optional mask model does not exist.")
            return
        row = self.image_list.currentRow()
        if not self.paths or row < 0:
            QMessageBox.information(self, "Select folder", "Open a folder containing images first.")
            return
        paths = list(self.paths) if whole_folder else [self.paths[row]]
        settings = {
            "model_path": str(model),
            "mask_path": str(Path(mask).expanduser()) if mask else None,
            "provider": self.provider.currentData(),
            "input_size": self.input_size.value(),
            "score_threshold": self.score.value(),
            "mask_threshold": self.mask_threshold.value(),
        }
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.worker = InferenceWorker(paths, settings, self)
        self.worker.ready.connect(self.model_info.setText)
        self.worker.result_ready.connect(self._on_result)
        self.worker.image_failed.connect(self._on_image_error)
        self.worker.fatal_error.connect(self._on_fatal_error)
        self.worker.progress.connect(lambda done, _total: self.progress.setValue(done))
        self.worker.finished.connect(self._on_finished)
        for widget in self._settings_widgets:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.statusBar().showMessage("Loading ONNX model…")
        self.worker.start()

    def _on_result(self, path: str, result: dict[str, Any]) -> None:
        self.results[path] = result
        self.hidden_instances.pop(path, None)
        self.hidden_boxes.pop(path, None)
        self.errors.pop(path, None)
        self._mark_path(path, "✓ ")
        row = self.image_list.currentRow()
        if row >= 0 and str(self.paths[row]) == path:
            self._show_current()
        self.statusBar().showMessage(f"Inferred {Path(path).name} · {result['elapsed_ms']:.1f} ms")

    def _on_image_error(self, path: str, error: str) -> None:
        self.results.pop(path, None)
        self.hidden_instances.pop(path, None)
        self.hidden_boxes.pop(path, None)
        self.errors[path] = error
        self._mark_path(path, "! ")
        row = self.image_list.currentRow()
        if row >= 0 and str(self.paths[row]) == path:
            self._show_current()

    def _mark_path(self, path: str, prefix: str) -> None:
        try:
            index = self.paths.index(Path(path))
        except ValueError:
            return
        self.image_list.item(index).setText(prefix + Path(path).name)

    def _on_fatal_error(self, error: str) -> None:
        if not self._closing:
            QMessageBox.critical(self, "ONNX inference failed", error)
        self.statusBar().showMessage(error)

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("Stopping after the current image finishes…")

    def _on_finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        for widget in self._settings_widgets:
            widget.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.statusBar().showMessage(
            f"Finished · {len(self.results)} images inferred · {len(self.errors)} image errors"
        )
        if self.errors and not self._closing:
            details = "\n".join(
                f"{Path(path).name}: {error}" for path, error in self.errors.items()
            )
            box = QMessageBox(self)
            box.setWindowTitle("Some images could not be inferred")
            box.setText(f"{len(self.errors)} images failed. Other results remain available.")
            box.setDetailedText(details)
            box.exec()
        if self._closing:
            self.close()

    def _save_results(self) -> None:
        if not self.results:
            QMessageBox.information(self, "No results", "Run inference before saving results.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Save overlays and measurements CSV")
        if not folder:
            return
        destination = Path(folder)
        csv_path = destination / "measurements.csv"
        overlay_paths = [destination / f"{Path(path).name}.overlay.png" for path in self.results]
        if csv_path.exists() or any(path.exists() for path in overlay_paths):
            answer = QMessageBox.question(
                self,
                "Replace saved results?",
                "This folder already contains result files. Replace matching files?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            fields = [
                "image",
                "circle",
                "label",
                "score",
                "center_x",
                "center_y",
                "diameter_px",
                "measurement_source",
                "elapsed_ms",
                "provider",
                "instance_visible",
                "box_visible",
            ]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for path, result in self.results.items():
                    options = self._overlay_options(path)
                    overlay = draw_overlay(load_rgb(path), result, **options)
                    output = destination / f"{Path(path).name}.overlay.png"
                    ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    if not ok:
                        raise OSError(f"Could not encode overlay: {output}")
                    encoded.tofile(str(output))
                    for index, obj in enumerate(result["objects"], 1):
                        writer.writerow(
                            {
                                "image": Path(path).name,
                                "circle": f"C{index}",
                                "label": obj.get("label", ""),
                                "score": obj["score"],
                                "center_x": obj.get("center_x"),
                                "center_y": obj.get("center_y"),
                                "diameter_px": obj.get("diameter_px"),
                                "measurement_source": obj.get("measurement_source", ""),
                                "elapsed_ms": result["elapsed_ms"],
                                "provider": result["provider"],
                                "instance_visible": index not in options["hidden_instances"],
                                "box_visible": (
                                    options["show_boxes"]
                                    and index not in options["hidden_instances"]
                                    and index not in options["hidden_boxes"]
                                ),
                            }
                        )
            self.statusBar().showMessage(
                f"Saved {len(self.results)} overlays with current visibility; "
                f"all measurements retained in {csv_path}"
            )
        except Exception as exc:  # noqa: BLE001 - surface image backend and filesystem errors in the GUI
            QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            self._closing = True
            self._cancel()
            self.setEnabled(False)
            event.ignore()
        else:
            event.accept()


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
