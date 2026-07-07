"""Standalone Z-stack viewer with ImageJ-style automatic Brightness & Contrast.

This is a self-contained application (independent of main.py). It loads TIFF
files from a folder (or a single multi-page TIFF) and lets you scroll through the
Z-stack while an automatic Brightness & Contrast adjustment is applied to every
slice -- reproducing the result of pressing "Auto" in ImageJ's B&C dialog
(Shift+C).

Run with:
    python tools/bc_viewer.py

The auto-contrast algorithm mirrors ImageJ's ContrastAdjuster.autoAdjust():
  1. Build a 256-bin histogram over the slice's [min, max] data range.
  2. Zero out bins that hold more than 10% of the pixels (the dominant
     background peak) so they don't skew the result.
  3. From each end of the histogram, find the first bin whose count exceeds
     pixelCount / 5000. Those bins define the display min / max.
  4. Linearly map [display_min, display_max] -> [0, 255] for viewing.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ImageJ's default auto-threshold divisor (first press of "Auto").
AUTO_THRESHOLD = 5000


def imagej_auto_minmax(pixels: np.ndarray, auto_threshold: int = AUTO_THRESHOLD):
    """Return the (display_min, display_max) ImageJ's "Auto" B&C would choose."""
    flat = pixels.ravel()
    pmin = float(flat.min())
    pmax = float(flat.max())
    if pmax <= pmin:
        return pmin, pmax

    n_bins = 256
    bin_size = (pmax - pmin) / n_bins
    hist, _ = np.histogram(flat, bins=n_bins, range=(pmin, pmax))

    pixel_count = flat.size
    threshold = pixel_count / auto_threshold
    limit = pixel_count / 10

    # Ignore over-full bins (e.g. the big dark background peak).
    hist = hist.copy()
    hist[hist > limit] = 0

    over = np.nonzero(hist > threshold)[0]
    if over.size == 0:
        return pmin, pmax

    hmin, hmax = int(over[0]), int(over[-1])
    disp_min = pmin + hmin * bin_size
    disp_max = pmin + hmax * bin_size
    if disp_max <= disp_min:
        return pmin, pmax
    return disp_min, disp_max


def apply_display_range(pixels: np.ndarray, disp_min: float, disp_max: float) -> np.ndarray:
    """Map [disp_min, disp_max] -> [0, 255] uint8, clipping outside the range."""
    if disp_max <= disp_min:
        scaled = np.zeros_like(pixels, dtype=np.float32)
    else:
        scaled = (pixels.astype(np.float32) - disp_min) * (255.0 / (disp_max - disp_min))
    return np.clip(scaled, 0, 255).astype(np.uint8)


class Frame:
    """A single Z-slice: a page inside a TIFF file, loaded lazily."""

    def __init__(self, tiff: tifffile.TiffFile, page: int, label: str):
        self._tiff = tiff
        self._page = page
        self.label = label
        self._data = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            arr = self._tiff.pages[self._page].asarray()
            # Collapse a singleton colour axis if present; keep it 2-D grayscale.
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            self._data = arr
        return self._data


class BCViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z-Stack Viewer - Auto Brightness & Contrast")
        self.resize(1000, 800)

        self._open_tiffs: list[tifffile.TiffFile] = []
        self.frames: list[Frame] = []
        self.index = 0
        # Cached (min, max) when using a single range for the whole stack.
        self._stack_range = None
        # Keeps the QImage backing buffer alive between paints.
        self._qbuf = None

        # --- widgets -------------------------------------------------------
        self.image_label = QLabel("Open a folder or a TIFF file to begin.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background:#101010; color:#888;")
        self.image_label.setMinimumSize(400, 300)

        open_folder_btn = QPushButton("Open Folder…")
        open_folder_btn.clicked.connect(self.open_folder)
        open_file_btn = QPushButton("Open File…")
        open_file_btn.clicked.connect(self.open_file)

        self.auto_cb = QCheckBox("Auto B&C")
        self.auto_cb.setChecked(True)
        self.auto_cb.toggled.connect(self.refresh)

        self.stack_cb = QCheckBox("Use stack histogram")
        self.stack_cb.setToolTip(
            "Compute one B&C range from the whole stack instead of per-slice."
        )
        self.stack_cb.toggled.connect(self._on_stack_toggled)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)

        self.pos_label = QLabel("0 / 0")
        self.pos_label.setMinimumWidth(90)
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.range_label = QLabel("")
        self.range_label.setStyleSheet("color:#aaa;")

        # --- layout --------------------------------------------------------
        top = QHBoxLayout()
        top.addWidget(open_folder_btn)
        top.addWidget(open_file_btn)
        top.addSpacing(20)
        top.addWidget(self.auto_cb)
        top.addWidget(self.stack_cb)
        top.addStretch(1)
        top.addWidget(self.range_label)

        nav = QHBoxLayout()
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.pos_label)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.image_label, 1)
        root.addLayout(nav)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    # --- loading -----------------------------------------------------------
    def _close_open_tiffs(self):
        for t in self._open_tiffs:
            try:
                t.close()
            except Exception:
                pass
        self._open_tiffs.clear()

    def _load_paths(self, paths: list[Path]):
        self._close_open_tiffs()
        frames: list[Frame] = []
        for path in paths:
            try:
                tiff = tifffile.TiffFile(str(path))
            except Exception as exc:  # noqa: BLE001 - report and skip bad files
                print(f"Skipping {path.name}: {exc}")
                continue
            self._open_tiffs.append(tiff)
            n = len(tiff.pages)
            for p in range(n):
                label = path.name if n == 1 else f"{path.name} [{p + 1}/{n}]"
                frames.append(Frame(tiff, p, label))

        self.frames = frames
        self.index = 0
        self._stack_range = None
        if not frames:
            self.image_label.setText("No readable TIFF frames found.")
            self.slider.setEnabled(False)
            self.pos_label.setText("0 / 0")
            return

        self.slider.blockSignals(True)
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(frames) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        if self.stack_cb.isChecked():
            self._compute_stack_range()
        self.refresh()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with TIFF files")
        if not folder:
            return
        folder = Path(folder)
        paths = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
            key=lambda p: p.name.lower(),
        )
        if not paths:
            self.image_label.setText("No .tif/.tiff files in that folder.")
            return
        self._load_paths(paths)

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select a TIFF file", "", "TIFF files (*.tif *.tiff)"
        )
        if not file:
            return
        self._load_paths([Path(file)])

    # --- B&C ---------------------------------------------------------------
    def _on_stack_toggled(self, checked: bool):
        self._stack_range = None
        if checked and self.frames:
            self._compute_stack_range()
        self.refresh()

    def _compute_stack_range(self):
        """One (min, max) range for the whole stack, ImageJ 'stack histogram' style."""
        gmin = np.inf
        gmax = -np.inf
        for f in self.frames:
            d = f.data
            gmin = min(gmin, float(d.min()))
            gmax = max(gmax, float(d.max()))
        if not np.isfinite(gmin) or gmax <= gmin:
            self._stack_range = (0.0, 1.0)
            return

        n_bins = 256
        bin_size = (gmax - gmin) / n_bins
        hist = np.zeros(n_bins, dtype=np.int64)
        for f in self.frames:
            h, _ = np.histogram(f.data.ravel(), bins=n_bins, range=(gmin, gmax))
            hist += h

        pixel_count = int(hist.sum())
        threshold = pixel_count / AUTO_THRESHOLD
        limit = pixel_count / 10
        hist[hist > limit] = 0
        over = np.nonzero(hist > threshold)[0]
        if over.size == 0:
            self._stack_range = (gmin, gmax)
            return
        self._stack_range = (
            gmin + int(over[0]) * bin_size,
            gmin + int(over[-1]) * bin_size,
        )

    # --- display -----------------------------------------------------------
    def _on_slider(self, value: int):
        self.index = value
        self.refresh()

    def refresh(self):
        if not self.frames:
            return
        frame = self.frames[self.index]
        pixels = frame.data

        if self.auto_cb.isChecked():
            if self.stack_cb.isChecked():
                if self._stack_range is None:
                    self._compute_stack_range()
                disp_min, disp_max = self._stack_range
            else:
                disp_min, disp_max = imagej_auto_minmax(pixels)
        else:
            disp_min, disp_max = float(pixels.min()), float(pixels.max())

        view = apply_display_range(pixels, disp_min, disp_max)
        self._show(view)

        self.pos_label.setText(f"{self.index + 1} / {len(self.frames)}")
        self.range_label.setText(
            f"{frame.label}   |   display: {disp_min:.4g} – {disp_max:.4g}"
        )

    def _show(self, view8: np.ndarray):
        self._qbuf = np.ascontiguousarray(view8)
        h, w = self._qbuf.shape
        qimg = QImage(self._qbuf.data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frames:
            self.refresh()

    def keyPressEvent(self, event):
        # Arrow keys / Page keys scroll the stack even when the slider isn't focused.
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Down, Qt.Key.Key_PageDown):
            self.slider.setValue(min(self.index + 1, self.slider.maximum()))
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            self.slider.setValue(max(self.index - 1, 0))
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._close_open_tiffs()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    viewer = BCViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
