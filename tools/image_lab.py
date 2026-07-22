"""DSI Image Lab — a two-channel post-processing workbench for DSI images.

A standalone application (independent of ``main.py``) for turning acquired data
into publication-quality images. It loads TIFF stacks (folder or multi-page) and
``.mat`` files at full float precision, then runs an ordered post-processing
pipeline whose every stage can be toggled and tuned live.

Run with::

    python tools/image_lab.py

--------------------------------------------------------------------------
Two channels, side by side
--------------------------------------------------------------------------
The window holds **two independent channels**, A and B — intended for the ORCA
and the EVK4 view of the same sample, which the detection beamsplitter lets you
acquire at once. Each channel keeps its own files, its own slice index and its
own full set of pipeline settings; the control panel edits whichever channel is
selected in **Editing**.

View modes: *A only*, *B only*, *Side by side*, and *Overlay* (green A / magenta
B — where the two detectors see the same beads, green + magenta reads white).

**Field matching.** The two cameras see the same sample through different
optics: the EVK4 footprint sits rotated (~43°) and scaled (~0.745, the
4.86/6.5 µm pixel-pitch ratio) inside the ORCA field, and EVK4 sees a smaller
patch of the sample. Four actions: *Measure field match* runs the project's
own masked-NCC registration (``core.register_evk4_to_orca``, the same routine
the acquisition GUI uses) on the two loaded images and returns the B→A affine
— while active, side-by-side and overlay live-warp B into A's frame so you can
check the alignment. *Warp ORCA onto EVK4's field of view* then permanently
resamples channel A through the inverse of that affine onto B's exact grid —
same rotation, scale and pixel dimensions as B — giving A pixel-for-pixel
EVK4's (smaller) field of view. *Reset match* clears the measured match and,
if A was warped, restores A's original frames too — the one way back. *Load
last session's match* brings the remembered one back after a reset.

**The match is not re-measured unless you ask for it.** It is remembered
between sessions (``MATCH_STATE_PATH``, env-override ``DSI_IMAGE_LAB_MATCH``):
the tool restores the last one at startup, so a fresh dataset from an
unchanged optical setup can be warped immediately. Acquisitions normally happen
at the same spot with the cameras untouched, so this is the usual case —
re-measure only when they have actually moved.

--------------------------------------------------------------------------
The 3-D volume view
--------------------------------------------------------------------------
*Open 3-D volume view* turns the active channel's z-stack into an interactive
volume: **drag to rotate**, **wheel to zoom into the pointer**, right-drag to
pan, **Shift+wheel to scroll the Z window** through the depth. *Fit* returns to
the whole volume. Nothing is *reconstructed* — a DSI stack is already
a 3-D image, because the optical sectioning is what makes each plane belong to
one depth. The view only resamples it onto **cubic voxels** (using the
channel's Z step and pixel size, so the shape is geometrically true rather than
stretched by the z step) and projects it from the chosen angle.

**Crop** has a from/to pair per axis: X and Y cut beads out of the picture at
the edge of the field, Z takes a slab through the depth. It changes only what
is projected, never the data, and what is left keeps its true position in the
frame — so nothing shifts as you close the box in.

**Zooming and real detail are two different things.** The wheel magnifies the
voxels the volume already holds; past a few screen pixels per voxel it is
showing you the grid, not the sample, and the status line says so. *Rebuild at
crop* is the answer: it rebuilds from just the cropped box, spending the same
voxel budget on less sample, so the voxel itself shrinks (typically 2-3x) and
the extra detail is genuinely there. *Whole volume* goes back to the full
field.

Frames are rendered at the canvas's own resolution and shown pixel-for-pixel,
rather than being drawn small and stretched to the widget; that second
resampling pass was the main reason the picture looked soft. *Detail* sets how
many voxels the volume gets, which is the real ceiling on sharpness — and the
half-resolution drafts drawn during a drag coarsen with it, so rotating stays
smooth even at Ultra.

Beads read as three-dimensional through three cues: parallax while rotating,
**hue keyed to depth in the sample** — rotation-invariant, so a bead keeps its
colour as the volume turns — and the projected **wireframe box**, without which
an orthographic projection of sparse beads gives the eye nothing to judge the
rotation against. *Solid* mode swaps the max-intensity projection for
front-to-back compositing, so nearer beads occlude further ones.

Be aware of what the optics allow: the axial resolution is the DSI sectioning
FWHM (2-4 um) against a ~0.16 um lateral pixel, so every bead is genuinely
elongated in z. That is physics, not a rendering artifact.

For the per-stage description of the processing pipeline itself — and for how
the renderer works — see ``tools/image_ops.py``.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.image_ops import (  # noqa: E402 — needs the sys.path line above
    EPS,
    LUT_NAMES,
    Frame,
    build_view_volume,
    depth_colour_legend,
    draw_scale_bar,
    draw_volume_box,
    fit_zoom,
    imagej_auto_minmax,
    process_slice,
    render_display,
    render_volume,
    scan_paths,
    view_rotation,
    volume_display_range,
)

# The measured EVK4->ORCA registration lives in the main application. Import it
# when available so the field match uses the *same* algorithm as the
# acquisition GUI; the tool still runs (matching just unavailable) if this
# checkout is incomplete or SciPy/OpenCV differ.
try:
    from core import register_evk4_to_orca
    REGISTRATION_AVAILABLE = True
except Exception:  # noqa: BLE001 — degrade gracefully, never fail to start
    register_evk4_to_orca = None
    REGISTRATION_AVAILABLE = False


IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Where a measured field match is remembered between sessions, so re-opening the
# tool does not mean re-running the (slow) NCC search. Stored next to the main
# application's own state; env-override with DSI_IMAGE_LAB_MATCH.
MATCH_STATE_PATH = os.environ.get(
    "DSI_IMAGE_LAB_MATCH",
    os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
        "DSIMicroscope", "image_lab_match.json",
    ),
)

MATCH_STATE_VERSION = 3          # v3 dropped the axial (Z) offset fields
MATCH_STATE_READABLE = (1, 2, 3)  # older records still load; extra fields ignored

# Where the rest of the session (loaded channels, per-channel pipeline settings,
# view mode) is remembered between runs, so the tool reopens exactly as it was
# left instead of needing to be reconfigured from scratch every time.
# Env-override with DSI_IMAGE_LAB_SESSION.
SESSION_STATE_PATH = os.environ.get(
    "DSI_IMAGE_LAB_SESSION",
    os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
        "DSIMicroscope", "image_lab_session.json",
    ),
)
SESSION_STATE_VERSION = 1


def _write_json_atomic(path, state):
    """Write ``state`` as JSON via a temp file + atomic replace.

    Writing in place would let an interrupted save leave a truncated JSON, which
    would then silently reset to defaults on the next start.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)
    return path


def write_match_state(path, state):
    """Persist a match record, via a temp file + atomic replace."""
    return _write_json_atomic(path, state)


def read_match_state(path):
    """Load a match record, or return None if absent / unreadable / wrong version."""
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:  # noqa: BLE001 — a missing or corrupt file is not an error
        return None
    if not isinstance(state, dict) or "affine" not in state:
        return None
    if state.get("version") not in MATCH_STATE_READABLE:
        return None
    return state


def write_session_state(path, state):
    """Persist a session record (loaded channels, pipeline params, view mode)."""
    return _write_json_atomic(path, state)


def read_session_state(path):
    """Load a session record, or return None if absent / unreadable / wrong version."""
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:  # noqa: BLE001 — a missing or corrupt file is not an error
        return None
    if not isinstance(state, dict) or state.get("version") != SESSION_STATE_VERSION:
        return None
    return state


# ===========================================================================
# Field matching helpers (pure)
# ===========================================================================
def warp_to(img, affine, shape):
    """Warp ``img`` through a 2x3 affine into an output of ``shape`` (h, w)."""
    h, w = int(shape[0]), int(shape[1])
    return cv2.warpAffine(np.asarray(img, dtype=np.float32),
                          np.asarray(affine, dtype=np.float64)[:2], (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=0.0)


def to_bgr(view):
    """Promote a display image to 3-channel BGR."""
    if view.ndim == 3:
        return view
    return cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)


def side_by_side(view_a, view_b, gap=8, labels=("A", "B")):
    """Lay two display images out horizontally, scaled to a common height."""
    a, b = to_bgr(view_a), to_bgr(view_b)
    h = max(a.shape[0], b.shape[0])
    out = []
    for img in (a, b):
        if img.shape[0] != h:
            scale = h / img.shape[0]
            img = cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), h),
                             interpolation=cv2.INTER_AREA)
        out.append(img)
    a, b = out
    divider = np.full((h, gap, 3), 40, np.uint8)
    canvas = np.hstack([a, divider, b])
    for text, x in ((labels[0], 10), (labels[1], a.shape[1] + gap + 10)):
        cv2.putText(canvas, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def green_magenta_overlay(view_a, view_b):
    """Green-A / magenta-B composite of two same-shape 8-bit views.

    Where both channels have signal, green + magenta sums to white — so the
    whiteness of a bead is a direct read-out of how well the fields are matched.
    """
    a = view_a if view_a.ndim == 2 else cv2.cvtColor(view_a, cv2.COLOR_BGR2GRAY)
    b = view_b if view_b.ndim == 2 else cv2.cvtColor(view_b, cv2.COLOR_BGR2GRAY)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((*a.shape, 3), np.uint8)
    out[..., 1] = a          # green  = A
    out[..., 0] = b          # blue  \
    out[..., 2] = b          # red   / = magenta = B
    return out


# ===========================================================================
# Per-channel state
# ===========================================================================
DEFAULT_PARAMS = {
    "hot_on": False, "hot_k": 6.0, "hot_win": 5,
    "bg_radius": 0,
    "denoise_method": "none", "denoise_strength": 1.0,
    "unsharp_amount": 0.0, "unsharp_radius": 2.0,
    "contrast_mode": "auto",
    "manual_min": 0.0, "manual_max": 255.0, "gamma": 1.0,
    "lut": "Grayscale", "scalebar_on": False,
    # geometry (per channel: the two cameras have different pixel pitches)
    "um_per_px": 0.1625, "bar_um": 10.0, "z_step": 0.5,
}


class Channel:
    """One loaded dataset with its own settings, references and view state."""

    def __init__(self, name):
        self.name = name
        self.frames: list[Frame] = []
        self.index = 0
        self.source_dir = None
        self.params = dict(DEFAULT_PARAMS)
        self.registered = None
        # Why ``registered`` is set — drift correction, footprint crop, exact-FOV
        # warp, or Z resample all reuse the same "materialised slices" slot, but
        # each needs its own label in the status line (see ImageLab._describe).
        self.registered_reason = None
        self.processed = None
        self.cache_key = None
        self.stack_range = None

    # -- data access ---------------------------------------------------
    @property
    def loaded(self):
        return bool(self.frames)

    @property
    def n_slices(self):
        return len(self.frames)

    def raw(self, i):
        if self.registered is not None:
            return self.registered[i]
        return self.frames[i].data

    def reset_derived(self):
        """Drop everything computed from the pixels (after a load / re-register)."""
        self.processed = None
        self.cache_key = None
        self.stack_range = None

    # -- pipeline ------------------------------------------------------
    _PIPELINE_KEYS = (
        "hot_on", "hot_k", "hot_win", "bg_radius",
        "denoise_method", "denoise_strength", "unsharp_amount", "unsharp_radius",
    )

    def pipeline_key(self):
        return ((self.index, self.registered is not None)
                + tuple(self.params[k] for k in self._PIPELINE_KEYS))

    def process_index(self, i, release=True):
        """Run the pipeline on slice ``i``."""
        out = process_slice(self.raw(i), self.params)
        if release and self.registered is None and i != self.index:
            self.frames[i].release()  # keep peak memory to ~one slice
        return out

    def ensure_processed(self):
        """Recompute the current slice only when a pipeline input changed."""
        key = self.pipeline_key()
        if key != self.cache_key or self.processed is None:
            self.processed = self.process_index(self.index, release=False)
            self.cache_key = key
        return self.processed

    def display_image(self):
        """The float image currently on show."""
        return self.processed

    def render(self):
        """Render this channel to an 8-bit view."""
        p = self.params
        self.ensure_processed()
        rng = None
        if self.use_stack_range and p["contrast_mode"] != "manual":
            rng = self.stack_range
        return render_display(self.processed, p, rng)

    use_stack_range = False


# ===========================================================================
# UI helpers
# ===========================================================================
def _spin(minimum, maximum, value, step=1.0, decimals=2, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    return w


def _int_spin(minimum, maximum, value, step=1, suffix=""):
    w = QSpinBox()
    w.setRange(minimum, maximum)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    return w


_HIGHLIGHT_STYLE = (
    "QPushButton { background-color: #2f6fb0; color: white; font-weight: 600; "
    "border: 1px solid #5a9bdb; border-radius: 4px; padding: 4px; }"
    "QPushButton:hover { background-color: #3a80c4; }"
    "QPushButton:pressed { background-color: #295f96; }"
    "QPushButton:disabled { background-color: #35404a; color: #888; border-color: #445; }"
)


def _highlight(button):
    """Mark a button as one of the tool's few essential, high-value actions."""
    button.setStyleSheet(_HIGHLIGHT_STYLE)
    return button


def _note(text):
    """A small, muted one-line explanation for the bottom of a section."""
    label = QLabel(text)
    label.setStyleSheet("color:#888; font-size: 8pt;")
    label.setWordWrap(True)
    return label


def _group(title, rows):
    """Build a QGroupBox from ``[(label_or_None, widget_or_layout), ...]``."""
    box = QGroupBox(title)
    grid = QGridLayout()
    grid.setContentsMargins(8, 6, 8, 6)
    grid.setVerticalSpacing(4)
    for r, (label, widget) in enumerate(rows):
        if label is None:
            if isinstance(widget, QWidget):
                grid.addWidget(widget, r, 0, 1, 2)
            else:
                grid.addLayout(widget, r, 0, 1, 2)
        else:
            grid.addWidget(QLabel(label), r, 0)
            if isinstance(widget, QWidget):
                grid.addWidget(widget, r, 1)
            else:
                grid.addLayout(widget, r, 1)
    grid.setColumnStretch(1, 1)
    box.setLayout(grid)
    return box


# ===========================================================================
# 3-D volume view
# ===========================================================================
class _Cancelled(Exception):
    """Raised out of a per-plane callback when the user cancels a long build."""


class VolumeCanvas(QLabel):
    """The 3-D view's drawing surface: drag to rotate, wheel to zoom.

    Reports gestures as deltas and holds no state, so the dialog stays the
    single owner of the camera.
    """

    rotated = pyqtSignal(float, float)
    zoomed = pyqtSignal(int, float, float)   # steps, cursor x, cursor y
    panned = pyqtSignal(float, float)
    slabbed = pyqtSignal(int)
    drag_finished = pyqtSignal()
    resized = pyqtSignal()

    # Left drag rotates; right (or middle) drag pans, which is what makes a
    # zoomed-in view usable — the detail you want is rarely at the centre.
    _PAN_BUTTONS = (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#101010;")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._last = None
        self._panning = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = event.position()
            self._panning = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() in self._PAN_BUTTONS:
            self._last = event.position()
            self._panning = True
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, event):
        if self._last is None:
            return
        pos = event.position()
        delta = (pos.x() - self._last.x(), pos.y() - self._last.y())
        (self.panned if self._panning else self.rotated).emit(*delta)
        self._last = pos

    def mouseReleaseEvent(self, event):
        if self._last is not None:
            self._last = None
            self._panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.drag_finished.emit()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        # Shift+wheel scrolls the slab through the stack — the 3-D equivalent of
        # scrolling the slice slider in the 2-D view.
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.slabbed.emit(int(np.sign(steps)))
        else:
            # The cursor goes with it: zooming towards the pointer is what lets
            # you drive into a particular bead instead of into the centre.
            pos = event.position()
            self.zoomed.emit(int(np.sign(steps)), pos.x(), pos.y())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The frame is rendered at this widget's own resolution, so its size is
        # a rendering parameter — not just a scaling factor applied afterwards.
        self.resized.emit()


class VolumeView(QDialog):
    """Interactive 3-D view of one channel's z-stack.

    The stack is already a 3-D image — DSI's optical sectioning is what makes
    each plane belong to one depth — so this only resamples it onto an isotropic
    grid once and then projects it from whatever angle the user drags to.

    Two things make sparse beads read as *3-D* rather than as a flat smear:
    hue keyed to depth in the sample (rotation-invariant, so a bead keeps its
    colour as the volume turns) and the projected wireframe box, which gives the
    eye a reference to judge the rotation against. The crop controls take a
    sub-box out of the volume — a slab through the depth, or an X/Y window that
    drops beads at the edge of the field — while what is left stays at its true
    position in the frame.

    Rendering is progressive: a half-resolution draft follows the mouse during a
    drag and a full-quality frame replaces it once the drag stops, so rotation
    stays smooth on volumes where a full frame takes a fraction of a second.
    Frames are produced at the canvas's own resolution (see
    ``image_ops.fit_zoom``) so nothing resamples them again on the way to the
    screen — the single biggest source of softness in the picture.
    """

    DETAIL = (("Fast (128)", 128), ("Balanced (256)", 256), ("Fine (384)", 384),
              ("Ultra (512)", 512))
    AXES = (("x", "X"), ("y", "Y"), ("z", "Z (depth)"))

    def __init__(self, parent, channel, builder, max_dim=256):
        super().__init__(parent)
        self.setWindowTitle(f"3-D volume — channel {channel.name}")
        self.resize(1180, 820)
        self.setWindowFlag(Qt.WindowType.Window, True)   # modeless, own taskbar entry

        self.channel = channel
        self._builder = builder
        self.vol = None
        self.voxel_um = 1.0
        # Sub-region of the stack the volume was built from — ``(roi, planes)``
        # in source pixels/planes, or None for the whole field.
        self.region = None
        self.range = (0.0, 1.0)
        self.azimuth, self.elevation = 32.0, 24.0
        self.zoom = 1.0
        self.pan = [0.0, 0.0]     # viewport offset in screen pixels
        self._buf = None
        self._dragging = False

        self.canvas = VolumeCanvas()
        self.canvas.rotated.connect(self._on_rotate)
        self.canvas.zoomed.connect(self._on_zoom)
        self.canvas.panned.connect(self._on_pan)
        self.canvas.slabbed.connect(self._on_slab_wheel)
        self.canvas.drag_finished.connect(self._on_drag_finished)
        self.canvas.resized.connect(self._on_canvas_resized)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#9c9;")
        self.status.setWordWrap(True)

        # Full-quality re-render after a gesture settles.
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(120)
        self._settle.timeout.connect(lambda: self._render(draft=False))

        panel = QVBoxLayout()
        panel.addWidget(self._build_view_group())
        panel.addWidget(self._build_crop_group())
        panel.addWidget(self._build_look_group())
        panel.addStretch(1)
        panel.addWidget(self.status)

        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(330)
        root = QHBoxLayout(self)
        root.addWidget(self.canvas, 1)
        root.addWidget(side)

        self._detail_index = next((i for i, (_, d) in enumerate(self.DETAIL)
                                   if d == max_dim), 1)
        self.detail.setCurrentIndex(self._detail_index)
        self.rebuild()

    # -- controls ------------------------------------------------------
    def _build_view_group(self):
        row = QHBoxLayout()
        for text, (az, el) in (("Top", (0, 0)), ("Front", (0, 90)),
                               ("Side", (90, 0)), ("Iso", (32, 24))):
            b = QPushButton(text)
            b.setToolTip(f"azimuth {az}°, elevation {el}°")
            b.clicked.connect(lambda _, a=az, e=el: self._set_view(a, e))
            row.addWidget(b)
        b_fit = QPushButton("Fit")
        b_fit.setToolTip("Back to zoom 1 and centred — the whole volume in view.")
        b_fit.clicked.connect(self._reset_view)
        row.addWidget(b_fit)

        self.az_slider = QSlider(Qt.Orientation.Horizontal)
        self.az_slider.setRange(-180, 180)
        self.az_slider.setValue(int(self.azimuth))
        self.az_slider.valueChanged.connect(self._on_angle_slider)
        self.el_slider = QSlider(Qt.Orientation.Horizontal)
        self.el_slider.setRange(-90, 90)
        self.el_slider.setValue(int(self.elevation))
        self.el_slider.valueChanged.connect(self._on_angle_slider)

        self.detail = QComboBox()
        self.detail.addItems([name for name, _ in self.DETAIL])
        self.detail.setToolTip(
            "Voxels along the longest axis — the real limit on how sharp the "
            "picture can be. Higher is slower to build and hungrier for memory "
            "(Ultra can run to a few hundred MB); rotating stays smooth either "
            "way, since the draft frames coarsen to match. The volume is "
            "rebuilt from the current processing settings.")
        self.detail.currentIndexChanged.connect(self._on_detail)

        hint = QLabel("Drag to rotate · wheel to zoom into the pointer · "
                      "right-drag to pan · Shift+wheel to scroll the Z window · "
                      "arrow keys nudge")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        return _group("View", [
            (None, row), ("Azimuth", self.az_slider),
            ("Elevation", self.el_slider), ("Detail", self.detail), (None, hint),
        ])

    def _build_crop_group(self):
        """Per-axis from/to sliders — the way to drop a bead out of the view.

        Cropping only changes what is *projected*; the volume itself is
        untouched, so nothing is lost by cutting a bead out and putting it back.
        """
        self.crop = {}
        rows = []
        for axis, title in self.AXES:
            lo = QSlider(Qt.Orientation.Horizontal)
            hi = QSlider(Qt.Orientation.Horizontal)
            lo.valueChanged.connect(lambda _v, a=axis: self._on_crop(a, "lo"))
            hi.valueChanged.connect(lambda _v, a=axis: self._on_crop(a, "hi"))
            self.crop[axis] = (lo, hi)
            rows.append((f"{title} from", lo))
            rows.append((f"{title} to", hi))

        b_finer = _highlight(QPushButton("Rebuild at crop (finer voxels)"))
        b_finer.setToolTip(
            "Rebuild the volume from just the cropped box, spending the same "
            "Detail budget on the smaller region — so the voxel shrinks and "
            "zooming reaches real detail instead of bigger blocks.\n"
            "Costs one pass over the stack. 'Whole volume' goes back.")
        b_finer.clicked.connect(self._rebuild_at_crop)
        b_all = QPushButton("Whole volume")
        b_all.setToolTip("Reset all six limits — and, after a rebuild, return "
                         "to the full field.")
        b_all.clicked.connect(self._crop_all)
        self.crop_label = QLabel("")
        self.crop_label.setStyleSheet("color:#888;")
        self.crop_label.setWordWrap(True)
        rows.append((None, b_finer))
        rows.append((None, b_all))
        rows.append((None, self.crop_label))
        rows.append((None, _note("Excludes beads outside the box from the "
                                 "picture. Shift+wheel slides the Z window "
                                 "through the stack at fixed thickness.")))
        return _group("Crop (exclude beads)", rows)

    def _build_look_group(self):
        self.mode = QComboBox()
        self.mode.addItems(["Max intensity (MIP)", "Solid (depth occlusion)"])
        self.mode.setToolTip(
            "MIP hides nothing — no bead can be occluded by one in front of "
            "it. Solid composites front-to-back, so nearer beads cover further "
            "ones: a stronger depth cue, but faint structure can disappear.")
        self.mode.currentIndexChanged.connect(self._queue)

        self.colour = QCheckBox("Colour by depth (hue = z in the sample)")
        self.colour.setChecked(True)
        self.colour.setToolTip(
            "Hue keyed to the bead's depth in the sample rather than its "
            "distance from the camera — so colours stay meaningful while you "
            "rotate.")
        self.colour.toggled.connect(self._queue)

        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(0, 95)
        self.threshold.setValue(12)
        self.threshold.setToolTip("Black point, as a % of the display range. "
                                  "Raise it until the haze between beads clears.")
        self.threshold.valueChanged.connect(self._queue)

        self.brightness = QSlider(Qt.Orientation.Horizontal)
        self.brightness.setRange(20, 400)
        self.brightness.setValue(100)
        self.brightness.valueChanged.connect(self._queue)

        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(2, 100)
        self.opacity.setValue(35)
        self.opacity.setToolTip("Solid mode only: how much each voxel absorbs.")
        self.opacity.valueChanged.connect(self._queue)

        self.show_box = QCheckBox("Bounding box + axes")
        self.show_box.setChecked(True)
        self.show_box.toggled.connect(self._queue)
        self.show_bar = QCheckBox("Scale bar")
        self.show_bar.setChecked(True)
        self.show_bar.toggled.connect(self._queue)

        b_png = QPushButton("Save view (PNG)")
        b_png.clicked.connect(self.save_png)
        return _group("Appearance", [
            ("Mode", self.mode), (None, self.colour),
            ("Threshold", self.threshold), ("Brightness", self.brightness),
            ("Opacity", self.opacity), (None, self.show_box),
            (None, self.show_bar), (None, b_png),
        ])

    # -- volume --------------------------------------------------------
    def rebuild(self):
        """(Re)build the working volume from the channel's current processing."""
        max_dim = self.DETAIL[self.detail.currentIndex()][1]
        built = self._builder(self.channel, max_dim, self.region)
        if built is None:
            if self.vol is None:
                QTimer.singleShot(0, self.reject)   # cancelled before first build
            return False
        self.vol, self.voxel_um = built
        self.range = volume_display_range(self.vol)
        self._reset_crop()        # a rebuilt volume has a new size: show all of it
        self._render(draft=False)
        return True

    def _axis_size(self, axis):
        """Length of the volume along ``axis`` (the array is (z, y, x))."""
        nz, ny, nx = self.vol.shape
        return {"x": nx, "y": ny, "z": nz}[axis]

    def _range(self, axis):
        """Current crop on ``axis`` as a half-open ``(a0, a1)`` in voxels."""
        lo, hi = self.crop[axis]
        return lo.value(), max(hi.value(), lo.value() + 1)

    def _reset_crop(self):
        """Size every axis's sliders to the volume and open them fully."""
        for axis, _ in self.AXES:
            n = self._axis_size(axis)
            for slider, bounds, value in ((self.crop[axis][0], (0, n - 1), 0),
                                          (self.crop[axis][1], (1, n), n)):
                slider.blockSignals(True)
                slider.setRange(*bounds)
                slider.setValue(value)
                slider.blockSignals(False)

    def _crop_all(self):
        """"Whole volume": drop the crop — and any region rebuild — and redraw."""
        if self.region is not None:
            self.region = None
            self.zoom, self.pan = 1.0, [0.0, 0.0]
            self.rebuild()        # back to the full field, at full extent
            return
        self._reset_crop()
        self._queue()

    def _rebuild_at_crop(self):
        """Rebuild the volume from just the cropped region, at finer voxels.

        Zooming magnifies the voxels the volume already has; it cannot show
        detail that was never sampled. Rebuilding spends the same voxel budget
        on the smaller region instead, so the voxel actually shrinks — this is
        the control that turns "bigger blobs" into "more detail".
        """
        if self.vol is None:
            return
        scale_xy = self.voxel_um / max(self.channel.params["um_per_px"], EPS)
        scale_z = self.voxel_um / max(self.channel.params["z_step"], EPS)
        (x0, x1), (y0, y1), (z0, z1) = (self._range(a) for a, _ in self.AXES)
        # Crop indices are voxels of the *current* volume, which may itself be a
        # region — so they compose onto the region already in force.
        base_roi, base_planes = self.region if self.region else ((0, 0, 0, 0), (0, 0))
        ox, oy = (base_roi[0], base_roi[2]) if self.region else (0, 0)
        op = base_planes[0] if self.region else 0
        roi = (ox + int(x0 * scale_xy), ox + int(np.ceil(x1 * scale_xy)),
               oy + int(y0 * scale_xy), oy + int(np.ceil(y1 * scale_xy)))
        planes = (op + int(z0 * scale_z), op + int(np.ceil(z1 * scale_z)))

        previous, self.region = self.region, (roi, planes)
        finer, zoom, pan = self.voxel_um, self.zoom, list(self.pan)
        self.zoom, self.pan = 1.0, [0.0, 0.0]   # the region now fills the view
        if not self.rebuild():          # cancelled — keep what is on screen
            self.region, self.zoom, self.pan = previous, zoom, pan
            return
        self.status.setText(
            f"Rebuilt from x {roi[0]}–{roi[1]}, y {roi[2]}–{roi[3]}, "
            f"planes {planes[0]}–{planes[1]}: voxel {finer:.3f} → "
            f"{self.voxel_um:.3f} µm. 'Whole volume' goes back.")

    # -- rendering -----------------------------------------------------
    def _queue(self, *_):
        self._render(draft=True)
        self._settle.start()

    def _viewport(self):
        """The canvas, in pixels — the frame is rendered to exactly this size."""
        size = self.canvas.size()
        return max(size.width(), 64), max(size.height(), 64)

    def _screen_zoom(self):
        """User zoom times the factor that fits the volume to the canvas.

        Rendering straight to the canvas's resolution is what keeps the picture
        crisp: the alternative is a small frame stretched to the widget, i.e. a
        second resampling pass whose only effect is blur. The fit is rotation-
        independent, so the volume keeps one apparent size as it turns, and it
        makes ``zoom`` mean "times the whole volume" — 1.0 is the whole thing in
        view whatever the detail level.
        """
        return self.zoom * fit_zoom(self.vol.shape, self._viewport())

    def _draft_stride(self):
        """Voxel stride for the frames drawn while a gesture is in progress.

        Fixed at 2 the draft cost still grows with the volume, so the finest
        Detail settings would drag at ~20 fps. Scaling the stride with the
        longest axis holds a draft frame near 10-20 ms whatever the volume size,
        which is what keeps rotating smooth; the quality frame that lands 120 ms
        later is unaffected.
        """
        return max(2, int(np.ceil(max(self.vol.shape) / 200.0)))

    def _render(self, draft=False):
        if self.vol is None:
            return
        started = time.perf_counter()
        lo, hi = self.range
        # Brightness pulls the white point in (>100%) or pushes it out (<100%).
        gain = self.brightness.value() / 100.0
        rng = (lo, lo + max(hi - lo, EPS) / gain)
        crop = tuple(self._range(axis) for axis, _ in self.AXES)  # x, y, z
        zoom = self._screen_zoom()
        pan = tuple(self.pan)
        viewport = self._viewport()
        rotation = view_rotation(self.azimuth, self.elevation)
        view = render_volume(
            self.vol, rotation, x_range=crop[0], y_range=crop[1], z_range=crop[2],
            mode="solid" if self.mode.currentIndex() == 1 else "mip",
            stride=self._draft_stride() if draft else 1, display_range=rng,
            threshold=self.threshold.value() / 100.0,
            opacity=self.opacity.value() / 100.0,
            colour_by_depth=self.colour.isChecked(), zoom=zoom,
            out_size=viewport, pan=pan)
        if self.show_box.isChecked():
            draw_volume_box(view, rotation, self.vol.shape, zoom, crop=crop,
                            pan=pan)
        if self.show_bar.isChecked():
            view = draw_scale_bar(view, self.voxel_um / max(zoom, EPS),
                                  self.channel.params["bar_um"])
        self._show(view)
        self._update_crop_label()
        if not draft:
            self._update_status(time.perf_counter() - started)

    def _show(self, view):
        self._buf = np.ascontiguousarray(view)
        h, w, _ = self._buf.shape
        qimg = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        # Only ever scale *down*. The frame is rendered to fit the canvas, so
        # blowing it up here would undo that and re-introduce the blur; a frame
        # that is momentarily too big (the canvas shrank) still has to fit.
        if pix.width() > self.canvas.width() or pix.height() > self.canvas.height():
            pix = pix.scaled(self.canvas.size(), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self.canvas.setPixmap(pix)

    def _update_crop_label(self):
        """Say what the crop is keeping. Updated on every frame, draft included,
        so the sliders read back immediately rather than 120 ms later."""
        parts = []
        for axis, title in self.AXES:
            a0, a1 = self._range(axis)
            if a0 == 0 and a1 >= self._axis_size(axis):
                continue
            parts.append(f"{title.split()[0]} {a0}–{a1 - 1} "
                         f"({(a1 - a0) * self.voxel_um:.1f} µm)")
        kept = "showing " + ", ".join(parts) if parts else "whole volume"
        z0, z1 = self._range("z")
        self.crop_label.setText(
            f"{kept}   ·   depth {z0 * self.voxel_um:.1f}–"
            f"{z1 * self.voxel_um:.1f} µm of "
            f"{self.vol.shape[0] * self.voxel_um:.1f} µm")

    def _update_status(self, seconds):
        nz, ny, nx = self.vol.shape
        z_step = self.channel.params["z_step"]
        note = ""
        if self.zoom > 1.5 and self._screen_zoom() > 3.0:
            # Past this the screen shows the voxel grid, not the sample: more
            # magnification cannot add detail the volume never sampled.
            note = ("   ⚠ zoomed past the voxel grid — 'Rebuild at crop' (or a "
                    "higher Detail) to resolve more")
        elif self.voxel_um > 1.5 * z_step:
            note = ("   ⚠ voxel is coarser than the z step — crop the field or "
                    "raise Detail for finer structure")
        region = "  ·  cropped region" if self.region is not None else ""
        self.status.setText(
            f"{nx}×{ny}×{nz} voxels at {self.voxel_um:.3f} µm{region}   ·   "
            f"az {self.azimuth:.0f}° el {self.elevation:.0f}° zoom {self.zoom:.2f}×"
            f"   ·   {seconds * 1000:.0f} ms{note}")

    # -- gestures ------------------------------------------------------
    def _on_rotate(self, dx, dy):
        self.azimuth = (self.azimuth + dx * 0.4 + 180.0) % 360.0 - 180.0
        self.elevation = float(np.clip(self.elevation - dy * 0.4, -90.0, 90.0))
        self._sync_angle_sliders()
        self._dragging = True
        self._render(draft=True)

    def _on_drag_finished(self):
        self._dragging = False
        self._render(draft=False)

    def _on_angle_slider(self, _value):
        if self._dragging:
            return
        self.azimuth = float(self.az_slider.value())
        self.elevation = float(self.el_slider.value())
        self._queue()

    def _sync_angle_sliders(self):
        for slider, value in ((self.az_slider, self.azimuth),
                              (self.el_slider, self.elevation)):
            slider.blockSignals(True)
            slider.setValue(int(round(value)))
            slider.blockSignals(False)

    def _set_view(self, azimuth, elevation):
        self.azimuth, self.elevation = float(azimuth), float(elevation)
        self._sync_angle_sliders()
        self._render(draft=False)

    ZOOM_LIMITS = (0.4, 20.0)

    def _on_zoom(self, steps, x=None, y=None):
        """Wheel zoom, anchored on the cursor.

        The picture is magnified about whatever is under the pointer rather than
        about the centre of the volume: keeping that point fixed is what lets
        you drive into one bead instead of watching it slide off the edge.
        """
        previous = self.zoom
        self.zoom = float(np.clip(self.zoom * (1.15 ** steps), *self.ZOOM_LIMITS))
        ratio = self.zoom / max(previous, EPS)
        if x is not None and ratio != 1.0:
            centre = (self.canvas.width() / 2.0, self.canvas.height() / 2.0)
            for i, (cursor, mid) in enumerate(zip((x, y), centre)):
                self.pan[i] = (cursor - mid) * (1.0 - ratio) + self.pan[i] * ratio
        self._clamp_pan()
        self._queue()

    def _on_pan(self, dx, dy):
        self.pan[0] += float(dx)
        self.pan[1] += float(dy)
        self._clamp_pan()
        self._dragging = True
        self._render(draft=True)

    def _clamp_pan(self):
        """Keep the pan within the volume's own projected radius.

        Far enough to bring any corner to the middle of the screen, never far
        enough to lose the volume off the edge and be left staring at black.
        """
        nz, ny, nx = self.vol.shape
        radius = 0.5 * self._screen_zoom() * float(
            np.linalg.norm(np.array([nx, ny, nz], dtype=np.float64) - 1.0))
        self.pan = [float(np.clip(p, -radius, radius)) for p in self.pan]

    def _reset_view(self):
        """Back to the whole volume, centred — the way out of a deep zoom."""
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self._render(draft=False)

    def _on_slab_wheel(self, steps):
        self._slide_crop("z", steps)

    def _on_crop(self, axis, moved):
        """Keep 'from' below 'to' by pushing the other end of the pair along."""
        lo, hi = self.crop[axis]
        if lo.value() >= hi.value():
            other, value = ((hi, lo.value() + 1) if moved == "lo"
                            else (lo, hi.value() - 1))
            other.blockSignals(True)
            other.setValue(value)
            other.blockSignals(False)
        self._queue()

    def _slide_crop(self, axis, steps):
        """Slide an axis's window without changing its width."""
        lo, hi = self.crop[axis]
        width = hi.value() - lo.value()
        start = int(np.clip(lo.value() + steps, 0, self._axis_size(axis) - width))
        for slider, value in ((lo, start), (hi, start + width)):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self._queue()

    def _on_canvas_resized(self):
        """Re-render at the new resolution once the resize settles."""
        if self.vol is not None:
            self._settle.start()

    def _on_detail(self, index):
        if index == self._detail_index:
            return
        previous, self._detail_index = self._detail_index, index
        if not self.rebuild():        # cancelled — keep showing the old volume
            self._detail_index = previous
            self.detail.blockSignals(True)
            self.detail.setCurrentIndex(previous)
            self.detail.blockSignals(False)

    def keyPressEvent(self, event):
        key = event.key()
        step = 5.0
        if key == Qt.Key.Key_Left:
            self._set_view(self.azimuth - step, self.elevation)
        elif key == Qt.Key.Key_Right:
            self._set_view(self.azimuth + step, self.elevation)
        elif key == Qt.Key.Key_Up:
            self._set_view(self.azimuth, min(90.0, self.elevation + step))
        elif key == Qt.Key.Key_Down:
            self._set_view(self.azimuth, max(-90.0, self.elevation - step))
        elif key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self._slide_crop("z", 1 if key == Qt.Key.Key_PageUp else -1)
        else:
            super().keyPressEvent(event)

    def save_png(self):
        if self._buf is None:
            return
        parent = self.parent()
        base = parent._out_base(stack_wide=True)
        path = parent._out_dir() / (
            f"{base}_volume_az{self.azimuth:+.0f}_el{self.elevation:+.0f}.png")
        cv2.imwrite(str(path), self._buf)
        msg = f"Saved {path}"
        if self.colour.isChecked():
            legend = parent._out_dir() / f"{base}_volume_depth_legend.png"
            cv2.imwrite(str(legend), depth_colour_legend(self.vol.shape[0]))
            msg += f"  (+ {legend.name})"
        self.status.setText(msg)
        parent._final_status(msg)


class ImageView(QLabel):
    """The main image display: mouse wheel scrolls through Z, over either pane.

    Side-by-side and overlay draw both channels into this one label, so
    hovering anywhere over it is "over the viewing area" for either stack —
    there is nothing to disambiguate. When the two stacks are linked
    (Link slice sliders / Z-link), the other channel follows automatically.
    """

    scrolled = pyqtSignal(int)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.scrolled.emit(int(np.sign(steps)))
        event.accept()


class _MatchWorker(QThread):
    """Runs the field-match search off the UI thread.

    ``register_evk4_to_orca``'s coarse search is a few-thousand-iteration loop
    with no yield to the Qt event loop, so running it inline freezes the window
    for the whole search. Here it runs in a background thread and reports back
    via signals; the images are copied in, so the main thread can keep
    re-processing without racing the worker.
    """

    progress = pyqtSignal(str)
    done = pyqtSignal(object, float, dict)   # affine, score, info
    failed = pyqtSignal(str)

    def __init__(self, img_a, img_b, seed_affine):
        super().__init__()
        self._img_a = np.asarray(img_a, dtype=np.float32).copy()
        self._img_b = np.asarray(img_b, dtype=np.float32).copy()
        self._seed = seed_affine

    def run(self):
        try:
            affine, score, info = register_evk4_to_orca(
                self._img_a, self._img_b, seed_affine=self._seed,
                status=self.progress.emit)
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the thread
            traceback.print_exc()
            self.failed.emit(str(exc))
            return
        self.done.emit(affine, float(score), info)


class ImageLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSI Image Lab — two-channel post-processing workbench")
        self.resize(1650, 980)

        self.channels = {"A": Channel("A"), "B": Channel("B")}
        self.active = "A"
        self._loading_params = False   # guard against feedback while repopulating
        self._qbuf = None
        self._volume_views = []        # open (modeless) 3-D windows

        # Field matching: the B -> A affine.
        #
        # Two separate facts, which must not be conflated: ``match_measured``
        # says a real match exists and is the thing worth remembering between
        # sessions; ``match_active`` says the live "warp B into A" pass should
        # run *right now*, and is switched off while channel A itself is sitting
        # on B's grid. That second one is a property of this session only —
        # a warp is never persisted — so persisting it once left a measured
        # match looking un-measured on the next start.
        self.base_affine = IDENTITY_AFFINE.copy()
        self.match_measured = False
        self.match_active = False
        self.match_score = None
        self._match_worker = None      # background field-match search, when running

        self._build_ui()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        # Coalesces a burst of rapid changes (slider drag, fast wheel spin) into
        # one redraw at the end, rather than one per event. A single tick of the
        # default pipeline (both channels: load+process+render+draw) measures
        # ~60-80ms, so this only needs to be a little above that to still catch
        # a genuine burst — 180ms was tacking on a flat, fully-avoidable extra
        # ~180ms of latency to every discrete wheel notch or arrow-key step.
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self.refresh)
        self._apply_params_to_widgets(self.ch().params)
        # Reopen exactly as it was left: same channels, same pipeline settings,
        # same view — so the tool doesn't need reconfiguring from scratch.
        self._restore_session_state()
        # A measured match is expensive; carry the last one across sessions so
        # cropping and overlaying work immediately on the next dataset.
        self._restore_match_state()

    # ------------------------------------------------------------------
    def ch(self, name=None):
        return self.channels[name or self.active]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        header = self._build_editing_header()   # first: the groups reference it
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(6)
        for build in (self._build_channel_group, self._build_view_group,
                      self._build_match_group,
                      self._build_hot_group, self._build_background_group,
                      self._build_denoise_group,
                      self._build_sharpen_group, self._build_contrast_group,
                      self._build_lut_group, self._build_stack_group,
                      self._build_export_group):
            col.addWidget(build())
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)

        # The header sits outside the scroll area, so it stays put while the
        # stages below it scroll.
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)
        left.addWidget(header)
        left.addWidget(scroll, 1)
        side = QWidget()
        side.setLayout(left)
        side.setMinimumWidth(370)
        side.setMaximumWidth(450)

        self.image_label = ImageView("Load a file into channel A (and optionally B) to begin.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background:#101010; color:#888;")
        self.image_label.setMinimumSize(560, 420)
        self.image_label.scrolled.connect(self._on_image_wheel)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.pos_label = QLabel("0 / 0")
        self.pos_label.setMinimumWidth(90)
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        nav = QHBoxLayout()
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.pos_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#aaa;")
        self.status_label.setWordWrap(True)

        right = QVBoxLayout()
        right.addWidget(self.image_label, 1)
        right.addLayout(nav)
        right.addWidget(self.status_label)

        root = QHBoxLayout()
        root.addWidget(side)
        root.addLayout(right, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _build_channel_group(self):
        rows = []
        self.chan_labels = {}
        for name, hint in (("A", "ORCA"), ("B", "EVK4")):
            b_folder = QPushButton("Folder…")
            b_folder.clicked.connect(lambda _, n=name: self.open_folder(n))
            b_files = QPushButton("File(s)…")
            b_files.clicked.connect(lambda _, n=name: self.open_files(n))
            b_clear = QPushButton("✕")
            b_clear.setMaximumWidth(30)
            b_clear.setToolTip(f"Unload channel {name}")
            b_clear.clicked.connect(lambda _, n=name: self.clear_channel(n))
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b>{name}</b> ({hint})"))
            row.addWidget(b_folder)
            row.addWidget(b_files)
            row.addWidget(b_clear)
            label = QLabel("empty")
            label.setStyleSheet("color:#888;")
            label.setWordWrap(True)
            self.chan_labels[name] = label
            rows.append((None, row))
            rows.append((None, label))
        # The "Editing" selector lives in the pinned header, not here — see
        # _build_editing_header.
        return _group("Channels  (.tif / .mat — prefer .mat for full precision)", rows)

    # Which camera each channel is for, and the colour that identifies it.
    _CHANNEL_HINTS = {"A": "ORCA", "B": "EVK4"}
    _CHANNEL_COLOURS = {"A": ("#1b3f63", "#5a9bdb"), "B": ("#5c4310", "#e0a33a")}

    def _build_editing_header(self):
        """The channel selector, pinned above the scroll area.

        Every stage in the panel below edits *one* channel, so which one is
        selected changes what all of them mean — and the panel is far taller
        than the window. Scrolled out of sight at the top, this was a standing
        invitation to tune the wrong camera's pipeline; pinned (and colour-coded
        per channel) the answer is always on screen.
        """
        self.editing = QComboBox()
        self.editing.addItems(["A", "B"])
        self.editing.setFixedWidth(64)
        self.editing.currentTextChanged.connect(self._on_editing_changed)

        self.editing_label = QLabel()
        self.editing_label.setWordWrap(True)
        row = QHBoxLayout()
        row.setContentsMargins(10, 6, 10, 6)
        row.addWidget(self.editing_label, 1)
        row.addWidget(self.editing)

        self.editing_header = QWidget()
        self.editing_header.setObjectName("editingHeader")
        # A plain QWidget ignores a stylesheet background unless it is told to
        # paint one.
        self.editing_header.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.editing_header.setLayout(row)
        self._header_channel = None
        self._update_editing_header()
        return self.editing_header

    def _update_editing_header(self):
        """Repaint the pinned header for the active channel and its state."""
        name = self.active
        if name != self._header_channel:
            background, border = self._CHANNEL_COLOURS[name]
            self.editing_header.setStyleSheet(
                f"#editingHeader {{ background:{background}; "
                f"border:1px solid {border}; border-radius:4px; }}"
                f"#editingHeader QLabel {{ color:#fff; font-weight:600; "
                f"border:none; background:transparent; }}")
            self._header_channel = name
        c = self.ch(name)
        state = f"{c.n_slices} slice(s)" if c.loaded else "empty"
        self.editing_label.setText(
            f"Editing channel {name} · {self._CHANNEL_HINTS[name]} · {state}")

    def _build_view_group(self):
        self.view_mode = QComboBox()
        self.view_mode.addItems(["A only", "B only", "Side by side", "Overlay (green A / magenta B)"])
        self.view_mode.currentIndexChanged.connect(self._schedule)
        self.link_slices = QCheckBox("Link slice sliders")
        self.link_slices.setToolTip("Scroll both stacks together (clamped to each "
                                    "channel's own length).")
        self.link_slices.toggled.connect(self._schedule)
        return _group("View", [("Mode", self.view_mode), (None, self.link_slices)])

    def _build_match_group(self):
        self.b_measure = _highlight(QPushButton("Measure field match (A ↔ B)"))
        self.b_measure.setToolTip("Masked-NCC registration of the two loaded images "
                             "— the same routine the acquisition GUI uses.\n"
                             "Load the ORCA into A and the EVK4 into B: the "
                             "search only shrinks B onto A (scale 0.55–0.96).")
        self.b_measure.clicked.connect(self.measure_match)
        if not REGISTRATION_AVAILABLE:
            self.b_measure.setEnabled(False)
            self.b_measure.setToolTip("core.register_evk4_to_orca could not be imported.")
        b_measure = self.b_measure

        b_last = _highlight(QPushButton("Load last session's match"))
        b_last.setToolTip(
            "Re-apply the last match measured on this machine. Acquisitions "
            "normally happen at the same spot with the cameras untouched, so "
            "the B → A map carries over — and it is restored automatically at "
            "startup. Use this to get it back after a Reset.")
        b_last.clicked.connect(self.load_last_match)

        b_reset = QPushButton("Reset match")
        b_reset.setToolTip("Clears the measured match and, if A was warped onto "
                           "B's field of view, restores A's original frames too.")
        b_reset.clicked.connect(self.reset_match)

        b_warp_a = _highlight(QPushButton("Warp ORCA (A) onto EVK4's (B) field of view"))
        b_warp_a.setToolTip(
            "Resample channel A through the inverse of the current match, onto "
            "B's own grid — same rotation, scale and pixel dimensions as B, so A "
            "ends up with exactly B's (smaller) field of view.\n"
            "Costs one interpolation pass over A's data. Needs a match measured "
            "first.")
        b_warp_a.clicked.connect(self.warp_a_to_b)

        self.match_label = QLabel("no match measured")
        self.match_label.setStyleSheet("color:#888;")
        self.match_label.setWordWrap(True)
        return _group("Field matching  (B → A) — remembered between sessions", [
            (None, b_measure), (None, b_last), (None, b_reset), (None, b_warp_a),
            (None, self.match_label),
        ])

    def _build_hot_group(self):
        self.hot_on = QCheckBox("Remove hot / crazy pixels")
        self.hot_k = _spin(1.0, 30.0, 6.0, 0.5, 1, " ·MAD")
        self.hot_win = _int_spin(3, 15, 5, 2, " px")
        for w in (self.hot_on, self.hot_k, self.hot_win):
            self._connect(w)
        return _group("1 · Sensor artifacts", [
            (None, self.hot_on), ("Threshold", self.hot_k), ("Window", self.hot_win),
            (None, _note("Fixes stray too-bright pixels from the camera itself — "
                        "not part of the sample.")),
        ])

    def _build_background_group(self):
        self.bg_radius = _int_spin(0, 400, 0, 5, " px")
        self.bg_radius.setToolTip("0 = off. Removes slowly-varying haze/glow "
                                  "(uneven illumination, tissue autofluorescence). "
                                  "Radius must exceed the largest structure you "
                                  "want to keep, or that gets subtracted too.")
        self._connect(self.bg_radius)
        return _group("2 · Background subtraction", [
            ("Ball radius", self.bg_radius),
            (None, _note("Removes slow, uneven glow so only real structure remains.")),
        ])

    def _build_denoise_group(self):
        self.denoise_method = QComboBox()
        self.denoise_method.addItems(["Off", "Gaussian", "Median"])
        self.denoise_method.setToolTip(
            "Gaussian: standard blur. Median: better at killing stray bright "
            "specks while rounding round structure (beads, nuclei) off less.")
        self.denoise_strength = _spin(0.1, 10.0, 1.0, 0.1, 2)
        for w in (self.denoise_method, self.denoise_strength):
            self._connect(w)
        return _group("3 · Denoising", [
            ("Method", self.denoise_method), ("Strength", self.denoise_strength),
            (None, _note("Smooths random pixel noise; higher strength = smoother "
                        "but softer detail.")),
        ])

    def _build_sharpen_group(self):
        self.unsharp_amount = _spin(0.0, 5.0, 0.0, 0.1, 2)
        self.unsharp_radius = _spin(0.5, 20.0, 2.0, 0.5, 1, " px")
        for w in (self.unsharp_amount, self.unsharp_radius):
            self._connect(w)
        return _group("4 · Sharpen (unsharp mask)", [
            ("Amount", self.unsharp_amount), ("Radius", self.unsharp_radius),
            (None, _note("Boosts edge contrast to make structures look crisper.")),
        ])

    def _build_contrast_group(self):
        self.auto_bc = QCheckBox("Auto brightness & contrast")
        self.auto_bc.setChecked(True)
        self.auto_bc.setToolTip("Let the tool pick the brightness/contrast from "
                                "the image itself (ImageJ's Auto B&C). Uncheck to "
                                "set the min/max by hand.")
        self.bc_both = QCheckBox("Apply to both images (A & B)")
        self.bc_both.setChecked(True)
        self.bc_both.setToolTip("Adjust the ORCA (A) and EVK4 (B) images together. "
                                "Because the two cameras have very different value "
                                "scales, each still fits its OWN range — only the "
                                "Auto/manual choice and gamma are shared, never one "
                                "camera's numbers forced onto the other.")
        self.stack_cb = QCheckBox("Same brightness across the whole stack")
        self.stack_cb.setToolTip("One range for every slice, so brightness stays "
                                 "comparable as you scroll through the depth.")
        self.manual_min = _spin(-1e6, 1e6, 0.0, 1.0, 3)
        self.manual_max = _spin(-1e6, 1e6, 255.0, 1.0, 3)
        self.gamma = _spin(0.1, 3.0, 1.0, 0.05, 2)
        self.gamma.setToolTip("< 1 lifts faint structure without clipping the bright end.")
        for w in (self.manual_min, self.manual_max, self.gamma):
            self._connect(w)
        self.stack_cb.toggled.connect(self._on_stack_toggled)
        self.bc_both.toggled.connect(self._on_bc_both_toggled)
        self.auto_bc.toggled.connect(self._on_auto_bc_toggled)
        self._update_manual_enabled(self.auto_bc.isChecked())
        return _group("5-6 · Brightness & contrast (display only)", [
            (None, self.auto_bc), (None, self.bc_both), (None, self.stack_cb),
            ("Manual min", self.manual_min), ("Manual max", self.manual_max),
            ("Gamma", self.gamma),
            (None, _note("Only changes how brightness is displayed — the saved "
                        "data is untouched.")),
        ])

    def _update_manual_enabled(self, auto_checked):
        """Grey out the manual min/max boxes while Auto B&C is on."""
        for w in (self.manual_min, self.manual_max):
            w.setEnabled(not auto_checked)

    def _on_auto_bc_toggled(self, checked):
        """Auto B&C on/off. Switching to manual seeds this channel's min/max
        from what Auto is showing right now — so manual never starts from a
        range meant for the other (differently-scaled) camera.
        """
        self._update_manual_enabled(checked)
        if self._loading_params:
            return
        if not checked:
            c = self.ch()
            c.ensure_processed()
            img = c.display_image()
            if img is not None:
                lo, hi = imagej_auto_minmax(img)
                for w, v in ((self.manual_min, lo), (self.manual_max, hi)):
                    w.blockSignals(True)
                    w.setValue(v)
                    w.blockSignals(False)
        self._on_param_changed()

    def _build_lut_group(self):
        self.lut = QComboBox()
        self.lut.addItems(LUT_NAMES)
        self.scalebar_on = QCheckBox("Scale bar")
        self.bar_um = _spin(0.1, 1000.0, 10.0, 1.0, 1, " um")
        self.pixel_um = _spin(0.001, 20.0, 0.1625, 0.005, 4, " um/px")
        self.pixel_um.setToolTip("Physical size of one pixel at the sample — sets "
                                 "the scale bar and the 3-D view's true proportions.")
        for w in (self.lut, self.scalebar_on, self.bar_um, self.pixel_um):
            self._connect(w)
        return _group("7-8 · Colour (LUT) & scale bar", [
            ("LUT", self.lut), (None, self.scalebar_on),
            ("Bar length", self.bar_um), ("Pixel size", self.pixel_um),
            (None, _note("Cosmetic only: pick a colour map and add a distance "
                        "marker for figures.")),
        ])

    def _build_stack_group(self):
        self.z_step = _spin(0.0, 100.0, 0.5, 0.1, 3, " um")
        self.z_step.setToolTip("Spacing between planes in the acquisition. Sets "
                               "the depth scale of the 3-D reconstruction, so the "
                               "volume has true proportions rather than being "
                               "stretched.")
        self._connect(self.z_step)
        b_3d = _highlight(QPushButton("Open 3-D volume view…"))
        b_3d.setToolTip("Rotate the stack in 3-D, scroll a slab through it, and "
                        "colour beads by depth. Uses the Z step and pixel size "
                        "to build cubic voxels, so the shape is geometrically "
                        "true rather than stretched.")
        b_3d.clicked.connect(self.open_volume_view)
        return _group("3-D view (active channel)", [
            ("Z step", self.z_step), (None, b_3d),
            (None, _note("Z step + pixel size (set with the scale bar) fix the "
                        "true proportions. Every other 3-D control lives inside "
                        "the viewer window.")),
        ])

    def _build_export_group(self):
        b_png = QPushButton("Save current view (PNG)")
        b_png.setToolTip("Saves exactly what is on screen — including a "
                         "side-by-side or overlay composite.")
        b_png.clicked.connect(self.save_view_png)
        b_tif32 = QPushButton("Save processed slice (32-bit TIFF)")
        b_tif32.clicked.connect(self.save_slice_tiff)
        b_stack = QPushButton("Process & save whole stack (32-bit TIFF)")
        b_stack.clicked.connect(self.save_stack_tiff)
        b_series = QPushButton("Save display series (8-bit PNGs)")
        b_series.clicked.connect(self.save_display_series)
        note = QLabel("Everything is written into the active channel's data folder.")
        note.setStyleSheet("color:#888;")
        note.setWordWrap(True)
        return _group("Export", [
            (None, b_png), (None, b_tif32), (None, b_stack), (None, b_series),
            (None, note),
        ])

    def _connect(self, widget):
        """Wire a pipeline control to the debounced refresh."""
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(self._on_param_changed)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_param_changed)
        else:
            widget.valueChanged.connect(self._on_param_changed)

    def _schedule(self):
        self._debounce.start()

    def _on_param_changed(self, *_):
        """A pipeline control moved: store it on the active channel, then redraw."""
        if self._loading_params:
            return
        self.ch().params = self._collect_params()
        self._mirror_brightness_contrast()
        self._schedule()

    # Brightness/contrast settings shared when "Apply to both images" is on.
    # NOTE: deliberately excludes ``manual_min``/``manual_max`` — the ORCA and
    # EVK4 have very different value scales (std vs event-rate), so copying one
    # camera's manual range onto the other blows it out to all-white/all-black.
    # Only the Auto/manual *choice* and gamma are shared; each channel keeps its
    # own numeric range (and its own auto fit).
    _BC_PARAM_KEYS = ("contrast_mode", "gamma")

    def _mirror_brightness_contrast(self):
        """Copy the active channel's Auto/manual choice + gamma to the other.

        Only when "Apply to both images (A & B)" is on. Each channel still fits
        its own range — in Auto to its own histogram, in Manual to its own
        min/max — so neither camera is forced onto the other's scale.
        """
        if not self.bc_both.isChecked():
            return
        src = self.ch()
        for c in self.channels.values():
            if c is src:
                continue
            for k in self._BC_PARAM_KEYS:
                c.params[k] = src.params[k]
            if c.use_stack_range != src.use_stack_range:
                c.use_stack_range = src.use_stack_range
                c.stack_range = None

    def _on_bc_both_toggled(self, checked):
        if checked:
            self._mirror_brightness_contrast()
        self._schedule()

    # ------------------------------------------------------------------
    # parameter <-> widget marshalling (per channel)
    # ------------------------------------------------------------------
    # Combo order, as stable keys — reordering the combo must not silently change
    # what a saved record means.
    _DENOISE_METHODS = ["none", "gaussian", "median"]
    _CONTRAST_MODES = ["auto", "manual"]

    def _collect_params(self):
        return {
            "hot_on": self.hot_on.isChecked(),
            "hot_k": self.hot_k.value(),
            "hot_win": self.hot_win.value(),
            "bg_radius": self.bg_radius.value(),
            "denoise_method": self._DENOISE_METHODS[self.denoise_method.currentIndex()],
            "denoise_strength": self.denoise_strength.value(),
            "unsharp_amount": self.unsharp_amount.value(),
            "unsharp_radius": self.unsharp_radius.value(),
            "contrast_mode": "auto" if self.auto_bc.isChecked() else "manual",
            "manual_min": self.manual_min.value(),
            "manual_max": self.manual_max.value(),
            "gamma": self.gamma.value(),
            "lut": self.lut.currentText(),
            "scalebar_on": self.scalebar_on.isChecked(),
            "um_per_px": self.pixel_um.value(),
            "bar_um": self.bar_um.value(),
            "z_step": self.z_step.value(),
        }

    def _apply_params_to_widgets(self, p):
        """Repopulate the panel from a channel's stored settings."""
        self._loading_params = True
        try:
            self.hot_on.setChecked(p["hot_on"])
            self.hot_k.setValue(p["hot_k"])
            self.hot_win.setValue(p["hot_win"])
            self.bg_radius.setValue(p["bg_radius"])
            self.denoise_method.setCurrentIndex(
                self._DENOISE_METHODS.index(p["denoise_method"]))
            self.denoise_strength.setValue(p["denoise_strength"])
            self.unsharp_amount.setValue(p["unsharp_amount"])
            self.unsharp_radius.setValue(p["unsharp_radius"])
            self.auto_bc.setChecked(p["contrast_mode"] == "auto")
            self.manual_min.setValue(p["manual_min"])
            self.manual_max.setValue(p["manual_max"])
            self.gamma.setValue(p["gamma"])
            self.lut.setCurrentText(p["lut"])
            self.scalebar_on.setChecked(p["scalebar_on"])
            self.pixel_um.setValue(p["um_per_px"])
            self.bar_um.setValue(p["bar_um"])
            self.z_step.setValue(p["z_step"])
            self.stack_cb.setChecked(self.ch().use_stack_range)
        finally:
            self._loading_params = False

    def _on_editing_changed(self, name):
        self.active = name
        self._apply_params_to_widgets(self.ch().params)
        self._sync_slider()
        self._schedule()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def open_folder(self, name):
        folder = QFileDialog.getExistingDirectory(self, f"Select a data folder for channel {name}")
        if not folder:
            return
        folder = Path(folder)
        paths = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".mat")],
            key=lambda p: p.name.lower(),
        )
        if not paths:
            self._status("No .tif/.tiff/.mat files in that folder.")
            return
        self._load(name, paths, folder)

    def open_files(self, name):
        files, _ = QFileDialog.getOpenFileNames(
            self, f"Select image files for channel {name}", "",
            "Images (*.tif *.tiff *.mat);;TIFF (*.tif *.tiff);;MATLAB (*.mat)")
        if not files:
            return
        self._load(name, [Path(f) for f in files], Path(files[0]).parent)

    def _load(self, name, paths, source_dir):
        c = self.ch(name)
        c.frames = scan_paths(paths)
        c.index = 0
        c.source_dir = source_dir
        c.registered = None
        c.registered_reason = None
        c.reset_derived()
        if not c.frames:
            self.chan_labels[name].setText("no readable frames")
            self._status(f"Channel {name}: nothing readable in that selection.")
            return

        n_files = len({f.path for f in c.frames})
        self.chan_labels[name].setText(
            f"{c.n_slices} slice(s), {n_files} file(s) — {source_dir.name}")
        # Loading into a channel is a strong hint you want to work on it.
        if self.editing.currentText() != name:
            self.editing.setCurrentText(name)
        else:
            self._sync_slider()
        # Two channels loaded and no view chosen yet: show them together.
        if all(ch.loaded for ch in self.channels.values()) and self.view_mode.currentIndex() == 0:
            self.view_mode.setCurrentIndex(2)  # side by side
        self.refresh()

    def clear_channel(self, name):
        c = self.ch(name)
        c.frames = []
        c.index = 0
        c.registered = None
        c.registered_reason = None
        c.reset_derived()
        self.chan_labels[name].setText("empty")
        self._sync_slider()
        self.refresh()

    # ------------------------------------------------------------------
    # field matching
    # ------------------------------------------------------------------
    def current_affine(self):
        """The current B->A map (measured, seeded, or reset to identity)."""
        return self.base_affine

    def _both_ready(self):
        a, b = self.ch("A"), self.ch("B")
        if not (a.loaded and b.loaded):
            self._status("Field matching needs data in both channels.")
            return None
        a.ensure_processed()
        b.ensure_processed()
        img_a = a.display_image()
        img_b = b.display_image()
        if img_a is None or img_b is None:
            self._status("Could not process both channels.")
            return None
        return img_a, img_b

    def measure_match(self):
        if register_evk4_to_orca is None:
            return
        if self._match_worker is not None:
            return  # a search is already running
        pair = self._both_ready()
        if pair is None:
            return
        img_a, img_b = pair
        seed = self.base_affine if self.match_measured else None

        # The search is a few-thousand-iteration loop; run it off the UI thread
        # so the window stays responsive (drag, scroll, resize all keep working).
        self.b_measure.setEnabled(False)
        self.b_measure.setText("Measuring field match…")
        worker = _MatchWorker(img_a, img_b, seed)
        worker.progress.connect(self.status_label.setText)
        worker.done.connect(self._on_match_done)
        worker.failed.connect(self._on_match_failed)
        worker.finished.connect(self._on_match_worker_finished)
        self._match_worker = worker
        worker.start()
        self._status("Measuring field match… (the window stays usable)")

    def _on_match_done(self, affine, score, info):
        self.base_affine = np.asarray(affine, dtype=np.float64)
        self.match_measured = True
        self.match_active = True
        self.match_score = score
        # The search only spans scale 0.55-0.96 (it expects B's finer EVK4 pixels
        # to *shrink* onto A), so the commonest cause of a poor score is the two
        # datasets being loaded into the wrong channels.
        warn = ("  ⚠ low confidence — check the overlay, and that A is the ORCA "
                "and B the EVK4 (the search only shrinks B onto A)"
                if score < 0.5 else "")
        self.match_label.setText(
            f"NCC = {score:.2f}, θ = {info['theta']:.2f}°, "
            f"scale = {info['scale']:.4f}, flip = {info['flip']}{warn}")
        self._save_match_state()
        self.refresh()
        self._final_status(f"Field match measured: NCC = {score:.2f}.{warn}")

    def _on_match_failed(self, message):
        self._status(f"Field match failed: {message}")

    def _on_match_worker_finished(self):
        """Re-enable the button once the thread has fully stopped (any outcome)."""
        self.b_measure.setEnabled(REGISTRATION_AVAILABLE)
        self.b_measure.setText("Measure field match (A ↔ B)")
        self._match_worker = None

    def reset_match(self):
        """Clear the match, and undo any crop/warp it was used for on A."""
        self.base_affine = IDENTITY_AFFINE.copy()
        self.match_measured = False
        self.match_active = False
        self.match_score = None
        self.match_label.setText("no match measured")
        a = self.ch("A")
        reverted = a.registered_reason in ("crop", "warp")
        if reverted:
            a.registered = None
            a.registered_reason = None
            a.reset_derived()
        self._save_match_state()
        self.refresh()
        if reverted:
            self._final_status("Match reset — channel A restored to its original frames.")

    def load_last_match(self):
        """Re-apply the match remembered from the last session.

        The same record the tool loads at startup: the cameras rarely move
        between acquisitions, so last time's B → A map is normally still the
        right one and re-measuring it is pure cost.
        """
        state = read_match_state(MATCH_STATE_PATH)
        if state is None:
            self._final_status(
                f"No remembered match to load ({MATCH_STATE_PATH}). "
                f"Measure one first.")
            return
        try:
            self._apply_match_state(state, "last session")
        except Exception as exc:  # noqa: BLE001 — a bad record is not a crash
            traceback.print_exc()
            self._final_status(f"Could not read the remembered match: {exc}")
            return
        self.refresh()
        self._final_status(f"Loaded the match saved "
                           f"{state.get('saved_utc', '?')} (UTC).")

    # -- persistence ---------------------------------------------------
    def _match_state(self):
        """The current match as a JSON-serialisable record."""
        return {
            "version": MATCH_STATE_VERSION,
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "affine": np.asarray(self.base_affine, dtype=float).tolist(),
            # The durable fact — that a match was measured — not whether the
            # live warp happens to be running (see __init__).
            "active": bool(self.match_measured),
            "score": None if self.match_score is None else float(self.match_score),
            "note": self.match_label.text(),
            # Provenance only — the match is not tied to these folders, but it is
            # the first thing to check when a restored match looks wrong.
            "source_a": str(self.ch("A").source_dir or ""),
            "source_b": str(self.ch("B").source_dir or ""),
        }

    def _apply_match_state(self, state, origin):
        self.base_affine = np.asarray(state["affine"], dtype=np.float64)
        self.match_score = state.get("score")
        # A record holding a non-identity affine *is* a measured match, whatever
        # the flag says — older builds persisted the transient "live warp is
        # running" flag here, so a match measured and then used to warp A came
        # back reading un-measured. Only "Reset match" writes identity back.
        self.match_measured = (
            bool(state.get("active", True))
            or not np.allclose(self.base_affine, IDENTITY_AFFINE, atol=1e-9))
        # The live warp is off only while A is itself sitting on B's grid —
        # never the case at startup, since a warp is not persisted.
        warped = self.ch("A").registered_reason in ("warp", "crop")
        self.match_active = self.match_measured and not warped
        # Records written before the axial section was removed carry z_* fields
        # too; they are simply ignored.
        if not self.match_measured:      # a remembered "Reset match"
            self.match_label.setText("no match measured")
            return
        when = state.get("saved_utc", "?")
        score = state.get("score")
        score_txt = f"NCC = {score:.2f}, " if isinstance(score, (int, float)) else ""
        # Carry the low-confidence warning across sessions too: a weak match is
        # just as misleading restored as freshly measured.
        weak = ("  ⚠ low confidence — check the overlay before trusting it"
                if isinstance(score, (int, float)) and score < 0.5 else "")
        self.match_label.setText(
            f"{score_txt}restored from {origin} (saved {when}){weak}")

    def _save_match_state(self):
        """Remember the match for the next session (best effort, never fatal)."""
        try:
            write_match_state(MATCH_STATE_PATH, self._match_state())
        except Exception:  # noqa: BLE001 — persistence must never break the tool
            traceback.print_exc()

    def _restore_match_state(self):
        state = read_match_state(MATCH_STATE_PATH)
        if state is None:
            return
        try:
            self._apply_match_state(state, "last session")
        except Exception:  # noqa: BLE001 — a bad record must not block startup
            traceback.print_exc()

    # -- session persistence (channels, pipeline params, view mode) ----
    def _collect_session_state(self):
        """The currently loaded channels + their settings, as JSON."""
        def chan_state(c):
            seen = []
            for f in c.frames:
                if f.path not in seen:
                    seen.append(f.path)
            return {
                "source_dir": str(c.source_dir) if c.source_dir else "",
                "files": [str(p) for p in seen],
                "index": c.index,
                "params": dict(c.params),
                "use_stack_range": bool(c.use_stack_range),
            }
        return {
            "version": SESSION_STATE_VERSION,
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "active": self.active,
            "view_mode": self.view_mode.currentIndex(),
            "link_slices": self.link_slices.isChecked(),
            "bc_both": self.bc_both.isChecked(),
            "channels": {name: chan_state(c) for name, c in self.channels.items()},
        }

    def _sanitize_params(self, saved):
        """Merge saved params onto the defaults, guarding against stale enum values.

        A session saved by an older build may name a denoise method, contrast
        mode or LUT that no longer exists (a pipeline stage since removed) —
        falling back to the default for just that key beats crashing the whole
        restore (and leaving the rest of the UI half-initialised) partway
        through ``_apply_params_to_widgets``.
        """
        out = dict(DEFAULT_PARAMS)
        out.update(saved or {})
        if out["denoise_method"] not in self._DENOISE_METHODS:
            out["denoise_method"] = DEFAULT_PARAMS["denoise_method"]
        if out["contrast_mode"] not in self._CONTRAST_MODES:
            out["contrast_mode"] = DEFAULT_PARAMS["contrast_mode"]
        if out["lut"] not in LUT_NAMES:
            out["lut"] = DEFAULT_PARAMS["lut"]
        return out

    def _apply_session_state(self, state):
        """Reload whatever channels/settings ``state`` describes.

        Best-effort per channel: a moved/deleted dataset just leaves that
        channel empty rather than blocking the rest of the restore.
        """
        for name, cfg in (state.get("channels") or {}).items():
            if name not in self.channels:
                continue
            paths = [Path(p) for p in cfg.get("files", []) if Path(p).exists()]
            if not paths:
                continue
            c = self.ch(name)
            c.frames = scan_paths(paths)
            if not c.frames:
                continue
            c.source_dir = Path(cfg.get("source_dir") or paths[0].parent)
            c.registered = None
            c.registered_reason = None
            c.reset_derived()
            c.params = self._sanitize_params(cfg.get("params"))
            c.use_stack_range = bool(cfg.get("use_stack_range", False))
            c.index = int(np.clip(cfg.get("index", 0), 0, c.n_slices - 1))
            n_files = len({f.path for f in c.frames})
            self.chan_labels[name].setText(
                f"{c.n_slices} slice(s), {n_files} file(s) — {c.source_dir.name}")

        self.active = state.get("active") if state.get("active") in self.channels else "A"
        self.editing.blockSignals(True)
        self.editing.setCurrentText(self.active)
        self.editing.blockSignals(False)
        self._apply_params_to_widgets(self.ch().params)

        self.view_mode.blockSignals(True)
        self.view_mode.setCurrentIndex(int(state.get("view_mode", 0)))
        self.view_mode.blockSignals(False)
        self.link_slices.blockSignals(True)
        self.link_slices.setChecked(bool(state.get("link_slices", False)))
        self.link_slices.blockSignals(False)
        self.bc_both.blockSignals(True)
        self.bc_both.setChecked(bool(state.get("bc_both", True)))
        self.bc_both.blockSignals(False)

        self._sync_slider()
        self.refresh()

    def _save_session_state(self):
        """Remember the loaded channels + settings for next time (best effort)."""
        try:
            write_session_state(SESSION_STATE_PATH, self._collect_session_state())
        except Exception:  # noqa: BLE001 — persistence must never break the tool
            traceback.print_exc()

    def _restore_session_state(self):
        state = read_session_state(SESSION_STATE_PATH)
        if state is None:
            return
        try:
            self._apply_session_state(state)
        except Exception:  # noqa: BLE001 — a bad record must not block startup
            traceback.print_exc()

    def warp_a_to_b(self):
        """Resample channel A onto B's exact grid — B's rotation, scale and shape.

        Gives A pixel-for-pixel the same field of view as B: the smaller of
        the two, since B is the narrower camera. Costs one interpolation pass
        over A's data per plane. Use "Reset match" to undo.
        """
        if self.ch("A").registered_reason in ("warp", "crop"):
            self._status("Channel A is already on B's field of view — "
                         "'Reset match' first if you want to redo it.")
            return
        if not self.match_measured:
            self._status("No field match measured — measure one first, or load "
                         "last session's.")
            return
        pair = self._both_ready()
        if pair is None:
            return
        img_a, img_b = pair
        inv = cv2.invertAffineTransform(self.current_affine().astype(np.float64))
        shape_b = img_b.shape[:2]
        if min(shape_b) < 8:
            self._status("B's frame is degenerately small — check the match.")
            return
        a = self.ch("A")
        warped = [warp_to(np.asarray(a.raw(i), np.float32), inv, shape_b)
                  for i in range(a.n_slices)]
        a.registered = warped  # reuse the "materialised slices" path
        a.registered_reason = "warp"
        a.reset_derived()
        # A now sits on B's own grid, so the live "warp B into A" pass in the
        # overlay/side-by-side view would re-apply the B->A map on top of an
        # already-aligned pair, shearing B for no reason — turn it off.
        self.match_active = False
        self.refresh()
        self._final_status(f"Channel A warped onto B's exact field of view "
                           f"({shape_b[1]}×{shape_b[0]} px, same grid as B). "
                           f"Use 'Reset match' to undo.")

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------
    def refresh(self):
        try:
            self._refresh()
        except Exception as exc:  # noqa: BLE001 — a bad parameter must not kill the app
            traceback.print_exc()
            self._status(f"Processing failed: {exc}")

    def _refresh(self):
        mode = self.view_mode.currentIndex()
        a, b = self.ch("A"), self.ch("B")

        if mode in (0, 1):
            c = a if mode == 0 else b
            if not c.loaded:
                self.image_label.setText(f"Channel {c.name} is empty.")
                self.image_label.setPixmap(QPixmap())
                return
            self._ensure_stack_range(c)
            view, (lo, hi) = c.render()
            self._show(view)
            self._status(self._describe(c, lo, hi))
            self._sync_slider()
            return

        if not (a.loaded and b.loaded):
            missing = "A" if not a.loaded else "B"
            self.image_label.setText(f"Channel {missing} is empty — "
                                     f"load it, or switch the view mode.")
            self.image_label.setPixmap(QPixmap())
            return

        self._ensure_stack_range(a)
        self._ensure_stack_range(b)
        view_a, (lo_a, hi_a) = a.render()
        view_b, (lo_b, hi_b) = b.render()

        if self.match_active:
            # Warp B into A's pixel frame so both show the same field.
            view_b = to_bgr(view_b)
            view_b = cv2.warpAffine(view_b, self.current_affine().astype(np.float64),
                                    (view_a.shape[1], view_a.shape[0]),
                                    flags=cv2.INTER_LINEAR)

        if mode == 2:
            composite = side_by_side(view_a, view_b, labels=("A", "B"))
        else:
            composite = green_magenta_overlay(view_a, view_b)

        self._show(composite)
        match = (f" · matched (NCC {self.match_score:.2f})" if self.match_score is not None
                 else " · matched" if self.match_active else "")
        self._status(f"A: {self._describe(a, lo_a, hi_a)}"
                     f"    ||    B: {self._describe(b, lo_b, hi_b)}{match}")
        self._sync_slider()

    _REGISTERED_LABELS = {
        "drift": "drift-corrected",
        "crop": "cropped to B's footprint",
        "warp": "warped onto B's frame",
        "resample": "Z-resampled",
    }

    def _describe(self, c, lo, hi):
        label = c.frames[c.index].label if c.loaded else "empty"
        reg = self._REGISTERED_LABELS.get(c.registered_reason, "")
        reg = f" · {reg}" if reg else ""
        return f"{label}{reg} | display {lo:.4g}–{hi:.4g}"

    def _ensure_stack_range(self, c):
        if not c.use_stack_range:
            return
        if c.stack_range is not None or c.params["contrast_mode"] == "manual":
            return
        lows, highs = [], []
        prog = QProgressDialog(f"Computing stack histogram (channel {c.name})…",
                               "Cancel", 0, c.n_slices, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        try:
            for i in range(c.n_slices):
                prog.setValue(i)
                QApplication.processEvents()
                if prog.wasCanceled():
                    break
                img = c.process_index(i)
                lo, hi = imagej_auto_minmax(img)
                lows.append(lo)
                highs.append(hi)
        finally:
            prog.close()
        c.stack_range = (min(lows), max(highs)) if lows else None

    def _process_all(self, c, title):
        """Process every slice of a channel, showing progress."""
        prog = QProgressDialog(title, "Cancel", 0, c.n_slices, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        out = []
        try:
            for i in range(c.n_slices):
                prog.setValue(i)
                QApplication.processEvents()
                if prog.wasCanceled():
                    return None
                out.append(c.process_index(i))
        finally:
            prog.close()
        return out

    def _show(self, view):
        self._qbuf = np.ascontiguousarray(view)
        if self._qbuf.ndim == 3:
            h, w, _ = self._qbuf.shape
            qimg = QImage(self._qbuf.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        else:
            h, w = self._qbuf.shape
            qimg = QImage(self._qbuf.data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _status(self, text):
        """Set the status line.

        Deliberately does NOT pump the event loop: doing so lets the pending
        debounced refresh fire and overwrite the message the user just asked for
        (an export confirmation would vanish the instant it appeared). Long
        operations use :meth:`_progress_status` instead.
        """
        self.status_label.setText(text)

    def _progress_status(self, text):
        """Status callback for long operations — repaints as they run."""
        self.status_label.setText(text)
        QApplication.processEvents()

    def _final_status(self, text):
        """A confirmation that must survive: cancel any queued redraw first."""
        self._debounce.stop()
        self.status_label.setText(text)

    def _sync_slider(self):
        # Everything that changes the active channel or its length goes through
        # here, so it is the one place the pinned header has to be refreshed.
        self._update_editing_header()
        c = self.ch()
        n = c.n_slices
        self.slider.blockSignals(True)
        self.slider.setEnabled(n > 1)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(min(c.index, max(0, n - 1)))
        self.slider.blockSignals(False)
        self.pos_label.setText(f"{c.index + 1} / {n}" if n else "0 / 0")

    def _on_image_wheel(self, steps):
        """Mouse wheel over the image view scrolls Z, same as the slider."""
        if not self.slider.isEnabled():
            return
        value = int(np.clip(self.slider.value() + steps, 0, self.slider.maximum()))
        self.slider.setValue(value)

    def _on_slider(self, value):
        self.ch().index = value
        if self.link_slices.isChecked():
            for c in self.channels.values():
                if c.loaded:
                    c.index = min(value, c.n_slices - 1)
        self._schedule()

    # ------------------------------------------------------------------
    # 3-D view (active channel)
    # ------------------------------------------------------------------
    def open_volume_view(self):
        """Open the interactive 3-D view of the active channel's stack."""
        c = self.ch()
        if c.n_slices < 3:
            self._status("The 3-D view needs a stack (at least 3 planes).")
            return
        if c.params["z_step"] <= 0:
            self._status("Set the Z step before opening the 3-D view — without "
                         "it the volume cannot be scaled to true proportions.")
            return
        view = VolumeView(self, c, self._build_volume)
        if view.vol is None:
            return
        # Modeless, so the pipeline can be tuned with the volume still open;
        # keep a reference or Python would collect the window immediately.
        self._volume_views.append(view)
        view.finished.connect(lambda _r, v=view: self._volume_views.remove(v)
                              if v in self._volume_views else None)
        view.show()

    def _build_volume(self, c, max_dim, region=None):
        """Build a render volume from ``c``'s processed planes (cancellable).

        Planes are processed and shrunk one at a time inside
        ``build_view_volume``, so this never holds the whole processed stack —
        which at full sensor would be gigabytes. ``region`` is an optional
        ``(roi, planes)`` restricting the build to part of the stack.
        """
        roi, planes = region if region else (None, None)
        prog = QProgressDialog(f"Building 3-D volume (channel {c.name})…",
                               "Cancel", 0, c.n_slices, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        cancelled = False

        def get_slice(i):
            nonlocal cancelled
            prog.setValue(i)
            QApplication.processEvents()
            if prog.wasCanceled():
                cancelled = True
                raise _Cancelled()
            return c.process_index(i)

        try:
            return build_view_volume(get_slice, c.n_slices, c.params["z_step"],
                                     c.params["um_per_px"], max_dim=max_dim,
                                     roi=roi, planes=planes)
        except _Cancelled:
            return None
        finally:
            prog.close()
            if not cancelled:
                scope = "a cropped region of " if region else ""
                self._status(f"3-D volume built from {scope}{c.n_slices} planes "
                             f"(channel {c.name}).")

    def _on_stack_toggled(self, checked):
        if self._loading_params:
            return
        c = self.ch()
        c.use_stack_range = checked
        c.stack_range = None
        self._mirror_brightness_contrast()
        self._schedule()

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def _out_dir(self):
        c = self.ch()
        return c.source_dir or Path.cwd()

    def _out_base(self, stack_wide=False):
        """Filename base for exports.

        Slice exports are named after the slice's own source file. Stack-wide
        exports fall back to the folder name when the frames span several files,
        so the result isn't named after whichever slice happened to be selected.
        """
        c = self.ch()
        if not c.loaded:
            return "image"
        if stack_wide and len({f.path for f in c.frames}) > 1:
            return self._out_dir().name or "stack"
        return Path(c.frames[c.index].label.split(" [")[0]).stem

    def save_view_png(self):
        if self._qbuf is None:
            return
        mode = self.view_mode.currentIndex()
        c = self.ch()
        if mode == 2:
            tag = "side_by_side"
        elif mode == 3:
            tag = "overlay"
        else:
            tag = f"z{c.index:03d}"
        base = self._out_base(stack_wide=(mode >= 2))
        path = self._out_dir() / f"{base}_{tag}_view.png"
        cv2.imwrite(str(path), self._qbuf)
        self._final_status(f"Saved {path}")

    def save_slice_tiff(self):
        c = self.ch()
        img = c.display_image()
        if img is None:
            return
        base, suffix = self._out_base(), f"z{c.index:03d}"
        path = self._out_dir() / f"{base}_{suffix}_processed.tif"
        tifffile.imwrite(str(path), np.asarray(img, dtype=np.float32), imagej=True)
        self._final_status(f"Saved {path}")

    def save_stack_tiff(self):
        c = self.ch()
        processed = self._process_all(c, f"Processing channel {c.name} for export…")
        if processed is None:
            return
        path = self._out_dir() / f"{self._out_base(stack_wide=True)}_processed_stack.tif"
        tifffile.imwrite(str(path), np.asarray(processed, dtype=np.float32), imagej=True)
        self._final_status(f"Saved {path} ({len(processed)} planes, 32-bit)")

    def save_display_series(self):
        c = self.ch()
        processed = self._process_all(c, f"Rendering channel {c.name} display series…")
        if processed is None:
            return
        out = self._out_dir() / f"{self._out_base(stack_wide=True)}_display"
        os.makedirs(out, exist_ok=True)
        rng = c.stack_range if c.use_stack_range else None
        for i, img in enumerate(processed):
            view, _ = render_display(img, c.params, rng)
            cv2.imwrite(str(out / f"z{i:03d}.png"), view)
        self._final_status(f"Saved {len(processed)} PNGs to {out}")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        # Catches nudges, which are not worth a write on every spinbox tick.
        self._save_match_state()
        self._save_session_state()
        # Let a running field-match search finish (a few seconds) so Qt doesn't
        # tear the thread down mid-run; the search can't be interrupted cleanly.
        if self._match_worker is not None:
            self._match_worker.wait()
        for view in list(self._volume_views):
            view.close()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if any(c.loaded for c in self.channels.values()):
            self._schedule()

    def keyPressEvent(self, event):
        # Arrow / Page keys scroll the stack even when the slider isn't focused.
        if not self.slider.isEnabled():
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Down, Qt.Key.Key_PageDown):
            self.slider.setValue(min(self.slider.value() + 1, self.slider.maximum()))
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            self.slider.setValue(max(self.slider.value() - 1, 0))
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    win = ImageLab()
    # Maximized rather than true full screen: the window keeps its title bar,
    # so it can still be moved and closed.
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
