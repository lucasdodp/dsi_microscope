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
acquire at once. Each channel keeps its own files, its own slice index, its own
calibration references and its own full set of pipeline settings; the control
panel edits whichever channel is selected in **Editing**.

View modes: *A only*, *B only*, *Side by side*, and *Overlay* (green A / magenta
B — where the two detectors see the same beads, green + magenta reads white).

**Field matching.** The two cameras see the same sample through different
optics: the EVK4 footprint sits rotated (~43°) and scaled (~0.745, the
4.86/6.5 µm pixel-pitch ratio) inside the ORCA field. *Measure field match*
runs the project's own masked-NCC registration (``core.register_evk4_to_orca``,
the same routine the acquisition GUI uses) on the two loaded images and returns
the B→A affine; B is then warped into A's frame, so side-by-side and overlay
show the same field at the same scale and orientation. Measuring from the image
content works regardless of how each file was cropped, which the stored
calibration alone cannot do. *Seed from calibration* preloads
``config.EVK4_TO_ORCA_AFFINE``, and the nudge controls (dx / dy / rotation /
scale) refine the result by hand.

**The match is not re-measured unless you ask for it.** Whatever match is in
force is reused for every warp and for *Crop A to B's footprint*, and it is
remembered between sessions (``MATCH_STATE_PATH``, env-override
``DSI_IMAGE_LAB_MATCH``): the tool restores the last one at startup, so a fresh
dataset from an unchanged optical setup can be cropped and overlaid immediately.
*Save / Load match to file* writes the same record next to the data, so a match
travels with the dataset. Re-measure only when the cameras have actually moved.

--------------------------------------------------------------------------
The 3-D volume view
--------------------------------------------------------------------------
*Stack tools -> Open 3-D volume view* turns the active channel's z-stack into
an interactive volume: **drag to rotate**, wheel to zoom, **Shift+wheel to
scroll a slab** through the depth. Nothing is *reconstructed* — a DSI stack is
already a 3-D image, because the optical sectioning is what makes each plane
belong to one depth. The view only resamples it onto **cubic voxels** (using
the channel's Z step and pixel size, so the shape is geometrically true rather
than stretched by the z step) and projects it from the chosen angle.

Beads read as three-dimensional through three cues: parallax while rotating,
**hue keyed to depth in the sample** — the same ramp as the 2-D depth-coded
projection, and rotation-invariant, so a bead keeps its colour as the volume
turns — and the projected **wireframe box**, without which an orthographic
projection of sparse beads gives the eye nothing to judge the rotation against.
*Solid* mode swaps the max-intensity projection for front-to-back compositing,
so nearer beads occlude further ones.

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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
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
    QMessageBox,
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
    apply_display_range,
    axial_profile,
    axial_profile_peakedness,
    build_view_volume,
    depth_colour_code,
    depth_colour_legend,
    draw_scale_bar,
    draw_volume_box,
    find_axial_offset,
    find_axial_offset_by_images,
    hot_pixel_mask_from_dark,
    imagej_auto_minmax,
    load_image_file,
    orthogonal_views,
    percentile_minmax,
    plane_positions,
    process_slice,
    project_stack,
    psf_sigma_px,
    read_axial_profile_csv,
    register_stack,
    render_display,
    render_volume,
    resample_stack_z,
    save_axial_comparison,
    scan_paths,
    view_rotation,
    volume_display_range,
)

# The measured EVK4->ORCA registration lives in the main application. Import it
# when available so the field match uses the *same* algorithm and calibration as
# the acquisition GUI; the tool still runs (with manual matching only) if this
# checkout is incomplete or SciPy/OpenCV differ.
try:
    from core import register_evk4_to_orca
    from config import EVK4_TO_ORCA_AFFINE
    REGISTRATION_AVAILABLE = True
except Exception:  # noqa: BLE001 — degrade to manual matching, never fail to start
    register_evk4_to_orca = None
    EVK4_TO_ORCA_AFFINE = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
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

MATCH_STATE_VERSION = 2          # v2 added the axial (Z) offset
MATCH_STATE_READABLE = (1, 2)    # older records still load; missing fields default


def write_match_state(path, state):
    """Persist a match record, via a temp file + atomic replace.

    Writing in place would let an interrupted save leave a truncated JSON, which
    would then silently reset the match to identity on the next start.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)
    return path


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


# ===========================================================================
# Field matching helpers (pure)
# ===========================================================================
def compose_match(base_affine, dx=0.0, dy=0.0, theta_deg=0.0, scale=1.0, centre=(0.0, 0.0)):
    """Compose a manual nudge onto a measured B->A affine.

    The nudge is applied *after* the base map, in A's coordinate frame, as a
    similarity about ``centre`` — so dragging rotation/scale pivots about the
    middle of the warped image rather than its corner.
    """
    base3 = np.vstack([np.asarray(base_affine, dtype=np.float64), [0, 0, 1]])
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                float(theta_deg), float(scale))
    M[0, 2] += float(dx)
    M[1, 2] += float(dy)
    nudge3 = np.vstack([M, [0, 0, 1]])
    return (nudge3 @ base3)[:2]


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


def footprint_bbox(affine, b_shape, a_shape):
    """Bounding box in A of B's footprint mapped through ``affine``.

    Because the EVK4 footprint is *rotated* in the ORCA field, the matching crop
    is the axis-aligned bounding box of the mapped corners — it contains the
    whole B field plus four corner triangles B does not see. Clamped to A.
    """
    h, w = int(b_shape[0]), int(b_shape[1])
    pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
    corners = pts @ np.asarray(affine, dtype=np.float64)[:2].T
    x0 = max(0, int(np.floor(corners[:, 0].min())))
    y0 = max(0, int(np.floor(corners[:, 1].min())))
    x1 = min(int(a_shape[1]), int(np.ceil(corners[:, 0].max())))
    y1 = min(int(a_shape[0]), int(np.ceil(corners[:, 1].max())))
    return (x0, y0, x1, y1), corners


# ===========================================================================
# Per-channel state
# ===========================================================================
DEFAULT_PARAMS = {
    "hot_on": False, "hot_k": 6.0, "hot_win": 5,
    "noise_mode": "off", "noise_strength": 1.0, "gain": 0.23,
    "offset": 100.0, "read_noise": 4.0,
    "norm_mode": "off", "flat_sigma": 60.0, "bg_radius": 0,
    "denoise_method": "none", "denoise_strength": 1.0,
    "decon_on": False, "decon_iters": 10, "decon_sigma": 1.2,
    "unsharp_amount": 0.0, "unsharp_radius": 2.0,
    "contrast_mode": "auto", "pct_low": 0.5, "pct_high": 99.5,
    "manual_min": 0.0, "manual_max": 255.0, "gamma": 1.0,
    "clahe_on": False, "clahe_clip": 2.0, "clahe_tiles": 8,
    "lut": "Grayscale", "scalebar_on": False,
    "um_per_px": 0.1625, "bar_um": 10.0,
    # optics (per channel: the two cameras have different pixel pitches)
    "na": 0.65, "wavelength": 580.0, "z_step": 0.5,
}


class Channel:
    """One loaded dataset with its own settings, references and view state."""

    def __init__(self, name):
        self.name = name
        self.frames: list[Frame] = []
        self.index = 0
        self.source_dir = None
        self.refs = {"avg": None, "noise": None, "hot_mask": None}
        self.ref_names = {}
        self.params = dict(DEFAULT_PARAMS)
        self.registered = None
        self.projection = None
        self.projection_is_bgr = False
        self.proj_label = ""
        self.processed = None
        self.cache_key = None
        self.stack_range = None
        # True stage positions per plane, when read from an acquisition's
        # ``_axial_profile_*.csv``; otherwise positions come from the z step.
        self.z_positions = None

    # -- data access ---------------------------------------------------
    @property
    def loaded(self):
        return bool(self.frames)

    def z_um(self):
        """Axial position (um) of every plane.

        Prefers the true stage positions read from the acquisition's axial CSV;
        falls back to ``plane index x z step``. The fallback puts plane 0 at
        z = 0, so an offset measured against it also absorbs any difference in
        where the two scans started.
        """
        if (self.z_positions is not None
                and len(self.z_positions) == self.n_slices):
            return np.asarray(self.z_positions, dtype=np.float64)
        return plane_positions(self.n_slices, self.params["z_step"])

    @property
    def n_slices(self):
        return len(self.frames)

    def raw(self, i):
        if self.registered is not None:
            return self.registered[i]
        return self.frames[i].data

    def refs_for(self, i):
        """References matching slice ``i`` (broadcast, or per-plane if a stack)."""
        out = {"hot_mask": self.refs.get("hot_mask")}
        for kind in ("avg", "noise"):
            ref = self.refs.get(kind)
            if ref is not None and ref.ndim == 3:
                ref = ref[min(i, ref.shape[0] - 1)]
            out[kind] = ref
        return out

    def reset_derived(self):
        """Drop everything computed from the pixels (after a load / re-register)."""
        self.processed = None
        self.cache_key = None
        self.projection = None
        self.projection_is_bgr = False
        self.stack_range = None

    # -- pipeline ------------------------------------------------------
    _PIPELINE_KEYS = (
        "hot_on", "hot_k", "hot_win", "noise_mode", "noise_strength", "gain",
        "offset", "read_noise", "norm_mode", "flat_sigma", "bg_radius",
        "denoise_method", "denoise_strength", "decon_on", "decon_iters",
        "decon_sigma", "unsharp_amount", "unsharp_radius",
    )

    def pipeline_key(self):
        return ((self.index, self.registered is not None)
                + tuple(self.params[k] for k in self._PIPELINE_KEYS))

    def process_index(self, i, release=True):
        """Run the pipeline on slice ``i``."""
        out = process_slice(self.raw(i), self.params, self.refs_for(i))
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
        """The float image currently on show (projection if one is active)."""
        if self.projection is not None and not self.projection_is_bgr:
            return self.projection
        return self.processed

    def render(self):
        """Render this channel to an 8-bit view, honouring an active projection."""
        p = self.params
        if self.projection is not None:
            if self.projection_is_bgr:
                view = self.projection
                if p["scalebar_on"]:
                    view = draw_scale_bar(view, p["um_per_px"], p["bar_um"])
                return view, (0.0, 255.0)
            return render_display(self.projection, p)

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
    zoomed = pyqtSignal(int)
    slabbed = pyqtSignal(int)
    drag_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#101010;")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._last = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._last is None:
            return
        pos = event.position()
        self.rotated.emit(pos.x() - self._last.x(), pos.y() - self._last.y())
        self._last = pos

    def mouseReleaseEvent(self, event):
        if self._last is not None:
            self._last = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.drag_finished.emit()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        # Shift+wheel scrolls the slab through the stack — the 3-D equivalent of
        # scrolling the slice slider in the 2-D view.
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.slabbed.emit(int(np.sign(steps)))
        else:
            self.zoomed.emit(int(np.sign(steps)))


class VolumeView(QDialog):
    """Interactive 3-D view of one channel's z-stack.

    The stack is already a 3-D image — DSI's optical sectioning is what makes
    each plane belong to one depth — so this only resamples it onto an isotropic
    grid once and then projects it from whatever angle the user drags to.

    Two things make sparse beads read as *3-D* rather than as a flat smear:
    hue keyed to depth in the sample (rotation-invariant, so a bead keeps its
    colour as the volume turns) and the projected wireframe box, which gives the
    eye a reference to judge the rotation against. The slab controls scroll a
    sub-range of the stack while it stays at its true position in the frame.

    Rendering is progressive: a half-resolution draft follows the mouse during a
    drag and a full-quality frame replaces it once the drag stops, so rotation
    stays smooth on volumes where a full frame takes a fraction of a second.
    """

    DETAIL = (("Fast (128)", 128), ("Balanced (256)", 256), ("Fine (384)", 384))

    def __init__(self, parent, channel, builder, max_dim=256):
        super().__init__(parent)
        self.setWindowTitle(f"3-D volume — channel {channel.name}")
        self.resize(1180, 820)
        self.setWindowFlag(Qt.WindowType.Window, True)   # modeless, own taskbar entry

        self.channel = channel
        self._builder = builder
        self.vol = None
        self.voxel_um = 1.0
        self.range = (0.0, 1.0)
        self.azimuth, self.elevation = 32.0, 24.0
        self.zoom = 1.0
        self._buf = None
        self._dragging = False

        self.canvas = VolumeCanvas()
        self.canvas.rotated.connect(self._on_rotate)
        self.canvas.zoomed.connect(self._on_zoom)
        self.canvas.slabbed.connect(self._on_slab_wheel)
        self.canvas.drag_finished.connect(self._on_drag_finished)

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
        panel.addWidget(self._build_slab_group())
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
            "Voxels along the longest axis. Higher is sharper but slower to "
            "build and to rotate; the volume is rebuilt from the current "
            "processing settings.")
        self.detail.currentIndexChanged.connect(self._on_detail)

        hint = QLabel("Drag to rotate · wheel to zoom · Shift+wheel to scroll "
                      "the slab · arrow keys nudge")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        return _group("View", [
            (None, row), ("Azimuth", self.az_slider),
            ("Elevation", self.el_slider), ("Detail", self.detail), (None, hint),
        ])

    def _build_slab_group(self):
        self.slab_centre = QSlider(Qt.Orientation.Horizontal)
        self.slab_centre.valueChanged.connect(self._on_slab)
        self.slab_depth = QSlider(Qt.Orientation.Horizontal)
        self.slab_depth.valueChanged.connect(self._on_slab)
        b_all = QPushButton("Whole volume")
        b_all.clicked.connect(self._slab_all)
        self.slab_label = QLabel("")
        self.slab_label.setStyleSheet("color:#888;")
        return _group("Slab (scroll through the stack)", [
            ("Centre", self.slab_centre), ("Thickness", self.slab_depth),
            (None, b_all), (None, self.slab_label),
        ])

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
            "Same hue ramp as the 2-D depth-coded projection, keyed to the "
            "bead's depth in the sample rather than its distance from the "
            "camera — so colours stay meaningful while you rotate.")
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
        built = self._builder(self.channel, max_dim)
        if built is None:
            if self.vol is None:
                QTimer.singleShot(0, self.reject)   # cancelled before first build
            return False
        self.vol, self.voxel_um = built
        self.range = volume_display_range(self.vol)

        nz = self.vol.shape[0]
        for slider, value in ((self.slab_centre, nz // 2), (self.slab_depth, nz)):
            slider.blockSignals(True)
            slider.setRange(0, nz)
            slider.setValue(value)
            slider.blockSignals(False)
        self._render(draft=False)
        return True

    def _slab(self):
        """Current slab as ``(z0, z1)`` volume planes."""
        nz = self.vol.shape[0]
        depth = max(1, self.slab_depth.value())
        if depth >= nz:
            return 0, nz
        centre = self.slab_centre.value()
        z0 = int(np.clip(centre - depth // 2, 0, nz - 1))
        return z0, int(min(nz, z0 + depth))

    # -- rendering -----------------------------------------------------
    def _queue(self, *_):
        self._render(draft=True)
        self._settle.start()

    def _render(self, draft=False):
        if self.vol is None:
            return
        started = time.perf_counter()
        lo, hi = self.range
        # Brightness pulls the white point in (>100%) or pushes it out (<100%).
        gain = self.brightness.value() / 100.0
        rng = (lo, lo + max(hi - lo, EPS) / gain)
        z0, z1 = self._slab()
        rotation = view_rotation(self.azimuth, self.elevation)
        view = render_volume(
            self.vol, rotation, z_range=(z0, z1),
            mode="solid" if self.mode.currentIndex() == 1 else "mip",
            stride=2 if draft else 1, display_range=rng,
            threshold=self.threshold.value() / 100.0,
            opacity=self.opacity.value() / 100.0,
            colour_by_depth=self.colour.isChecked(), zoom=self.zoom)
        if self.show_box.isChecked():
            draw_volume_box(view, rotation, self.vol.shape, self.zoom,
                            z_range=(z0, z1))
        if self.show_bar.isChecked():
            view = draw_scale_bar(view, self.voxel_um / max(self.zoom, EPS),
                                  self.channel.params["bar_um"])
        self._show(view)
        if not draft:
            self._update_status(time.perf_counter() - started)

    def _show(self, view):
        self._buf = np.ascontiguousarray(view)
        h, w, _ = self._buf.shape
        qimg = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.canvas.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.canvas.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _update_status(self, seconds):
        nz, ny, nx = self.vol.shape
        z0, z1 = self._slab()
        z_step = self.channel.params["z_step"]
        self.slab_label.setText(
            f"planes {z0}-{z1 - 1} of {nz}   "
            f"({z0 * self.voxel_um:.1f}–{z1 * self.voxel_um:.1f} µm deep, "
            f"{(z1 - z0) * self.voxel_um:.1f} µm thick)")
        note = ""
        if self.voxel_um > 1.5 * z_step:
            note = ("   ⚠ voxel is coarser than the z step — crop the field or "
                    "raise Detail for finer structure")
        self.status.setText(
            f"{nx}×{ny}×{nz} voxels at {self.voxel_um:.3f} µm   ·   "
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

    def _on_zoom(self, steps):
        self.zoom = float(np.clip(self.zoom * (1.15 ** steps), 0.4, 3.0))
        self._queue()

    def _on_slab_wheel(self, steps):
        self.slab_centre.setValue(self.slab_centre.value() + steps)

    def _on_slab(self, _value):
        self._queue()

    def _slab_all(self):
        self.slab_depth.setValue(self.slab_depth.maximum())

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
            delta = 1 if key == Qt.Key.Key_PageUp else -1
            self.slab_centre.setValue(self.slab_centre.value() + delta)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._buf is not None:
            self._show(self._buf)

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

        # Field matching: the B -> A affine, plus the manual nudge on top of it.
        self.base_affine = IDENTITY_AFFINE.copy()
        self.match_active = False
        self.match_score = None
        # Axial: the two camera ports focus at slightly different stage positions.
        self.z_offset_score = None
        self._axial_cache = None

        self._build_ui()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self.refresh)
        self._apply_params_to_widgets(self.ch().params)
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
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(6)
        for build in (self._build_channel_group, self._build_view_group,
                      self._build_match_group, self._build_axial_group,
                      self._build_reference_group,
                      self._build_hot_group, self._build_noise_group,
                      self._build_norm_group, self._build_background_group,
                      self._build_denoise_group, self._build_decon_group,
                      self._build_sharpen_group, self._build_contrast_group,
                      self._build_lut_group, self._build_stack_group,
                      self._build_export_group):
            col.addWidget(build())
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(370)
        scroll.setMaximumWidth(450)

        self.image_label = QLabel("Load a file into channel A (and optionally B) to begin.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background:#101010; color:#888;")
        self.image_label.setMinimumSize(560, 420)

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
        root.addWidget(scroll)
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

        self.editing = QComboBox()
        self.editing.addItems(["A", "B"])
        self.editing.currentTextChanged.connect(self._on_editing_changed)
        rows.append(("Editing", self.editing))
        return _group("Channels  (.tif / .mat — prefer .mat for full precision)", rows)

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
        b_measure = QPushButton("Measure field match (A ↔ B)")
        b_measure.setToolTip("Masked-NCC registration of the two loaded images — "
                             "the same routine the acquisition GUI uses.\n"
                             "Load the ORCA into A and the EVK4 into B: the "
                             "search only shrinks B onto A (scale 0.55–0.96).")
        b_measure.clicked.connect(self.measure_match)
        if not REGISTRATION_AVAILABLE:
            b_measure.setEnabled(False)
            b_measure.setToolTip("core.register_evk4_to_orca could not be imported.")
        b_seed = QPushButton("Seed from stored calibration")
        b_seed.setToolTip("Load config.EVK4_TO_ORCA_AFFINE as the starting map "
                          "(valid for full-sensor, uncropped images).")
        b_seed.clicked.connect(self.seed_match)
        b_reset = QPushButton("Reset match (identity)")
        b_reset.clicked.connect(self.reset_match)

        self.match_on = QCheckBox("Warp B into A's frame")
        self.match_on.toggled.connect(self._on_match_toggled)
        self.nudge_dx = _spin(-4000, 4000, 0.0, 1.0, 1, " px")
        self.nudge_dy = _spin(-4000, 4000, 0.0, 1.0, 1, " px")
        self.nudge_rot = _spin(-180.0, 180.0, 0.0, 0.5, 2, " °")
        self.nudge_scale = _spin(0.05, 20.0, 1.0, 0.01, 3, " ×")
        for w in (self.nudge_dx, self.nudge_dy, self.nudge_rot, self.nudge_scale):
            w.valueChanged.connect(self._schedule)

        b_crop = QPushButton("Crop A to B's footprint")
        b_crop.setToolTip("Crop channel A to the bounding box of B's mapped field, "
                          "so both channels show the same region.\n"
                          "Uses whatever match is currently loaded — a restored "
                          "one needs no re-measurement.")
        b_crop.clicked.connect(self.crop_a_to_b)
        b_uncrop = QPushButton("Undo crop")
        b_uncrop.clicked.connect(self.undo_crop)

        b_save = QPushButton("Save match to file…")
        b_save.setToolTip("Write the match next to the data, so it travels with "
                          "the dataset.")
        b_save.clicked.connect(self.save_match_as)
        b_load = QPushButton("Load match from file…")
        b_load.clicked.connect(self.load_match_from)

        self.match_label = QLabel("no match measured")
        self.match_label.setStyleSheet("color:#888;")
        self.match_label.setWordWrap(True)
        return _group("Field matching  (B → A)", [
            (None, b_measure), (None, b_seed), (None, b_reset),
            (None, self.match_on),
            ("Nudge dx", self.nudge_dx), ("Nudge dy", self.nudge_dy),
            ("Rotate", self.nudge_rot), ("Scale", self.nudge_scale),
            (None, b_crop), (None, b_uncrop),
            (None, b_save), (None, b_load), (None, self.match_label),
        ])

    def _build_axial_group(self):
        self.axial_method = QComboBox()
        self.axial_method.addItems([
            "Image cross-correlation (best for 3D samples)",
            "Axial profile (mean intensity)",
            "Focus metric (Laplacian energy)",
        ])
        self.axial_method.setToolTip(
            "Image cross-correlation matches bead patterns plane against plane, "
            "so it works on a 3D sample whose beads span the whole thickness.\n"
            "The profile methods align the shape of the axial curve, which needs "
            "a single clear focal peak — i.e. a thin / 2-D bead layer.")
        b_measure = QPushButton("Measure Z offset (A ↔ B)")
        b_measure.setToolTip("The two camera ports sit at slightly different "
                             "optical path lengths, so plane k of one stack is "
                             "not plane k of the other.")
        b_measure.clicked.connect(self.measure_axial_offset)

        self.z_offset = _spin(-1000.0, 1000.0, 0.0, 0.1, 3, " um")
        self.z_offset.setToolTip("z_B − z_A for the same physical plane. "
                                 "Editable, so you can set a known offset by hand.")
        self.z_offset.valueChanged.connect(self._on_z_offset_changed)

        self.z_link = QCheckBox("Link sliders through the Z offset")
        self.z_link.setToolTip("With 'Link slice sliders' on, B follows A to the "
                               "matching physical plane instead of the same index.")
        self.z_link.setChecked(True)
        self.z_link.toggled.connect(self._schedule)

        b_csv = QPushButton("Read true Z from axial CSVs")
        b_csv.setToolTip("Reads <name>_axial_profile_*.csv from each channel's "
                         "folder to get the real stage positions, so the offset "
                         "is the optical one rather than a scan-start difference.")
        b_csv.clicked.connect(self.read_axial_csvs)

        b_resample = QPushButton("Resample B onto A's Z grid")
        b_resample.setToolTip("Interpolate B along z so plane k of both channels "
                              "is the same physical plane.")
        b_resample.clicked.connect(self.resample_b_to_a)
        b_undo = QPushButton("Undo resample")
        b_undo.clicked.connect(self.undo_resample)

        b_plot = QPushButton("Save axial comparison (CSV + PNG)")
        b_plot.clicked.connect(self.save_axial_comparison_plot)

        self.axial_label = QLabel("no Z offset measured")
        self.axial_label.setStyleSheet("color:#888;")
        self.axial_label.setWordWrap(True)
        return _group("Axial (Z) plane matching", [
            ("Method", self.axial_method), (None, b_measure),
            ("Z offset", self.z_offset), (None, self.z_link),
            (None, b_csv), (None, b_resample), (None, b_undo),
            (None, b_plot), (None, self.axial_label),
        ])

    def _build_reference_group(self):
        b_avg = QPushButton("Load average image…")
        b_avg.setToolTip("The companion *_average.mat/.tif — enables std/mean "
                         "normalization and the analytic noise model.")
        b_avg.clicked.connect(lambda: self._load_reference("avg"))
        b_noise = QPushButton("Load noise reference…")
        b_noise.setToolTip("A frozen-speckle std image (AWG off, same exposure): "
                           "its variance IS the per-pixel noise floor.")
        b_noise.clicked.connect(lambda: self._load_reference("noise"))
        b_dark = QPushButton("Load dark frame → hot-pixel mask…")
        b_dark.setToolTip("Illumination blocked. Hot pixels are a fixed sensor "
                          "property, so a measured mask beats per-image detection.")
        b_dark.clicked.connect(lambda: self._load_reference("hot_mask"))
        b_clear = QPushButton("Clear references")
        b_clear.clicked.connect(self._clear_references)
        self.ref_label = QLabel("none loaded")
        self.ref_label.setWordWrap(True)
        self.ref_label.setStyleSheet("color:#888;")
        return _group("Calibration references (active channel)", [
            (None, b_avg), (None, b_noise), (None, b_dark),
            (None, b_clear), (None, self.ref_label),
        ])

    def _build_hot_group(self):
        self.hot_on = QCheckBox("Remove hot / crazy pixels")
        self.hot_k = _spin(1.0, 30.0, 6.0, 0.5, 1, " ·MAD")
        self.hot_win = _int_spin(3, 15, 5, 2, " px")
        for w in (self.hot_on, self.hot_k, self.hot_win):
            self._connect(w)
        return _group("1 · Sensor artifacts", [
            (None, self.hot_on), ("Threshold", self.hot_k), ("Window", self.hot_win),
        ])

    def _build_noise_group(self):
        self.noise_mode = QComboBox()
        self.noise_mode.addItems(["Off", "Measured reference", "Analytic (needs average)"])
        self.noise_strength = _spin(0.0, 2.0, 1.0, 0.05, 2)
        self.gain = _spin(0.001, 10.0, 0.23, 0.01, 3, " ADU/e-")
        self.offset = _spin(0.0, 5000.0, 100.0, 10.0, 1, " ADU")
        self.read_noise = _spin(0.0, 200.0, 4.0, 0.5, 1, " ADU")
        for w in (self.noise_mode, self.noise_strength, self.gain, self.offset,
                  self.read_noise):
            self._connect(w)
        return _group("2 · Noise floor (variance domain)", [
            ("Mode", self.noise_mode), ("Strength", self.noise_strength),
            ("Gain", self.gain), ("Offset", self.offset), ("Read noise", self.read_noise),
        ])

    def _build_norm_group(self):
        self.norm_mode = QComboBox()
        self.norm_mode.addItems([
            "Off",
            "std / mean  (speckle contrast)",
            "std / sqrt(mean)  (shot-weighted)",
            "Self flat-field (blur divide)",
        ])
        self.flat_sigma = _spin(5.0, 400.0, 60.0, 5.0, 0, " px")
        for w in (self.norm_mode, self.flat_sigma):
            self._connect(w)
        return _group("3 · Normalization / flat field", [
            ("Mode", self.norm_mode), ("Self-FF sigma", self.flat_sigma),
        ])

    def _build_background_group(self):
        self.bg_radius = _int_spin(0, 400, 0, 5, " px")
        self.bg_radius.setToolTip("0 = off. Rolling-ball radius; must exceed the "
                                  "largest structure you want to keep.")
        self._connect(self.bg_radius)
        return _group("4 · Background subtraction", [("Ball radius", self.bg_radius)])

    def _build_denoise_group(self):
        self.denoise_method = QComboBox()
        self.denoise_method.addItems([
            "Off", "Gaussian", "Median", "Bilateral (edge-preserving)",
            "Non-local means", "Anscombe + NLM (Poisson data)",
        ])
        self.denoise_strength = _spin(0.1, 10.0, 1.0, 0.1, 2)
        for w in (self.denoise_method, self.denoise_strength):
            self._connect(w)
        return _group("5 · Denoising", [
            ("Method", self.denoise_method), ("Strength", self.denoise_strength),
        ])

    def _build_decon_group(self):
        self.decon_on = QCheckBox("Richardson–Lucy deconvolution")
        self.decon_iters = _int_spin(1, 100, 10)
        self.decon_sigma = _spin(0.3, 20.0, 1.2, 0.1, 2, " px")
        self.na = _spin(0.05, 1.6, 0.65, 0.05, 2)
        self.wavelength = _spin(300.0, 900.0, 580.0, 10.0, 0, " nm")
        self.pixel_um = _spin(0.001, 20.0, 0.1625, 0.005, 4, " um/px")
        b_psf = QPushButton("Compute sigma from optics")
        b_psf.clicked.connect(self._compute_psf_sigma)
        for w in (self.decon_on, self.decon_iters, self.decon_sigma,
                  self.na, self.wavelength):
            self._connect(w)
        return _group("6 · Deconvolution (contrast enhancement, not restoration)", [
            (None, self.decon_on), ("Iterations", self.decon_iters),
            ("PSF sigma", self.decon_sigma), ("NA", self.na),
            ("Emission λ", self.wavelength), ("Pixel size", self.pixel_um),
            (None, b_psf),
        ])

    def _build_sharpen_group(self):
        self.unsharp_amount = _spin(0.0, 5.0, 0.0, 0.1, 2)
        self.unsharp_radius = _spin(0.5, 20.0, 2.0, 0.5, 1, " px")
        for w in (self.unsharp_amount, self.unsharp_radius):
            self._connect(w)
        return _group("7 · Unsharp mask", [
            ("Amount", self.unsharp_amount), ("Radius", self.unsharp_radius),
        ])

    def _build_contrast_group(self):
        self.contrast_mode = QComboBox()
        self.contrast_mode.addItems(["ImageJ Auto B&C", "Percentile clip", "Manual"])
        self.stack_cb = QCheckBox("Use whole-stack histogram")
        self.stack_cb.setToolTip("One range for every slice, so brightness stays "
                                 "comparable across the volume.")
        self.pct_low = _spin(0.0, 20.0, 0.5, 0.1, 2, " %")
        self.pct_high = _spin(80.0, 100.0, 99.5, 0.1, 2, " %")
        self.manual_min = _spin(-1e6, 1e6, 0.0, 1.0, 3)
        self.manual_max = _spin(-1e6, 1e6, 255.0, 1.0, 3)
        b_fit = QPushButton("Set manual = current auto range")
        b_fit.clicked.connect(self._fill_manual_from_auto)
        self.gamma = _spin(0.1, 3.0, 1.0, 0.05, 2)
        self.gamma.setToolTip("< 1 lifts faint structure without clipping the bright end.")
        self.clahe_on = QCheckBox("CLAHE (display only — non-quantitative)")
        self.clahe_clip = _spin(0.5, 20.0, 2.0, 0.5, 1)
        self.clahe_tiles = _int_spin(2, 32, 8)
        for w in (self.contrast_mode, self.pct_low, self.pct_high, self.manual_min,
                  self.manual_max, self.gamma, self.clahe_on, self.clahe_clip,
                  self.clahe_tiles):
            self._connect(w)
        self.stack_cb.toggled.connect(self._on_stack_toggled)
        return _group("8-10 · Contrast, gamma, CLAHE (display only)", [
            ("Mode", self.contrast_mode), (None, self.stack_cb),
            ("Low pct", self.pct_low), ("High pct", self.pct_high),
            ("Manual min", self.manual_min), ("Manual max", self.manual_max),
            (None, b_fit), ("Gamma", self.gamma),
            (None, self.clahe_on), ("CLAHE clip", self.clahe_clip),
            ("CLAHE tiles", self.clahe_tiles),
        ])

    def _build_lut_group(self):
        self.lut = QComboBox()
        self.lut.addItems(LUT_NAMES)
        self.scalebar_on = QCheckBox("Scale bar")
        self.bar_um = _spin(0.1, 1000.0, 10.0, 1.0, 1, " um")
        for w in (self.lut, self.scalebar_on, self.bar_um, self.pixel_um):
            self._connect(w)
        return _group("11-12 · LUT & scale bar", [
            ("LUT", self.lut), (None, self.scalebar_on), ("Bar length", self.bar_um),
        ])

    def _build_stack_group(self):
        b_reg = QPushButton("Correct lateral drift (register stack)")
        b_reg.setToolTip("Phase-correlation alignment between planes — stops "
                         "orthogonal views smearing.")
        b_reg.clicked.connect(self.register)
        b_unreg = QPushButton("Undo registration")
        b_unreg.clicked.connect(self._undo_registration)

        self.proj_mode = QComboBox()
        self.proj_mode.addItems([
            "Max intensity (MIP)", "Mean", "Std dev",
            "Extended depth of field", "Depth colour-coded",
        ])
        b_proj = QPushButton("Compute projection")
        b_proj.clicked.connect(self.compute_projection)
        b_back = QPushButton("Back to slices")
        b_back.clicked.connect(self._clear_projection)

        self.z_step = _spin(0.0, 100.0, 0.5, 0.1, 3, " um")
        self._connect(self.z_step)
        b_ortho = QPushButton("Save orthogonal views (XZ / YZ)")
        b_ortho.clicked.connect(self.save_orthogonal)
        b_3d = QPushButton("Open 3-D volume view…")
        b_3d.setToolTip("Rotate the stack in 3-D, scroll a slab through it, and "
                        "colour beads by depth. Uses the Z step and pixel size "
                        "to build cubic voxels, so the shape is geometrically "
                        "true rather than stretched.")
        b_3d.clicked.connect(self.open_volume_view)
        return _group("Stack tools (active channel)", [
            (None, b_reg), (None, b_unreg),
            ("Projection", self.proj_mode), (None, b_proj), (None, b_back),
            ("Z step", self.z_step), (None, b_ortho), (None, b_3d),
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
        self._schedule()

    # ------------------------------------------------------------------
    # parameter <-> widget marshalling (per channel)
    # ------------------------------------------------------------------
    # Combo order, as stable keys — reordering the combo must not silently change
    # what a saved record means.
    _AXIAL_METHODS = ["images", "profile", "focus"]

    _NOISE_MODES = ["off", "reference", "analytic"]
    _NORM_MODES = ["off", "ratio", "sqrt", "self"]
    _DENOISE_METHODS = ["none", "gaussian", "median", "bilateral", "nlm", "anscombe"]
    _CONTRAST_MODES = ["auto", "percentile", "manual"]

    def _collect_params(self):
        return {
            "hot_on": self.hot_on.isChecked(),
            "hot_k": self.hot_k.value(),
            "hot_win": self.hot_win.value(),
            "noise_mode": self._NOISE_MODES[self.noise_mode.currentIndex()],
            "noise_strength": self.noise_strength.value(),
            "gain": self.gain.value(),
            "offset": self.offset.value(),
            "read_noise": self.read_noise.value(),
            "norm_mode": self._NORM_MODES[self.norm_mode.currentIndex()],
            "flat_sigma": self.flat_sigma.value(),
            "bg_radius": self.bg_radius.value(),
            "denoise_method": self._DENOISE_METHODS[self.denoise_method.currentIndex()],
            "denoise_strength": self.denoise_strength.value(),
            "decon_on": self.decon_on.isChecked(),
            "decon_iters": self.decon_iters.value(),
            "decon_sigma": self.decon_sigma.value(),
            "unsharp_amount": self.unsharp_amount.value(),
            "unsharp_radius": self.unsharp_radius.value(),
            "contrast_mode": self._CONTRAST_MODES[self.contrast_mode.currentIndex()],
            "pct_low": self.pct_low.value(),
            "pct_high": self.pct_high.value(),
            "manual_min": self.manual_min.value(),
            "manual_max": self.manual_max.value(),
            "gamma": self.gamma.value(),
            "clahe_on": self.clahe_on.isChecked(),
            "clahe_clip": self.clahe_clip.value(),
            "clahe_tiles": self.clahe_tiles.value(),
            "lut": self.lut.currentText(),
            "scalebar_on": self.scalebar_on.isChecked(),
            "um_per_px": self.pixel_um.value(),
            "bar_um": self.bar_um.value(),
            "na": self.na.value(),
            "wavelength": self.wavelength.value(),
            "z_step": self.z_step.value(),
        }

    def _apply_params_to_widgets(self, p):
        """Repopulate the panel from a channel's stored settings."""
        self._loading_params = True
        try:
            self.hot_on.setChecked(p["hot_on"])
            self.hot_k.setValue(p["hot_k"])
            self.hot_win.setValue(p["hot_win"])
            self.noise_mode.setCurrentIndex(self._NOISE_MODES.index(p["noise_mode"]))
            self.noise_strength.setValue(p["noise_strength"])
            self.gain.setValue(p["gain"])
            self.offset.setValue(p["offset"])
            self.read_noise.setValue(p["read_noise"])
            self.norm_mode.setCurrentIndex(self._NORM_MODES.index(p["norm_mode"]))
            self.flat_sigma.setValue(p["flat_sigma"])
            self.bg_radius.setValue(p["bg_radius"])
            self.denoise_method.setCurrentIndex(
                self._DENOISE_METHODS.index(p["denoise_method"]))
            self.denoise_strength.setValue(p["denoise_strength"])
            self.decon_on.setChecked(p["decon_on"])
            self.decon_iters.setValue(p["decon_iters"])
            self.decon_sigma.setValue(p["decon_sigma"])
            self.unsharp_amount.setValue(p["unsharp_amount"])
            self.unsharp_radius.setValue(p["unsharp_radius"])
            self.contrast_mode.setCurrentIndex(
                self._CONTRAST_MODES.index(p["contrast_mode"]))
            self.pct_low.setValue(p["pct_low"])
            self.pct_high.setValue(p["pct_high"])
            self.manual_min.setValue(p["manual_min"])
            self.manual_max.setValue(p["manual_max"])
            self.gamma.setValue(p["gamma"])
            self.clahe_on.setChecked(p["clahe_on"])
            self.clahe_clip.setValue(p["clahe_clip"])
            self.clahe_tiles.setValue(p["clahe_tiles"])
            self.lut.setCurrentText(p["lut"])
            self.scalebar_on.setChecked(p["scalebar_on"])
            self.pixel_um.setValue(p["um_per_px"])
            self.bar_um.setValue(p["bar_um"])
            self.na.setValue(p["na"])
            self.wavelength.setValue(p["wavelength"])
            self.z_step.setValue(p["z_step"])
            self.stack_cb.setChecked(self.ch().use_stack_range)
        finally:
            self._loading_params = False

    def _on_editing_changed(self, name):
        self.active = name
        c = self.ch()
        self._apply_params_to_widgets(c.params)
        self.ref_label.setText(
            "\n".join(f"{k}: {v}" for k, v in c.ref_names.items()) or "none loaded")
        self._sync_slider()
        self._schedule()

    def _compute_psf_sigma(self):
        na, lam = self.na.value(), self.wavelength.value()
        sigma = psf_sigma_px(lam, na, self.pixel_um.value())
        self.decon_sigma.setValue(round(sigma, 3))
        self._status(f"Channel {self.active}: PSF sigma = {sigma:.3f} px "
                     f"(lateral FWHM = {0.51 * lam * 1e-3 / na:.3f} µm)")

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
        c.reset_derived()
        c.refs = {"avg": None, "noise": None, "hot_mask": None}
        c.ref_names = {}
        self.chan_labels[name].setText("empty")
        if name == self.active:
            self.ref_label.setText("none loaded")
        self._sync_slider()
        self.refresh()

    def _load_reference(self, kind):
        c = self.ch()
        if not c.loaded:
            self._status(f"Load data into channel {self.active} first.")
            return
        titles = {"avg": "Select the average image",
                  "noise": "Select the frozen-speckle (noise) std image",
                  "hot_mask": "Select a dark frame"}
        start = str(c.source_dir) if c.source_dir else ""
        file, _ = QFileDialog.getOpenFileName(self, titles[kind], start,
                                              "Images (*.tif *.tiff *.mat)")
        if not file:
            return
        try:
            arr = load_image_file(file)
        except Exception as exc:  # noqa: BLE001 — surface load failures to the user
            QMessageBox.warning(self, "Could not load", str(exc))
            return

        img = arr[0] if arr.shape[0] == 1 else arr  # keep a per-plane stack if it is one
        if kind == "hot_mask":
            flat = img if img.ndim == 2 else img.mean(axis=0)
            ref = hot_pixel_mask_from_dark(flat)
            self._status(f"Channel {self.active}: hot-pixel mask, "
                         f"{int(ref.sum())} pixels flagged.")
        else:
            ref = img
        c.refs[kind] = ref
        c.ref_names[kind] = Path(file).name
        self.ref_label.setText("\n".join(f"{k}: {v}" for k, v in c.ref_names.items()))
        c.cache_key = None
        self.refresh()

    def _clear_references(self):
        c = self.ch()
        c.refs = {"avg": None, "noise": None, "hot_mask": None}
        c.ref_names = {}
        self.ref_label.setText("none loaded")
        c.cache_key = None
        self.refresh()

    # ------------------------------------------------------------------
    # field matching
    # ------------------------------------------------------------------
    def current_affine(self):
        """The full B->A map: measured base plus the manual nudge."""
        b = self.ch("B")
        img = b.display_image()
        centre = (img.shape[1] / 2.0, img.shape[0] / 2.0) if img is not None else (0.0, 0.0)
        return compose_match(self.base_affine, self.nudge_dx.value(),
                             self.nudge_dy.value(), self.nudge_rot.value(),
                             self.nudge_scale.value(), centre)

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
        pair = self._both_ready()
        if pair is None:
            return
        img_a, img_b = pair
        seed = self.base_affine if self.match_active else None
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            affine, score, info = register_evk4_to_orca(
                np.asarray(img_a, dtype=np.float32),
                np.asarray(img_b, dtype=np.float32),
                seed_affine=seed, status=self._progress_status)
        except Exception as exc:  # noqa: BLE001 — a failed match must not kill the app
            traceback.print_exc()
            self._status(f"Field match failed: {exc}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self.base_affine = np.asarray(affine, dtype=np.float64)
        self.match_active = True
        self.match_score = score
        self._reset_nudge()
        self.match_on.setChecked(True)
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

    def seed_match(self):
        self.base_affine = np.asarray(EVK4_TO_ORCA_AFFINE, dtype=np.float64)
        self.match_active = True
        self.match_score = None
        self._reset_nudge()
        self.match_on.setChecked(True)
        self.match_label.setText("seeded from config.EVK4_TO_ORCA_AFFINE "
                                 "(assumes full-sensor, uncropped images)")
        self._save_match_state()
        self.refresh()

    def reset_match(self):
        self.base_affine = IDENTITY_AFFINE.copy()
        self.match_active = False
        self.match_score = None
        self._reset_nudge()
        self.match_on.setChecked(False)
        self.match_label.setText("no match measured")
        self._save_match_state()
        self.refresh()

    # ------------------------------------------------------------------
    # axial (Z) matching
    # ------------------------------------------------------------------
    def _on_z_offset_changed(self, _value):
        self.z_offset_score = None
        self._update_axial_label()
        self._schedule()

    def _update_axial_label(self, extra=""):
        dz = self.z_offset.value()
        b = self.ch("B")
        step = b.params["z_step"] if b.loaded else 0.0
        planes = f"{dz / step:+.2f} planes of B" if step else "—"
        score = (f", score {self.z_offset_score:.2f}"
                 if self.z_offset_score is not None else "")
        self.axial_label.setText(f"Z offset = {dz:+.3f} µm ({planes}){score}{extra}")

    def _axial_method(self):
        """Selected method as a stable key (combo order may change; keys don't)."""
        return self._AXIAL_METHODS[self.axial_method.currentIndex()]

    def _axial_profiles(self):
        """Processed axial profiles + z positions for both channels."""
        a, b = self.ch("A"), self.ch("B")
        metric = "focus" if self._axial_method() == "focus" else "mean"
        slices_a = self._process_all(a, "Profiling channel A…")
        if slices_a is None:
            return None
        slices_b = self._process_all(b, "Profiling channel B…")
        if slices_b is None:
            return None
        return (a.z_um(), axial_profile(slices_a, metric),
                b.z_um(), axial_profile(slices_b, metric), slices_a, slices_b)

    def measure_axial_offset(self):
        a, b = self.ch("A"), self.ch("B")
        if not (a.loaded and b.loaded):
            self._status("Axial matching needs a stack in both channels.")
            return
        if a.n_slices < 3 or b.n_slices < 3:
            self._status("Axial matching needs at least 3 planes per channel.")
            return

        bundle = self._axial_profiles()
        if bundle is None:
            return
        z_a, prof_a, z_b, prof_b, slices_a, slices_b = bundle

        if self._axial_method() == "images":
            affine = self.current_affine() if self.match_active else None
            shift, score = find_axial_offset_by_images(
                slices_a, slices_b, affine=affine, status=self._progress_status)
            # The image search returns a shift in B planes; convert to microns.
            step_b = float(np.median(np.diff(z_b))) if len(z_b) > 1 else 1.0
            dz = shift * step_b
        else:
            dz, score = find_axial_offset(z_a, prof_a, z_b, prof_b)

        self.z_offset.blockSignals(True)
        self.z_offset.setValue(round(dz, 3))
        self.z_offset.blockSignals(False)
        self.z_offset_score = score
        self._axial_cache = (z_a, prof_a, z_b, prof_b)

        # On a 3D sample the beads span the whole thickness, so the axial curve
        # has no single peak to align. The correlation score does NOT catch this
        # (two flat profiles correlate happily at zero shift and score ~0.7), so
        # judge the profile methods on peakedness instead.
        weak = ""
        if self._axial_method() != "images":
            spread = max(axial_profile_peakedness(prof_a),
                         axial_profile_peakedness(prof_b))
            if spread > 0.5:
                weak = (f"  ⚠ UNRELIABLE — the axial profile has no distinct peak "
                        f"({spread:.0%} of planes above half-max: a 3D sample). "
                        f"Use image cross-correlation; this number is probably ~0.")
            elif score < 0.5:
                weak = "  ⚠ weak — low profile correlation"
        elif score < 0.5:
            weak = "  ⚠ weak — check the lateral field match and the Z range"
        self._update_axial_label(weak)
        self._save_match_state()
        self.refresh()
        self._final_status(f"Z offset = {dz:+.3f} µm (score {score:.2f}).{weak}")

    def read_axial_csvs(self):
        """Pick up true stage positions from each channel's acquisition CSV."""
        found = []
        for name in ("A", "B"):
            c = self.ch(name)
            if not (c.loaded and c.source_dir):
                continue
            matches = sorted(Path(c.source_dir).glob("*axial_profile*.csv"))
            for path in matches:
                z, _ = read_axial_profile_csv(path)
                if z is not None and len(z) == c.n_slices:
                    c.z_positions = z
                    found.append(f"{name}: {path.name} ({len(z)} planes, "
                                 f"{z[0]:g}–{z[-1]:g} µm)")
                    break
            else:
                if matches:
                    found.append(f"{name}: {matches[0].name} has "
                                 f"{len(read_axial_profile_csv(matches[0])[0] or [])} "
                                 f"planes ≠ {c.n_slices} loaded — ignored")
                else:
                    found.append(f"{name}: no axial CSV found")
        self._final_status(" | ".join(found) if found else "No channels loaded.")

    def _matching_b_index(self, a_index):
        """B's plane index showing the same physical plane as A's ``a_index``."""
        a, b = self.ch("A"), self.ch("B")
        if not (a.loaded and b.loaded):
            return a_index
        if not self.z_link.isChecked():
            return min(a_index, b.n_slices - 1)
        z_a = a.z_um()
        z_b = b.z_um()
        target = z_a[min(a_index, len(z_a) - 1)] + self.z_offset.value()
        return int(np.argmin(np.abs(z_b - target)))

    def resample_b_to_a(self):
        """Interpolate B along z so plane k of both channels is the same plane."""
        a, b = self.ch("A"), self.ch("B")
        if not (a.loaded and b.loaded):
            self._status("Resampling needs a stack in both channels.")
            return
        if b.n_slices < 2:
            self._status("Channel B needs at least 2 planes to resample.")
            return
        slices_b = [np.asarray(b.raw(i), np.float32) for i in range(b.n_slices)]
        z_b = b.z_um()
        targets = a.z_um() + self.z_offset.value()
        self._resample_backup = (b.registered, b.z_positions,
                                 [f._data for f in b.frames])
        resampled = resample_stack_z(slices_b, z_b, targets)
        b.registered = resampled
        # B now lives on A's grid: same plane count, same nominal positions.
        b.z_positions = a.z_um().copy()
        b.index = min(b.index, len(resampled) - 1)
        b.reset_derived()
        self._sync_slider()
        self.refresh()
        self._final_status(
            f"Channel B resampled onto A's Z grid ({len(resampled)} planes, "
            f"offset {self.z_offset.value():+.3f} µm). Plane k is now the same "
            f"physical plane in both channels.")

    def undo_resample(self):
        backup = getattr(self, "_resample_backup", None)
        if backup is None:
            self._status("No resample to undo.")
            return
        b = self.ch("B")
        b.registered, b.z_positions, cached = backup
        for f, data in zip(b.frames, cached):
            f._data = data
        b.reset_derived()
        self._resample_backup = None
        self._sync_slider()
        self.refresh()
        self._final_status("Channel B resample undone.")

    def save_axial_comparison_plot(self):
        cache = getattr(self, "_axial_cache", None)
        if cache is None:
            bundle = self._axial_profiles()
            if bundle is None:
                return
            cache = bundle[:4]
            self._axial_cache = cache
        z_a, prof_a, z_b, prof_b = cache
        csv_path, png_path = save_axial_comparison(
            z_a, prof_a, z_b, prof_b, self.z_offset.value(),
            self._out_dir(), self._out_base(stack_wide=True))
        extra = f" (+ {png_path.name})" if png_path else " (no matplotlib — CSV only)"
        self._final_status(f"Saved {csv_path}{extra}")

    # -- persistence ---------------------------------------------------
    def _match_state(self):
        """The current match as a JSON-serialisable record."""
        return {
            "version": MATCH_STATE_VERSION,
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "affine": np.asarray(self.base_affine, dtype=float).tolist(),
            "active": bool(self.match_active),
            "score": None if self.match_score is None else float(self.match_score),
            "nudge": {
                "dx": self.nudge_dx.value(), "dy": self.nudge_dy.value(),
                "rot": self.nudge_rot.value(), "scale": self.nudge_scale.value(),
            },
            "note": self.match_label.text(),
            "z_offset_um": self.z_offset.value(),
            "z_offset_score": (None if self.z_offset_score is None
                               else float(self.z_offset_score)),
            "z_method": self._axial_method(),
            "z_link": self.z_link.isChecked(),
            # Provenance only — the match is not tied to these folders, but it is
            # the first thing to check when a restored match looks wrong.
            "source_a": str(self.ch("A").source_dir or ""),
            "source_b": str(self.ch("B").source_dir or ""),
        }

    def _apply_match_state(self, state, origin):
        self.base_affine = np.asarray(state["affine"], dtype=np.float64)
        self.match_score = state.get("score")
        nudge = state.get("nudge") or {}
        for w, key, default in ((self.nudge_dx, "dx", 0.0), (self.nudge_dy, "dy", 0.0),
                                (self.nudge_rot, "rot", 0.0),
                                (self.nudge_scale, "scale", 1.0)):
            w.blockSignals(True)
            w.setValue(float(nudge.get(key, default)))
            w.blockSignals(False)
        self.match_active = bool(state.get("active", True))
        self.match_on.blockSignals(True)
        self.match_on.setChecked(self.match_active)
        self.match_on.blockSignals(False)

        # Axial offset (absent from v1 records — default to "not measured").
        self.z_offset.blockSignals(True)
        self.z_offset.setValue(float(state.get("z_offset_um", 0.0) or 0.0))
        self.z_offset.blockSignals(False)
        self.z_offset_score = state.get("z_offset_score")
        # v2 stored the combo index under the old ordering; v2.1+ stores the key.
        method = state.get("z_method", "images")
        if isinstance(method, int):
            method = ["profile", "focus", "images"][method] if 0 <= method <= 2 else "images"
        idx = (self._AXIAL_METHODS.index(method)
               if method in self._AXIAL_METHODS else 0)
        self.axial_method.blockSignals(True)
        self.axial_method.setCurrentIndex(idx)
        self.axial_method.blockSignals(False)
        self.z_link.blockSignals(True)
        self.z_link.setChecked(bool(state.get("z_link", True)))
        self.z_link.blockSignals(False)
        self._axial_cache = None
        self._update_axial_label()

        when = state.get("saved_utc", "?")
        score = state.get("score")
        score_txt = f"NCC = {score:.2f}, " if isinstance(score, (int, float)) else ""
        self.match_label.setText(f"{score_txt}restored from {origin} (saved {when})")

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

    def save_match_as(self):
        start = self.ch().source_dir or Path.cwd()
        file, _ = QFileDialog.getSaveFileName(
            self, "Save field match", str(Path(start) / "field_match.json"),
            "JSON (*.json)")
        if not file:
            return
        try:
            path = write_match_state(file, self._match_state())
        except Exception as exc:  # noqa: BLE001 — report, don't crash
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self._final_status(f"Field match saved to {path}")

    def load_match_from(self):
        start = self.ch().source_dir or Path.cwd()
        file, _ = QFileDialog.getOpenFileName(self, "Load field match", str(start),
                                              "JSON (*.json)")
        if not file:
            return
        state = read_match_state(file)
        if state is None:
            QMessageBox.warning(self, "Could not load",
                                "That file is not a valid field-match record "
                                f"(version {MATCH_STATE_VERSION} expected).")
            return
        self._apply_match_state(state, Path(file).name)
        self._save_match_state()
        self.refresh()
        self._final_status(f"Field match loaded from {file} — no re-measurement needed.")

    def _reset_nudge(self):
        for w, v in ((self.nudge_dx, 0.0), (self.nudge_dy, 0.0),
                     (self.nudge_rot, 0.0), (self.nudge_scale, 1.0)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def _on_match_toggled(self, checked):
        self.match_active = checked
        self._schedule()

    def crop_a_to_b(self):
        """Crop channel A to the bounding box of B's mapped footprint."""
        pair = self._both_ready()
        if pair is None:
            return
        img_a, img_b = pair
        (x0, y0, x1, y1), _ = footprint_bbox(self.current_affine(),
                                             img_b.shape[:2], img_a.shape[:2])
        if x1 - x0 < 8 or y1 - y0 < 8:
            self._status("B's mapped footprint barely overlaps A — measure the "
                         "field match first.")
            return
        a = self.ch("A")
        self._crop_backup = (a.registered, [f._data for f in a.frames])
        cropped = [np.asarray(a.raw(i), np.float32)[y0:y1, x0:x1]
                   for i in range(a.n_slices)]
        a.registered = cropped  # reuse the "materialised slices" path
        a.reset_derived()
        self.refresh()
        self._final_status(f"Channel A cropped to B's footprint: "
                           f"x {x0}–{x1}, y {y0}–{y1} ({x1 - x0}×{y1 - y0} px).")

    def undo_crop(self):
        backup = getattr(self, "_crop_backup", None)
        if backup is None:
            self._status("No crop to undo.")
            return
        a = self.ch("A")
        a.registered, cached = backup
        for f, data in zip(a.frames, cached):
            f._data = data
        a.reset_derived()
        self._crop_backup = None
        self.refresh()
        self._final_status("Channel A crop undone.")

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

    def _describe(self, c, lo, hi):
        if c.projection is not None:
            return f"{c.proj_label}"
        label = c.frames[c.index].label if c.loaded else "empty"
        reg = " · drift-corrected" if c.registered is not None else ""
        return f"{label}{reg} | display {lo:.4g}–{hi:.4g}"

    def _ensure_stack_range(self, c):
        if not (c.use_stack_range and c.projection is None):
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
                if c.params["contrast_mode"] == "percentile":
                    lo, hi = percentile_minmax(img, c.params["pct_low"], c.params["pct_high"])
                else:
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
        c = self.ch()
        n = c.n_slices
        showing_projection = c.projection is not None
        self.slider.blockSignals(True)
        self.slider.setEnabled(n > 1 and not showing_projection)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(min(c.index, max(0, n - 1)))
        self.slider.blockSignals(False)
        self.pos_label.setText(f"{c.index + 1} / {n}" if n else "0 / 0")

    def _on_slider(self, value):
        self.ch().index = value
        if self.link_slices.isChecked():
            a, b = self.ch("A"), self.ch("B")
            if a.loaded and b.loaded:
                # Follow the *physical* plane through the axial offset, so the
                # two feeds stay on the same depth rather than the same index.
                if self.active == "A":
                    b.index = self._matching_b_index(value)
                else:
                    z_b = b.z_um()
                    target = z_b[min(value, len(z_b) - 1)] - self.z_offset.value()
                    a.index = int(np.argmin(np.abs(a.z_um() - target)))
        self._schedule()

    # ------------------------------------------------------------------
    # stack tools (active channel)
    # ------------------------------------------------------------------
    def register(self):
        c = self.ch()
        if c.n_slices < 2:
            self._status("Need at least 2 slices to register.")
            return
        prog = QProgressDialog(f"Loading stack (channel {c.name})…", "Cancel",
                               0, c.n_slices, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        slices = []
        try:
            for i, f in enumerate(c.frames):
                prog.setValue(i)
                QApplication.processEvents()
                if prog.wasCanceled():
                    return
                slices.append(np.asarray(f.data, dtype=np.float32))
        finally:
            prog.close()

        aligned, shifts = register_stack(slices, status=self._progress_status)
        c.registered = aligned
        c.reset_derived()
        max_shift = max(abs(dx) + abs(dy) for dx, dy in shifts)
        self.refresh()
        self._status(f"Channel {c.name}: drift corrected — "
                     f"max cumulative shift {max_shift:.1f} px.")

    def _undo_registration(self):
        c = self.ch()
        c.registered = None
        c.reset_derived()
        self.refresh()

    def compute_projection(self):
        c = self.ch()
        if not c.loaded:
            return
        modes = ["mip", "mean", "std", "edf", "depth"]
        mode = modes[self.proj_mode.currentIndex()]
        processed = self._process_all(c, f"Processing {c.n_slices} slices "
                                         f"(channel {c.name})…")
        if processed is None:
            return
        if mode == "depth":
            c.projection = depth_colour_code(processed)
            c.projection_is_bgr = True
        else:
            c.projection = project_stack(processed, mode)
            c.projection_is_bgr = False
        c.proj_label = f"{self.proj_mode.currentText()} (ch {c.name})"
        self._sync_slider()
        self.refresh()

    def _clear_projection(self):
        c = self.ch()
        c.projection = None
        c.projection_is_bgr = False
        self._sync_slider()
        self.refresh()

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

    def _build_volume(self, c, max_dim):
        """Build a render volume from ``c``'s processed planes (cancellable).

        Planes are processed and shrunk one at a time inside
        ``build_view_volume``, so this never holds the whole processed stack —
        which at full sensor would be gigabytes.
        """
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
                                     c.params["um_per_px"], max_dim=max_dim)
        except _Cancelled:
            return None
        finally:
            prog.close()
            if not cancelled:
                self._status(f"3-D volume built from {c.n_slices} planes "
                             f"(channel {c.name}).")

    def _on_stack_toggled(self, checked):
        if self._loading_params:
            return
        c = self.ch()
        c.use_stack_range = checked
        c.stack_range = None
        self._schedule()

    def _fill_manual_from_auto(self):
        c = self.ch()
        img = c.display_image()
        if img is None:
            return
        lo, hi = imagej_auto_minmax(img)
        self.manual_min.setValue(lo)
        self.manual_max.setValue(hi)
        self.contrast_mode.setCurrentIndex(2)

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

    def _proj_slug(self):
        return "".join(ch if ch.isalnum() else "_"
                       for ch in self.ch().proj_label.lower()).strip("_")

    def save_view_png(self):
        if self._qbuf is None:
            return
        mode = self.view_mode.currentIndex()
        c = self.ch()
        if mode == 2:
            tag = "side_by_side"
        elif mode == 3:
            tag = "overlay"
        elif c.projection is not None:
            tag = self._proj_slug()
        else:
            tag = f"z{c.index:03d}"
        base = self._out_base(stack_wide=(mode >= 2 or c.projection is not None))
        path = self._out_dir() / f"{base}_{tag}_view.png"
        cv2.imwrite(str(path), self._qbuf)
        msg = f"Saved {path}"
        if c.projection is not None and c.projection_is_bgr and mode < 2:
            legend_path = self._out_dir() / f"{base}_depth_legend.png"
            cv2.imwrite(str(legend_path), depth_colour_legend(c.n_slices))
            msg += f"  (+ {legend_path.name})"
        self._final_status(msg)

    def save_slice_tiff(self):
        c = self.ch()
        if c.projection is not None and c.projection_is_bgr:
            self._status("The depth-coded view is RGB — save it as a PNG instead.")
            return
        img = c.display_image()
        if img is None:
            return
        if c.projection is not None:
            base, suffix = self._out_base(stack_wide=True), self._proj_slug()
        else:
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

    def save_orthogonal(self):
        c = self.ch()
        if c.n_slices < 2:
            self._status("Orthogonal views need a stack.")
            return
        processed = self._process_all(c, f"Processing channel {c.name} "
                                         f"for orthogonal views…")
        if processed is None:
            return
        xz, yz = orthogonal_views(processed, c.params["z_step"], c.params["um_per_px"])
        for name, img in (("xz", xz), ("yz", yz)):
            view, _ = render_display(img, c.params)
            path = self._out_dir() / f"{self._out_base(stack_wide=True)}_{name}.png"
            cv2.imwrite(str(path), view)
        self._final_status(f"Saved XZ / YZ views to {self._out_dir()}")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        # Catches nudges, which are not worth a write on every spinbox tick.
        self._save_match_state()
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
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
