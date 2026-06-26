"""Custom composite UI widgets: AWG control panel and PI stage control panel.

These widgets own their visuals/inputs and delegate all instrument I/O to the
hardware-layer controllers (AWGController, StageController) and workers (PIMoveWorker).
"""

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)
from PyQt6.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen

from config import (
    DCAM_BINNING_OPTIONS, DCAM_DEFECTCORRECT_OPTIONS, DCAM_READOUTSPEED_OPTIONS,
    DCAM_TRIGGER_MODE_OPTIONS, DCAM_TRIGGERSOURCE_OPTIONS,
    EVK4_SENSOR_WIDTH, EVK4_SENSOR_HEIGHT,
    ORCA_DSI_PROCESS_PIXELS_PER_S, ORCA_ROW_READOUT_US,
    ORCA_SENSOR_WIDTH, ORCA_SENSOR_HEIGHT, ZSTACK_DISK_BYTES_PER_S,
)
from hardware.awg_control import AWGController
from hardware.stage_control import PI_AVAILABLE, PIMoveWorker, StageController


def make_dcam_combo(options, default_label):
    """Build a QComboBox from an {label: dcam value} map, storing the value as
    item data and selecting ``default_label``."""
    combo = QComboBox()
    for label, value in options.items():
        combo.addItem(label, value)
    idx = combo.findText(default_label)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    return combo


def make_offset_slider(limit, step=4):
    """Build a draggable centre-offset slider (range ±limit px) paired with a
    live value label. Returns (slider, row_layout) for a form row.

    Shared by the ORCA and EVK4 ROI controls so the crop offset behaves
    identically for both cameras — drag, click-to-jump, or scroll-wheel, with a
    live "N px" readout instead of typing.
    """
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(-limit, limit)
    slider.setSingleStep(step)    # arrow keys / scroll wheel step
    slider.setPageStep(step * 10)  # click-in-trough jump
    slider.setMinimumWidth(140)   # wide enough to grab and drag comfortably
    slider.setValue(0)
    value_lbl = QLabel("0 px")
    value_lbl.setMinimumWidth(56)
    value_lbl.setStyleSheet("color: #cccccc;")
    slider.valueChanged.connect(lambda v: value_lbl.setText(f"{v} px"))
    row = QHBoxLayout()
    row.addWidget(slider, stretch=1)
    row.addWidget(value_lbl)
    return slider, row


class Evk4ParamsWidget(QWidget):
    """Reusable Prophesee EVK4 parameter controls: biases, per-acquisition
    duration, ROI crop, and post-processing options. Used in both the Event
    Camera tab and the Z-Stack tab so each has independent, fully selectable
    parameters."""

    # Emitted (debounced) whenever the ROI geometry — size or centre offset —
    # changes, so a running live feed can be re-cropped without clicking Apply.
    # Mirrors OrcaParamsWidget so the interactive crop tool drives both cameras.
    roi_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bias_group = QGroupBox("EVK4 Biases")
        bias_form = QFormLayout()
        # Ranges per the IMX636 datasheet (cf. EVK4 reference, Table II).
        self.spin_fo = QSpinBox(); self.spin_fo.setRange(-35, 55); self.spin_fo.setValue(5)
        self.spin_hpf = QSpinBox(); self.spin_hpf.setRange(0, 120); self.spin_hpf.setValue(30)
        self.spin_on = QSpinBox(); self.spin_on.setRange(-85, 140); self.spin_on.setValue(5)
        self.spin_off = QSpinBox(); self.spin_off.setRange(-35, 190); self.spin_off.setValue(5)
        bias_form.addRow("bias_fo (low-pass):", self.spin_fo)
        bias_form.addRow("bias_hpf (high-pass):", self.spin_hpf)
        bias_form.addRow("bias_on (positive):", self.spin_on)
        bias_form.addRow("bias_off (negative):", self.spin_off)
        self.btn_apply_biases = QPushButton("Apply Biases to Live Feed")
        self.btn_apply_biases.setEnabled(False)
        bias_form.addRow("", self.btn_apply_biases)
        bias_group.setLayout(bias_form)
        layout.addWidget(bias_group)

        dur_group = QGroupBox("EVK4 Acquisition Duration")
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration (s):"))
        self.spin_time = QDoubleSpinBox(); self.spin_time.setRange(0.1, 3600); self.spin_time.setDecimals(2); self.spin_time.setSingleStep(0.5); self.spin_time.setValue(5)
        dur_layout.addWidget(self.spin_time)
        dur_group.setLayout(dur_layout)
        layout.addWidget(dur_group)

        post_group = QGroupBox("EVK4 Post-Processing")
        post_layout = QVBoxLayout()
        self.chk_crazy = QCheckBox("Remove 'Crazy' Pixels (Top 0.1%)"); self.chk_crazy.setChecked(True)
        self.chk_smooth = QCheckBox("Apply Spatial Smoothing (Gaussian)"); self.chk_smooth.setChecked(True)
        post_layout.addWidget(self.chk_crazy)
        post_layout.addWidget(self.chk_smooth)
        post_group.setLayout(post_layout)
        layout.addWidget(post_group)

        # ROI crop — the same width/height + centre-offset model as the ORCA, so
        # the interactive crop tool and live re-cropping work identically for the
        # event camera. The crop is expressed in IMX636 full-sensor pixels; the
        # event stream is always full-sensor, so the region is software-cropped
        # from the live frame and the accumulated image (and, best-effort, used to
        # drive a hardware ROI to cut the event rate).
        roi_group = QGroupBox("Region of Interest — Crop")
        roi_form = QFormLayout()
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(1, EVK4_SENSOR_WIDTH)
        self.spin_roi_width.setValue(EVK4_SENSOR_WIDTH)
        self.spin_roi_width.setSuffix(" px")
        self.spin_roi_height = QSpinBox()
        self.spin_roi_height.setRange(1, EVK4_SENSOR_HEIGHT)
        self.spin_roi_height.setValue(EVK4_SENSOR_HEIGHT)
        self.spin_roi_height.setSuffix(" px")
        self.slider_offset_x, offset_x_row = make_offset_slider(EVK4_SENSOR_WIDTH // 2, step=1)
        self.slider_offset_y, offset_y_row = make_offset_slider(EVK4_SENSOR_HEIGHT // 2, step=1)
        roi_form.addRow("Width:", self.spin_roi_width)
        roi_form.addRow("Height:", self.spin_roi_height)
        roi_form.addRow("Centre offset X:", offset_x_row)
        roi_form.addRow("Centre offset Y:", offset_y_row)
        lbl_roi = QLabel(
            f"Full sensor: {EVK4_SENSOR_WIDTH} × {EVK4_SENSOR_HEIGHT} px. The crop "
            "is centred and shifted by the offsets, clamped to the sensor. Drag a "
            "box on the live feed (crop tool) to set it visually."
        )
        lbl_roi.setWordWrap(True)
        lbl_roi.setStyleSheet("color: #888888; font-size: 11px;")
        roi_form.addRow(lbl_roi)
        roi_group.setLayout(roi_form)
        layout.addWidget(roi_group)

        # Debounced live re-cropping: coalesce the burst of valueChanged events
        # from a drag into a single re-apply (the worker just updates its software
        # crop window — no device restart, unlike the ORCA hardware subarray).
        self._roi_debounce = QTimer(self)
        self._roi_debounce.setSingleShot(True)
        self._roi_debounce.setInterval(200)
        self._roi_debounce.timeout.connect(self.roi_changed.emit)
        for signal in (
            self.spin_roi_width.valueChanged,
            self.spin_roi_height.valueChanged,
            self.slider_offset_x.valueChanged,
            self.slider_offset_y.valueChanged,
        ):
            signal.connect(lambda *_: self._roi_debounce.start())

    def _compute_roi(self):
        """Compute x_min/x_max/y_min/y_max from the width, height and centre-offset
        controls, clamped to the IMX636 sensor.

        Unlike the ORCA there is no multiple-of-4 alignment requirement: the crop
        is applied as an exact software slice (and a best-effort hardware ROI),
        so any window is valid.
        """
        sw, sh = EVK4_SENSOR_WIDTH, EVK4_SENSOR_HEIGHT
        w = min(self.spin_roi_width.value(), sw)
        h = min(self.spin_roi_height.value(), sh)
        x_min = (sw - w) // 2 + self.slider_offset_x.value()
        y_min = (sh - h) // 2 + self.slider_offset_y.value()
        x_min = max(0, min(x_min, sw - w))
        y_min = max(0, min(y_min, sh - h))
        return {"x_min": x_min, "x_max": x_min + w, "y_min": y_min, "y_max": y_min + h}

    def get_params(self):
        return {
            "bias_fo": self.spin_fo.value(),
            "bias_hpf": self.spin_hpf.value(),
            "bias_on": self.spin_on.value(),
            "bias_off": self.spin_off.value(),
            "acqu_time": self.spin_time.value(),
            "filter_crazy_pixels": self.chk_crazy.isChecked(),
            "apply_smoothing": self.chk_smooth.isChecked(),
            "evk4_roi": self._compute_roi(),
        }

    def get_preset(self):
        """Return all widget values as a JSON-serialisable dict for preset files."""
        data = self.get_params()
        data.pop("evk4_roi", None)  # store the control values, not the derived window
        data.update({
            "roi_width": self.spin_roi_width.value(),
            "roi_height": self.spin_roi_height.value(),
            "roi_offset_x": self.slider_offset_x.value(),
            "roi_offset_y": self.slider_offset_y.value(),
        })
        return data

    def set_preset(self, data):
        """Restore widget values from a preset dict. Unknown keys are ignored."""
        if "bias_fo" in data:
            self.spin_fo.setValue(int(data["bias_fo"]))
        if "bias_hpf" in data:
            self.spin_hpf.setValue(int(data["bias_hpf"]))
        if "bias_on" in data:
            self.spin_on.setValue(int(data["bias_on"]))
        if "bias_off" in data:
            self.spin_off.setValue(int(data["bias_off"]))
        if "acqu_time" in data:
            self.spin_time.setValue(float(data["acqu_time"]))
        if "filter_crazy_pixels" in data:
            self.chk_crazy.setChecked(bool(data["filter_crazy_pixels"]))
        if "apply_smoothing" in data:
            self.chk_smooth.setChecked(bool(data["apply_smoothing"]))
        if "roi_width" in data:
            self.spin_roi_width.setValue(int(data["roi_width"]))
        if "roi_height" in data:
            self.spin_roi_height.setValue(int(data["roi_height"]))
        if "roi_offset_x" in data:
            self.slider_offset_x.setValue(int(data["roi_offset_x"]))
        if "roi_offset_y" in data:
            self.slider_offset_y.setValue(int(data["roi_offset_y"]))


class OrcaParamsWidget(QWidget):
    """Reusable Hamamatsu ORCA parameter controls: exposure, frame count, ROI
    cropping, and camera mode/readout. Used in both the Scientific Camera tab and
    the Z-Stack tab."""

    # Emitted (debounced) whenever ROI geometry — size or centre offset — changes,
    # so a running live feed can be re-framed/re-centred without clicking Apply.
    roi_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        setup_group = QGroupBox("Hamamatsu ORCA Setup")
        setup_form = QFormLayout()
        # Min 0.017 ms = 17 µs (the Fast-scan exposure floor from the manual).
        # 3 decimals so sub-millisecond exposures can be dialled in — required to
        # reach the high framerates (low exposure + small ROI on Fast scan).
        self.spin_exp = QDoubleSpinBox(); self.spin_exp.setRange(0.017, 10000); self.spin_exp.setDecimals(3); self.spin_exp.setValue(50)
        self.spin_exp.setSuffix(" ms")
        self.spin_frames = QSpinBox(); self.spin_frames.setRange(2, 1000); self.spin_frames.setValue(100)
        setup_form.addRow("Exposure Time (texp):", self.spin_exp)
        setup_form.addRow("Frames per Stack (N):", self.spin_frames)
        setup_group.setLayout(setup_form)
        layout.addWidget(setup_group)

        mode_group = QGroupBox("Camera Mode & Readout")
        mode_form = QFormLayout()
        self.combo_readout = make_dcam_combo(DCAM_READOUTSPEED_OPTIONS, "Standard (2)")
        self.combo_binning = make_dcam_combo(DCAM_BINNING_OPTIONS, "1 x 1")
        self.combo_trigsrc = make_dcam_combo(DCAM_TRIGGERSOURCE_OPTIONS, "Internal")
        self.combo_trigmode = make_dcam_combo(DCAM_TRIGGER_MODE_OPTIONS, "Normal")
        self.combo_defect = make_dcam_combo(DCAM_DEFECTCORRECT_OPTIONS, "On")
        mode_form.addRow("Readout Speed (mode):", self.combo_readout)
        mode_form.addRow("Binning:", self.combo_binning)
        mode_form.addRow("Trigger Source:", self.combo_trigsrc)
        mode_form.addRow("Trigger Mode:", self.combo_trigmode)
        mode_form.addRow("Defect Correction:", self.combo_defect)
        mode_group.setLayout(mode_form)
        layout.addWidget(mode_group)

        roi_group = QGroupBox("Region of Interest — Hardware Crop")
        roi_form = QFormLayout()
        # The crop defaults to the sensor centre and can be shifted by the centre
        # offset below. The camera uses DCAM subarray mode so only the selected
        # region is read out, increasing the maximum framerate proportionally.
        # Size and position must be multiples of 4; the camera rejects anything
        # else (and falls back to a slow software crop), so _compute_roi aligns them.
        self.spin_roi_width = QSpinBox()
        self.spin_roi_width.setRange(4, 2304)
        self.spin_roi_width.setSingleStep(4)
        self.spin_roi_width.setValue(2304)
        self.spin_roi_width.setSuffix(" px")
        self.spin_roi_height = QSpinBox()
        self.spin_roi_height.setRange(4, 2304)
        self.spin_roi_height.setSingleStep(4)
        self.spin_roi_height.setValue(2304)
        self.spin_roi_height.setSuffix(" px")
        # Offset of the crop centre relative to the sensor centre, in pixels.
        # The laser is not always centred on the sensor, so the crop can be
        # shifted to follow it. Positive X = right, positive Y = down. These are
        # draggable sliders (drag, click-to-jump, or scroll-wheel over them) with
        # a live value label — far quicker than typing. Step 4 keeps the
        # resulting subarray position aligned to the required grid.
        self.slider_offset_x, offset_x_row = make_offset_slider(ORCA_SENSOR_WIDTH // 2)
        self.slider_offset_y, offset_y_row = make_offset_slider(ORCA_SENSOR_HEIGHT // 2)
        roi_form.addRow("Width:", self.spin_roi_width)
        roi_form.addRow("Height:", self.spin_roi_height)
        roi_form.addRow("Centre offset X:", offset_x_row)
        roi_form.addRow("Centre offset Y:", offset_y_row)
        lbl_roi = QLabel(
            "Full sensor: 2304 × 2304 px. Reducing the size enables hardware "
            "subarray mode, increasing the framerate. The centre offset shifts "
            "the crop off-centre to follow the laser; the region is clamped to "
            "the sensor and aligned to a multiple of 4 px."
        )
        lbl_roi.setWordWrap(True)
        lbl_roi.setStyleSheet("color: #888888; font-size: 11px;")
        roi_form.addRow(lbl_roi)

        self.lbl_framerate = QLabel()
        self.lbl_framerate.setStyleSheet("color: #4daaf2; font-size: 11px; font-weight: bold;")
        roi_form.addRow(self.lbl_framerate)
        roi_group.setLayout(roi_form)
        layout.addWidget(roi_group)

        # Update the framerate label whenever any of the three influencing
        # parameters change: ROI height, exposure time, or readout speed.
        self.spin_roi_height.valueChanged.connect(self._update_framerate)
        self.spin_exp.valueChanged.connect(self._update_framerate)
        self.combo_readout.currentIndexChanged.connect(self._update_framerate)
        self._update_framerate()

        # Debounced live re-framing: changing any ROI geometry (size or centre
        # offset) emits `roi_changed` shortly after the user stops adjusting, so
        # main_window can re-apply it to a running live feed and the crop follows
        # the slider in real time (used to centre the crop on the speckle). The
        # debounce coalesces the burst of valueChanged events from a drag into a
        # single re-apply, avoiding a capture restart on every pixel.
        self._roi_debounce = QTimer(self)
        self._roi_debounce.setSingleShot(True)
        self._roi_debounce.setInterval(200)
        self._roi_debounce.timeout.connect(self.roi_changed.emit)
        for signal in (
            self.spin_roi_width.valueChanged,
            self.spin_roi_height.valueChanged,
            self.slider_offset_x.valueChanged,
            self.slider_offset_y.valueChanged,
        ):
            signal.connect(lambda *_: self._roi_debounce.start())

        # Single button that sends all ORCA settings (exposure, mode, ROI) to the
        # running live-feed worker. Enabled only while live mode is active.
        self.btn_apply_live = QPushButton("Apply All Settings to Live Feed")
        self.btn_apply_live.setEnabled(False)
        layout.addWidget(self.btn_apply_live)

    def estimated_raw_save_s(self, n):
        """Estimated time to write the N-frame raw 16-bit stack to disk.

        The raw TIFF dominates the wall-clock time of a full-frame acquisition,
        so the duration estimate must include it. Bytes = N · width · height · 2
        (uint16), divided by an assumed sustained disk write rate.
        """
        roi = self._compute_roi()
        w = roi["x_max"] - roi["x_min"]
        h = roi["y_max"] - roi["y_min"]
        return (n * w * h * 2) / ZSTACK_DISK_BYTES_PER_S

    def estimated_compute_s(self, n):
        """Estimated time to reconstruct the DSI images for an N-frame stack.

        The average + standard-deviation reconstruction runs once per plane over
        N·width·height pixels; at full sensor it is several seconds and was
        previously omitted from the duration estimate, which is the main cause of
        the Z-stack running longer than predicted.
        """
        roi = self._compute_roi()
        w = roi["x_max"] - roi["x_min"]
        h = roi["y_max"] - roi["y_min"]
        return (n * w * h) / ORCA_DSI_PROCESS_PIXELS_PER_S

    def estimated_frame_time_s(self):
        """Estimated time per frame (seconds) — the longer of exposure and readout.

        The readout term follows the manual's free-running formula (Vn+1)*1H,
        where Vn is the subarray height in rows and 1H is the per-row readout
        time for the selected scan speed. Used by main_window to compute
        acquisition duration estimates without importing ORCA constants directly.
        """
        texp_s = self.spin_exp.value() / 1000.0
        row_us = ORCA_ROW_READOUT_US.get(self.combo_readout.currentData(), 18.64706)
        readout_s = (self.spin_roi_height.value() + 1) * row_us * 1e-6
        return max(texp_s, readout_s)

    def _update_framerate(self):
        """Recompute and display the estimated camera framerate in the ROI group."""
        frame_s = self.estimated_frame_time_s()
        fps = 1.0 / frame_s
        h = self.spin_roi_height.value()
        texp_s = self.spin_exp.value() / 1000.0
        row_us = ORCA_ROW_READOUT_US.get(self.combo_readout.currentData(), 18.64706)
        readout_s = (h + 1) * row_us * 1e-6

        if readout_s <= texp_s:
            limit = "exposure-limited"
        else:
            limit = "readout-limited"

        if h == 2304:
            self.lbl_framerate.setText(f"Max framerate: ≈ {fps:.0f} fps (full sensor, {limit})")
        else:
            self.lbl_framerate.setText(f"Max framerate: ≈ {fps:.0f} fps ({h} px height, {limit})")

    def _compute_roi(self):
        """Compute x_min/x_max/y_min/y_max from the width, height and centre-offset
        spinboxes.

        The crop starts centred on the sensor, is shifted by the user offset,
        then clamped so it stays fully on the sensor and floored to a multiple of
        4 px. DCAM requires the subarray position to be a multiple of 4; an
        unaligned position is rejected and silently degrades to a slow software
        crop, so the alignment here is what keeps the hardware framerate gain.
        """
        sw, sh = ORCA_SENSOR_WIDTH, ORCA_SENSOR_HEIGHT
        w = self.spin_roi_width.value() // 4 * 4
        h = self.spin_roi_height.value() // 4 * 4
        x_min = (sw - w) // 2 + self.slider_offset_x.value()
        y_min = (sh - h) // 2 + self.slider_offset_y.value()
        # Keep the region on the sensor, then align the position to a 4 px grid.
        x_min = max(0, min(x_min, sw - w)) // 4 * 4
        y_min = max(0, min(y_min, sh - h)) // 4 * 4
        return {"x_min": x_min, "x_max": x_min + w, "y_min": y_min, "y_max": y_min + h}

    def get_params(self):
        return {
            "orca_exposure": self.spin_exp.value(),
            "orca_frames": self.spin_frames.value(),
            "orca_roi": self._compute_roi(),
            "readout_speed": self.combo_readout.currentData(),
            "binning": self.combo_binning.currentData(),
            "trigger_source": self.combo_trigsrc.currentData(),
            "trigger_mode": self.combo_trigmode.currentData(),
            "defect_correct": self.combo_defect.currentData(),
        }

    def mode_labels(self):
        """Human-readable mode selections for the parameter log."""
        return {
            "readout_speed": self.combo_readout.currentText(),
            "binning": self.combo_binning.currentText(),
            "trigger_source": self.combo_trigsrc.currentText(),
            "trigger_mode": self.combo_trigmode.currentText(),
            "defect_correction": self.combo_defect.currentText(),
        }

    def get_preset(self):
        """Return all widget values as a JSON-serialisable dict for preset files."""
        return {
            "exposure_ms": self.spin_exp.value(),
            "frames": self.spin_frames.value(),
            "roi_width": self.spin_roi_width.value(),
            "roi_height": self.spin_roi_height.value(),
            "roi_offset_x": self.slider_offset_x.value(),
            "roi_offset_y": self.slider_offset_y.value(),
            "readout_speed": self.combo_readout.currentData(),
            "binning": self.combo_binning.currentData(),
            "trigger_source": self.combo_trigsrc.currentData(),
            "trigger_mode": self.combo_trigmode.currentData(),
            "defect_correct": self.combo_defect.currentData(),
        }

    def set_preset(self, data):
        """Restore widget values from a preset dict. Unknown keys are ignored."""
        if "exposure_ms" in data:
            self.spin_exp.setValue(float(data["exposure_ms"]))
        if "frames" in data:
            self.spin_frames.setValue(int(data["frames"]))
        if "roi_width" in data:
            self.spin_roi_width.setValue(int(data["roi_width"]))
        if "roi_height" in data:
            self.spin_roi_height.setValue(int(data["roi_height"]))
        if "roi_offset_x" in data:
            self.slider_offset_x.setValue(int(data["roi_offset_x"]))
        if "roi_offset_y" in data:
            self.slider_offset_y.setValue(int(data["roi_offset_y"]))
        for key, combo in [
            ("readout_speed", self.combo_readout),
            ("binning", self.combo_binning),
            ("trigger_source", self.combo_trigsrc),
            ("trigger_mode", self.combo_trigmode),
            ("defect_correct", self.combo_defect),
        ]:
            if key not in data:
                continue
            for i in range(combo.count()):
                if combo.itemData(i) == data[key]:
                    combo.setCurrentIndex(i)
                    break


class AWGWidget(QGroupBox):
    """Siglent AWG control panel with independent CH1 / CH2 control (LC speckle)."""

    def __init__(self):
        super().__init__("Siglent AWG Control (LC Speckle)")
        self.controller = AWGController()
        self.channels = {}  # channel number -> dict of its widgets
        self.init_ui()

    # Backwards-compatible accessor so MainWindow teardown can reach the device.
    @property
    def awg(self):
        return self.controller.awg

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Shared connection controls --------------------------------------
        conn_form = QFormLayout()
        self.combo_visa = QComboBox()
        self.refresh_resources()

        btn_refresh = QPushButton("⟳")
        btn_refresh.setFixedWidth(30)
        btn_refresh.clicked.connect(self.refresh_resources)

        conn_layout = QHBoxLayout()
        conn_layout.addWidget(self.combo_visa, stretch=1)
        conn_layout.addWidget(btn_refresh)

        self.btn_connect = QPushButton("Connect AWG")
        self.btn_connect.clicked.connect(self.connect_awg)

        conn_form.addRow("VISA Address:", conn_layout)
        conn_form.addRow("", self.btn_connect)
        layout.addLayout(conn_form)

        # --- One independent control block per channel -----------------------
        for ch in (1, 2):
            layout.addWidget(self._build_channel_group(ch))

        self.setLayout(layout)

    def _build_channel_group(self, channel):
        """Build the freq / amplitude / apply / output controls for one channel."""
        group = QGroupBox(f"Channel {channel}")
        form = QFormLayout()

        spin_freq = QDoubleSpinBox()
        spin_freq.setRange(100, 5000)
        spin_freq.setValue(2000)
        spin_freq.setSuffix(" Hz")
        spin_freq.setDecimals(0)

        spin_amp = QDoubleSpinBox()
        spin_amp.setRange(0.1, 20.0)
        spin_amp.setValue(9.0)
        spin_amp.setSuffix(" Vpp")
        spin_amp.setSingleStep(0.5)

        btn_apply = QPushButton("Apply Parameters")
        btn_apply.clicked.connect(lambda _, c=channel: self.update_awg_params(c))
        btn_apply.setEnabled(False)

        btn_output = QPushButton("Output OFF")
        btn_output.setCheckable(True)
        btn_output.clicked.connect(lambda checked, c=channel: self.toggle_output(c, checked))
        btn_output.setEnabled(False)

        form.addRow("Frequency:", spin_freq)
        form.addRow("Amplitude:", spin_amp)
        form.addRow("", btn_apply)
        form.addRow("", btn_output)
        group.setLayout(form)

        self.channels[channel] = {
            "freq": spin_freq,
            "amp": spin_amp,
            "apply": btn_apply,
            "output": btn_output,
        }
        return group

    def refresh_resources(self):
        self.combo_visa.clear()
        resources = self.controller.list_resources()
        if resources:
            self.combo_visa.addItems(resources)
        else:
            self.combo_visa.addItem("No VISA devices found")

    def connect_awg(self):
        addr = self.combo_visa.currentText()
        if "No VISA" in addr:
            return

        try:
            idn = self.controller.connect(addr)
            print(f"Connected to: {idn}")

            for widgets in self.channels.values():
                widgets["apply"].setEnabled(True)
                widgets["output"].setEnabled(True)
            self.btn_connect.setText("Connected")
            self.btn_connect.setStyleSheet("background-color: #3a3f44; color: #4daaf2;")
            self.btn_connect.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "AWG Connection Error", f"Failed to connect to {addr}.\n\nError: {str(e)}")

    def update_awg_params(self, channel):
        if self.controller.is_connected:
            try:
                widgets = self.channels[channel]
                self.controller.set_params(widgets["freq"].value(), widgets["amp"].value(), channel)
            except Exception as e:
                print(f"Error updating AWG CH{channel} parameters: {e}")

    def toggle_output(self, channel, checked):
        if self.controller.is_connected:
            try:
                self.controller.set_output(checked, channel)
                btn_output = self.channels[channel]["output"]
                if checked:
                    btn_output.setText("Output ON")
                    btn_output.setStyleSheet("background-color: #2e7d32; color: white;")
                else:
                    btn_output.setText("Output OFF")
                    btn_output.setStyleSheet("")
            except Exception as e:
                print(f"Error toggling AWG CH{channel} output: {e}")

    def get_settings(self):
        """Return the current AWG UI state for the acquisition parameter log."""
        settings = {
            "connected": self.controller.is_connected,
            "visa_address": self.combo_visa.currentText(),
        }
        for channel, widgets in self.channels.items():
            settings[f"ch{channel}_frequency_hz"] = widgets["freq"].value()
            settings[f"ch{channel}_amplitude_vpp"] = widgets["amp"].value()
            settings[f"ch{channel}_output"] = "ON" if widgets["output"].isChecked() else "OFF"
        return settings

    def get_preset(self):
        """Return the AWG UI parameters for presets / session restore.

        Persists the VISA address and per-channel frequency/amplitude. The live
        output ON/OFF state is deliberately *not* saved — it drives the hardware
        and must only be turned on after the user connects the device.
        """
        preset = {"visa_address": self.combo_visa.currentText()}
        for channel, widgets in self.channels.items():
            preset[f"ch{channel}_freq"] = widgets["freq"].value()
            preset[f"ch{channel}_amp"] = widgets["amp"].value()
        return preset

    def set_preset(self, data):
        """Restore AWG UI parameters from a preset dict. Unknown keys are ignored;
        the VISA address is only selected if it is still in the resource list."""
        addr = data.get("visa_address")
        if addr:
            idx = self.combo_visa.findText(addr)
            if idx >= 0:
                self.combo_visa.setCurrentIndex(idx)
        for channel, widgets in self.channels.items():
            if f"ch{channel}_freq" in data:
                widgets["freq"].setValue(float(data[f"ch{channel}_freq"]))
            if f"ch{channel}_amp" in data:
                widgets["amp"].setValue(float(data[f"ch{channel}_amp"]))

    def close_device(self):
        self.controller.close()


class PIStageWidget(QWidget):
    """PI Z-stage motor control panel (connect, focus move, manual stepping)."""

    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.controller = StageController()
        self.move_worker = None
        # Guards the GCS serial link: pause idle polling while a worker thread
        # (manual move or Z-stack) is driving the stage, to avoid concurrent qPOS.
        self._device_busy = False
        self.init_ui()

        self.position_timer = QTimer(self)
        self.position_timer.setInterval(200)  # 5 Hz live readout
        self.position_timer.timeout.connect(self.update_position)

    # Accessors used by MainWindow / the Z-stack orchestrator.
    @property
    def pidevice(self):
        return self.controller.pidevice

    @property
    def axis(self):
        return self.controller.axis

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        z_group = QGroupBox("PI Stage Motor Control")
        z_form = QFormLayout()

        # The PIFOC piezo stage works in micrometers (e.g. P-725.4: 400 µm
        # travel). Ranges/units are refined from the controller (qPUN/qTMX) on
        # connect; these defaults are sensible starting points for a 400 µm stage.
        self.spin_focus = QDoubleSpinBox()
        self.spin_focus.setRange(0, 400)
        self.spin_focus.setDecimals(4)
        self.spin_focus.setSingleStep(1.0)
        self.spin_focus.setValue(200.0)
        self.spin_focus.setSuffix(" µm")

        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(0.001, 100)
        self.spin_step_size.setSingleStep(0.1)
        self.spin_step_size.setDecimals(3)
        self.spin_step_size.setValue(0.5)
        self.spin_step_size.setSuffix(" µm")

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 1000)
        self.spin_steps.setValue(60)

        self.btn_connect = QPushButton("Connect PI Stage")
        self.btn_connect.clicked.connect(self.connect_stage)

        self.lbl_position = QLabel("-- (not connected)")
        self.lbl_position.setStyleSheet("color: #4daaf2; font-weight: bold; background-color: transparent;")

        self.btn_move_focus = QPushButton("Move to Target Focus")
        self.btn_move_focus.setObjectName("btnLive")
        self.btn_move_focus.clicked.connect(self.move_to_focus)
        self.btn_move_focus.setEnabled(False)

        self.btn_move_init = QPushButton("Calculate and Move to Z-Stack Start")
        self.btn_move_init.setObjectName("btnAcquire")
        self.btn_move_init.clicked.connect(self.move_to_initial_pos)
        self.btn_move_init.setEnabled(False)

        step_layout = QHBoxLayout()
        self.btn_step_bwd = QPushButton("Step Backward (-)")
        self.btn_step_bwd.clicked.connect(lambda: self.step_manual(-1))
        self.btn_step_bwd.setEnabled(False)

        self.btn_step_fwd = QPushButton("Step Forward (+)")
        self.btn_step_fwd.clicked.connect(lambda: self.step_manual(1))
        self.btn_step_fwd.setEnabled(False)

        step_layout.addWidget(self.btn_step_bwd)
        step_layout.addWidget(self.btn_step_fwd)

        z_form.addRow("", self.btn_connect)
        z_form.addRow("Live Position:", self.lbl_position)
        z_form.addRow("Target Focus Position:", self.spin_focus)
        z_form.addRow("", self.btn_move_focus)
        z_form.addRow("Step Size:", self.spin_step_size)
        z_form.addRow("Number of Steps:", self.spin_steps)
        z_form.addRow("", self.btn_move_init)
        z_form.addRow("Manual Step:", step_layout)

        z_group.setLayout(z_form)
        layout.addWidget(z_group)

    def connect_stage(self):
        if not PI_AVAILABLE:
            QMessageBox.critical(self, "Library Missing", "The 'pipython' library is not installed.")
            return

        try:
            self.status_update.emit("Connecting to PI E-709 (USB / RS-232)...")
            info = self.controller.connect()

            self.btn_connect.setText("Connected")
            self.btn_connect.setStyleSheet("background-color: #3a3f44; color: #4daaf2;")
            self.btn_connect.setEnabled(False)

            self.enable_controls(True)
            self._apply_controller_units()
            self.status_update.emit(f"PI E-709 connected and servo enabled ({info}).")
            self.update_position()
            self.position_timer.start()
        except Exception as e:
            QMessageBox.critical(self, "PI Stage Error", f"Failed to connect or initialize PI Stage.\n\nError: {str(e)}")
            self.status_update.emit("PI Stage connection failed.")

    def enable_controls(self, state):
        self.btn_move_focus.setEnabled(state)
        self.btn_move_init.setEnabled(state)
        self.btn_step_fwd.setEnabled(state)
        self.btn_step_bwd.setEnabled(state)

    def execute_movement(self, target):
        if self.controller.is_connected:
            self.enable_controls(False)
            self._device_busy = True  # pause idle polling while the worker drives the stage
            self.move_worker = PIMoveWorker(self.controller.pidevice, self.controller.axis, target)
            self.move_worker.status_update.connect(self.status_update.emit)
            self.move_worker.finished_signal.connect(self._on_move_finished)
            self.move_worker.start()

    def _on_move_finished(self, pos):
        self._device_busy = False
        self.show_position(pos)
        self.enable_controls(True)

    # -- live position readout ----------------------------------------------
    def update_position(self):
        """Poll the controller for the current position (idle GUI-thread poll)."""
        if not self.controller.is_connected or self._device_busy:
            return
        try:
            self.show_position(self.controller.position())
        except Exception:
            # Lost the link; stop polling rather than spamming errors.
            self.position_timer.stop()
            self.lbl_position.setText("-- (read error)")

    def show_position(self, pos):
        """Display a position value (NaN -> placeholder), used by polling, moves
        and the Z-stack."""
        if pos != pos:  # NaN
            self.lbl_position.setText("--")
        else:
            self.lbl_position.setText(f"{pos:.4f} {self.controller.unit}")

    def _apply_controller_units(self):
        """Reflect the controller's real position unit and travel range in the UI."""
        unit = self.controller.unit
        self.spin_focus.setSuffix(f" {unit}")
        self.spin_step_size.setSuffix(f" {unit}")
        tmin, tmax = self.controller.travel_min, self.controller.travel_max
        if tmin is not None and tmax is not None and tmax > tmin:
            self.spin_focus.setRange(tmin, tmax)

    def pause_position_updates(self):
        """Suspend idle polling while another thread owns the device (e.g. Z-stack)."""
        self._device_busy = True

    def resume_position_updates(self):
        self._device_busy = False

    def move_to_focus(self):
        self.execute_movement(self.spin_focus.value())

    def move_to_initial_pos(self):
        focus = self.spin_focus.value()
        step_size = self.spin_step_size.value()
        steps = self.spin_steps.value()
        init_pos = focus - (step_size * steps / 2)
        self.execute_movement(init_pos)

    def step_manual(self, direction):
        if self.controller.is_connected and not self._device_busy:
            step_size = self.spin_step_size.value() * direction
            current_pos = self.controller.position()
            self.execute_movement(current_pos + step_size)

    def close_device(self):
        self.position_timer.stop()
        self.controller.close()


class VideoFeedLabel(QLabel):
    """Video-feed display with an interactive crop-region selector.

    Normally it behaves like the plain QLabel that showed the live image. In
    *crop mode* the user drags a rectangle over the (full-sensor) image; the
    selection is drawn as a bright box with dashed guide lines extending across
    the frame, and is kept on screen after the mouse is released so the crop can
    be reviewed *before* it is applied. ``region_drawn`` reports the selection in
    source-image pixel coordinates (x, y, w, h) of the frame currently displayed.

    The widget knows the source frame size (``set_source_size``) so it can map
    between widget coordinates and image pixels; the pixmap is assumed centred
    (the label uses ``AlignCenter``) and pre-scaled with KeepAspectRatio.
    """

    region_drawn = pyqtSignal(int, int, int, int)  # x, y, w, h in image pixels

    def __init__(self, text=""):
        super().__init__(text)
        self._src_w = None        # source frame width in pixels
        self._src_h = None        # source frame height in pixels
        self._orig_pixmap = None  # full-resolution frame, re-scaled to fit on resize
        self._crop_mode = False
        self._origin = None       # drag start (widget coords)
        self._cur = None          # current drag point (widget coords)
        self._selection = None    # committed selection as QRect in image pixels

    def set_source_size(self, w, h):
        """Tell the label the pixel size of the frame currently shown."""
        if (w, h) != (self._src_w, self._src_h):
            self._src_w, self._src_h = w, h
            # A frame of a different size means the old selection no longer maps.
            self._selection = None

    def set_frame_pixmap(self, pixmap):
        """Display a full-resolution frame, scaled to fit the current label size.

        The *original* pixmap is kept so it can be re-scaled whenever the label is
        resized (a tab switch, a Display-mode change, or a window resize). Scaling
        only at render time left a stale, wrongly-sized pixmap after a layout
        change until the next frame arrived — which is the flicker fixed here.
        """
        self._orig_pixmap = pixmap
        self._rescale_pixmap()

    def _rescale_pixmap(self):
        """Re-scale the stored frame to the current label size (KeepAspectRatio)."""
        if self._orig_pixmap is None or self._orig_pixmap.isNull():
            return
        super().setPixmap(
            self._orig_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Layout changed the label size — re-fit the current frame immediately so
        # it never stays scaled to the previous size.
        self._rescale_pixmap()

    def set_crop_mode(self, on):
        self._crop_mode = bool(on)
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        if not on:
            self._origin = self._cur = None
        self.update()

    def clear_selection(self):
        self._selection = None
        self._origin = self._cur = None
        self.update()

    def _image_rect(self):
        """Return (rect, scale): the widget-coordinate rectangle the pixmap
        occupies (centred) and the image-px -> widget-px scale, or (None, 1.0)."""
        pm = self.pixmap()
        if pm is None or pm.isNull() or not self._src_w or not self._src_h:
            return None, 1.0
        pw, ph = pm.width(), pm.height()
        x = (self.width() - pw) // 2
        y = (self.height() - ph) // 2
        return QRect(x, y, pw, ph), (pw / self._src_w)

    def _clamp(self, point):
        """Clamp a widget-coordinate point to the displayed image rectangle."""
        rect, _ = self._image_rect()
        if rect is None:
            return point
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return point.__class__(x, y)

    def mousePressEvent(self, event):
        rect, _ = self._image_rect()
        if (self._crop_mode and event.button() == Qt.MouseButton.LeftButton
                and rect is not None and rect.contains(event.pos())):
            self._origin = event.pos()
            self._cur = event.pos()
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._crop_mode and self._origin is not None:
            self._cur = self._clamp(event.pos())
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._crop_mode and self._origin is not None:
            self._cur = self._clamp(event.pos())
            self._commit_selection()
            self._origin = None
            self.update()
        else:
            super().mouseReleaseEvent(event)

    def _commit_selection(self):
        """Convert the dragged widget rectangle into image pixels, store and emit."""
        rect, scale = self._image_rect()
        if rect is None or self._origin is None or self._cur is None or scale <= 0:
            return
        wr = QRect(self._origin, self._cur).normalized()
        ix = int(round((wr.left() - rect.left()) / scale))
        iy = int(round((wr.top() - rect.top()) / scale))
        iw = int(round(wr.width() / scale))
        ih = int(round(wr.height() / scale))
        # Clamp to the frame.
        ix = max(0, min(ix, self._src_w - 1))
        iy = max(0, min(iy, self._src_h - 1))
        iw = max(1, min(iw, self._src_w - ix))
        ih = max(1, min(ih, self._src_h - iy))
        if iw < 4 or ih < 4:  # ignore an accidental click / tiny drag
            return
        self._selection = QRect(ix, iy, iw, ih)
        self.region_drawn.emit(ix, iy, iw, ih)

    def paintEvent(self, event):
        super().paintEvent(event)  # draws the pixmap / text as usual
        if not self._crop_mode:
            return
        rect, scale = self._image_rect()
        if rect is None:
            return

        # The box to draw: the live drag if one is in progress, otherwise the
        # committed selection mapped back into widget coordinates.
        if self._origin is not None and self._cur is not None:
            box = QRect(self._origin, self._cur).normalized()
        elif self._selection is not None:
            box = QRect(
                int(rect.left() + self._selection.left() * scale),
                int(rect.top() + self._selection.top() * scale),
                int(self._selection.width() * scale),
                int(self._selection.height() * scale),
            )
        else:
            return

        painter = QPainter(self)
        # Dashed guide lines spanning the frame through the selection edges.
        painter.setPen(QPen(QColor(0, 230, 118, 130), 1, Qt.PenStyle.DashLine))
        painter.drawLine(box.left(), rect.top(), box.left(), rect.bottom())
        painter.drawLine(box.right(), rect.top(), box.right(), rect.bottom())
        painter.drawLine(rect.left(), box.top(), rect.right(), box.top())
        painter.drawLine(rect.left(), box.bottom(), rect.right(), box.bottom())
        # The selection rectangle itself.
        painter.setPen(QPen(QColor("#00e676"), 2, Qt.PenStyle.SolidLine))
        painter.drawRect(box)
        painter.end()
