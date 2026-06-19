"""Custom composite UI widgets: AWG control panel and PI stage control panel.

These widgets own their visuals/inputs and delegate all instrument I/O to the
hardware-layer controllers (AWGController, StageController) and workers (PIMoveWorker).
"""

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)
from PyQt6.QtCore import QTimer, pyqtSignal

from config import (
    DCAM_BINNING_OPTIONS, DCAM_DEFECTCORRECT_OPTIONS, DCAM_READOUTSPEED_OPTIONS,
    DCAM_TRIGGER_MODE_OPTIONS, DCAM_TRIGGERSOURCE_OPTIONS,
    ORCA_ROW_READOUT_US,
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


class Evk4ParamsWidget(QWidget):
    """Reusable Prophesee EVK4 parameter controls: biases, per-acquisition
    duration, and post-processing options. Used in both the Event Camera tab and
    the Z-Stack tab so each has independent, fully selectable parameters."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bias_group = QGroupBox("EVK4 Biases")
        bias_form = QFormLayout()
        self.spin_fo = QSpinBox(); self.spin_fo.setRange(-100, 100); self.spin_fo.setValue(5)
        self.spin_hpf = QSpinBox(); self.spin_hpf.setRange(-100, 100); self.spin_hpf.setValue(30)
        self.spin_on = QSpinBox(); self.spin_on.setRange(0, 255); self.spin_on.setValue(5)
        self.spin_off = QSpinBox(); self.spin_off.setRange(0, 255); self.spin_off.setValue(5)
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
        self.spin_time = QSpinBox(); self.spin_time.setRange(1, 3600); self.spin_time.setValue(5)
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

    def get_params(self):
        return {
            "bias_fo": self.spin_fo.value(),
            "bias_hpf": self.spin_hpf.value(),
            "bias_on": self.spin_on.value(),
            "bias_off": self.spin_off.value(),
            "acqu_time": self.spin_time.value(),
            "filter_crazy_pixels": self.chk_crazy.isChecked(),
            "apply_smoothing": self.chk_smooth.isChecked(),
        }

    def get_preset(self):
        """Return all widget values as a JSON-serialisable dict for preset files."""
        return self.get_params()

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
            self.spin_time.setValue(int(data["acqu_time"]))
        if "filter_crazy_pixels" in data:
            self.chk_crazy.setChecked(bool(data["filter_crazy_pixels"]))
        if "apply_smoothing" in data:
            self.chk_smooth.setChecked(bool(data["apply_smoothing"]))


class OrcaParamsWidget(QWidget):
    """Reusable Hamamatsu ORCA parameter controls: exposure, frame count, ROI
    cropping, and camera mode/readout. Used in both the Scientific Camera tab and
    the Z-Stack tab."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        setup_group = QGroupBox("Hamamatsu ORCA Setup")
        setup_form = QFormLayout()
        self.spin_exp = QDoubleSpinBox(); self.spin_exp.setRange(0.1, 10000); self.spin_exp.setDecimals(1); self.spin_exp.setValue(50)
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

        roi_group = QGroupBox("Region of Interest — Centered Hardware Crop")
        roi_form = QFormLayout()
        # Crop is always centred on the sensor (1152, 1152 for ORCA-Fusion).
        # The camera uses DCAM subarray mode so only the selected region is read
        # out, increasing the maximum framerate proportionally. Values must be
        # multiples of 4; the camera rejects anything else.
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
        roi_form.addRow("Width:", self.spin_roi_width)
        roi_form.addRow("Height:", self.spin_roi_height)
        lbl_roi = QLabel(
            "Full sensor: 2304 × 2304 px. Reducing the size crops to the centre "
            "and enables hardware subarray mode, increasing the framerate."
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

        # Single button that sends all ORCA settings (exposure, mode, ROI) to the
        # running live-feed worker. Enabled only while live mode is active.
        self.btn_apply_live = QPushButton("Apply All Settings to Live Feed")
        self.btn_apply_live.setEnabled(False)
        layout.addWidget(self.btn_apply_live)

    def estimated_frame_time_s(self):
        """Estimated time per frame (seconds) — the longer of exposure and readout.

        Used by main_window to compute acquisition duration estimates without
        importing ORCA hardware constants directly.
        """
        texp_s = self.spin_exp.value() / 1000.0
        row_us = ORCA_ROW_READOUT_US.get(self.combo_readout.currentData(), 4.34)
        readout_s = self.spin_roi_height.value() * row_us * 1e-6
        return max(texp_s, readout_s)

    def _update_framerate(self):
        """Recompute and display the estimated camera framerate in the ROI group."""
        frame_s = self.estimated_frame_time_s()
        fps = 1.0 / frame_s
        h = self.spin_roi_height.value()
        texp_s = self.spin_exp.value() / 1000.0
        row_us = ORCA_ROW_READOUT_US.get(self.combo_readout.currentData(), 4.34)
        readout_s = h * row_us * 1e-6

        if readout_s <= texp_s:
            limit = "exposure-limited"
        else:
            limit = "readout-limited"

        if h == 2304:
            self.lbl_framerate.setText(f"Max framerate: ≈ {fps:.0f} fps (full sensor, {limit})")
        else:
            self.lbl_framerate.setText(f"Max framerate: ≈ {fps:.0f} fps ({h} px height, {limit})")

    def _compute_roi(self):
        """Compute centred x_min/x_max/y_min/y_max from width and height spinboxes."""
        sw, sh = 2304, 2304
        w = self.spin_roi_width.value()
        h = self.spin_roi_height.value()
        x_min = (sw - w) // 2
        y_min = (sh - h) // 2
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
