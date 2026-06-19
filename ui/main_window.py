"""Main application window: layout, tab management, worker wiring and feed display.

This is the only place that instantiates the hardware workers and the Z-stack
orchestrator, routing every worker's status_update / error_signal back to the single
`lbl_status` bar and managing button enable/disable state during acquisition.
"""

import json
import time

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QTabWidget,
    QVBoxLayout, QWidget,
)

from hardware.event_camera import CameraWorker, METAVISION_AVAILABLE
from hardware.orca_camera import OrcaWorker, DCAM_AVAILABLE
from ui.orchestrator import AutomatedZStackWorker
from ui.widgets import (
    AWGWidget, Evk4ParamsWidget, OrcaParamsWidget, PIStageWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Institut Fresnel - DSI Microscope Control")
        self.resize(1300, 950)
        self.evk4_worker = None
        self.orca_worker = None
        self.zstack_worker = None
        self.zstack_live_worker = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Left column: a fixed-width container holding a scrollable control stack
        # plus a status bar pinned to the bottom. The scroll area is essential so
        # that, on shorter screens or higher DPI, the tab contents keep their
        # natural size and scroll instead of being squashed into each other.
        control_panel = QWidget()
        control_panel.setFixedWidth(460)
        control_panel_layout = QVBoxLayout(control_panel)
        control_panel_layout.setContentsMargins(0, 0, 0, 0)
        control_panel_layout.setSpacing(10)

        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        scroll_content.setStyleSheet("QWidget#scrollContent { background-color: #1e1e1e; }")
        control_layout = QVBoxLayout(scroll_content)
        control_layout.setContentsMargins(0, 0, 8, 0)  # right pad clears the scrollbar
        control_layout.setSpacing(10)

        lbl_header = QLabel("DSI Hardware Control")
        lbl_header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lbl_header.setStyleSheet("color: #ffffff; margin-bottom: 5px; background-color: transparent;")
        control_layout.addWidget(lbl_header)

        preset_layout = QHBoxLayout()
        btn_save_preset = QPushButton("Save Preset")
        btn_save_preset.setToolTip(
            "Save all camera parameters (biases, exposure, ROI, mode, Z-stack geometry) "
            "to a JSON file so they can be reloaded in a future session."
        )
        btn_save_preset.clicked.connect(self.save_preset)
        btn_load_preset = QPushButton("Load Preset")
        btn_load_preset.setToolTip("Restore previously saved camera parameters from a JSON file.")
        btn_load_preset.clicked.connect(self.load_preset)
        preset_layout.addWidget(btn_save_preset)
        preset_layout.addWidget(btn_load_preset)
        control_layout.addLayout(preset_layout)

        # AWG control is set rarely, so keep it in a collapsible section to free
        # up vertical space for the acquisition tabs (starts collapsed).
        self.btn_awg_toggle = QPushButton("▸  Siglent AWG Control (LC Speckle)")
        self.btn_awg_toggle.setCheckable(True)
        self.btn_awg_toggle.setStyleSheet("text-align: left; padding: 8px; background-color: #3a3f44;")
        self.btn_awg_toggle.toggled.connect(self._toggle_awg)
        control_layout.addWidget(self.btn_awg_toggle)

        self.awg_widget = AWGWidget()
        self.awg_widget.setVisible(False)
        control_layout.addWidget(self.awg_widget)

        self.tabs = QTabWidget()
        self._build_evk4_tab()
        self._build_orca_tab()
        self._build_zstack_tab()
        # Populate the time-estimate labels with the default parameter values.
        self._update_evk4_time()
        self._update_orca_time()
        self._update_zstack_time()
        control_layout.addWidget(self.tabs)
        control_layout.addStretch()

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setWidget(scroll_content)
        control_panel_layout.addWidget(control_scroll, stretch=1)

        self.lbl_status = QLabel("System Ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            "color: #4daaf2; font-weight: bold; padding: 10px; "
            "background-color: #252526; border-radius: 6px;"
        )
        self.pi_stage_widget.status_update.connect(self.lbl_status.setText)
        control_panel_layout.addWidget(self.lbl_status)

        main_layout.addWidget(control_panel)

        feed_panel = QFrame()
        feed_panel.setStyleSheet("background-color: #000000; border: 2px solid #3a3a3a; border-radius: 8px;")
        feed_layout = QVBoxLayout(feed_panel)
        feed_layout.setContentsMargins(2, 2, 2, 2)

        self.video_label = QLabel("Video Feed Offline")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("color: #555555; font-size: 24px; font-weight: bold; border: none;")
        feed_layout.addWidget(self.video_label)

        main_layout.addWidget(feed_panel, stretch=1)

    def _toggle_awg(self, expanded):
        self.awg_widget.setVisible(expanded)
        arrow = "▾" if expanded else "▸"
        self.btn_awg_toggle.setText(f"{arrow}  Siglent AWG Control (LC Speckle)")

    # ==========================
    # Tab construction
    # ==========================
    def _build_evk4_tab(self):
        self.tab_evk4 = QWidget()
        evk4_layout = QVBoxLayout(self.tab_evk4)

        # 1. Live controls — always visible at the top so the user can start
        #    watching events immediately before touching any other setting.
        action_group = QGroupBox("EVK4 Controls")
        action_layout = QVBoxLayout()
        self.btn_live = QPushButton("▶  Start Live Mode")
        self.btn_live.setObjectName("btnLive")
        self.btn_live.clicked.connect(lambda: self.start_evk4("live"))
        self.btn_acquire = QPushButton("⬤  Start Acquisition")
        self.btn_acquire.setObjectName("btnAcquire")
        self.btn_acquire.clicked.connect(lambda: self.start_evk4("acquire"))
        self.btn_stop = QPushButton("■  Stop EVK4")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.clicked.connect(self.stop_evk4)
        self.btn_stop.setEnabled(False)
        action_layout.addWidget(self.btn_live)
        action_layout.addWidget(self.btn_acquire)
        action_layout.addWidget(self.btn_stop)
        self.lbl_evk4_time = QLabel()
        self.lbl_evk4_time.setStyleSheet("color: #4daaf2; font-size: 11px; font-weight: bold;")
        action_layout.addWidget(self.lbl_evk4_time)
        action_group.setLayout(action_layout)
        evk4_layout.addWidget(action_group)

        # 2. Camera parameters — adjust biases while watching the live feed,
        #    then click "Apply Biases to Live Feed" to push them to hardware.
        self.evk4_params = Evk4ParamsWidget()
        self.evk4_params.spin_time.valueChanged.connect(self._update_evk4_time)
        evk4_layout.addWidget(self.evk4_params)

        # 3. Output settings — set before starting an acquisition.
        out_group = QGroupBox("EVK4 Output")
        out_layout = QVBoxLayout()
        out_layout.addWidget(QLabel("Output Directory:"))
        dir_layout = QHBoxLayout()
        self.txt_dir = QLineEdit()
        btn_dir = QPushButton("Browse")
        btn_dir.clicked.connect(self.browse_dir)
        dir_layout.addWidget(self.txt_dir); dir_layout.addWidget(btn_dir)
        out_layout.addLayout(dir_layout)
        out_layout.addWidget(QLabel("Filename Base:"))
        self.txt_filename = QLineEdit(f"recording_{time.strftime('%y%m%d_%H%M%S')}")
        out_layout.addWidget(self.txt_filename)
        self.chk_evk4_raw = QCheckBox("Save raw event recording (.raw)")
        self.chk_evk4_raw.setChecked(False)
        out_layout.addWidget(self.chk_evk4_raw)
        out_group.setLayout(out_layout)
        evk4_layout.addWidget(out_group)

        evk4_layout.addStretch()
        self.tabs.addTab(self.tab_evk4, "Event Camera (2D)")

    def _build_orca_tab(self):
        self.tab_orca = QWidget()
        orca_layout = QVBoxLayout(self.tab_orca)

        # 1. Live controls at the top — start the live feed first to focus the
        #    sample, then scroll down to adjust parameters and click Apply.
        orca_actions_group = QGroupBox("ORCA Controls")
        orca_actions_layout = QHBoxLayout()
        self.btn_orca_live = QPushButton("▶  Start Live Focus Mode")
        self.btn_orca_live.setObjectName("btnLive")
        self.btn_orca_live.clicked.connect(lambda: self.start_orca("live"))
        self.btn_orca_stop = QPushButton("■  Stop ORCA")
        self.btn_orca_stop.setObjectName("btnStop")
        self.btn_orca_stop.clicked.connect(self.stop_orca)
        self.btn_orca_stop.setEnabled(False)
        orca_actions_layout.addWidget(self.btn_orca_live)
        orca_actions_layout.addWidget(self.btn_orca_stop)
        orca_actions_group.setLayout(orca_actions_layout)
        orca_layout.addWidget(orca_actions_group)

        # 2. Camera parameters — tune exposure, mode and ROI while watching live,
        #    then click "Apply All Settings to Live Feed" (at the bottom of the
        #    widget) to push them to the camera without restarting.
        self.orca_params = OrcaParamsWidget()
        self.orca_params.spin_frames.valueChanged.connect(self._update_orca_time)
        self.orca_params.spin_exp.valueChanged.connect(self._update_orca_time)
        self.orca_params.spin_roi_height.valueChanged.connect(self._update_orca_time)
        self.orca_params.combo_readout.currentIndexChanged.connect(self._update_orca_time)
        orca_layout.addWidget(self.orca_params)

        # 3. DSI acquisition — set output path, then acquire when ready.
        orca_dsi_group = QGroupBox("DSI Acquisition (Single Z-Plane)")
        orca_dsi_layout = QVBoxLayout()
        orca_dsi_layout.addWidget(QLabel("Output Directory:"))
        orca_dir_layout = QHBoxLayout()
        self.txt_orca_dir = QLineEdit()
        btn_orca_dir = QPushButton("Browse")
        btn_orca_dir.clicked.connect(self.browse_orca_dir)
        orca_dir_layout.addWidget(self.txt_orca_dir); orca_dir_layout.addWidget(btn_orca_dir)
        orca_dsi_layout.addLayout(orca_dir_layout)
        orca_dsi_layout.addWidget(QLabel("Filename Base:"))
        self.txt_orca_filename = QLineEdit(f"dsi_{time.strftime('%y%m%d_%H%M%S')}")
        orca_dsi_layout.addWidget(self.txt_orca_filename)
        self.chk_orca_raw = QCheckBox("Save raw 16-bit speckle stack (3D TIFF)")
        self.chk_orca_raw.setChecked(True)
        orca_dsi_layout.addWidget(self.chk_orca_raw)
        self.btn_orca_acquire = QPushButton("⬤  Start DSI Acquisition")
        self.btn_orca_acquire.setObjectName("btnAcquire")
        self.btn_orca_acquire.clicked.connect(lambda: self.start_orca("acquire"))
        orca_dsi_layout.addWidget(self.btn_orca_acquire)
        self.lbl_orca_time = QLabel()
        self.lbl_orca_time.setStyleSheet("color: #4daaf2; font-size: 11px; font-weight: bold;")
        orca_dsi_layout.addWidget(self.lbl_orca_time)
        lbl_dsi_info = QLabel(
            "Records N frames at the current focus, then computes the average "
            "(widefield) and standard-deviation (optically-sectioned DSI) images. "
            "Tune the speckle decorrelation time (AWG) close to or slightly slower "
            "than texp."
        )
        lbl_dsi_info.setWordWrap(True)
        lbl_dsi_info.setStyleSheet("color: #888888; font-size: 11px;")
        orca_dsi_layout.addWidget(lbl_dsi_info)
        orca_dsi_group.setLayout(orca_dsi_layout)
        orca_layout.addWidget(orca_dsi_group)

        orca_layout.addStretch()
        self.tabs.addTab(self.tab_orca, "Scientific Camera (ORCA)")

    def _build_zstack_tab(self):
        self.tab_zstack = QWidget()
        zstack_layout = QVBoxLayout(self.tab_zstack)

        # 1. Live / Stop always at the top — start live first so you can watch
        #    the sample while moving the stage to the starting focal plane.
        live_group = QGroupBox("Live Preview")
        live_layout = QHBoxLayout()
        self.btn_zstack_live = QPushButton("▶  Start Live (selected camera)")
        self.btn_zstack_live.setObjectName("btnLive")
        self.btn_zstack_live.clicked.connect(self.start_zstack_live)
        self.btn_stop_zstack = QPushButton("■  Stop")
        self.btn_stop_zstack.setObjectName("btnStop")
        self.btn_stop_zstack.clicked.connect(self.stop_zstack)
        self.btn_stop_zstack.setEnabled(False)
        live_layout.addWidget(self.btn_zstack_live)
        live_layout.addWidget(self.btn_stop_zstack)
        live_group.setLayout(live_layout)
        zstack_layout.addWidget(live_group)

        # 2. PI stage — position the sample while watching the live feed above.
        self.pi_stage_widget = PIStageWidget()
        zstack_layout.addWidget(self.pi_stage_widget)

        # 3. Acquisition group — select camera, set parameters and output path,
        #    then launch the stack once the sample is positioned correctly.
        auto_z_group = QGroupBox("Automated 3D DSI Acquisition")
        auto_z_layout = QVBoxLayout()

        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Acquisition Camera:"))
        self.combo_zstack_camera = QComboBox()
        self.combo_zstack_camera.addItem("Scientific Camera (ORCA)", "orca")
        self.combo_zstack_camera.addItem("Event Camera (EVK4)", "event")
        self.combo_zstack_camera.currentIndexChanged.connect(self._on_zstack_camera_changed)
        cam_layout.addWidget(self.combo_zstack_camera, stretch=1)
        auto_z_layout.addLayout(cam_layout)

        # Full, independent parameter controls for each camera; only the selected
        # camera's controls are shown.
        self.zstack_orca_params = OrcaParamsWidget()
        self.zstack_evk4_params = Evk4ParamsWidget()
        self.zstack_evk4_params.setVisible(False)
        auto_z_layout.addWidget(self.zstack_orca_params)
        auto_z_layout.addWidget(self.zstack_evk4_params)

        # Keep the Z-Stack time label updated whenever any relevant parameter changes.
        for sig in (
            self.zstack_orca_params.spin_frames.valueChanged,
            self.zstack_orca_params.spin_exp.valueChanged,
            self.zstack_orca_params.spin_roi_height.valueChanged,
            self.zstack_orca_params.combo_readout.currentIndexChanged,
            self.zstack_evk4_params.spin_time.valueChanged,
        ):
            sig.connect(self._update_zstack_time)
        self.pi_stage_widget.spin_steps.valueChanged.connect(self._update_zstack_time)
        self.combo_zstack_camera.currentIndexChanged.connect(self._update_zstack_time)

        auto_z_layout.addWidget(QLabel("Output Directory:"))
        zstack_dir_layout = QHBoxLayout()
        self.txt_zstack_dir = QLineEdit()
        btn_zstack_dir = QPushButton("Browse")
        btn_zstack_dir.clicked.connect(self.browse_zstack_dir)
        zstack_dir_layout.addWidget(self.txt_zstack_dir); zstack_dir_layout.addWidget(btn_zstack_dir)
        auto_z_layout.addLayout(zstack_dir_layout)

        auto_z_layout.addWidget(QLabel("Filename Base:"))
        self.txt_zstack_filename = QLineEdit(f"zstack_{time.strftime('%y%m%d_%H%M%S')}")
        auto_z_layout.addWidget(self.txt_zstack_filename)

        # Raw data (ORCA: all planes' speckle frames in one multi-page TIFF for
        # the MATLAB RIM algorithm / EVK4: one .raw event file per plane).
        self.chk_zstack_raw = QCheckBox("Save raw data (for RIM / re-processing)")
        self.chk_zstack_raw.setChecked(True)
        auto_z_layout.addWidget(self.chk_zstack_raw)

        lbl_zstack_info = QLabel(
            "Moves through Z and acquires with the selected camera at each plane, "
            "saving the per-plane sectioned images as a 3D TIFF depth volume plus a "
            "parameter log."
        )
        lbl_zstack_info.setWordWrap(True)
        lbl_zstack_info.setStyleSheet("color: #888888; font-size: 11px;")
        auto_z_layout.addWidget(lbl_zstack_info)

        self.btn_start_zstack = QPushButton("⬤  Start Z-Stack Acquisition")
        self.btn_start_zstack.setObjectName("btnAcquire")
        self.btn_start_zstack.clicked.connect(self.start_zstack)
        auto_z_layout.addWidget(self.btn_start_zstack)
        self.lbl_zstack_time = QLabel()
        self.lbl_zstack_time.setStyleSheet("color: #4daaf2; font-size: 11px; font-weight: bold;")
        auto_z_layout.addWidget(self.lbl_zstack_time)

        auto_z_group.setLayout(auto_z_layout)
        zstack_layout.addWidget(auto_z_group)
        zstack_layout.addStretch()

        self.tabs.addTab(self.tab_zstack, "3D Z-Stack")

    def _on_zstack_camera_changed(self):
        camera = self.combo_zstack_camera.currentData()
        self.zstack_orca_params.setVisible(camera == "orca")
        self.zstack_evk4_params.setVisible(camera == "event")
        self._update_zstack_time()

    # ==========================
    # Acquisition time estimates
    # ==========================
    def _update_evk4_time(self):
        t = self.evk4_params.spin_time.value()
        self.lbl_evk4_time.setText(f"Duration: {t} s")

    def _update_orca_time(self):
        n = self.orca_params.spin_frames.value()
        frame_s = self.orca_params.estimated_frame_time_s()
        total_s = n * frame_s
        fps = 1.0 / frame_s
        self.lbl_orca_time.setText(
            f"≈ {total_s:.0f} s  ({n} frames at ≈ {fps:.0f} fps)"
        )

    def _update_zstack_time(self):
        steps = self.pi_stage_widget.spin_steps.value()
        camera = self.combo_zstack_camera.currentData()
        if camera == "orca":
            n = self.zstack_orca_params.spin_frames.value()
            frame_s = self.zstack_orca_params.estimated_frame_time_s()
            plane_s = n * frame_s
        else:
            plane_s = self.zstack_evk4_params.spin_time.value()
        overhead_s = 1.0  # rough: ~0.5 s motor move + 0.5 s mechanical settle
        total_s = steps * (plane_s + overhead_s)
        mins, secs = divmod(int(round(total_s)), 60)
        if mins > 0:
            self.lbl_zstack_time.setText(
                f"≈ {mins} min {secs:02d} s  "
                f"({steps} planes × {plane_s + overhead_s:.1f} s/plane)"
            )
        else:
            self.lbl_zstack_time.setText(
                f"≈ {secs} s  ({steps} planes × {plane_s + overhead_s:.1f} s/plane)"
            )

    # ==========================
    # Lifecycle / teardown
    # ==========================
    def closeEvent(self, event):
        if self.evk4_worker is not None and self.evk4_worker.isRunning():
            self.evk4_worker.stop()
            self.evk4_worker.wait(2000)

        if self.orca_worker is not None and self.orca_worker.isRunning():
            self.orca_worker.stop()
            self.orca_worker.wait(2000)

        if self.zstack_worker is not None and self.zstack_worker.isRunning():
            self.zstack_worker.stop()
            self.zstack_worker.wait(2000)

        if self.zstack_live_worker is not None and self.zstack_live_worker.isRunning():
            self.zstack_live_worker.stop()
            self.zstack_live_worker.wait(2000)

        self.awg_widget.close_device()
        self.pi_stage_widget.close_device()
        event.accept()

    # ==========================
    # Parameter presets
    # ==========================
    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameter Preset", "", "DSI Preset (*.json)"
        )
        if not path:
            return
        preset = {
            "version": 1,
            "evk4": self.evk4_params.get_preset(),
            "orca": self.orca_params.get_preset(),
            "zstack": {
                "camera": self.combo_zstack_camera.currentData(),
                "step_size": self.pi_stage_widget.spin_step_size.value(),
                "num_steps": self.pi_stage_widget.spin_steps.value(),
            },
            "save_raw": {
                "evk4": self.chk_evk4_raw.isChecked(),
                "orca": self.chk_orca_raw.isChecked(),
                "zstack": self.chk_zstack_raw.isChecked(),
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset, f, indent=2)
            self.lbl_status.setText(f"Preset saved: {path}")
        except OSError as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameter Preset", "", "DSI Preset (*.json)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                preset = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Load Failed", f"Could not read preset:\n{e}")
            return

        if preset.get("version") != 1:
            QMessageBox.warning(
                self, "Incompatible Preset",
                "This file was saved by a different version of the software and cannot be loaded."
            )
            return

        if "evk4" in preset:
            self.evk4_params.set_preset(preset["evk4"])
            self.zstack_evk4_params.set_preset(preset["evk4"])

        if "orca" in preset:
            self.orca_params.set_preset(preset["orca"])
            self.zstack_orca_params.set_preset(preset["orca"])

        if "zstack" in preset:
            z = preset["zstack"]
            if "camera" in z:
                idx = self.combo_zstack_camera.findData(z["camera"])
                if idx >= 0:
                    self.combo_zstack_camera.setCurrentIndex(idx)
            if "step_size" in z:
                self.pi_stage_widget.spin_step_size.setValue(float(z["step_size"]))
            if "num_steps" in z:
                self.pi_stage_widget.spin_steps.setValue(int(z["num_steps"]))

        if "save_raw" in preset:
            sr = preset["save_raw"]
            if "evk4" in sr:
                self.chk_evk4_raw.setChecked(bool(sr["evk4"]))
            if "orca" in sr:
                self.chk_orca_raw.setChecked(bool(sr["orca"]))
            if "zstack" in sr:
                self.chk_zstack_raw.setChecked(bool(sr["zstack"]))

        self.lbl_status.setText(f"Preset loaded: {path}")

    def browse_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.txt_dir.setText(folder)

    def browse_orca_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.txt_orca_dir.setText(folder)

    def browse_zstack_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.txt_zstack_dir.setText(folder)

    # ==========================
    # Parameter collection
    # ==========================
    def collect_acquisition_metadata(self, source, output_dir, filename, evk4_widget, orca_widget):
        """Gather the full UI state for the per-acquisition parameter log.

        Both cameras' parameters are read from the widgets relevant to this
        acquisition, alongside illumination and stage state; ``source`` records
        which acquisition triggered the log.
        """
        e = evk4_widget.get_params()
        o = orca_widget.get_params()
        modes = orca_widget.mode_labels()
        roi = o["orca_roi"]
        return {
            "Acquisition": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": source,
                "filename_base": filename,
                "output_directory": output_dir,
            },
            "Event Camera (Prophesee EVK4)": {
                "bias_fo": e["bias_fo"],
                "bias_hpf": e["bias_hpf"],
                "bias_on": e["bias_on"],
                "bias_off": e["bias_off"],
                "acquisition_time_s": e["acqu_time"],
                "filter_crazy_pixels": e["filter_crazy_pixels"],
                "apply_smoothing": e["apply_smoothing"],
            },
            "Scientific Camera (Hamamatsu ORCA-Fusion)": {
                "exposure_time_ms": o["orca_exposure"],
                "frames_per_stack_N": o["orca_frames"],
                "readout_speed": modes["readout_speed"],
                "binning": modes["binning"],
                "trigger_source": modes["trigger_source"],
                "trigger_mode": modes["trigger_mode"],
                "defect_correction": modes["defect_correction"],
                "roi_x_min": roi["x_min"],
                "roi_x_max": roi["x_max"],
                "roi_y_min": roi["y_min"],
                "roi_y_max": roi["y_max"],
            },
            "Illumination (Siglent AWG / LC speckle)": self.awg_widget.get_settings(),
            "PI Stage": {
                "position_unit": self.pi_stage_widget.controller.unit,
                "target_focus": self.pi_stage_widget.spin_focus.value(),
                "step_size": self.pi_stage_widget.spin_step_size.value(),
                "num_steps": self.pi_stage_widget.spin_steps.value(),
            },
        }

    def get_motor_params(self):
        return {
            "focus": self.pi_stage_widget.spin_focus.value(),
            "step_size": self.pi_stage_widget.spin_step_size.value(),
            "steps": self.pi_stage_widget.spin_steps.value(),
        }

    # ==========================
    # EVK4 Logic
    # ==========================
    def start_evk4(self, mode):
        if not METAVISION_AVAILABLE:
            QMessageBox.warning(
                self, "Event Camera Unavailable",
                "The Prophesee Metavision SDK was not found.\n\n"
                "Install the Metavision SDK for your Python version and ensure "
                "all its dependencies (including h5py) are installed in the same environment.\n\n"
                "See README.md for setup instructions.",
            )
            return
        params = self.evk4_params.get_params()
        params["output_dir"] = self.txt_dir.text()
        params["filename"] = self.txt_filename.text()
        params["save_raw"] = self.chk_evk4_raw.isChecked()
        if mode == "acquire" and not params["output_dir"]:
            QMessageBox.warning(self, "Missing Parameter", "Please select an output directory for the acquisition.")
            return

        if mode == "acquire":
            params["metadata"] = self.collect_acquisition_metadata(
                "Event Camera (EVK4) - 2D event-DSI", params["output_dir"], params["filename"],
                self.evk4_params, self.orca_params,
            )

        self.btn_live.setEnabled(False)
        self.btn_acquire.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.evk4_worker = CameraWorker(mode, params)
        self.evk4_worker.frame_ready.connect(self.update_image)
        self.evk4_worker.status_update.connect(self.lbl_status.setText)
        self.evk4_worker.error_signal.connect(self.show_error)
        self.evk4_worker.finished_signal.connect(self.on_evk4_finished)
        self.evk4_worker.start()

        if mode == "live":
            self.evk4_params.btn_apply_biases.setEnabled(True)
            self.evk4_params.btn_apply_biases.clicked.connect(self._apply_evk4_biases_live)

    def stop_evk4(self):
        if self.evk4_worker is not None:
            self.evk4_worker.stop()
            self.lbl_status.setText("Stopping EVK4...")
            self.evk4_worker.wait(3000)

    def on_evk4_finished(self):
        self.btn_live.setEnabled(True)
        self.btn_acquire.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.txt_filename.setText(f"recording_{time.strftime('%y%m%d_%H%M%S')}")
        self.evk4_params.btn_apply_biases.setEnabled(False)
        try:
            self.evk4_params.btn_apply_biases.clicked.disconnect(self._apply_evk4_biases_live)
        except (RuntimeError, TypeError):
            pass

    def _apply_evk4_biases_live(self):
        if self.evk4_worker is not None and self.evk4_worker.isRunning():
            self.evk4_worker.apply_biases(self.evk4_params.get_params())

    # ==========================
    # ORCA Logic
    # ==========================
    def start_orca(self, mode):
        if not DCAM_AVAILABLE:
            QMessageBox.warning(
                self, "ORCA Camera Unavailable",
                "The Hamamatsu DCAM API (dcam.py) was not found.\n\n"
                "Download and install the DCAM-API SDK from Hamamatsu, then set the "
                "HAMAMATSU_SDK_PATH environment variable to the folder containing dcam.py "
                "(e.g. …\\dcamsdk4\\samples\\python).\n\n"
                "Open a new terminal after running 'setx' and restart the app.\n\n"
                "See README.md for full setup instructions.",
            )
            return
        params = self.orca_params.get_params()
        params["save_raw_stack"] = self.chk_orca_raw.isChecked()
        params["output_dir"] = self.txt_orca_dir.text()
        params["filename"] = self.txt_orca_filename.text()
        if mode == "acquire" and not params["output_dir"]:
            QMessageBox.warning(self, "Missing Parameter", "Please select an output directory for the DSI acquisition.")
            return

        if mode == "acquire":
            params["metadata"] = self.collect_acquisition_metadata(
                "Scientific Camera (ORCA) - Single-Z DSI", params["output_dir"], params["filename"],
                self.evk4_params, self.orca_params,
            )

        self.btn_orca_live.setEnabled(False)
        self.btn_orca_acquire.setEnabled(False)
        self.btn_orca_stop.setEnabled(True)

        self.orca_worker = OrcaWorker(mode, params)
        self.orca_worker.image_ready.connect(self.update_image)
        self.orca_worker.status_update.connect(self.lbl_status.setText)
        self.orca_worker.error_signal.connect(self.show_error)
        self.orca_worker.finished_signal.connect(self.on_orca_finished)
        self.orca_worker.start()

        if mode == "live":
            self.orca_params.btn_apply_live.setEnabled(True)
            self.orca_params.btn_apply_live.clicked.connect(self._apply_orca_params_live)

    def stop_orca(self):
        if self.orca_worker is not None:
            self.orca_worker.stop()
            self.orca_worker.wait(3000)

    def on_orca_finished(self):
        self.btn_orca_live.setEnabled(True)
        self.btn_orca_acquire.setEnabled(True)
        self.btn_orca_stop.setEnabled(False)
        self.txt_orca_filename.setText(f"dsi_{time.strftime('%y%m%d_%H%M%S')}")
        self.orca_params.btn_apply_live.setEnabled(False)
        try:
            self.orca_params.btn_apply_live.clicked.disconnect(self._apply_orca_params_live)
        except (RuntimeError, TypeError):
            pass

    def _apply_orca_params_live(self):
        if self.orca_worker is not None and self.orca_worker.isRunning():
            self.orca_worker.apply_params(self.orca_params.get_params())

    # ==========================
    # Automated Z-Stack Logic
    # ==========================
    def _set_zstack_busy(self, busy):
        self.btn_zstack_live.setEnabled(not busy)
        self.btn_start_zstack.setEnabled(not busy)
        self.btn_stop_zstack.setEnabled(busy)

    def start_zstack_live(self):
        """Live preview with the selected camera, for focusing before a stack."""
        camera = self.combo_zstack_camera.currentData()
        if camera == "orca" and not DCAM_AVAILABLE:
            QMessageBox.warning(self, "ORCA Camera Unavailable",
                "The Hamamatsu DCAM API was not found. See README.md for setup instructions.")
            return
        if camera == "event" and not METAVISION_AVAILABLE:
            QMessageBox.warning(self, "Event Camera Unavailable",
                "The Prophesee Metavision SDK was not found. See README.md for setup instructions.")
            return
        self._set_zstack_busy(True)

        if camera == "orca":
            self.zstack_live_worker = OrcaWorker("live", self.zstack_orca_params.get_params())
            self.zstack_live_worker.image_ready.connect(self.update_image)
            self.zstack_orca_params.btn_apply_live.setEnabled(True)
            self.zstack_orca_params.btn_apply_live.clicked.connect(self._apply_zstack_orca_params_live)
        else:
            self.zstack_live_worker = CameraWorker("live", self.zstack_evk4_params.get_params())
            self.zstack_live_worker.frame_ready.connect(self.update_image)
            self.zstack_evk4_params.btn_apply_biases.setEnabled(True)
            self.zstack_evk4_params.btn_apply_biases.clicked.connect(self._apply_zstack_evk4_biases_live)

        self.zstack_live_worker.status_update.connect(self.lbl_status.setText)
        self.zstack_live_worker.error_signal.connect(self.show_error)
        self.zstack_live_worker.finished_signal.connect(self.on_zstack_finished)
        self.zstack_live_worker.start()

    def start_zstack(self):
        camera = self.combo_zstack_camera.currentData()
        if camera == "orca" and not DCAM_AVAILABLE:
            QMessageBox.warning(self, "ORCA Camera Unavailable",
                "The Hamamatsu DCAM API was not found. See README.md for setup instructions.")
            return
        if camera == "event" and not METAVISION_AVAILABLE:
            QMessageBox.warning(self, "Event Camera Unavailable",
                "The Prophesee Metavision SDK was not found. See README.md for setup instructions.")
            return
        if not self.pi_stage_widget.pidevice:
            QMessageBox.warning(self, "Connection Error", "Please connect the PI Stage before starting a Z-Stack.")
            return

        output_dir = self.txt_zstack_dir.text()
        if not output_dir:
            QMessageBox.warning(self, "Missing Parameter", "Please select an output directory for the Z-Stack.")
            return
        filename = self.txt_zstack_filename.text()
        camera = self.combo_zstack_camera.currentData()
        source = (
            "3D Z-Stack (ORCA) - per-plane DSI" if camera == "orca"
            else "3D Z-Stack (EVK4) - per-plane event-DSI"
        )

        save_params = {
            "output_dir": output_dir,
            "filename": filename,
            "save_raw": self.chk_zstack_raw.isChecked(),
            "metadata": self.collect_acquisition_metadata(
                source, output_dir, filename, self.zstack_evk4_params, self.zstack_orca_params
            ),
        }

        self._set_zstack_busy(True)

        # The orchestrator thread now owns the stage; pause the widget's idle
        # polling so the two don't query the GCS link concurrently.
        self.pi_stage_widget.pause_position_updates()

        self.zstack_worker = AutomatedZStackWorker(
            self.pi_stage_widget.pidevice,
            self.pi_stage_widget.axis,
            self.get_motor_params(),
            self.zstack_orca_params.get_params(),
            save_params,
            camera=camera,
            evk4_params=self.zstack_evk4_params.get_params(),
        )
        self.zstack_worker.image_ready.connect(self.update_image)
        self.zstack_worker.status_update.connect(self.lbl_status.setText)
        self.zstack_worker.z_profile_update.connect(self.handle_z_profile)
        self.zstack_worker.position_update.connect(self.pi_stage_widget.show_position)
        self.zstack_worker.error_signal.connect(self.show_error)
        self.zstack_worker.finished_signal.connect(self.on_zstack_finished)
        self.zstack_worker.start()

    def stop_zstack(self):
        if self.zstack_worker is not None and self.zstack_worker.isRunning():
            self.zstack_worker.stop()
            self.lbl_status.setText("Aborting Z-Stack...")
        if self.zstack_live_worker is not None and self.zstack_live_worker.isRunning():
            self.zstack_live_worker.stop()
            self.zstack_live_worker.wait(3000)

    def handle_z_profile(self, z_val, step_num):
        print(f"Step {step_num} | Computed Z-Profile value: {z_val}")

    def on_zstack_finished(self):
        self._set_zstack_busy(False)
        self.pi_stage_widget.resume_position_updates()
        self.txt_zstack_filename.setText(f"zstack_{time.strftime('%y%m%d_%H%M%S')}")
        self.zstack_orca_params.btn_apply_live.setEnabled(False)
        self.zstack_evk4_params.btn_apply_biases.setEnabled(False)
        for slot in (self._apply_zstack_orca_params_live, self._apply_zstack_evk4_biases_live):
            try:
                self.zstack_orca_params.btn_apply_live.clicked.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
            try:
                self.zstack_evk4_params.btn_apply_biases.clicked.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _apply_zstack_orca_params_live(self):
        if self.zstack_live_worker is not None and self.zstack_live_worker.isRunning():
            self.zstack_live_worker.apply_params(self.zstack_orca_params.get_params())

    def _apply_zstack_evk4_biases_live(self):
        if self.zstack_live_worker is not None and self.zstack_live_worker.isRunning():
            self.zstack_live_worker.apply_biases(self.zstack_evk4_params.get_params())

    # ==========================
    # General UI Handling
    # ==========================
    def show_error(self, err_msg):
        QMessageBox.critical(self, "System Error", err_msg)
        self.on_evk4_finished()
        self.on_orca_finished()
        self.on_zstack_finished()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        self._current_frame = cv_img
        # PyQt6 requires bytes, not memoryview; ascontiguousarray packs the
        # ROI-cropped slice before tobytes() serialises it row-by-row.
        img = np.ascontiguousarray(cv_img)
        if len(img.shape) == 3:
            h, w, ch = img.shape
            qt_img = QImage(img.tobytes(), w, h, ch * w, QImage.Format.Format_BGR888)
        else:
            h, w = img.shape
            qt_img = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        )
