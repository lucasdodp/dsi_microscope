"""Main application window: layout, tab management, worker wiring and feed display.

This is the only place that instantiates the hardware workers and the Z-stack
orchestrator, routing every worker's status_update / error_signal back to the single
`lbl_status` bar and managing button enable/disable state during acquisition.
"""

import json
import os
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QTabWidget,
    QVBoxLayout, QWidget,
)

from config import (
    ACQUISITION_HISTORY_MAX, ACQUISITION_HISTORY_PATH, EVK4_PLANE_OVERHEAD_S,
    ORCA_CAMERA_INIT_S, ORCA_PLANE_OVERHEAD_S, ORCA_SENSOR_HEIGHT, ORCA_SENSOR_WIDTH,
    SESSION_STATE_PATH,
)
from hardware.event_camera import CameraWorker, METAVISION_AVAILABLE
from hardware.orca_camera import OrcaWorker, DCAM_AVAILABLE
from ui.orchestrator import AutomatedZStackWorker
from ui.widgets import (
    AWGWidget, Evk4ParamsWidget, OrcaParamsWidget, PIStageWidget, VideoFeedLabel,
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
        self._current_frame = None   # last frame shown (image pixels), for the crop tool
        self._crop_region = None     # pending crop selection (x, y, w, h) in frame px
        # Remembers the last auto-generated timestamp default for each filename
        # field, so the field is only re-stamped while it still holds that default
        # (a name the user typed themselves is never overwritten).
        self._fn_defaults = {}

        # Live elapsed-time readout: while an acquisition runs, the relevant tab's
        # time label shows "Elapsed: mm:ss" (ground truth vs the rough estimate).
        self._acq_timer = QTimer(self)
        self._acq_timer.setInterval(1000)
        self._acq_timer.timeout.connect(self._tick_elapsed)
        self._acq_start = None      # time.time() when the acquisition started
        self._acq_label = None      # QLabel to update with the elapsed time
        self._acq_restore = None    # callable that restores the estimate text

        # Acquisition-time learning: each completed run's actual elapsed time is
        # recorded and used to calibrate future estimates (per acquisition type).
        self._acq_record = None     # context of the running acquisition (type, predicted_s, …)
        self._acq_aborted = False   # set if the current run was stopped / errored (don't learn from it)
        self._acq_history = self._load_acq_history()  # {type: [{predicted_s, actual_s, …}]}

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

        self.video_label = VideoFeedLabel("Video Feed Offline")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("color: #555555; font-size: 24px; font-weight: bold; border: none;")
        self.video_label.region_drawn.connect(self._on_crop_region_drawn)
        feed_layout.addWidget(self.video_label, stretch=1)

        feed_layout.addWidget(self._build_crop_bar())

        main_layout.addWidget(feed_panel, stretch=1)

        # Restore the parameters saved when the app was last closed, so a session
        # starts where the previous one left off (silent if there is no state file).
        self._load_session()

        # Persist parameters continuously, so the last session is remembered no
        # matter how the app exits — a clean window close, an app quit, or an
        # abrupt stop (e.g. the IDE's stop button) that skips closeEvent. A
        # periodic autosave backs up the save-on-close, and writes are atomic.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._save_session)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(15000)  # 15 s; the file is tiny
        self._autosave_timer.timeout.connect(self._save_session)
        self._autosave_timer.start()

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
        self.txt_filename = QLineEdit()
        self._stamp_default_filename(self.txt_filename, "recording")
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
        self.orca_params.spin_roi_width.valueChanged.connect(self._update_orca_time)
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
        self.txt_orca_filename = QLineEdit()
        self._stamp_default_filename(self.txt_orca_filename, "dsi")
        orca_dsi_layout.addWidget(self.txt_orca_filename)
        self.chk_orca_raw = QCheckBox("Save raw 16-bit speckle stack (3D TIFF)")
        self.chk_orca_raw.setChecked(True)
        self.chk_orca_raw.stateChanged.connect(self._update_orca_time)
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

        # 1. Camera selection at the very top — the live preview, parameters and
        #    acquisition below all act on the selected camera, so it is more
        #    intuitive to choose it first.
        cam_group = QGroupBox("Acquisition Camera")
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Acquisition Camera:"))
        self.combo_zstack_camera = QComboBox()
        self.combo_zstack_camera.addItem("Scientific Camera (ORCA)", "orca")
        self.combo_zstack_camera.addItem("Event Camera (EVK4)", "event")
        cam_layout.addWidget(self.combo_zstack_camera, stretch=1)
        cam_group.setLayout(cam_layout)
        zstack_layout.addWidget(cam_group)

        # 2. Live / Stop — start live first so you can watch the sample while
        #    moving the stage to the starting focal plane.
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

        # 3. PI stage — position the sample while watching the live feed above.
        self.pi_stage_widget = PIStageWidget()
        zstack_layout.addWidget(self.pi_stage_widget)

        # 4. Acquisition group — set parameters and output path, then launch the
        #    stack (for the camera chosen at the top) once the sample is
        #    positioned correctly.
        auto_z_group = QGroupBox("Automated 3D DSI Acquisition")
        auto_z_layout = QVBoxLayout()

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
            self.zstack_orca_params.spin_roi_width.valueChanged,
            self.zstack_orca_params.combo_readout.currentIndexChanged,
            self.zstack_evk4_params.spin_time.valueChanged,
        ):
            sig.connect(self._update_zstack_time)
        self.pi_stage_widget.spin_steps.valueChanged.connect(self._update_zstack_time)
        # Connected here (not at combo creation) so the slots run only once the
        # per-camera parameter widgets above exist to be shown/hidden.
        self.combo_zstack_camera.currentIndexChanged.connect(self._on_zstack_camera_changed)
        self.combo_zstack_camera.currentIndexChanged.connect(self._update_zstack_time)

        auto_z_layout.addWidget(QLabel("Output Directory:"))
        zstack_dir_layout = QHBoxLayout()
        self.txt_zstack_dir = QLineEdit()
        btn_zstack_dir = QPushButton("Browse")
        btn_zstack_dir.clicked.connect(self.browse_zstack_dir)
        zstack_dir_layout.addWidget(self.txt_zstack_dir); zstack_dir_layout.addWidget(btn_zstack_dir)
        auto_z_layout.addLayout(zstack_dir_layout)

        auto_z_layout.addWidget(QLabel("Filename Base:"))
        self.txt_zstack_filename = QLineEdit()
        self._stamp_default_filename(self.txt_zstack_filename, "zstack")
        auto_z_layout.addWidget(self.txt_zstack_filename)

        # Raw data (ORCA: each plane's speckle frames in its own multi-page TIFF —
        # one file per plane — for the MATLAB RIM algorithm; EVK4: not saved).
        self.chk_zstack_raw = QCheckBox("Save raw speckle stack — ORCA only (for RIM / re-processing)")
        self.chk_zstack_raw.setChecked(True)
        self.chk_zstack_raw.stateChanged.connect(self._update_zstack_time)
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
    def _calibrate(self, predicted_s, acq_type):
        """Scale a cold-start physics estimate by what past runs of this type
        actually took. Returns ``(adjusted_s, n_runs)``.

        The correction is the **median** of the recorded ``actual_s / predicted_s``
        ratios (robust to the odd outlier). With no history it returns the estimate
        unchanged, so the model degrades gracefully on a fresh machine and sharpens
        as runs accumulate.
        """
        records = self._acq_history.get(acq_type, [])
        ratios = sorted(
            r["actual_s"] / r["predicted_s"]
            for r in records
            if r.get("predicted_s", 0) > 0 and r.get("actual_s", 0) > 0
        )
        if not ratios:
            return predicted_s, 0
        mid = len(ratios) // 2
        factor = ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])
        return predicted_s * factor, len(ratios)

    @staticmethod
    def _calib_note(runs):
        """Suffix noting how many runs the estimate was calibrated from."""
        return f"  · calibrated from {runs} run{'s' if runs != 1 else ''}" if runs else ""

    def _evk4_predicted_s(self):
        """Cold-start estimate (s) for a single EVK4 acquisition: the fixed
        recording duration plus device init + post-processing overhead."""
        return EVK4_PLANE_OVERHEAD_S + self.evk4_params.spin_time.value()

    def _update_evk4_time(self):
        if self._acq_label is self.lbl_evk4_time:
            return  # acquisition running: leave the live elapsed readout alone
        t = self.evk4_params.spin_time.value()
        total_s, runs = self._calibrate(self._evk4_predicted_s(), "evk4_single")
        self.lbl_evk4_time.setText(
            f"Estimated: ≈ {self._fmt_dur(total_s)}  ({t} s recording + overhead){self._calib_note(runs)}"
        )

    @staticmethod
    def _fmt_dur(seconds):
        """Format a duration in seconds as a compact human-readable string."""
        total = int(round(seconds))
        m, s = divmod(total, 60)
        if m >= 60:
            h, m = divmod(m, 60)
            return f"{h} h {m:02d} min"
        if m > 0:
            return f"{m} min {s:02d} s"
        return f"{s} s"

    def _orca_predicted_s(self):
        """Cold-start estimate (s) for a single-Z ORCA DSI acquisition: camera
        start-up + capture + DSI reconstruction + (optional) raw-stack write."""
        n = self.orca_params.spin_frames.value()
        frame_s = self.orca_params.estimated_frame_time_s()
        compute_s = self.orca_params.estimated_compute_s(n)
        save_s = self.orca_params.estimated_raw_save_s(n) if self.chk_orca_raw.isChecked() else 0.0
        return ORCA_CAMERA_INIT_S + n * frame_s + compute_s + save_s

    def _update_orca_time(self):
        if self._acq_label is self.lbl_orca_time:
            return  # acquisition running: leave the live elapsed readout alone
        n = self.orca_params.spin_frames.value()
        fps = 1.0 / self.orca_params.estimated_frame_time_s()
        total_s, runs = self._calibrate(self._orca_predicted_s(), "orca_single")
        self.lbl_orca_time.setText(
            f"Estimated: ≈ {self._fmt_dur(total_s)}  ({n} frames @ ≈ {fps:.0f} fps + overhead)"
            f"{self._calib_note(runs)}"
        )

    def _zstack_predicted_s(self):
        """Cold-start estimate (s) for a Z-stack with the selected camera."""
        steps = self.pi_stage_widget.spin_steps.value()
        if self.combo_zstack_camera.currentData() == "orca":
            n = self.zstack_orca_params.spin_frames.value()
            frame_s = self.zstack_orca_params.estimated_frame_time_s()
            compute_s = self.zstack_orca_params.estimated_compute_s(n)
            save_s = (
                self.zstack_orca_params.estimated_raw_save_s(n)
                if self.chk_zstack_raw.isChecked() else 0.0
            )
            # Per plane: motor move + settle, capture, DSI reconstruction, raw write.
            plane_s = ORCA_PLANE_OVERHEAD_S + n * frame_s + compute_s + save_s
            init_s = ORCA_CAMERA_INIT_S
        else:
            # Per plane the EVK4 is re-opened and the events accumulated.
            plane_s = EVK4_PLANE_OVERHEAD_S + self.zstack_evk4_params.spin_time.value()
            init_s = 0.0
        return init_s + steps * plane_s

    def _update_zstack_time(self):
        if self._acq_label is self.lbl_zstack_time:
            return  # acquisition running: leave the live elapsed readout alone
        steps = self.pi_stage_widget.spin_steps.value()
        acq_type = "orca_zstack" if self.combo_zstack_camera.currentData() == "orca" else "evk4_zstack"
        total_s, runs = self._calibrate(self._zstack_predicted_s(), acq_type)
        plane_s = total_s / max(1, steps)
        self.lbl_zstack_time.setText(
            f"Estimated: ≈ {self._fmt_dur(total_s)}  ({steps} planes × ≈ {plane_s:.1f} s/plane)"
            f"{self._calib_note(runs)}"
        )

    # ----- live elapsed timer (ground truth during an acquisition) -----
    def _start_elapsed(self, label, restore):
        """Begin ticking 'Elapsed: mm:ss' in ``label``; ``restore`` is called on
        stop to put the estimate back."""
        self._acq_label = label
        self._acq_restore = restore
        self._acq_start = time.time()
        self._acq_timer.start()
        self._tick_elapsed()

    def _tick_elapsed(self):
        if self._acq_start is None or self._acq_label is None:
            return
        m, s = divmod(int(time.time() - self._acq_start), 60)
        self._acq_label.setText(f"Elapsed: {m:02d}:{s:02d}")

    def _stop_elapsed(self):
        """Stop the elapsed timer, record/save the real elapsed time, and restore
        the (now calibrated) estimate label."""
        if self._acq_start is None:
            return
        actual_s = time.time() - self._acq_start
        self._acq_timer.stop()
        self._acq_start = None
        self._acq_label = None
        restore, self._acq_restore = self._acq_restore, None

        self._record_acquisition(actual_s)
        # Keep the real duration visible alongside the worker's completion message.
        self.lbl_status.setText(self.lbl_status.text() + f"  (elapsed {self._fmt_dur(actual_s)})")
        if restore is not None:
            restore()  # recomputes the estimate, now using the run we just learned from

    # ----- acquisition-time recording + learning -----
    def _begin_acq_record(self, acq_type, predicted_s, planes, frames, out_dir, filename):
        """Mark the start of a timed acquisition so its elapsed time can be saved
        and learned from when it finishes."""
        self._acq_aborted = False
        self._acq_record = {
            "type": acq_type,
            "predicted_s": predicted_s,
            "planes": planes,
            "frames": frames,
            "out_dir": out_dir,
            "filename": filename,
        }

    def _record_acquisition(self, actual_s):
        """Persist the elapsed time of the just-finished acquisition: write it into
        the parameter log next to the data, and (for full runs) add it to the
        learning history so future estimates of this type are calibrated."""
        rec, self._acq_record = self._acq_record, None
        if rec is None:
            return
        self._append_elapsed_to_log(rec, actual_s)
        # Only learn from runs that actually completed — an aborted/errored run's
        # elapsed time is for a partial acquisition and would skew the model.
        if self._acq_aborted or rec.get("predicted_s", 0) <= 0:
            return
        entry = {
            "predicted_s": round(rec["predicted_s"], 2),
            "actual_s": round(actual_s, 2),
            "planes": rec.get("planes"),
            "frames": rec.get("frames"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        history = self._acq_history.setdefault(rec["type"], [])
        history.append(entry)
        del history[:-ACQUISITION_HISTORY_MAX]  # keep only the most recent runs
        self._save_acq_history()

    def _append_elapsed_to_log(self, rec, actual_s):
        """Append the estimated vs actual duration to the acquisition's parameter
        log file, so each dataset carries a permanent record of how long it took."""
        out_dir, filename = rec.get("out_dir"), rec.get("filename")
        if not out_dir or not filename:
            return
        path = os.path.join(out_dir, f"parameters_{filename}.txt")
        if not os.path.exists(path):
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n[Acquisition timing]\n")
                f.write(f"status = {'aborted' if self._acq_aborted else 'completed'}\n")
                f.write(f"estimated_s = {rec.get('predicted_s', 0):.1f}\n")
                f.write(f"actual_elapsed_s = {actual_s:.1f}\n")
                f.write(f"actual_elapsed = {self._fmt_dur(actual_s)}\n")
        except OSError:
            pass

    def _load_acq_history(self):
        """Load the recorded actual/predicted times used to calibrate estimates."""
        try:
            with open(ACQUISITION_HISTORY_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _save_acq_history(self):
        """Persist the learning history atomically (never leaves a corrupt file)."""
        try:
            directory = os.path.dirname(ACQUISITION_HISTORY_PATH)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = ACQUISITION_HISTORY_PATH + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._acq_history, f, indent=2)
            os.replace(tmp_path, ACQUISITION_HISTORY_PATH)
        except Exception:
            pass

    def _stop_worker_silently(self, worker, finished_slot):
        """Stop a running worker and wait for it to release its device, WITHOUT
        firing its finished handler — so a queued ``finished_signal`` from the
        live feed can't reset the UI state of the acquisition we're about to
        start. Returns True if a worker was actually stopped."""
        if worker is None or not worker.isRunning():
            return False
        try:
            worker.finished_signal.disconnect(finished_slot)
        except (TypeError, RuntimeError):
            pass
        worker.stop()
        worker.wait(3000)
        return True

    # ==========================
    # Lifecycle / teardown
    # ==========================
    def closeEvent(self, event):
        # Persist the current parameters so the next launch restores them.
        self._save_session()

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
    def _collect_preset(self):
        """Gather all camera / Z-stack parameters into a JSON-serialisable dict.

        Shared by the manual *Save Preset* button and the automatic session-state
        save on exit, so both write exactly the same structure.
        """
        return {
            "version": 1,
            "evk4": self.evk4_params.get_preset(),
            "orca": self.orca_params.get_preset(),
            "awg": self.awg_widget.get_preset(),
            "zstack": {
                "camera": self.combo_zstack_camera.currentData(),
                "step_size": self.pi_stage_widget.spin_step_size.value(),
                "num_steps": self.pi_stage_widget.spin_steps.value(),
                "focus": self.pi_stage_widget.spin_focus.value(),
            },
            "save_raw": {
                "evk4": self.chk_evk4_raw.isChecked(),
                "orca": self.chk_orca_raw.isChecked(),
                "zstack": self.chk_zstack_raw.isChecked(),
            },
            "output_dirs": {
                "evk4": self.txt_dir.text(),
                "orca": self.txt_orca_dir.text(),
                "zstack": self.txt_zstack_dir.text(),
            },
        }

    def _apply_preset(self, preset):
        """Apply a preset/session dict to every control. Unknown keys are ignored."""
        if "evk4" in preset:
            self.evk4_params.set_preset(preset["evk4"])
            self.zstack_evk4_params.set_preset(preset["evk4"])

        if "orca" in preset:
            self.orca_params.set_preset(preset["orca"])
            self.zstack_orca_params.set_preset(preset["orca"])

        if "awg" in preset:
            self.awg_widget.set_preset(preset["awg"])

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
            if "focus" in z:
                self.pi_stage_widget.spin_focus.setValue(float(z["focus"]))

        if "save_raw" in preset:
            sr = preset["save_raw"]
            if "evk4" in sr:
                self.chk_evk4_raw.setChecked(bool(sr["evk4"]))
            if "orca" in sr:
                self.chk_orca_raw.setChecked(bool(sr["orca"]))
            if "zstack" in sr:
                self.chk_zstack_raw.setChecked(bool(sr["zstack"]))

        # Output directories are restored (the filename bases stay timestamped).
        if "output_dirs" in preset:
            od = preset["output_dirs"]
            if od.get("evk4"):
                self.txt_dir.setText(od["evk4"])
            if od.get("orca"):
                self.txt_orca_dir.setText(od["orca"])
            if od.get("zstack"):
                self.txt_zstack_dir.setText(od["zstack"])

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameter Preset", "", "DSI Preset (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._collect_preset(), f, indent=2)
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

        self._apply_preset(preset)
        self.lbl_status.setText(f"Preset loaded: {path}")

    # ----- automatic session state (restored on the next launch) -----
    def _save_session(self):
        """Persist the current parameters so the next launch starts where this one
        left off.

        Robust by design — parameter persistence must not depend on a clean
        shutdown: this is called periodically, on app quit, and on window close.
        The dict is built *before* any file is touched (so a problem there can't
        truncate the file), and written to a temp file that is atomically renamed
        into place (so an interrupted write can never leave an empty/corrupt
        session file that would silently reset everything to defaults). Every
        failure mode is swallowed so a save attempt can never crash the app.
        """
        try:
            preset = self._collect_preset()
        except Exception:
            return
        try:
            directory = os.path.dirname(SESSION_STATE_PATH)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = SESSION_STATE_PATH + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(preset, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, SESSION_STATE_PATH)  # atomic on Windows + POSIX
        except Exception:
            pass

    def _load_session(self):
        """Restore the parameters saved by the previous session, if any. Silent and
        best-effort — a missing/corrupt file just leaves the defaults in place."""
        try:
            with open(SESSION_STATE_PATH, encoding="utf-8") as f:
                preset = json.load(f)
        except Exception:
            return
        try:
            if isinstance(preset, dict) and preset.get("version") == 1:
                self._apply_preset(preset)
                self.lbl_status.setText("Restored parameters from the last session.")
        except Exception:
            pass

    # ==========================
    # Filename base handling
    # ==========================
    def _stamp_default_filename(self, field, prefix):
        """Fill a filename field with a fresh ``<prefix>_<timestamp>`` default and
        remember it, so a later auto-refresh can tell the default apart from a name
        the user typed."""
        name = f"{prefix}_{time.strftime('%y%m%d_%H%M%S')}"
        field.setText(name)
        self._fn_defaults[field] = name

    def _refresh_default_filename(self, field, prefix):
        """Re-stamp the field with the current time, but only if it still holds the
        previous auto-generated default — a custom name the user typed is left
        untouched. Called at the start of an acquisition so the saved files carry
        the acquisition time (not the app-launch time)."""
        if self._fn_defaults.get(field) == field.text():
            self._stamp_default_filename(field, prefix)

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
        params["save_raw"] = self.chk_evk4_raw.isChecked()
        if mode == "acquire" and not params["output_dir"]:
            QMessageBox.warning(self, "Missing Parameter", "Please select an output directory for the acquisition.")
            return
        if mode == "acquire":
            # Stamp a default filename with the acquisition time; a custom name is kept.
            self._refresh_default_filename(self.txt_filename, "recording")
        params["filename"] = self.txt_filename.text()

        if mode == "acquire":
            params["metadata"] = self.collect_acquisition_metadata(
                "Event Camera (EVK4) - 2D event-DSI", params["output_dir"], params["filename"],
                self.evk4_params, self.orca_params,
            )
            # Auto-stop a running live feed so the user doesn't have to click Stop
            # first; this releases the camera before the acquisition opens it.
            if self._stop_worker_silently(self.evk4_worker, self.on_evk4_finished):
                self._reset_evk4_apply()

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
        else:
            self._begin_acq_record(
                "evk4_single", self._evk4_predicted_s(), planes=1,
                frames=params.get("acqu_time"), out_dir=params["output_dir"],
                filename=params["filename"],
            )
            self._start_elapsed(self.lbl_evk4_time, self._update_evk4_time)

    def stop_evk4(self):
        if self.evk4_worker is not None:
            self._acq_aborted = True  # a stopped run is partial — don't learn its time
            self.evk4_worker.stop()
            self.lbl_status.setText("Stopping EVK4...")
            self.evk4_worker.wait(3000)

    def on_evk4_finished(self):
        self._stop_elapsed()
        self.btn_live.setEnabled(True)
        self.btn_acquire.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # The filename is re-stamped at the start of the next acquisition (if it is
        # still the auto-default), so stopping a live feed never wipes a typed name.
        self._reset_evk4_apply()

    def _reset_evk4_apply(self):
        """Disable + disconnect the EVK4 live bias-apply button."""
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
        if mode == "acquire" and not params["output_dir"]:
            QMessageBox.warning(self, "Missing Parameter", "Please select an output directory for the DSI acquisition.")
            return
        if mode == "acquire":
            # Stamp a default filename with the acquisition time; a custom name is kept.
            self._refresh_default_filename(self.txt_orca_filename, "dsi")
        params["filename"] = self.txt_orca_filename.text()

        if mode == "acquire":
            # Pre-flight memory check (see start_zstack): the frame stack is held
            # in RAM (camera ring buffer + our copy); a high frame count at full
            # sensor can exceed physical memory. Warn rather than freeze.
            roi = params["orca_roi"]
            px = max(1, roi["x_max"] - roi["x_min"]) * max(1, roi["y_max"] - roi["y_min"])
            need = 2 * params["orca_frames"] * px * 2 + 128 * 1024 * 1024
            avail = self._available_memory_bytes()
            if avail is not None and need > 0.85 * avail:
                resp = QMessageBox.warning(
                    self, "Acquisition may exhaust memory",
                    f"This acquisition is estimated to need about {self._format_bytes(need)} of RAM, "
                    f"but only {self._format_bytes(avail)} is currently free.\n\n"
                    "Running it may make the computer unresponsive. Reduce the frame count (N) "
                    "or the ROI size before retrying.\n\nStart anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if resp != QMessageBox.StandardButton.Yes:
                    return

            params["metadata"] = self.collect_acquisition_metadata(
                "Scientific Camera (ORCA) - Single-Z DSI", params["output_dir"], params["filename"],
                self.evk4_params, self.orca_params,
            )
            # Auto-stop a running live feed so the user doesn't have to click Stop
            # first; this releases the camera before the acquisition opens it.
            if self._stop_worker_silently(self.orca_worker, self.on_orca_finished):
                self._reset_orca_live_apply()

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
            # Live re-framing: dragging the ROI size / centre-offset controls
            # re-applies to the live feed automatically (debounced in the widget).
            self.orca_params.roi_changed.connect(self._apply_orca_params_live)
        else:
            self._begin_acq_record(
                "orca_single", self._orca_predicted_s(), planes=1,
                frames=params["orca_frames"], out_dir=params["output_dir"],
                filename=params["filename"],
            )
            self._start_elapsed(self.lbl_orca_time, self._update_orca_time)

    def stop_orca(self):
        if self.orca_worker is not None:
            self._acq_aborted = True  # a stopped run is partial — don't learn its time
            self.orca_worker.stop()
            self.orca_worker.wait(3000)

    def on_orca_finished(self):
        self._stop_elapsed()
        self.btn_orca_live.setEnabled(True)
        self.btn_orca_acquire.setEnabled(True)
        self.btn_orca_stop.setEnabled(False)
        # Filename is re-stamped at the next acquisition start (if still the auto-
        # default), so stopping live focus mode never wipes a typed name.
        self._reset_orca_live_apply()

    def _reset_orca_live_apply(self):
        """Disable + disconnect the ORCA live-apply button and live re-framing."""
        self.orca_params.btn_apply_live.setEnabled(False)
        for sig in (self.orca_params.btn_apply_live.clicked, self.orca_params.roi_changed):
            try:
                sig.disconnect(self._apply_orca_params_live)
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
            self.zstack_orca_params.roi_changed.connect(self._apply_zstack_orca_params_live)
        else:
            self.zstack_live_worker = CameraWorker("live", self.zstack_evk4_params.get_params())
            self.zstack_live_worker.frame_ready.connect(self.update_image)
            self.zstack_evk4_params.btn_apply_biases.setEnabled(True)
            self.zstack_evk4_params.btn_apply_biases.clicked.connect(self._apply_zstack_evk4_biases_live)

        self.zstack_live_worker.status_update.connect(self.lbl_status.setText)
        self.zstack_live_worker.error_signal.connect(self.show_error)
        self.zstack_live_worker.finished_signal.connect(self.on_zstack_finished)
        self.zstack_live_worker.start()

    def _zstack_orca_peak_bytes(self):
        """Rough estimate of the peak RAM an ORCA Z-stack needs, in bytes.

        Dominated by the per-plane frame stack held in RAM (the camera's own
        N-frame ring buffer + our single host copy, both uint16) and the depth
        volumes that accumulate over every plane (average + DSI, float32, plus a
        transient copy when they are written). Deliberately rough — it only needs
        to be good enough to catch a configuration that clearly won't fit.
        """
        p = self.zstack_orca_params.get_params()
        roi = p["orca_roi"]
        w = max(1, roi["x_max"] - roi["x_min"])
        h = max(1, roi["y_max"] - roi["y_min"])
        n = p["orca_frames"]
        z = self.pi_stage_widget.spin_steps.value()
        px = w * h
        frame_stack = 2 * n * px * 2     # SDK ring buffer + host copy (uint16 = 2 B)
        volumes = 3 * z * px * 4         # avg + DSI float32 volumes + a save-time copy
        chunk = 128 * 1024 * 1024        # compute_dsi_images working-set budget
        return frame_stack + volumes + chunk

    @staticmethod
    def _available_memory_bytes():
        """Best-effort free physical memory in bytes, or None if undetermined."""
        try:
            import ctypes

            class _MemStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatusEx()
            stat.dwLength = ctypes.sizeof(_MemStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
        try:
            import os
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return None

    @staticmethod
    def _format_bytes(num):
        """Human-readable byte size (e.g. ``3.4 GB``)."""
        value = float(num)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024.0 or unit == "GB":
                return f"{value:.1f} {unit}"
            value /= 1024.0

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

        # Pre-flight memory check: an ORCA Z-stack holds a whole frame stack in
        # RAM at each plane (the camera ring buffer plus our copy) and grows the
        # depth volumes plane by plane. A high frame count at full sensor can
        # exceed physical memory and swap the machine to a standstill, so warn
        # before launching a run that would not fit.
        if camera == "orca":
            need = self._zstack_orca_peak_bytes()
            avail = self._available_memory_bytes()
            if avail is not None and need > 0.85 * avail:
                resp = QMessageBox.warning(
                    self, "Z-Stack may exhaust memory",
                    f"This Z-Stack is estimated to need about {self._format_bytes(need)} of RAM, "
                    f"but only {self._format_bytes(avail)} is currently free.\n\n"
                    "Running it may make the computer unresponsive. Reduce the frame count "
                    "(N), the ROI size, or the number of Z steps before retrying.\n\n"
                    "Start anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if resp != QMessageBox.StandardButton.Yes:
                    return

        # Stamp a default filename with the acquisition time; a custom name is kept.
        self._refresh_default_filename(self.txt_zstack_filename, "zstack")
        filename = self.txt_zstack_filename.text()
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

        # Auto-stop a running live preview so its camera handle is released before
        # the orchestrator opens the camera; the user no longer needs to click Stop.
        if self._stop_worker_silently(self.zstack_live_worker, self.on_zstack_finished):
            self._reset_zstack_live_apply()

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
        frames = (self.zstack_orca_params.get_params()["orca_frames"] if camera == "orca"
                  else self.zstack_evk4_params.get_params()["acqu_time"])
        self._begin_acq_record(
            "orca_zstack" if camera == "orca" else "evk4_zstack",
            self._zstack_predicted_s(), planes=self.pi_stage_widget.spin_steps.value(),
            frames=frames, out_dir=output_dir, filename=filename,
        )
        self._start_elapsed(self.lbl_zstack_time, self._update_zstack_time)

    def stop_zstack(self):
        if self.zstack_worker is not None and self.zstack_worker.isRunning():
            self._acq_aborted = True  # a stopped stack is partial — don't learn its time
            self.zstack_worker.stop()
            self.lbl_status.setText("Aborting Z-Stack...")
        if self.zstack_live_worker is not None and self.zstack_live_worker.isRunning():
            self.zstack_live_worker.stop()
            self.zstack_live_worker.wait(3000)

    def handle_z_profile(self, z_val, step_num):
        print(f"Step {step_num} | Computed Z-Profile value: {z_val}")

    def on_zstack_finished(self):
        self._stop_elapsed()
        self._set_zstack_busy(False)
        self.pi_stage_widget.resume_position_updates()
        # Filename is re-stamped at the next Z-stack start (if still the auto-
        # default), so stopping the live preview never wipes a typed name.
        self._reset_zstack_live_apply()

    def _reset_zstack_live_apply(self):
        """Disable + disconnect both Z-stack live-apply buttons and live re-framing."""
        self.zstack_orca_params.btn_apply_live.setEnabled(False)
        self.zstack_evk4_params.btn_apply_biases.setEnabled(False)
        try:
            self.zstack_orca_params.roi_changed.disconnect(self._apply_zstack_orca_params_live)
        except (RuntimeError, TypeError):
            pass
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
        self._acq_aborted = True  # an errored run is partial — don't learn its time
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
        # Let the crop overlay map widget <-> image pixels for this frame size.
        self.video_label.set_source_size(w, h)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        )

    # ==========================
    # Interactive crop tool
    # ==========================
    def _build_crop_bar(self):
        """Crop-tool control bar shown beneath the video feed."""
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)

        self.btn_crop_select = QPushButton("⛶  Select Crop Region")
        self.btn_crop_select.setCheckable(True)
        self.btn_crop_select.setToolTip(
            "Drag a rectangle on the full-sensor ORCA live image to mark a crop "
            "region. The box stays on screen so you can review it, then click "
            "'Apply Crop' to set the camera ROI to that region."
        )
        self.btn_crop_select.toggled.connect(self._toggle_crop_mode)

        self.btn_crop_apply = QPushButton("Apply Crop")
        self.btn_crop_apply.setObjectName("btnAcquire")
        self.btn_crop_apply.setEnabled(False)
        self.btn_crop_apply.clicked.connect(self._apply_crop)

        self.btn_crop_reset = QPushButton("Reset to Full")
        self.btn_crop_reset.clicked.connect(self._reset_crop)

        self.lbl_crop = QLabel("")
        self.lbl_crop.setStyleSheet(
            "color: #00e676; font-size: 11px; font-weight: bold; border: none;"
        )

        layout.addWidget(self.btn_crop_select)
        layout.addWidget(self.btn_crop_apply)
        layout.addWidget(self.btn_crop_reset)
        layout.addWidget(self.lbl_crop, stretch=1)
        return bar

    def _active_orca_params(self):
        """The ORCA parameter widget the crop tool should drive: the Z-stack copy
        when that tab is active with the ORCA camera, otherwise the standalone one."""
        if (self.tabs.currentWidget() is self.tab_zstack
                and self.combo_zstack_camera.currentData() == "orca"):
            return self.zstack_orca_params
        return self.orca_params

    def _toggle_crop_mode(self, on):
        self.video_label.set_crop_mode(on)
        self.btn_crop_select.setText(
            "⛶  Selecting… (drag on image)" if on else "⛶  Select Crop Region"
        )
        if not on:
            self._crop_region = None
            self.btn_crop_apply.setEnabled(False)
            self.lbl_crop.setText("")

    def _frame_region_to_sensor(self, x, y, w, h):
        """Map a region in the displayed frame's pixels to absolute sensor pixels.

        Assumes the frame was read out 1:1 (no binning) from the active ROI, so the
        frame origin is that ROI's top-left — at full sensor this is the identity.
        """
        roi = self._active_orca_params()._compute_roi()
        return roi["x_min"] + x, roi["y_min"] + y, w, h

    def _on_crop_region_drawn(self, x, y, w, h):
        """A rectangle was drawn on the feed: remember it and preview the ROI."""
        self._crop_region = (x, y, w, h)
        sx, sy, sw, sh = self._frame_region_to_sensor(x, y, w, h)
        self.btn_crop_apply.setEnabled(True)
        self.lbl_crop.setText(f"Selection: {sw} × {sh} px  @ ({sx}, {sy}) — click Apply Crop")

    def _apply_crop(self):
        """Set the active ORCA ROI to the drawn region (re-applies live if running)."""
        if self._crop_region is None:
            return
        if self._current_frame is None:
            self.lbl_status.setText("Start an ORCA live feed before cropping.")
            return
        x, y, w, h = self._crop_region
        sx, sy, sw, sh = self._frame_region_to_sensor(x, y, w, h)
        params = self._active_orca_params()
        # Convert the absolute sensor rectangle into the widget's width/height +
        # centre-offset model; _compute_roi re-aligns to multiples of 4 and clamps.
        offset_x = int(round(sx + sw / 2.0 - ORCA_SENSOR_WIDTH / 2.0))
        offset_y = int(round(sy + sh / 2.0 - ORCA_SENSOR_HEIGHT / 2.0))
        params.spin_roi_width.setValue(min(sw, ORCA_SENSOR_WIDTH))
        params.spin_roi_height.setValue(min(sh, ORCA_SENSOR_HEIGHT))
        params.slider_offset_x.setValue(offset_x)   # QSlider clamps to its range
        params.slider_offset_y.setValue(offset_y)

        applied = params._compute_roi()
        aw = applied["x_max"] - applied["x_min"]
        ah = applied["y_max"] - applied["y_min"]
        self.lbl_status.setText(
            f"Crop applied: {aw} × {ah} px @ ({applied['x_min']}, {applied['y_min']})."
        )
        # Leave crop mode; the live feed now shows the cropped subarray.
        self.btn_crop_select.setChecked(False)  # fires _toggle_crop_mode(False)
        self.video_label.clear_selection()

    def _reset_crop(self):
        """Restore the active ORCA ROI to the full sensor."""
        params = self._active_orca_params()
        params.spin_roi_width.setValue(ORCA_SENSOR_WIDTH)
        params.spin_roi_height.setValue(ORCA_SENSOR_HEIGHT)
        params.slider_offset_x.setValue(0)
        params.slider_offset_y.setValue(0)
        self._crop_region = None
        self.btn_crop_apply.setEnabled(False)
        self.lbl_crop.setText("")
        self.video_label.clear_selection()
        self.lbl_status.setText(
            f"ROI reset to full sensor ({ORCA_SENSOR_WIDTH} × {ORCA_SENSOR_HEIGHT})."
        )
