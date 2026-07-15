"""Main application window: layout, tab management, worker wiring and feed display.

This is the only place that instantiates the hardware workers and the Z-stack
orchestrator, routing every worker's status_update / error_signal back to the single
`lbl_status` bar and managing button enable/disable state during acquisition.

The application is Z-Stack-only: there is one tab per camera (ORCA and EVK4), each
driving the automated 3D Z-stack for that detector. The shared instruments (AWG and
PI Z-stage) live in collapsible sections of the left control panel so both tabs can
reach them.
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
    EVK4_SENSOR_HEIGHT, EVK4_SENSOR_WIDTH,
    ORCA_CAMERA_INIT_S, ORCA_PLANE_OVERHEAD_S, ORCA_SENSOR_HEIGHT, ORCA_SENSOR_WIDTH,
    SESSION_STATE_PATH,
)
from hardware.event_camera import CameraWorker, METAVISION_AVAILABLE
from hardware.orca_camera import OrcaWorker, DCAM_AVAILABLE
from ui.orchestrator import AutomatedZStackWorker
from ui.widgets import (
    AWGWidget, Evk4ParamsWidget, Evk4QueueWidget, OrcaParamsWidget, PIStageWidget,
    VideoFeedLabel,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Institut Fresnel - DSI Microscope Control")
        self.resize(1300, 950)
        # Live preview workers (one per camera) and the single Z-stack orchestrator.
        self.orca_worker = None     # ORCA live-focus worker
        self.evk4_worker = None     # EVK4 live worker
        self.zstack_worker = None   # automated Z-stack orchestrator (one at a time)
        # EVK4 batch queue: expanded per-acquisition configs run back-to-back.
        self._queue = []
        self._queue_index = 0
        self._queue_active = False
        self._zstack_camera = None  # "orca"/"event" while a Z-stack runs, else None
        # Snapshot of the ORCA settings currently applied to the live feed, so a
        # real-time crop change re-applies only the ROI (other edits wait for Apply).
        self._orca_live_params = None
        # Last frame shown per camera (image pixels), for the crop tool. The two
        # cameras have independent feeds so both can be viewed at once.
        self._orca_frame = None
        self._evk4_frame = None
        self._crop_region = None     # pending crop selection (x, y, w, h) in frame px
        self._crop_label = None      # the feed label currently in crop mode (if any)
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

        # Shared instruments live in collapsible sections so both Z-stack tabs can
        # reach them. The AWG is set rarely, so it starts collapsed; the PI stage is
        # used for every stack, so it starts expanded.
        self.btn_awg_toggle = QPushButton("▸  Siglent AWG Control (LC Speckle)")
        self.btn_awg_toggle.setCheckable(True)
        self.btn_awg_toggle.setStyleSheet("text-align: left; padding: 8px; background-color: #3a3f44;")
        self.btn_awg_toggle.toggled.connect(self._toggle_awg)
        control_layout.addWidget(self.btn_awg_toggle)

        self.awg_widget = AWGWidget()
        self.awg_widget.setVisible(False)
        control_layout.addWidget(self.awg_widget)

        # PI Z-stage: shared between both tabs, in its own collapsible section
        # (mirroring the AWG) so it is reachable from either camera's tab.
        self.pi_stage_widget = PIStageWidget()
        self.btn_pi_toggle = QPushButton("▸  PI Z-Stage Control")
        self.btn_pi_toggle.setCheckable(True)
        self.btn_pi_toggle.setStyleSheet("text-align: left; padding: 8px; background-color: #3a3f44;")
        self.btn_pi_toggle.toggled.connect(self._toggle_pi)
        control_layout.addWidget(self.btn_pi_toggle)
        self.pi_stage_widget.setVisible(False)  # starts collapsed, like the AWG section
        control_layout.addWidget(self.pi_stage_widget)
        # The number of Z steps drives both tabs' acquisition-time estimates.
        self.pi_stage_widget.spin_steps.valueChanged.connect(self._update_orca_time)
        self.pi_stage_widget.spin_steps.valueChanged.connect(self._update_evk4_time)

        self.tabs = QTabWidget()
        self._build_zstack_orca_tab()
        self._build_zstack_evk4_tab()
        # Switching tabs changes which camera the crop tool targets and which feed
        # is shown; cancel any in-progress crop and follow the tab's camera.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # Populate the time-estimate labels with the default parameter values.
        self._update_orca_time()
        self._update_evk4_time()
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

        # The two cameras share the sample via a beamsplitter, so both can stream
        # at once. A view selector chooses what is shown: one camera at a time
        # (toggle) or both feeds stacked. Each feed is an independent VideoFeedLabel
        # so the two never overwrite each other.
        view_bar = QHBoxLayout()
        view_bar.setContentsMargins(6, 4, 6, 0)
        lbl_view = QLabel("Display:")
        lbl_view.setStyleSheet("color: #cccccc; border: none;")
        self.combo_feed_view = QComboBox()
        self.combo_feed_view.addItem("ORCA only", "orca")
        self.combo_feed_view.addItem("Event camera only", "evk4")
        self.combo_feed_view.addItem("Both (stacked)", "both")
        # Launch showing the ORCA feed only; the feed follows the active tab, and
        # the user can pick Both or use Start Both Cameras.
        self.combo_feed_view.setCurrentIndex(self.combo_feed_view.findData("orca"))
        self.combo_feed_view.currentIndexChanged.connect(self._apply_feed_view)
        view_bar.addWidget(lbl_view)
        view_bar.addWidget(self.combo_feed_view)
        view_bar.addStretch()

        # Start/stop both live feeds at once (beamsplitter feeds both). Lives in the
        # shared feed panel because it acts on both cameras together.
        self.btn_both_live = QPushButton("▶  Start Both Cameras")
        self.btn_both_live.setObjectName("btnLive")
        self.btn_both_live.setToolTip(
            "Start the ORCA and event-camera live feeds together (or stop both if "
            "both are running). Switches the display to Both (stacked)."
        )
        self.btn_both_live.clicked.connect(self._toggle_both_live)
        view_bar.addWidget(self.btn_both_live)
        feed_layout.addLayout(view_bar)

        self.lbl_orca_feed_title = QLabel("Scientific Camera (ORCA)")
        self.lbl_evk4_feed_title = QLabel("Event Camera (EVK4)")
        self.video_label_orca = VideoFeedLabel("ORCA Feed Offline")
        self.video_label_evk4 = VideoFeedLabel("Event Camera Feed Offline")
        for title in (self.lbl_orca_feed_title, self.lbl_evk4_feed_title):
            title.setStyleSheet(
                "color: #4daaf2; font-size: 12px; font-weight: bold; border: none; padding: 2px 6px;"
            )
        for lbl, title in (
            (self.video_label_orca, self.lbl_orca_feed_title),
            (self.video_label_evk4, self.lbl_evk4_feed_title),
        ):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #555555; font-size: 24px; font-weight: bold; border: none;")
            lbl.region_drawn.connect(self._on_crop_region_drawn)
            feed_layout.addWidget(title)
            feed_layout.addWidget(lbl, stretch=1)

        feed_layout.addWidget(self._build_crop_bar())

        self._apply_feed_view()

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

        self._refresh_buttons()

    def _toggle_awg(self, expanded):
        self.awg_widget.setVisible(expanded)
        arrow = "▾" if expanded else "▸"
        self.btn_awg_toggle.setText(f"{arrow}  Siglent AWG Control (LC Speckle)")

    def _toggle_pi(self, expanded):
        self.pi_stage_widget.setVisible(expanded)
        arrow = "▾" if expanded else "▸"
        self.btn_pi_toggle.setText(f"{arrow}  PI Z-Stage Control")

    # ==========================
    # Tab construction
    # ==========================
    def _build_zstack_orca_tab(self):
        """Z-Stack tab for the Hamamatsu ORCA (per-plane DSI)."""
        self.tab_orca = QWidget()
        layout = QVBoxLayout(self.tab_orca)

        # 1. Live preview — start it first to focus the sample and set the crop.
        live_group = QGroupBox("Live Preview")
        live_layout = QHBoxLayout()
        self.btn_orca_live = QPushButton("▶  Start Live")
        self.btn_orca_live.setObjectName("btnLive")
        self.btn_orca_live.clicked.connect(self.start_orca_live)
        self.btn_orca_stop = QPushButton("■  Stop")
        self.btn_orca_stop.setObjectName("btnStop")
        self.btn_orca_stop.clicked.connect(self.stop_orca)
        self.btn_orca_stop.setEnabled(False)
        live_layout.addWidget(self.btn_orca_live)
        live_layout.addWidget(self.btn_orca_stop)
        live_group.setLayout(live_layout)
        layout.addWidget(live_group)

        # 2. Camera parameters — full control. Crop applies live; other settings
        #    apply when "Apply All Settings to Live Feed" is clicked.
        self.orca_params = OrcaParamsWidget()
        for sig in (
            self.orca_params.spin_frames.valueChanged,
            self.orca_params.spin_exp.valueChanged,
            self.orca_params.spin_roi_height.valueChanged,
            self.orca_params.spin_roi_width.valueChanged,
            self.orca_params.combo_readout.currentIndexChanged,
        ):
            sig.connect(self._update_orca_time)
        layout.addWidget(self.orca_params)

        # 3. Automated Z-stack acquisition.
        acq_group = QGroupBox("Automated 3D Z-Stack (ORCA DSI)")
        acq_layout = QVBoxLayout()
        acq_layout.addWidget(QLabel("Output Directory:"))
        dir_layout = QHBoxLayout()
        self.txt_orca_dir = QLineEdit()
        btn_dir = QPushButton("Browse")
        btn_dir.clicked.connect(self.browse_orca_dir)
        dir_layout.addWidget(self.txt_orca_dir); dir_layout.addWidget(btn_dir)
        acq_layout.addLayout(dir_layout)
        acq_layout.addWidget(QLabel("Filename Base:"))
        self.txt_orca_filename = QLineEdit()
        self._stamp_default_filename(self.txt_orca_filename, "zstack_orca")
        acq_layout.addWidget(self.txt_orca_filename)
        self.chk_orca_raw = QCheckBox("Save raw 16-bit speckle stack per plane (for RIM / re-processing)")
        self.chk_orca_raw.setChecked(True)
        self.chk_orca_raw.stateChanged.connect(self._update_orca_time)
        acq_layout.addWidget(self.chk_orca_raw)
        info = QLabel(
            "Moves through Z and acquires an ORCA speckle stack at each plane, saving "
            "the per-plane sectioned images as a 3D TIFF depth volume, the raw stacks "
            "(one file per plane), and a parameter log."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888888; font-size: 11px;")
        acq_layout.addWidget(info)
        self.btn_orca_zstack = QPushButton("⬤  Start ORCA Z-Stack")
        self.btn_orca_zstack.setObjectName("btnAcquire")
        self.btn_orca_zstack.clicked.connect(lambda: self.start_zstack("orca"))
        acq_layout.addWidget(self.btn_orca_zstack)
        # Live pause/resume: appears only while a running ORCA stack is paused waiting
        # for the camera to be restored (e.g. after replugging the USB).
        self.btn_orca_live_resume = QPushButton("⟳  Resume Acquisition")
        self.btn_orca_live_resume.setObjectName("btnLive")
        self.btn_orca_live_resume.clicked.connect(self._resume_live_acquisition)
        self.btn_orca_live_resume.setVisible(False)
        acq_layout.addWidget(self.btn_orca_live_resume)
        self.lbl_orca_time = QLabel()
        self.lbl_orca_time.setStyleSheet("color: #4daaf2; font-size: 11px; font-weight: bold;")
        acq_layout.addWidget(self.lbl_orca_time)
        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)

        layout.addStretch()
        self.tabs.addTab(self.tab_orca, "Z-Stack (ORCA)")

    def _build_zstack_evk4_tab(self):
        """Z-Stack tab for the Prophesee EVK4 (per-plane event-DSI)."""
        self.tab_evk4 = QWidget()
        layout = QVBoxLayout(self.tab_evk4)

        # 1. Live preview.
        live_group = QGroupBox("Live Preview")
        live_layout = QHBoxLayout()
        self.btn_evk4_live = QPushButton("▶  Start Live")
        self.btn_evk4_live.setObjectName("btnLive")
        self.btn_evk4_live.clicked.connect(self.start_evk4_live)
        self.btn_evk4_stop = QPushButton("■  Stop")
        self.btn_evk4_stop.setObjectName("btnStop")
        self.btn_evk4_stop.clicked.connect(self.stop_evk4)
        self.btn_evk4_stop.setEnabled(False)
        live_layout.addWidget(self.btn_evk4_live)
        live_layout.addWidget(self.btn_evk4_stop)
        live_group.setLayout(live_layout)
        layout.addWidget(live_group)

        # 2. Camera parameters — full control. Crop applies live; biases apply when
        #    "Apply Biases to Live Feed" is clicked.
        self.evk4_params = Evk4ParamsWidget()
        self.evk4_params.spin_time.valueChanged.connect(self._update_evk4_time)
        layout.addWidget(self.evk4_params)

        # 3. Automated Z-stack acquisition.
        acq_group = QGroupBox("Automated 3D Z-Stack (Event-DSI)")
        acq_layout = QVBoxLayout()
        acq_layout.addWidget(QLabel("Output Directory:"))
        dir_layout = QHBoxLayout()
        self.txt_evk4_dir = QLineEdit()
        btn_dir = QPushButton("Browse")
        btn_dir.clicked.connect(self.browse_evk4_dir)
        dir_layout.addWidget(self.txt_evk4_dir); dir_layout.addWidget(btn_dir)
        acq_layout.addLayout(dir_layout)
        acq_layout.addWidget(QLabel("Filename Base:"))
        self.txt_evk4_filename = QLineEdit()
        self._stamp_default_filename(self.txt_evk4_filename, "zstack_evk4")
        acq_layout.addWidget(self.txt_evk4_filename)
        info = QLabel(
            "Moves through Z and records events at each plane. Each plane's raw event "
            "stream (.raw) is always saved, alongside the accumulated event depth "
            "volume (3D TIFF) and a parameter log."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888888; font-size: 11px;")
        acq_layout.addWidget(info)
        self.btn_evk4_zstack = QPushButton("⬤  Start EVK4 Z-Stack")
        self.btn_evk4_zstack.setObjectName("btnAcquire")
        self.btn_evk4_zstack.clicked.connect(lambda: self.start_zstack("event"))
        acq_layout.addWidget(self.btn_evk4_zstack)
        # Shown only when an acquisition has paused after losing the event camera:
        # replug the USB, then click this to continue from the plane it stopped on.
        self.btn_evk4_resume = QPushButton("⟳  Resume Acquisition")
        self.btn_evk4_resume.setObjectName("btnLive")
        self.btn_evk4_resume.clicked.connect(self._resume_live_acquisition)
        self.btn_evk4_resume.setVisible(False)
        acq_layout.addWidget(self.btn_evk4_resume)
        self.lbl_evk4_time = QLabel()
        self.lbl_evk4_time.setStyleSheet("color: #4daaf2; font-size: 11px; font-weight: bold;")
        acq_layout.addWidget(self.lbl_evk4_time)
        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)

        # 4. Batch queue — configure several acquisitions and run them unattended.
        self.evk4_queue = Evk4QueueWidget()
        self.evk4_queue.run_requested.connect(self._start_evk4_queue)
        self.evk4_queue.stop_requested.connect(self._stop_evk4_queue)
        # Keep the queue's total-time estimate live as rows or the step count change.
        self.evk4_queue.changed.connect(self._update_queue_estimate)
        self.pi_stage_widget.spin_steps.valueChanged.connect(self._update_queue_estimate)
        layout.addWidget(self.evk4_queue)
        self._update_queue_estimate()

        layout.addStretch()
        self.tabs.addTab(self.tab_evk4, "Z-Stack (EVK4)")

    def _on_tab_changed(self):
        """Tab switched: cancel any in-progress crop and follow the tab's camera in
        the feed panel (unless the user has chosen Both)."""
        self._cancel_crop_mode()
        if getattr(self, "combo_feed_view", None) is None:
            return
        if self.combo_feed_view.currentData() != "both":
            cam = "evk4" if self.tabs.currentWidget() is self.tab_evk4 else "orca"
            idx = self.combo_feed_view.findData(cam)
            if idx >= 0:
                self.combo_feed_view.setCurrentIndex(idx)

    def _cancel_crop_mode(self):
        """Leave crop mode if active — used when the crop target (active tab)
        changes, so the overlay can't be stranded on the wrong feed."""
        if getattr(self, "btn_crop_select", None) is not None and self.btn_crop_select.isChecked():
            self.btn_crop_select.setChecked(False)  # fires _toggle_crop_mode(False)

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

    def _orca_zstack_predicted_s(self):
        """Cold-start estimate (s) for an ORCA Z-stack: camera start-up plus, per
        plane, motor move + settle, capture, DSI reconstruction and raw write."""
        steps = self.pi_stage_widget.spin_steps.value()
        n = self.orca_params.spin_frames.value()
        frame_s = self.orca_params.estimated_frame_time_s()
        compute_s = self.orca_params.estimated_compute_s(n)
        save_s = self.orca_params.estimated_raw_save_s(n) if self.chk_orca_raw.isChecked() else 0.0
        plane_s = ORCA_PLANE_OVERHEAD_S + n * frame_s + compute_s + save_s
        return ORCA_CAMERA_INIT_S + steps * plane_s

    def _evk4_zstack_predicted_s(self):
        """Cold-start estimate (s) for an EVK4 Z-stack: per plane the device is
        re-opened and events accumulated for the fixed recording duration."""
        steps = self.pi_stage_widget.spin_steps.value()
        plane_s = EVK4_PLANE_OVERHEAD_S + self.evk4_params.spin_time.value()
        return steps * plane_s

    def _update_orca_time(self):
        if self._acq_label is self.lbl_orca_time:
            return  # acquisition running: leave the live elapsed readout alone
        steps = self.pi_stage_widget.spin_steps.value()
        total_s, runs = self._calibrate(self._orca_zstack_predicted_s(), "orca_zstack")
        plane_s = total_s / max(1, steps)
        self.lbl_orca_time.setText(
            f"Estimated: ≈ {self._fmt_dur(total_s)}  ({steps} planes × ≈ {plane_s:.1f} s/plane)"
            f"{self._calib_note(runs)}"
        )

    def _update_evk4_time(self):
        if self._acq_label is self.lbl_evk4_time:
            return  # acquisition running: leave the live elapsed readout alone
        steps = self.pi_stage_widget.spin_steps.value()
        total_s, runs = self._calibrate(self._evk4_zstack_predicted_s(), "evk4_zstack")
        plane_s = total_s / max(1, steps)
        self.lbl_evk4_time.setText(
            f"Estimated: ≈ {self._fmt_dur(total_s)}  ({steps} planes × ≈ {plane_s:.1f} s/plane)"
            f"{self._calib_note(runs)}"
        )

    def _evk4_queue_total_predicted_s(self):
        """Total predicted time for the whole batch queue, summed over every row and
        repeat. Each acquisition uses its row's duration and the shared step count;
        the same EVK4 model + calibration as a single Z-stack is applied."""
        steps = self.pi_stage_widget.spin_steps.value()
        total_raw, n_acq = 0.0, 0
        for r in self.evk4_queue.rows():
            total_raw += r["repeats"] * steps * (EVK4_PLANE_OVERHEAD_S + r["acqu_time"])
            n_acq += r["repeats"]
        total_s, runs = self._calibrate(total_raw, "evk4_zstack")  # calibration is linear
        return total_s, n_acq, runs

    def _update_queue_estimate(self):
        """Refresh the batch-queue total-time label (rows or step count changed)."""
        if getattr(self, "evk4_queue", None) is None:
            return
        total_s, n_acq, runs = self._evk4_queue_total_predicted_s()
        if n_acq == 0:
            self.evk4_queue.set_estimate("")
            return
        self.evk4_queue.set_estimate(
            f"Estimated total: ≈ {self._fmt_dur(total_s)}  "
            f"({n_acq} acquisition{'s' if n_acq != 1 else ''}){self._calib_note(runs)}"
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
        path = os.path.join(out_dir, f"{filename}_parameters.txt")
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

        for worker in (self.orca_worker, self.evk4_worker, self.zstack_worker):
            if worker is not None and worker.isRunning():
                worker.stop()
                worker.wait(2000)

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
                "step_size": self.pi_stage_widget.spin_step_size.value(),
                "num_steps": self.pi_stage_widget.spin_steps.value(),
                "focus": self.pi_stage_widget.spin_focus.value(),
            },
            "save_raw": {
                # EVK4 .raw is always saved; only the ORCA raw-stack toggle persists.
                "orca": self.chk_orca_raw.isChecked(),
            },
            "output_dirs": {
                "evk4": self.txt_evk4_dir.text(),
                "orca": self.txt_orca_dir.text(),
            },
        }

    def _apply_preset(self, preset):
        """Apply a preset/session dict to every control. Unknown keys are ignored.

        Older files stored separate standalone/Z-stack copies per camera
        (``orca``/``orca_zstack``); prefer the Z-stack copy, falling back to the
        standalone one, so they still load.
        """
        if "orca" in preset or "orca_zstack" in preset:
            self.orca_params.set_preset(preset.get("orca_zstack") or preset.get("orca"))
        if "evk4" in preset or "evk4_zstack" in preset:
            self.evk4_params.set_preset(preset.get("evk4_zstack") or preset.get("evk4"))

        if "awg" in preset:
            self.awg_widget.set_preset(preset["awg"])

        if "zstack" in preset:
            z = preset["zstack"]
            if "step_size" in z:
                self.pi_stage_widget.spin_step_size.setValue(float(z["step_size"]))
            if "num_steps" in z:
                self.pi_stage_widget.spin_steps.setValue(int(z["num_steps"]))
            if "focus" in z:
                self.pi_stage_widget.spin_focus.setValue(float(z["focus"]))

        if "save_raw" in preset:
            sr = preset["save_raw"]
            val = sr.get("orca", sr.get("zstack"))  # old files used "zstack" for the ORCA toggle
            if val is not None:
                self.chk_orca_raw.setChecked(bool(val))

        # Output directories are restored (the filename bases stay timestamped).
        if "output_dirs" in preset:
            od = preset["output_dirs"]
            if od.get("orca") or od.get("zstack"):
                self.txt_orca_dir.setText(od.get("orca") or od.get("zstack"))
            if od.get("evk4") or od.get("zstack"):
                self.txt_evk4_dir.setText(od.get("evk4") or od.get("zstack"))

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

    def browse_orca_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.txt_orca_dir.setText(folder)

    def browse_evk4_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.txt_evk4_dir.setText(folder)

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
        eroi = e["evk4_roi"]
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
                "roi_x_min": eroi["x_min"],
                "roi_x_max": eroi["x_max"],
                "roi_y_min": eroi["y_min"],
                "roi_y_max": eroi["y_max"],
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
    # Button state
    # ==========================
    def _refresh_buttons(self):
        """Centralised enable/disable for the per-tab live and Z-stack buttons,
        derived from the current worker states.

        While a Z-stack runs (on either camera) no new live feed or stack can be
        started; the running camera's Stop button is the abort. A camera's live
        Start is disabled while that camera is already live; its Stop then aborts
        the live feed.
        """
        orca_live = self._orca_live_running()
        evk4_live = self._evk4_live_running()
        zstack = self.zstack_worker is not None and self.zstack_worker.isRunning()
        zcam = self._zstack_camera if zstack else None
        # A running batch queue blocks new manual runs just like a live Z-stack.
        busy = zstack or self._queue_active

        self.btn_orca_live.setEnabled(not orca_live and not busy)
        self.btn_evk4_live.setEnabled(not evk4_live and not busy)
        self.btn_orca_zstack.setEnabled(not busy)
        self.btn_evk4_zstack.setEnabled(not busy)
        self.btn_orca_stop.setEnabled(orca_live or (zstack and zcam == "orca"))
        self.btn_evk4_stop.setEnabled(evk4_live or (zstack and zcam == "event") or self._queue_active)
        # Outside a queue, disable 'Run Queue' while any manual Z-stack runs.
        if getattr(self, "evk4_queue", None) is not None and not self._queue_active:
            self.evk4_queue.btn_run.setEnabled(not zstack)

    # ==========================
    # ORCA live preview
    # ==========================
    def start_orca_live(self):
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
        if self._orca_live_running():
            return

        params = self.orca_params.get_params()
        self.orca_worker = OrcaWorker("live", params)
        self.orca_worker.image_ready.connect(self.update_orca_image)
        self.orca_worker.status_update.connect(self.lbl_status.setText)
        self.orca_worker.error_signal.connect(self.show_error)
        self.orca_worker.finished_signal.connect(self.on_orca_live_finished)
        self.orca_worker.start()

        # Live-apply wiring: the Apply button pushes ALL settings; only the crop
        # is applied in real time as the ROI controls change.
        self._orca_live_params = params
        self.orca_params.btn_apply_live.setEnabled(True)
        self.orca_params.btn_apply_live.clicked.connect(self._apply_orca_params_live)
        self.orca_params.roi_changed.connect(self._apply_orca_roi_live)

        self._refresh_buttons()
        self._sync_both_live_button()

    def stop_orca(self):
        """Stop the ORCA — its running Z-stack if one is active, else its live feed."""
        self._acq_aborted = True  # a stopped run is partial — don't learn its time
        if self.zstack_worker is not None and self.zstack_worker.isRunning() and self._zstack_camera == "orca":
            self.zstack_worker.stop()
            self.lbl_status.setText("Aborting Z-Stack...")
            return
        if self.orca_worker is not None and self.orca_worker.isRunning():
            self.orca_worker.stop()
            self.orca_worker.wait(3000)

    def on_orca_live_finished(self):
        # Live mode never owns the elapsed timer (only a Z-stack does), so don't
        # touch it here — a Z-stack on the other camera may be timing.
        self._reset_orca_live_apply()
        self._refresh_buttons()
        self._sync_both_live_button()

    def _reset_orca_live_apply(self):
        """Disable + disconnect the ORCA live-apply button and live re-cropping."""
        self.orca_params.btn_apply_live.setEnabled(False)
        try:
            self.orca_params.btn_apply_live.clicked.disconnect(self._apply_orca_params_live)
        except (RuntimeError, TypeError):
            pass
        try:
            self.orca_params.roi_changed.disconnect(self._apply_orca_roi_live)
        except (RuntimeError, TypeError):
            pass

    def _apply_orca_params_live(self):
        """Apply ALL ORCA settings to the live feed — triggered by the Apply button."""
        if self.orca_worker is not None and self.orca_worker.isRunning():
            params = self.orca_params.get_params()
            self._orca_live_params = params
            self.orca_worker.apply_params(params)

    def _apply_orca_roi_live(self):
        """Apply only the crop/ROI to the running ORCA live feed, in real time.

        Other parameter edits (exposure, mode, readout) are deliberately left for
        the Apply button: a live crop change merges the new ROI onto the
        last-applied settings, so dragging the crop never pushes un-applied edits.
        """
        if not (self.orca_worker is not None and self.orca_worker.isRunning()):
            return
        base = self._orca_live_params or self.orca_params.get_params()
        params = dict(base)
        params["orca_roi"] = self.orca_params._compute_roi()
        self._orca_live_params = params
        self.orca_worker.apply_params(params)

    # ==========================
    # EVK4 live preview
    # ==========================
    def start_evk4_live(self):
        if not METAVISION_AVAILABLE:
            QMessageBox.warning(
                self, "Event Camera Unavailable",
                "The Prophesee Metavision SDK was not found.\n\n"
                "Install the Metavision SDK for your Python version and ensure "
                "all its dependencies (including h5py) are installed in the same environment.\n\n"
                "See README.md for setup instructions.",
            )
            return
        if self._evk4_live_running():
            return

        params = self.evk4_params.get_params()
        self.evk4_worker = CameraWorker("live", params)
        self.evk4_worker.frame_ready.connect(self.update_evk4_image)
        self.evk4_worker.status_update.connect(self.lbl_status.setText)
        self.evk4_worker.error_signal.connect(self.show_error)
        self.evk4_worker.finished_signal.connect(self.on_evk4_live_finished)
        self.evk4_worker.start()

        # Live-apply wiring: biases on the Apply button; crop in real time.
        self.evk4_params.btn_apply_biases.setEnabled(True)
        self.evk4_params.btn_apply_biases.clicked.connect(self._apply_evk4_biases_live)
        self.evk4_params.roi_changed.connect(self._apply_evk4_roi_live)

        self._refresh_buttons()
        self._sync_both_live_button()

    def stop_evk4(self):
        """Stop the EVK4 — its running Z-stack if one is active, else its live feed."""
        self._acq_aborted = True
        # If a batch queue is running, the camera Stop button stops the whole queue.
        if self._queue_active:
            self._stop_evk4_queue()
            self.lbl_status.setText("Aborting queue...")
            return
        if self.zstack_worker is not None and self.zstack_worker.isRunning() and self._zstack_camera == "event":
            self.zstack_worker.stop()
            self.lbl_status.setText("Aborting Z-Stack...")
            return
        if self.evk4_worker is not None and self.evk4_worker.isRunning():
            self.evk4_worker.stop()
            self.lbl_status.setText("Stopping EVK4...")
            self.evk4_worker.wait(3000)

    def on_evk4_live_finished(self):
        self._reset_evk4_apply()
        self._refresh_buttons()
        self._sync_both_live_button()

    def _reset_evk4_apply(self):
        """Disable + disconnect the EVK4 live bias-apply button and live re-cropping."""
        self.evk4_params.btn_apply_biases.setEnabled(False)
        try:
            self.evk4_params.btn_apply_biases.clicked.disconnect(self._apply_evk4_biases_live)
        except (RuntimeError, TypeError):
            pass
        try:
            self.evk4_params.roi_changed.disconnect(self._apply_evk4_roi_live)
        except (RuntimeError, TypeError):
            pass

    def _apply_evk4_biases_live(self):
        if self.evk4_worker is not None and self.evk4_worker.isRunning():
            self.evk4_worker.apply_biases(self.evk4_params.get_params())

    def _apply_evk4_roi_live(self):
        if self.evk4_worker is not None and self.evk4_worker.isRunning():
            self.evk4_worker.apply_roi(self.evk4_params._compute_roi())

    # ==========================
    # Dual-camera live (both feeds at once)
    # ==========================
    def _orca_live_running(self):
        return self.orca_worker is not None and self.orca_worker.isRunning()

    def _evk4_live_running(self):
        return self.evk4_worker is not None and self.evk4_worker.isRunning()

    def _toggle_both_live(self):
        """Start both camera live feeds, or stop both if both are already running."""
        if self._orca_live_running() and self._evk4_live_running():
            self.stop_orca()
            self.stop_evk4()
            self._sync_both_live_button()
            return
        # Show both feeds so the result is visible immediately.
        self.combo_feed_view.setCurrentIndex(self.combo_feed_view.findData("both"))
        if not self._orca_live_running():
            self.start_orca_live()
        if not self._evk4_live_running():
            self.start_evk4_live()
        self._sync_both_live_button()

    def _sync_both_live_button(self):
        """Reflect the combined live state in the Start/Stop Both button label, so
        it stays correct even when feeds are started/stopped from their own tabs."""
        both = self._orca_live_running() and self._evk4_live_running()
        self.btn_both_live.setText("■  Stop Both Cameras" if both else "▶  Start Both Cameras")

    # ==========================
    # Automated Z-Stack
    # ==========================
    def _zstack_orca_peak_bytes(self):
        """Rough estimate of the peak RAM an ORCA Z-stack needs, in bytes.

        Dominated by the per-plane frame stack held in RAM (the camera's own
        N-frame ring buffer + our single host copy, both uint16) and the depth
        volumes that accumulate over every plane (average + DSI, float32, plus a
        transient copy when they are written). Deliberately rough — it only needs
        to be good enough to catch a configuration that clearly won't fit.
        """
        p = self.orca_params.get_params()
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

    def start_zstack(self, camera):
        """Launch the automated Z-stack for ``camera`` ("orca" or "event")."""
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
        if self.zstack_worker is not None and self.zstack_worker.isRunning():
            self.lbl_status.setText("A Z-Stack is already running — stop it before starting another.")
            return

        out_dir = self.txt_orca_dir.text() if camera == "orca" else self.txt_evk4_dir.text()
        if not out_dir:
            QMessageBox.warning(self, "Missing Parameter", "Please select an output directory for the Z-Stack.")
            return

        # Pre-flight memory check for ORCA (each plane holds a whole frame stack in
        # RAM; a high frame count at full sensor can exceed physical memory).
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

        fn_field = self.txt_orca_filename if camera == "orca" else self.txt_evk4_filename
        fn_prefix = "zstack_orca" if camera == "orca" else "zstack_evk4"
        self._refresh_default_filename(fn_field, fn_prefix)
        filename = fn_field.text()
        source = (
            "3D Z-Stack (ORCA) - per-plane DSI" if camera == "orca"
            else "3D Z-Stack (EVK4) - per-plane event-DSI"
        )

        save_params = {
            "output_dir": out_dir,
            "filename": filename,
            # ORCA raw-stack TIFF is optional; the EVK4 per-plane .raw is always saved.
            "save_raw": self.chk_orca_raw.isChecked() if camera == "orca" else True,
            "metadata": self.collect_acquisition_metadata(
                source, out_dir, filename, self.evk4_params, self.orca_params
            ),
        }

        # Auto-stop a running live preview for this camera so its handle is released
        # before the orchestrator opens the camera.
        if camera == "orca":
            if self._stop_worker_silently(self.orca_worker, self.on_orca_live_finished):
                self._reset_orca_live_apply()
        else:
            if self._stop_worker_silently(self.evk4_worker, self.on_evk4_live_finished):
                self._reset_evk4_apply()

        self._zstack_camera = camera

        # The orchestrator thread now owns the stage; pause the widget's idle
        # polling so the two don't query the GCS link concurrently.
        self.pi_stage_widget.pause_position_updates()

        self.zstack_worker = AutomatedZStackWorker(
            self.pi_stage_widget.pidevice,
            self.pi_stage_widget.axis,
            self.get_motor_params(),
            self.orca_params.get_params(),
            save_params,
            camera=camera,
            evk4_params=self.evk4_params.get_params(),
        )
        self.zstack_worker.image_ready.connect(
            self.update_orca_image if camera == "orca" else self.update_evk4_image
        )
        self.zstack_worker.status_update.connect(self.lbl_status.setText)
        self.zstack_worker.z_profile_update.connect(self.handle_z_profile)
        self.zstack_worker.position_update.connect(self.pi_stage_widget.show_position)
        self.zstack_worker.error_signal.connect(self.show_error)
        self.zstack_worker.awaiting_reconnect.connect(self._on_awaiting_reconnect)
        self.zstack_worker.finished_signal.connect(self.on_zstack_finished)
        self.zstack_worker.start()
        self._refresh_buttons()
        self._sync_both_live_button()

        if camera == "orca":
            predicted = self._orca_zstack_predicted_s()
            frames = self.orca_params.get_params()["orca_frames"]
            label, restore = self.lbl_orca_time, self._update_orca_time
        else:
            predicted = self._evk4_zstack_predicted_s()
            frames = self.evk4_params.get_params()["acqu_time"]
            label, restore = self.lbl_evk4_time, self._update_evk4_time
        self._begin_acq_record(
            "orca_zstack" if camera == "orca" else "evk4_zstack",
            predicted, planes=self.pi_stage_widget.spin_steps.value(),
            frames=frames, out_dir=out_dir, filename=filename,
        )
        self._start_elapsed(label, restore)

    def handle_z_profile(self, z_val, step_num):
        print(f"Step {step_num} | Computed Z-Profile value: {z_val}")

    def _on_awaiting_reconnect(self, waiting):
        """Show/hide the live-resume button when the running acquisition pauses
        waiting for its camera to be restored. The button appears on the active
        camera's tab (either camera can pause and resume in place now)."""
        btn = self.btn_orca_live_resume if self._zstack_camera == "orca" else self.btn_evk4_resume
        btn.setVisible(waiting)
        btn.setEnabled(waiting)

    def _resume_live_acquisition(self):
        """User clicked a live Resume button after restoring the paused camera."""
        if self.zstack_worker is not None and self.zstack_worker.isRunning():
            # Debounce both buttons; the active one is re-shown if it pauses again.
            self.btn_orca_live_resume.setEnabled(False)
            self.btn_evk4_resume.setEnabled(False)
            self.zstack_worker.resume()

    def on_zstack_finished(self):
        self._stop_elapsed()
        self.btn_evk4_resume.setVisible(False)
        self.btn_orca_live_resume.setVisible(False)
        self.pi_stage_widget.resume_position_updates()
        self._zstack_camera = None
        self._refresh_buttons()
        self._sync_both_live_button()
        # If a batch queue is running, advance to the next acquisition.
        if self._queue_active:
            self._queue_index += 1
            QTimer.singleShot(800, self._run_next_queue_item)

    # ==========================
    # EVK4 batch queue
    # ==========================
    def _start_evk4_queue(self):
        """Validate and launch the EVK4 batch queue (run from the queue widget)."""
        if not METAVISION_AVAILABLE:
            QMessageBox.warning(self, "Event Camera Unavailable",
                "The Prophesee Metavision SDK was not found. See README.md for setup instructions.")
            return
        if not self.pi_stage_widget.pidevice:
            QMessageBox.warning(self, "Connection Error", "Please connect the PI Stage before running the queue.")
            return
        if self._queue_active or (self.zstack_worker is not None and self.zstack_worker.isRunning()):
            self.lbl_status.setText("An acquisition is already running — stop it first.")
            return
        if not self.txt_evk4_dir.text():
            QMessageBox.warning(self, "Missing Parameter", "Please select an output directory for the queue.")
            return
        rows = self.evk4_queue.rows()
        if not rows:
            QMessageBox.warning(self, "Empty Queue", "Add at least one acquisition to the queue.")
            return
        if any(not r["filename"] for r in rows):
            QMessageBox.warning(self, "Missing Filename",
                "Every queue row needs a filename. Type one, or pick a parameter and click 'Apply names'.")
            return

        # Expand each row's repeats into uniquely-named acquisitions.
        self._queue = []
        for r in rows:
            n = max(1, r["repeats"])
            for k in range(n):
                name = r["filename"] if n == 1 else f"{r['filename']}_rep{k + 1:02d}"
                self._queue.append({
                    "filename": name, "bias_fo": r["bias_fo"], "bias_hpf": r["bias_hpf"],
                    "bias_on": r["bias_on"], "bias_off": r["bias_off"], "acqu_time": r["acqu_time"],
                })
        self._queue_index = 0
        self._queue_active = True
        self.evk4_queue.set_running(True)
        self._refresh_buttons()
        self._run_next_queue_item()

    def _run_next_queue_item(self):
        """Apply the next queued acquisition's parameters and launch it."""
        if not self._queue_active:
            return
        # Wait for the previous worker thread to fully release before starting.
        if self.zstack_worker is not None and self.zstack_worker.isRunning():
            QTimer.singleShot(200, self._run_next_queue_item)
            return
        if self._queue_index >= len(self._queue):
            self._finish_queue(f"Queue complete — {len(self._queue)} acquisition(s) done.")
            return
        item = self._queue[self._queue_index]
        # Push the row's biases + duration into the EVK4 widget; start_zstack reads them.
        self.evk4_params.set_preset({
            "bias_fo": item["bias_fo"], "bias_hpf": item["bias_hpf"],
            "bias_on": item["bias_on"], "bias_off": item["bias_off"],
            "acqu_time": item["acqu_time"],
        })
        self.txt_evk4_filename.setText(item["filename"])
        self.evk4_queue.set_status(
            f"Running {self._queue_index + 1} / {len(self._queue)}:  {item['filename']}")
        self.start_zstack("event")

    def _stop_evk4_queue(self):
        """Stop the queue: no further acquisitions start; the current one is aborted."""
        if not self._queue_active:
            return
        self._queue_active = False
        if self.zstack_worker is not None and self.zstack_worker.isRunning() and self._zstack_camera == "event":
            self.zstack_worker.stop()
        self._finish_queue(f"Queue stopped at {self._queue_index + 1} / {len(self._queue)}.")

    def _finish_queue(self, message):
        self._queue_active = False
        self._queue = []
        self.evk4_queue.set_running(False)
        self.evk4_queue.set_status(message)
        self.lbl_status.setText(message)
        self._refresh_buttons()

    # ==========================
    # General UI Handling
    # ==========================
    def show_error(self, err_msg):
        self._acq_aborted = True  # an errored run is partial — don't learn its time
        # Halt the batch queue before the finish handlers run (so it won't advance).
        if self._queue_active:
            self._queue_active = False
            self.evk4_queue.set_running(False)
            self.evk4_queue.set_status(
                f"Queue stopped — error at {self._queue_index + 1} / {len(self._queue)}.")
        QMessageBox.critical(self, "System Error", err_msg)
        self.on_orca_live_finished()
        self.on_evk4_live_finished()
        self.on_zstack_finished()

    def _apply_feed_view(self):
        """Show/hide each camera feed per the view selector (ORCA, EVK4, or both)."""
        mode = self.combo_feed_view.currentData()
        show_orca = mode in ("orca", "both")
        show_evk4 = mode in ("evk4", "both")
        self.lbl_orca_feed_title.setVisible(show_orca)
        self.video_label_orca.setVisible(show_orca)
        self.lbl_evk4_feed_title.setVisible(show_evk4)
        self.video_label_evk4.setVisible(show_evk4)

    def _render_to_label(self, label, cv_img):
        """Render a NumPy frame into a feed label, scaled with KeepAspectRatio."""
        # PyQt6 requires bytes, not memoryview; ascontiguousarray packs the
        # ROI-cropped slice before tobytes() serialises it row-by-row.
        img = np.ascontiguousarray(cv_img)
        # Keep the pixel buffer in a *named* local: a QImage built on a buffer does
        # not own it, and QPixmap.fromImage copies only on the next line. If the
        # buffer were an unnamed temporary (``QImage(img.tobytes(), …)``) it could
        # be freed before that copy, showing garbled frames — intermittently, and
        # worst under dual-camera load where buffer churn reuses the freed memory.
        buf = img.tobytes()
        if img.ndim == 3:
            h, w, ch = img.shape
            qt_img = QImage(buf, w, h, ch * w, QImage.Format.Format_BGR888)
        else:
            h, w = img.shape
            qt_img = QImage(buf, w, h, w, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qt_img)  # copies now, while ``buf`` is still alive
        # Let the crop overlay map widget <-> image pixels for this frame size.
        label.set_source_size(w, h)
        # Hand the full-resolution pixmap to the label, which scales it to fit and
        # re-scales on resize — so switching tabs / Display mode never leaves a
        # stale, wrongly-sized frame.
        label.set_frame_pixmap(pixmap)

    @pyqtSlot(np.ndarray)
    def update_orca_image(self, cv_img):
        self._orca_frame = cv_img
        self._render_to_label(self.video_label_orca, cv_img)

    @pyqtSlot(np.ndarray)
    def update_evk4_image(self, cv_img):
        self._evk4_frame = cv_img
        self._render_to_label(self.video_label_evk4, cv_img)

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
            "Drag a rectangle on the live image (ORCA or event camera) to mark a "
            "crop region. The box stays on screen so you can review it, then click "
            "'Apply Crop' to set that camera's ROI to the region."
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

    def _active_crop_target(self):
        """Return (params_widget, sensor_w, sensor_h) for the camera the crop tool
        should drive, based on the active tab.

        Both ORCA and EVK4 params widgets expose the same ROI interface
        (``spin_roi_width``/``spin_roi_height``, ``slider_offset_x``/``_y``,
        ``_compute_roi``), so the crop tool treats them uniformly — only the
        sensor dimensions differ between cameras."""
        if self.tabs.currentWidget() is self.tab_evk4:
            return self.evk4_params, EVK4_SENSOR_WIDTH, EVK4_SENSOR_HEIGHT
        return self.orca_params, ORCA_SENSOR_WIDTH, ORCA_SENSOR_HEIGHT

    def _active_crop_label(self):
        """Return the VideoFeedLabel for the camera the crop tool should drive."""
        if self.tabs.currentWidget() is self.tab_evk4:
            return self.video_label_evk4
        return self.video_label_orca

    def _active_crop_frame(self):
        """Last frame of the camera the crop tool targets, or None if no feed yet."""
        return (self._evk4_frame if self._active_crop_label() is self.video_label_evk4
                else self._orca_frame)

    def _toggle_crop_mode(self, on):
        label = self._active_crop_label()
        # Only one feed draws the crop overlay at a time; clear the other.
        for lbl in (self.video_label_orca, self.video_label_evk4):
            if lbl is not label:
                lbl.set_crop_mode(False)
        # Make sure the targeted feed is visible so the overlay can be seen/dragged.
        if on and not label.isVisible():
            cam = "evk4" if label is self.video_label_evk4 else "orca"
            self.combo_feed_view.setCurrentIndex(self.combo_feed_view.findData(cam))
        label.set_crop_mode(on)
        self._crop_label = label if on else None
        self.btn_crop_select.setText(
            "⛶  Selecting… (drag on image)" if on else "⛶  Select Crop Region"
        )
        if not on:
            self._crop_region = None
            self.btn_crop_apply.setEnabled(False)
            self.lbl_crop.setText("")

    def _frame_region_to_sensor(self, x, y, w, h):
        """Map a region in the displayed frame's pixels to absolute sensor pixels.

        The frame shown is the active ROI window (1:1, no binning), so the frame
        origin is that ROI's top-left — at full sensor this is the identity. Works
        for either camera via the active crop target's ROI.
        """
        params, _, _ = self._active_crop_target()
        roi = params._compute_roi()
        return roi["x_min"] + x, roi["y_min"] + y, w, h

    def _on_crop_region_drawn(self, x, y, w, h):
        """A rectangle was drawn on the feed: remember it and preview the ROI."""
        self._crop_region = (x, y, w, h)
        sx, sy, sw, sh = self._frame_region_to_sensor(x, y, w, h)
        self.btn_crop_apply.setEnabled(True)
        self.lbl_crop.setText(f"Selection: {sw} × {sh} px  @ ({sx}, {sy}) — click Apply Crop")

    def _apply_crop(self):
        """Set the active camera's ROI to the drawn region (re-applies live if running)."""
        if self._crop_region is None:
            return
        if self._active_crop_frame() is None:
            self.lbl_status.setText("Start a live feed before cropping.")
            return
        x, y, w, h = self._crop_region
        sx, sy, sw, sh = self._frame_region_to_sensor(x, y, w, h)
        params, sensor_w, sensor_h = self._active_crop_target()
        # Convert the absolute sensor rectangle into the widget's width/height +
        # centre-offset model; _compute_roi clamps (and, for ORCA, 4-px-aligns).
        offset_x = int(round(sx + sw / 2.0 - sensor_w / 2.0))
        offset_y = int(round(sy + sh / 2.0 - sensor_h / 2.0))
        params.spin_roi_width.setValue(min(sw, sensor_w))
        params.spin_roi_height.setValue(min(sh, sensor_h))
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
        self._active_crop_label().clear_selection()

    def _reset_crop(self):
        """Restore the active camera's ROI to its full sensor."""
        params, sensor_w, sensor_h = self._active_crop_target()
        params.spin_roi_width.setValue(sensor_w)
        params.spin_roi_height.setValue(sensor_h)
        params.slider_offset_x.setValue(0)
        params.slider_offset_y.setValue(0)
        self._crop_region = None
        self.btn_crop_apply.setEnabled(False)
        self.lbl_crop.setText("")
        self._active_crop_label().clear_selection()
        self.lbl_status.setText(
            f"ROI reset to full sensor ({sensor_w} × {sensor_h})."
        )
