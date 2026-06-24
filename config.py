"""Global configuration: SDK paths, hardware constants, and the Qt stylesheet.

This module must remain dependency-light (only the stdlib) so it can be safely
imported from every layer (core / hardware / ui).

Machine-specific settings (the Hamamatsu SDK path and the PI controller
identifiers) can be overridden with environment variables, so the same checkout
runs on the laptop and the lab PC without editing this file. See README.md.
"""

import os

# ---------------------------------------------------------------------------
# SDK / Driver paths
# ---------------------------------------------------------------------------
# Hamamatsu DCAM-API ships its Python wrapper (`dcam.py`) inside the SDK samples
# folder. We append this to sys.path at import time in hardware/orca_camera.py.
# The default looks for dcamsdk4/samples/python/ relative to this file, which
# works when the SDK folder is copied alongside the project (the recommended layout).
# Override with the HAMAMATSU_SDK_PATH environment variable if the SDK lives elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
HAMAMATSU_SDK_PATH = os.environ.get(
    "HAMAMATSU_SDK_PATH",
    os.path.join(_HERE, "dcamsdk4", "samples", "python"),
)

# ---------------------------------------------------------------------------
# Physik Instrumente (PI) E-709 piezo controller
# ---------------------------------------------------------------------------
PI_CONTROLLER_NAME = os.environ.get("PI_CONTROLLER_NAME", "E-709")  # GCS controller model
PI_SERIAL_NUM = os.environ.get("PI_SERIAL_NUM", "0023550769")  # PI native-USB serial (tried if reachable)
PI_AXIS = "1"                  # default axis id; auto-detected via qSAI() on connect

# The E-709 is commonly accessed over RS-232 through a USB-to-serial adapter
# (e.g. the PI US232R, which enumerates as a virtual COM port) rather than PI
# native USB. Set PI_RS232_PORT to a COM number (e.g. 5) to force that exact
# port; leave it None to auto-scan the available serial ports.
# Set the PI_RS232_PORT environment variable to a COM number (e.g. "5") to force
# that exact port on a given machine; leave it unset to auto-scan.
PI_RS232_PORT = int(os.environ["PI_RS232_PORT"]) if os.environ.get("PI_RS232_PORT") else None
PI_BAUDRATE = 115200

# The E-709 piezo controller has NO reference switches, so there is no FNL/FRF
# referencing step. Closed-loop control is enabled via the servo (SVO). Auto-zero
# (ATZ) is optional and off by default: it is slow and unnecessary for the
# absolute capacitive/SGS sensor.
PI_AUTOZERO = False

# ---------------------------------------------------------------------------
# Hamamatsu ORCA (DCAM)
# ---------------------------------------------------------------------------
# DCAM property ids (from dcamprop.h, stable across DCAM-API 4). EXPOSURETIME
# takes a value in seconds; the others take one of the enum values listed in the
# *_OPTIONS maps below.
DCAM_EXPOSURE_PROP = 0x001F0110
DCAM_READOUTSPEED_PROP = 0x00400110
DCAM_BINNING_PROP = 0x00401110
DCAM_TRIGGERSOURCE_PROP = 0x00100110
DCAM_TRIGGER_MODE_PROP = 0x00100210
DCAM_DEFECTCORRECT_PROP = 0x00470010

# Per-row readout time ("1H") for the ORCA-Fusion BT C15440-20UP at each DCAM
# readout-speed setting. Taken directly from the instruction manual (§11-1-4
# "Frame rate Calculation" and §14-1), NOT estimated:
#   Fast scan        1H = 4.86765 µs  → 1/((2304+1)*1H) ≈ 89.1 fps full frame
#   Standard scan    1H = 18.64706 µs → 1/((2304+1)*1H) ≈ 23.2 fps full frame
#   Ultra quiet scan 1H = 80.0 µs     → 1/((2304+1)*1H) ≈  5.42 fps full frame
# The free-running frame time is (Vn+1)*1H where Vn is the number of vertical
# lines read out, so reducing the subarray HEIGHT is what raises the framerate
# (e.g. 144 rows on Fast scan → ~1400 fps, matching HCImageLive).
ORCA_ROW_READOUT_US = {
    1.0: 80.0,      # Ultra Quiet (1)
    2.0: 18.64706,  # Standard (2)
    3.0: 4.86765,   # Fast (3)
}

# Hardware subarray (ROI) property ids — ORCA-Fusion C14440-20UP sensor is
# 2304 × 2304 px. SUBARRAYMODE 1=OFF 2=ON; must be turned OFF before changing
# position/size, then turned ON. Values must be multiples of 4 and satisfy
# HPOS+HSIZE ≤ ORCA_SENSOR_WIDTH, VPOS+VSIZE ≤ ORCA_SENSOR_HEIGHT.
ORCA_SENSOR_WIDTH  = 2304
ORCA_SENSOR_HEIGHT = 2304
DCAM_SUBARRAY_HPOS_PROP  = 0x00402110
DCAM_SUBARRAY_HSIZE_PROP = 0x00402120
DCAM_SUBARRAY_VPOS_PROP  = 0x00402130
DCAM_SUBARRAY_VSIZE_PROP = 0x00402140
DCAM_SUBARRAY_MODE_PROP  = 0x00402150
DCAM_SUBARRAY_MODE_OFF   = 1.0
DCAM_SUBARRAY_MODE_ON    = 2.0

# Selectable enum values exposed in the ORCA tab, as {label: dcam value}.
# These match the ORCA-Fusion C14440-20UP. prop_setvalue() failures are caught
# and surfaced as a status warning, so an unsupported value can't crash a run.
DCAM_READOUTSPEED_OPTIONS = {
    "Ultra Quiet (1)": 1.0,
    "Standard (2)": 2.0,
    "Fast (3)": 3.0,
}
DCAM_BINNING_OPTIONS = {
    "1 x 1": 1.0,
    "2 x 2": 2.0,
    "4 x 4": 4.0,
}
DCAM_TRIGGERSOURCE_OPTIONS = {
    "Internal": 1.0,
    "External": 2.0,
    "Software": 3.0,
    "Master Pulse": 4.0,
}
DCAM_TRIGGER_MODE_OPTIONS = {
    "Normal": 1.0,
    "Start": 6.0,
}
DCAM_DEFECTCORRECT_OPTIONS = {
    "Off": 1.0,
    "On": 2.0,
}

# ---------------------------------------------------------------------------
# Duration-estimate model (the "≈ … s" labels only)
# ---------------------------------------------------------------------------
# The estimate is a rough model so users know roughly how long to wait; the
# *live elapsed timer* shown during an acquisition is the ground truth. These
# constants are only the COLD-START model (before any run has been recorded):
# once acquisitions complete, their measured elapsed times calibrate the estimate
# per acquisition type (see ACQUISITION_HISTORY_PATH and MainWindow._calibrate),
# so machine-specific overheads the constants can't predict are learned. They
# account for the costs a naive N×frame_time estimate ignores: camera start-up,
# the per-plane motor move + settle, the EVK4's per-plane device re-init, and the
# raw-stack disk write. Deliberately conservative — better to slightly over- than
# under-estimate (and the per-plane overhead, dominated by motor settle + serial
# round-trips + DCAM buffer setup + first-frame latency, was measured higher than
# the original 1.3 s).
ORCA_CAMERA_INIT_S = 1.5         # Dcamapi.init + dev_open + buffer allocation
ORCA_PLANE_OVERHEAD_S = 2.0      # piezo move + ~0.5 s settle + serial round-trips + buffer/cap setup
EVK4_PLANE_OVERHEAD_S = 4.0      # per-plane initiate_device + raw readback/accumulate
ZSTACK_DISK_BYTES_PER_S = 150e6  # assumed sustained disk write rate for the raw TIFF
# Throughput of the DSI reconstruction (per-pixel average + std across the N-frame
# stack, computed in memory-bounded float64 chunks). This per-plane compute was
# previously left out of the estimate entirely, which is the main reason a full-
# sensor Z-stack ran much longer than predicted: at full sensor it is ~5–6 s PER
# PLANE (≈90–100 Mpx/s measured; the conservative value here slightly over-estimates).
ORCA_DSI_PROCESS_PIXELS_PER_S = 85e6

# ---------------------------------------------------------------------------
# Session state — parameters from the last run, auto-restored on startup
# ---------------------------------------------------------------------------
# On exit the app writes the current camera / Z-stack parameters here and reloads
# them on the next launch, so a session starts where the previous one left off
# (the manual Save/Load Preset buttons still work for named presets). Stored under
# the per-user app-data dir to avoid cluttering the repo. Override with the
# DSI_SESSION_STATE env var.
SESSION_STATE_PATH = os.environ.get(
    "DSI_SESSION_STATE",
    os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
        "DSIMicroscope", "last_session.json",
    ),
)

# ---------------------------------------------------------------------------
# Acquisition-time learning — measured elapsed times calibrate the estimate
# ---------------------------------------------------------------------------
# Each completed acquisition records (predicted_s, actual_s) here, keyed by type
# (orca_single / orca_zstack / evk4_single / evk4_zstack). The duration estimate
# is then scaled by the learned median actual/predicted ratio for that type, so
# it converges to the real time on this machine after a couple of runs instead of
# relying on hand-tuned constants. Stored next to the session state; env-override
# with DSI_ACQ_HISTORY.
ACQUISITION_HISTORY_PATH = os.environ.get(
    "DSI_ACQ_HISTORY",
    os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
        "DSIMicroscope", "acquisition_history.json",
    ),
)
ACQUISITION_HISTORY_MAX = 40  # most recent runs kept per acquisition type

# ---------------------------------------------------------------------------
# Prophesee EVK4 (Metavision)
# ---------------------------------------------------------------------------
EVK4_ERC_RATE = 20_000_000          # Event Rate Controller cap (events/s)
EVK4_FPS = 25                       # PeriodicFrameGenerationAlgorithm display fps
EVK4_CRAZY_PIXEL_PERCENTILE = 99.9  # Hot-pixel rejection threshold
# IMX636 sensor geometry — the EVK4 streams events in this full-sensor pixel
# space regardless of any ROI, so the crop window is expressed against it.
EVK4_SENSOR_WIDTH = 1280
EVK4_SENSOR_HEIGHT = 720

# ---------------------------------------------------------------------------
# Qt stylesheet (dark theme)
# ---------------------------------------------------------------------------
STYLESHEET = """
QMainWindow { background-color: #1e1e1e; }
QWidget { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #e0e0e0; }
QGroupBox { font-weight: bold; border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 12px; padding-top: 15px; background-color: #252526; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #4daaf2; left: 10px; }
QLabel { color: #cccccc; }

QLineEdit, QComboBox {
    background-color: #333333;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    color: #ffffff;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #4daaf2; }
QComboBox::drop-down { border-left: 1px solid #555555; width: 20px; }

QSpinBox, QDoubleSpinBox {
    background-color: #333333;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 25px 4px 8px;
    min-height: 24px;
    color: #ffffff;
}
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #4daaf2; }

QPushButton {
    background-color: #3a3f44;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 24px;
    color: white;
    font-weight: bold;
}
QPushButton:hover { background-color: #4a5056; }
QPushButton:pressed { background-color: #2b2f33; }
QPushButton#btnLive { background-color: #007acc; }
QPushButton#btnLive:hover { background-color: #0098ff; }
QPushButton#btnAcquire { background-color: #2e7d32; }
QPushButton#btnAcquire:hover { background-color: #388e3c; }
QPushButton#btnStop { background-color: #c62828; }
QPushButton#btnStop:hover { background-color: #d32f2f; }
QPushButton:disabled { background-color: #555555; color: #888888; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; background-color: #333333; border: 1px solid #555555; border-radius: 3px; }
QCheckBox::indicator:checked { background-color: #4daaf2; border: 1px solid #4daaf2; }
QTabWidget::pane { border: 1px solid #3a3a3a; border-radius: 4px; background-color: #252526; }
QTabBar::tab { background: #333333; color: #cccccc; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
QTabBar::tab:selected { background: #4daaf2; color: white; font-weight: bold; }

QScrollArea { background-color: transparent; border: none; }
QScrollArea > QWidget { background-color: #1e1e1e; }
QScrollBar:vertical { background: #252526; width: 10px; margin: 0; border-radius: 5px; }
QScrollBar::handle:vertical { background: #4a5056; min-height: 30px; border-radius: 5px; }
QScrollBar::handle:vertical:hover { background: #5a616b; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""
