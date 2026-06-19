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

# Approximate per-row readout times for the ORCA-Fusion C14440-20UP at each
# DCAM readout-speed setting. Derived from the Hamamatsu datasheet:
# Standard (2) → ~100 fps full-frame (2304 rows) → 10 ms / 2304 ≈ 4.34 µs/row
# Ultra Quiet (1) → ~30 fps → 33 ms / 2304 ≈ 14.3 µs/row
# Fast (3) → ~200 fps → 5 ms / 2304 ≈ 2.17 µs/row
ORCA_ROW_READOUT_US = {
    1.0: 14.3,   # Ultra Quiet
    2.0:  4.34,  # Standard (2)
    3.0:  2.17,  # Fast (3)
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
# Prophesee EVK4 (Metavision)
# ---------------------------------------------------------------------------
EVK4_ERC_RATE = 20_000_000          # Event Rate Controller cap (events/s)
EVK4_FPS = 25                       # PeriodicFrameGenerationAlgorithm display fps
EVK4_CRAZY_PIXEL_PERCENTILE = 99.9  # Hot-pixel rejection threshold

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
