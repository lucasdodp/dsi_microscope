"""Hamamatsu ORCA-fusion scientific camera — DCAM API wrapper and live worker.

The DCAM Python wrapper (`dcam.py`) lives in the SDK samples folder, so we extend
sys.path from config before importing. `OrcaWorker` runs the live-focus loop on a
dedicated thread and guarantees buffer release / device close via try/finally.
"""

import sys

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    DCAM_BINNING_PROP, DCAM_DEFECTCORRECT_PROP, DCAM_EXPOSURE_PROP,
    DCAM_READOUTSPEED_PROP, DCAM_TRIGGER_MODE_PROP, DCAM_TRIGGERSOURCE_PROP,
    DCAM_SUBARRAY_HPOS_PROP, DCAM_SUBARRAY_HSIZE_PROP,
    DCAM_SUBARRAY_VPOS_PROP, DCAM_SUBARRAY_VSIZE_PROP,
    DCAM_SUBARRAY_MODE_PROP, DCAM_SUBARRAY_MODE_OFF, DCAM_SUBARRAY_MODE_ON,
    ORCA_SENSOR_WIDTH, ORCA_SENSOR_HEIGHT,
    HAMAMATSU_SDK_PATH,
)
from core import (
    compute_dsi_images, normalize_to_8bit, save_dsi_results, save_parameter_log,
    save_raw_stack_tiff, scale_16bit_image,
)

# Extend the path to the Hamamatsu SDK sample wrapper before importing it.
if HAMAMATSU_SDK_PATH not in sys.path:
    sys.path.append(HAMAMATSU_SDK_PATH)

try:
    from dcam import Dcam, Dcamapi
    DCAM_AVAILABLE = True
except ImportError:
    Dcam = None
    Dcamapi = None
    DCAM_AVAILABLE = False


class OrcaWorker(QThread):
    """ORCA acquisition worker: live-focus display or single-z DSI acquisition."""

    image_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, mode, params):
        super().__init__()
        self.mode = mode
        self.params = params
        self._is_running = True
        self._pending_params = None  # set by apply_params(), consumed in _run_live
        self._hw_roi_active = False  # True when DCAM subarray mode is ON

    def apply_params(self, params):
        """Queue a parameter update; applied on the next live frame (live mode only)."""
        self._pending_params = params

    def run(self):
        if not DCAM_AVAILABLE:
            self.error_signal.emit("DCAM API not found.")
            return

        exposure_time = self.params["orca_exposure"] / 1000.0
        _initialized = False
        _opened = False
        dcam = None
        try:
            self.status_update.emit("Initializing ORCA DCAM API...")
            if not Dcamapi.init():
                raise RuntimeError(f"Dcamapi.init() failed with error {Dcamapi.lasterr()}")
            _initialized = True

            dcam = Dcam(0)
            if not dcam.dev_open():
                raise RuntimeError(f"dev_open() failed with error {dcam.lasterr()}")
            _opened = True

            dcam.prop_setvalue(DCAM_EXPOSURE_PROP, exposure_time)
            self._apply_camera_settings(dcam)
            if self.mode == "live":
                self._run_live(dcam)
            elif self.mode == "acquire":
                self._run_acquire(dcam)

            self.status_update.emit("ORCA stopped.")
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"ORCA Hardware Error: {str(e)}")

        finally:
            # Always close the device and uninit in the correct order,
            # regardless of which code path raised. Guards prevent double-calls.
            if _opened and dcam is not None:
                try:
                    dcam.dev_close()
                except Exception:
                    pass
            if _initialized:
                try:
                    Dcamapi.uninit()
                except Exception:
                    pass

    def _apply_camera_settings(self, dcam, params=None):
        """Push the user-selected DCAM mode/readout properties to the camera.

        Applied before buffer allocation so size-changing settings (e.g. binning
        and subarray) take effect. Rejected values are reported but do not abort.
        Accepts an explicit params dict for live updates; falls back to self.params.
        """
        p = params if params is not None else self.params
        settings = [
            ("Readout speed", DCAM_READOUTSPEED_PROP, p.get("readout_speed")),
            ("Binning", DCAM_BINNING_PROP, p.get("binning")),
            ("Trigger source", DCAM_TRIGGERSOURCE_PROP, p.get("trigger_source")),
            ("Trigger mode", DCAM_TRIGGER_MODE_PROP, p.get("trigger_mode")),
            ("Defect correction", DCAM_DEFECTCORRECT_PROP, p.get("defect_correct")),
        ]
        for name, prop_id, value in settings:
            if value is None:
                continue
            if not dcam.prop_setvalue(prop_id, value):
                self.status_update.emit(f"Warning: ORCA rejected {name} = {value} (err {dcam.lasterr()})")

        # Hardware subarray must be applied AFTER binning so coordinates are
        # relative to the (possibly binned) sensor. Updates self._hw_roi_active.
        self._hw_roi_active = self._apply_subarray(dcam, p.get("orca_roi"))

    def _apply_subarray(self, dcam, roi):
        """Configure DCAM hardware subarray mode. Returns True if active."""
        # Always turn MODE OFF first — required before changing position/size.
        dcam.prop_setvalue(DCAM_SUBARRAY_MODE_PROP, DCAM_SUBARRAY_MODE_OFF)

        if roi is None:
            return False

        w = roi["x_max"] - roi["x_min"]
        h = roi["y_max"] - roi["y_min"]

        # Skip hardware subarray for full-sensor ROI.
        if w >= ORCA_SENSOR_WIDTH and h >= ORCA_SENSOR_HEIGHT:
            return False

        ok = (
            dcam.prop_setvalue(DCAM_SUBARRAY_HPOS_PROP,  float(roi["x_min"])) and
            dcam.prop_setvalue(DCAM_SUBARRAY_HSIZE_PROP, float(w)) and
            dcam.prop_setvalue(DCAM_SUBARRAY_VPOS_PROP,  float(roi["y_min"])) and
            dcam.prop_setvalue(DCAM_SUBARRAY_VSIZE_PROP, float(h)) and
            dcam.prop_setvalue(DCAM_SUBARRAY_MODE_PROP,  DCAM_SUBARRAY_MODE_ON)
        )
        if not ok:
            self.status_update.emit(
                f"Warning: hardware subarray ({roi['x_min']},{roi['y_min']}) "
                f"{w}×{h} px rejected (err {dcam.lasterr()}); using software crop."
            )
            dcam.prop_setvalue(DCAM_SUBARRAY_MODE_PROP, DCAM_SUBARRAY_MODE_OFF)
            return False
        return True

    def _run_live(self, dcam):
        """Continuous live-focus display loop (3-frame ring buffer).

        Supports live parameter updates via apply_params(): on the next frame
        the loop stops capture, applies the new exposure/mode settings, and
        restarts — producing a brief blink that is much faster than a full
        worker restart.
        """
        buf_allocated = False
        try:
            if not dcam.buf_alloc(3):
                raise RuntimeError(f"buf_alloc(3) failed: {dcam.lasterr()}")
            buf_allocated = True
            if dcam.cap_start():
                while self._is_running:
                    if self._pending_params is not None:
                        pending = self._pending_params
                        self._pending_params = None
                        self.params = pending  # keep ROI and other display settings current
                        dcam.cap_stop()
                        dcam.buf_release()
                        buf_allocated = False
                        dcam.prop_setvalue(DCAM_EXPOSURE_PROP, pending["orca_exposure"] / 1000.0)
                        self._apply_camera_settings(dcam, pending)
                        if not dcam.buf_alloc(3):
                            raise RuntimeError(f"buf_alloc(3) failed after param update: {dcam.lasterr()}")
                        buf_allocated = True
                        if not dcam.cap_start():
                            raise RuntimeError(f"cap_start() failed after param update: {dcam.lasterr()}")
                        self.status_update.emit("ORCA parameters updated.")
                        continue
                    if dcam.wait_capevent_frameready(100):
                        data = dcam.buf_getlastframedata()
                        if data is not False:
                            img = scale_16bit_image(data)
                            # Hardware subarray already delivers the cropped frame;
                            # only apply software crop when the camera rejected it.
                            if not self._hw_roi_active:
                                roi = self.params.get("orca_roi")
                                if roi:
                                    fh, fw = img.shape[:2]
                                    y1 = max(0, min(roi["y_min"], fh))
                                    y2 = max(y1 + 1, min(roi["y_max"], fh))
                                    x1 = max(0, min(roi["x_min"], fw))
                                    x2 = max(x1 + 1, min(roi["x_max"], fw))
                                    img = img[y1:y2, x1:x2]
                            self.image_ready.emit(img)
                    else:
                        if not dcam.lasterr().is_timeout():
                            break
                dcam.cap_stop()
        finally:
            if buf_allocated:
                dcam.buf_release()

    def _run_acquire(self, dcam):
        """Single-z DSI acquisition: record N speckle frames, then compute and
        save the average (widefield) and standard-deviation (DSI) images.

        The objective is held at its current focus throughout; the optical
        sectioning comes from the statistics across the stack.
        """
        num_frames = self.params["orca_frames"]
        roi = self.params["orca_roi"]
        out_dir = self.params.get("output_dir", "")
        filename = self.params.get("filename", "dsi")

        if not dcam.buf_alloc(num_frames):
            raise RuntimeError(f"buf_alloc({num_frames}) failed: {dcam.lasterr()}")

        acquired_stack = []
        try:
            if dcam.cap_start():
                for i in range(num_frames):
                    if not self._is_running:
                        break
                    self.status_update.emit(f"Acquiring speckle frame {i + 1}/{num_frames}...")
                    if dcam.wait_capevent_frameready(2000):
                        data = dcam.buf_getframedata(i)
                        acquired_stack.append(np.copy(data))
                        self.image_ready.emit(scale_16bit_image(data))  # live preview
                    else:
                        raise RuntimeError(f"Frame timeout: {dcam.lasterr()}")
                dcam.cap_stop()
        finally:
            dcam.buf_release()

        if not self._is_running:
            self.status_update.emit("DSI acquisition cancelled.")
            return
        if len(acquired_stack) < 2:
            self.status_update.emit("Not enough frames acquired to compute DSI statistics.")
            return

        raw_stack = np.array(acquired_stack)

        self.status_update.emit("Computing average (widefield) and standard-deviation (DSI) images...")
        # If hardware subarray was active the frames are already the correct size;
        # pass roi=None so compute_dsi_images does not double-crop them.
        dsi_roi = None if self._hw_roi_active else roi
        avg_img, std_img = compute_dsi_images(raw_stack, dsi_roi)

        # Display the optically-sectioned DSI image.
        self.image_ready.emit(normalize_to_8bit(std_img))

        if out_dir:
            if self.params.get("save_raw_stack", True):
                self.status_update.emit("Saving raw 16-bit speckle stack (3D TIFF)...")
                save_raw_stack_tiff(raw_stack, out_dir, filename, roi)
            save_dsi_results(avg_img, std_img, out_dir, filename)
            metadata = self.params.get("metadata")
            if metadata:
                save_parameter_log(out_dir, filename, metadata)
            self.status_update.emit(f"Saved raw stack + average + DSI images and parameter log to {out_dir}")
        else:
            self.status_update.emit("DSI images computed (not saved: no output directory set).")

    def stop(self):
        self._is_running = False
