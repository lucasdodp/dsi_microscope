"""Hamamatsu ORCA-fusion scientific camera — DCAM API wrapper and live worker.

The DCAM Python wrapper (`dcam.py`) lives in the SDK samples folder, so we extend
sys.path from config before importing. `OrcaWorker` runs the live-focus loop on a
dedicated thread and guarantees buffer release / device close via try/finally.
"""

import sys
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    DCAM_BINNING_PROP, DCAM_DEFECTCORRECT_PROP, DCAM_EXPOSURE_PROP,
    DCAM_READOUTSPEED_PROP, DCAM_TRIGGER_MODE_PROP, DCAM_TRIGGERSOURCE_PROP,
    DCAM_TRIGGERSOURCE_INTERNAL,
    DCAM_SUBARRAY_HPOS_PROP, DCAM_SUBARRAY_HSIZE_PROP,
    DCAM_SUBARRAY_VPOS_PROP, DCAM_SUBARRAY_VSIZE_PROP,
    DCAM_SUBARRAY_MODE_PROP, DCAM_SUBARRAY_MODE_OFF, DCAM_SUBARRAY_MODE_ON,
    ORCA_SENSOR_WIDTH, ORCA_SENSOR_HEIGHT, PREVIEW_MAX_FPS,
    HAMAMATSU_SDK_PATH,
)
from core import (
    autocontrast_8bit, compute_dsi_images, crop_to_roi, normalize_to_8bit,
    save_dsi_results, save_parameter_log, save_raw_stack_tiff, save_single_image,
    scale_16bit_image,
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


# The ORCA (DCAM) allows only ONE process to hold the camera at a time. The most
# common reason a connect that works in HCImage Live fails here is that HCImage
# Live — or a previous, still-running instance of this app — is holding the
# device open. This hint is appended to the init/open error so the cause is
# actionable rather than a bare error code.
_ORCA_BUSY_HINT = (
    "The ORCA can be opened by only one program at a time — close HCImage Live "
    "(and any other instance of this app) completely, then try again. If it "
    "stays locked after a crash, unplug/replug the camera USB or power-cycle it."
)


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
        # Display-only auto-contrast (HCImage-style percentile stretch). Toggled
        # live via set_auto_contrast() without a capture restart, since it only
        # changes how frames are mapped to 8-bit for the preview.
        self._auto_contrast = bool(params.get("auto_contrast", True))

    def apply_params(self, params):
        """Queue a parameter update; applied on the next live frame (live mode only)."""
        self._pending_params = params

    def set_auto_contrast(self, on):
        """Toggle display auto-contrast on the running live feed (no restart)."""
        self._auto_contrast = bool(on)

    def _scale_for_display(self, data):
        """Map a raw 16-bit frame to 8-bit for the preview, honouring the
        auto-contrast toggle. Never touches the data used for the saved stack /
        DSI statistics — this is display only."""
        if self._auto_contrast:
            return autocontrast_8bit(data)
        return scale_16bit_image(data)

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
                raise RuntimeError(
                    f"Dcamapi.init() failed with error {Dcamapi.lasterr()}. "
                    f"{_ORCA_BUSY_HINT}")
            _initialized = True

            dcam = Dcam(0)
            if not dcam.dev_open():
                raise RuntimeError(
                    f"dev_open() failed with error {dcam.lasterr()}. "
                    f"{_ORCA_BUSY_HINT}")
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
        # A live focus preview must free-run. The app never fires a software (or
        # external) trigger, so if the selected trigger source is anything other
        # than Internal the camera opens and cap_start() succeeds but every
        # wait_capevent_frameready() times out — no frame is ever delivered and
        # the feed stays on "ORCA Feed Offline". Force Internal for the live feed
        # regardless of the acquisition trigger source the user picked.
        trigger_source = p.get("trigger_source")
        if self.mode == "live":
            trigger_source = DCAM_TRIGGERSOURCE_INTERNAL
        settings = [
            ("Readout speed", DCAM_READOUTSPEED_PROP, p.get("readout_speed")),
            ("Binning", DCAM_BINNING_PROP, p.get("binning")),
            ("Trigger source", DCAM_TRIGGERSOURCE_PROP, trigger_source),
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
                # Measured display framerate, reported ~twice a second so the
                # real rate can be compared against the ROI prediction. This is
                # the *display* rate; the true camera throughput is what the
                # acquisition reports below. The preview is capped at the monitor
                # refresh (PREVIEW_MAX_FPS) — drawing faster is wasted work — so at
                # or below that cap it now tracks the real camera rate.
                min_interval = 1.0 / PREVIEW_MAX_FPS if PREVIEW_MAX_FPS > 0 else 0.0
                last_emit = 0.0
                fps_count = 0
                fps_t0 = time.perf_counter()
                while self._is_running:
                    if self._pending_params is not None:
                        pending = self._pending_params
                        self._pending_params = None
                        self.params = pending  # keep ROI and other display settings current
                        self._auto_contrast = bool(pending.get("auto_contrast", self._auto_contrast))
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
                        fps_count = 0
                        fps_t0 = time.perf_counter()
                        continue
                    if dcam.wait_capevent_frameready(100):
                        # Cap the preview at the monitor refresh. When a frame is
                        # ready but the last emit was under one interval ago, skip
                        # it (the camera still captures it) without the heavy
                        # read/scale/emit, sleeping briefly so the loop doesn't spin
                        # a core waiting out the interval.
                        now = time.perf_counter()
                        if min_interval and now - last_emit < min_interval:
                            time.sleep(0.002)
                            continue
                        data = dcam.buf_getlastframedata()
                        if data is not False:
                            last_emit = now
                            img = self._scale_for_display(data)
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
                            fps_count += 1
                            elapsed = time.perf_counter() - fps_t0
                            if elapsed >= 0.5:
                                self.status_update.emit(
                                    f"Live display: ≈ {fps_count / elapsed:.0f} fps"
                                )
                                fps_count = 0
                                fps_t0 = time.perf_counter()
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

        # Copy each frame straight into one preallocated array instead of a
        # growing list of np.copy'd frames followed by a np.array() of it (that
        # kept two full copies of the stack in RAM). The camera's ring buffer
        # already holds N frames, so this keeps host-side overhead to a single
        # copy and avoids the memory spike that could swap the machine.
        raw_stack = None
        count = 0
        # Throttle the live preview + status updates: at high frame rates,
        # scaling and emitting every single frame would cap the read loop well
        # below the camera's true rate. The camera fills the ring buffer at full
        # speed regardless; we just read it out as fast as possible and refresh
        # the preview ~20 times over the run.
        preview_every = max(1, num_frames // 20)
        capture_start = None
        capture_elapsed = 0.0
        try:
            if dcam.cap_start():
                for i in range(num_frames):
                    if not self._is_running:
                        break
                    if dcam.wait_capevent_frameready(2000):
                        if capture_start is None:
                            capture_start = time.perf_counter()  # first frame in hand
                        data = dcam.buf_getframedata(i)
                        if raw_stack is None:
                            raw_stack = np.empty((num_frames,) + data.shape, dtype=data.dtype)
                        raw_stack[i] = data  # copies out of the SDK ring buffer
                        count += 1
                        if i % preview_every == 0:
                            self.status_update.emit(f"Acquiring speckle frame {i + 1}/{num_frames}...")
                            self.image_ready.emit(self._scale_for_display(data))  # live preview
                    else:
                        raise RuntimeError(f"Frame timeout: {dcam.lasterr()}")
                if capture_start is not None:
                    capture_elapsed = time.perf_counter() - capture_start
                dcam.cap_stop()
        finally:
            dcam.buf_release()

        if not self._is_running:
            self.status_update.emit("Acquisition cancelled.")
            return
        if count < 1:
            self.status_update.emit("No frames acquired.")
            return

        # Trim if the run was cut short (e.g. stopped mid-capture).
        if count != num_frames:
            raw_stack = raw_stack[:count]

        # Hardware subarray already delivers the cropped frame; only crop in
        # software when the camera rejected the subarray (so the saved data is
        # never double-cropped).
        proc_roi = None if self._hw_roi_active else roi

        # A single frame has no stack to compute a standard deviation over, so
        # there is no optical sectioning: save it as a plain widefield snapshot.
        if count == 1:
            img = crop_to_roi(raw_stack[0], proc_roi)
            self.image_ready.emit(self._scale_for_display(img))
            if out_dir:
                self.status_update.emit("Saving single image...")
                save_single_image(img, out_dir, filename)
                metadata = self.params.get("metadata")
                if metadata:
                    save_parameter_log(out_dir, filename, metadata)
                self.status_update.emit(
                    f"Saved single image (no DSI — one frame) and parameter log to {out_dir}"
                )
            else:
                self.status_update.emit(
                    "Single image acquired (not saved: no output directory set)."
                )
            return

        # Measured capture throughput — the number that actually determines how
        # long an acquisition takes. Appended to the final status line so it
        # stays visible after the computing/saving messages.
        n = count
        fps_msg = ""
        if capture_elapsed > 0 and n > 1:
            fps_msg = (
                f"  [captured {n} frames in {capture_elapsed:.3f} s "
                f"≈ {(n - 1) / capture_elapsed:.0f} fps]"
            )

        self.status_update.emit("Computing average (widefield) and standard-deviation (DSI) images...")
        # proc_roi is None when the hardware subarray already cropped the frames,
        # so compute_dsi_images does not double-crop them.
        avg_img, std_img = compute_dsi_images(raw_stack, proc_roi)

        # Display the optically-sectioned DSI image.
        self.image_ready.emit(normalize_to_8bit(std_img))

        if out_dir:
            if self.params.get("save_raw_stack", True):
                self.status_update.emit("Saving raw 16-bit speckle stack (3D TIFF)...")
                save_raw_stack_tiff(raw_stack, out_dir, filename, roi)
            save_dsi_results(avg_img, std_img, out_dir, filename)
            metadata = self.params.get("metadata")
            if metadata:
                # Record the *measured* capture framerate alongside the settings,
                # so the log carries the real rate the frames were recorded at.
                if capture_elapsed > 0 and n > 1:
                    metadata = dict(metadata)
                    metadata["Measured performance (ORCA)"] = {
                        "measured_framerate_fps": f"{(n - 1) / capture_elapsed:.1f}",
                        "total_capture_time_s": f"{capture_elapsed:.3f}",
                        "frames_timed": n,
                    }
                save_parameter_log(out_dir, filename, metadata)
            self.status_update.emit(
                f"Saved raw stack + average + DSI images and parameter log to {out_dir}{fps_msg}"
            )
        else:
            self.status_update.emit(
                f"DSI images computed (not saved: no output directory set).{fps_msg}"
            )

    def stop(self):
        self._is_running = False
