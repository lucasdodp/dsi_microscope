"""Prophesee EVK4 event-based camera — Metavision SDK wrapper and acquisition worker.

`CameraWorker` streams events, generates live frames, optionally logs raw data, and
(for acquisition mode) reconstructs a 2D event-count image via the pure routines in
core.image_processing. Raw-data logging is always stopped in a finally block.
"""

import os
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import EVK4_ERC_RATE, EVK4_FPS
from core import (
    accumulate_event_frame, apply_smoothing, crop_to_roi, filter_crazy_pixels,
    save_mat_tif, save_parameter_log,
)

try:
    from metavision_core.event_io.raw_reader import initiate_device
    from metavision_core.event_io import EventsIterator
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    METAVISION_AVAILABLE = True
except ImportError:
    initiate_device = None
    EventsIterator = None
    PeriodicFrameGenerationAlgorithm = None
    ColorPalette = None
    METAVISION_AVAILABLE = False


def apply_event_roi(device, roi):
    """Best-effort: restrict the EVK4 sensor to the ROI window to cut the event
    rate. Returns True if a hardware ROI was set, False otherwise.

    Correctness never depends on this — the caller always software-crops the
    frame and the accumulated image to the same window (mirroring the ORCA's
    "hardware subarray, else software crop" fallback). The Metavision ROI API
    differs across SDK versions, so every step is guarded and a failure simply
    leaves the full sensor streaming.
    """
    if roi is None:
        return False
    w = roi["x_max"] - roi["x_min"]
    h = roi["y_max"] - roi["y_min"]
    try:
        i_roi = device.get_i_roi()
    except Exception:
        i_roi = None
    if not i_roi:
        return False
    try:
        try:
            window = i_roi.Window(roi["x_min"], roi["y_min"], w, h)
        except Exception:
            from metavision_hal import I_ROI
            window = I_ROI.Window(roi["x_min"], roi["y_min"], w, h)
        i_roi.set_window(window)
        i_roi.enable(True)
        return True
    except Exception:
        return False


class CameraWorker(QThread):
    """Event streaming / acquisition loop for the Prophesee EVK4."""

    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, mode, params):
        super().__init__()
        self.mode = mode
        self.params = params
        self._is_running = True
        self._pending_biases = None  # set by apply_biases(), consumed in the event loop
        # Software crop window (full-sensor pixels). None = full frame. Live drags
        # update it via apply_roi(); the frame callback crops to it each frame.
        self._roi = params.get("evk4_roi")

    def apply_biases(self, params):
        """Queue a bias update; applied on the next event batch (live mode only)."""
        self._pending_biases = params

    def apply_roi(self, roi):
        """Update the live crop window (live mode). Takes effect on the next frame.

        Only the software crop is changed — no device reconfiguration — so live
        dragging stays smooth and can re-frame to any sub-window of the full
        sensor. The hardware ROI (event-rate reduction) is set once at acquire
        startup, where the window is fixed.
        """
        self._roi = roi

    def run(self):
        if not METAVISION_AVAILABLE:
            self.error_signal.emit("Metavision SDK not found.")
            return

        try:
            self.status_update.emit("Initializing EVK4 device...")
            device = initiate_device("")

            # Low-level bias tuning (pixel cut-off frequencies & contrast thresholds).
            if device.get_i_ll_biases():
                device.get_i_ll_biases().set("bias_fo", self.params["bias_fo"])
                device.get_i_ll_biases().set("bias_hpf", self.params["bias_hpf"])
                device.get_i_ll_biases().set("bias_diff_on", self.params["bias_on"])
                device.get_i_ll_biases().set("bias_diff_off", self.params["bias_off"])

            erc = device.get_i_erc_module()
            if erc:
                erc.enable(True)
                erc.set_cd_event_rate(EVK4_ERC_RATE)

            # Acquire mode has a fixed ROI for the whole run, so set the hardware
            # ROI to cut the event rate (best-effort; the software crop below makes
            # the output correct regardless). Live mode keeps the full sensor
            # streaming so the user can re-crop to any sub-window on the fly.
            if self.mode == "acquire" and self._roi is not None:
                if apply_event_roi(device, self._roi):
                    self.status_update.emit("Hardware ROI enabled (reduced event rate).")

            log_path = ""
            if self.mode == "acquire":
                log_path = os.path.join(self.params["output_dir"], self.params["filename"] + ".raw")
                if device.get_i_events_stream():
                    device.get_i_events_stream().log_raw_data(log_path)
                    self.status_update.emit(f"Recording to {log_path}")

            mv_iterator = EventsIterator.from_device(device=device)
            height, width = mv_iterator.get_size()

            event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=width, sensor_height=height, fps=EVK4_FPS, palette=ColorPalette.Dark
            )

            def on_cd_frame_cb(ts, cd_frame):
                if self._is_running:
                    # Software-crop to the active ROI so the live view shows the
                    # cropped region (full sensor when no crop is set).
                    self.frame_ready.emit(crop_to_roi(cd_frame, self._roi).copy())

            event_frame_gen.set_output_callback(on_cd_frame_cb)

            self.status_update.emit(f"Running in {self.mode.upper()} mode...")
            start_time = time.time()
            acqu_time = self.params["acqu_time"]

            try:
                for evs in mv_iterator:
                    if not self._is_running:
                        break
                    if self._pending_biases is not None:
                        biases = self._pending_biases
                        self._pending_biases = None
                        if device.get_i_ll_biases():
                            device.get_i_ll_biases().set("bias_fo", biases["bias_fo"])
                            device.get_i_ll_biases().set("bias_hpf", biases["bias_hpf"])
                            device.get_i_ll_biases().set("bias_diff_on", biases["bias_on"])
                            device.get_i_ll_biases().set("bias_diff_off", biases["bias_off"])
                            self.status_update.emit("EVK4 biases updated.")
                    event_frame_gen.process_events(evs)

                    if self.mode == "acquire":
                        elapsed = time.time() - start_time
                        if elapsed >= acqu_time:
                            self.status_update.emit("Acquisition time reached. Stopping stream...")
                            break
            finally:
                # Only stop logging if we actually started it; calling stop without
                # a prior log_raw_data can crash the Metavision native library.
                if log_path and device.get_i_events_stream():
                    device.get_i_events_stream().stop_log_raw_data()

            if self.mode == "acquire" and self._is_running:
                self.status_update.emit("Processing final image from RAW data...")
                self.process_final_image(log_path, width, height)
                if log_path and not self.params.get("save_raw", True):
                    try:
                        os.remove(log_path)
                    except OSError:
                        pass

            self.status_update.emit("Camera stopped.")
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"EVK4 Hardware Error: {str(e)}")

    def process_final_image(self, raw_path, width, height):
        """Reconstruct, post-process and save the 2D event image from a raw log.

        Event iteration is hardware-specific (Metavision), but the accumulation,
        filtering, smoothing and saving are delegated to the pure core layer.
        """
        iterator = EventsIterator(input_path=raw_path, delta_t=1000000)
        final_image = accumulate_event_frame(iterator, width, height)
        # Crop to the same ROI the live view used, so the saved image matches the
        # framing the user selected (the raw log keeps every event for re-use).
        final_image = crop_to_roi(final_image, self._roi)

        # Always log the acquisition parameters, even if no events were recorded.
        metadata = self.params.get("metadata")
        if metadata:
            save_parameter_log(self.params["output_dir"], self.params["filename"], metadata)

        if np.max(final_image) == 0:
            self.status_update.emit("Warning: No events recorded. Final image is blank.")
            return

        if self.params["filter_crazy_pixels"]:
            self.status_update.emit("Filtering 'crazy' pixels...")
            final_image = filter_crazy_pixels(final_image)

        if self.params["apply_smoothing"]:
            self.status_update.emit("Applying spatial smoothing...")
            final_image = apply_smoothing(final_image)

        out_dir = save_mat_tif(final_image, self.params["output_dir"], self.params["filename"])
        self.status_update.emit(f"Saved processed .mat and .tif to {out_dir}")

    def stop(self):
        self._is_running = False
