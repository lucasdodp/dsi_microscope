"""Prophesee EVK4 event-based camera — Metavision SDK wrapper and acquisition worker.

`CameraWorker` streams events, generates live frames, records the event data, and
(for acquisition mode) reconstructs a 2D event-count image via the pure routines in
core.image_processing.

An acquisition records its events in one of two formats, selected by the
``evk4_save_format`` parameter (see config.EVK4_SAVE_FORMAT_*):

  * ``evt3`` — the SDK logs the camera's encoded stream to ``<name>.raw`` and the
    image is reconstructed afterwards from that complete file. Lossless.
  * ``csv``  — no ``.raw``; events are streamed to ``<name>_xytp.csv`` by
    ``core.EventCsvWriter`` as they arrive, and the image counts exactly what was
    written. Lossy under load, and the loss is reported.

Whichever is in use is always shut down in a finally block: raw logging must be
stopped before the device is released, and the CSV writer holds the only copy of
the events it has queued.
"""

import gc
import os
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    EVK4_ERC_RATE, EVK4_FPS, EVK4_SAVE_FORMAT_CSV, EVK4_SAVE_FORMAT_DEFAULT,
)
from core import (
    EventCsvWriter, accumulate_event_frame, apply_smoothing, crop_to_roi,
    filter_crazy_pixels, save_mat_tif, save_parameter_log,
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

    def _set_biases(self, device, params):
        """Push the four biases to the device, one at a time, never raising.

        Each bias is set independently and its failure isolated: the SDK rejects a
        value for reasons the spin-box ranges don't capture (per-device limits, and
        ordering constraints between bias_diff_on/off), and a rejected *parameter*
        must not be able to kill the stream. Previously any failure here escaped
        into ``run()``'s handler, which reported it as an "EVK4 Hardware Error",
        tore the live feed down and orphaned the USB handle — so a single bad value
        left the camera unopenable until the app restarted.

        Returns a list of human-readable failures (empty when all four applied).
        """
        biases = device.get_i_ll_biases()
        if not biases:
            return ["this device exposes no low-level bias interface"]
        failures = []
        for name, key in (("bias_fo", "bias_fo"), ("bias_hpf", "bias_hpf"),
                          ("bias_diff_on", "bias_on"), ("bias_diff_off", "bias_off")):
            value = params[key]
            try:
                biases.set(name, value)
            except Exception as e:
                # Report the UI's name for the bias, not the SDK's, so the message
                # points at the field the user actually typed into.
                failures.append(f"{key}={value} rejected ({e})")
        return failures

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

        device = None
        mv_iterator = None
        try:
            self.status_update.emit("Initializing EVK4 device...")
            device = initiate_device("")

            # Low-level bias tuning (pixel cut-off frequencies & contrast thresholds).
            # A rejected value is reported but does not abort the run: the camera
            # simply keeps that bias at its previous setting.
            failures = self._set_biases(device, self.params)
            if failures:
                self.status_update.emit("EVK4 bias warning: " + "; ".join(failures))

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

            # EVT3 vs CSV: the raw logger must be armed *before* the iterator is
            # created, whereas the CSV writer needs the sensor geometry that only
            # the iterator can report — hence the two are set up on either side of
            # it rather than together.
            save_format = self.params.get("evk4_save_format", EVK4_SAVE_FORMAT_DEFAULT)
            csv_mode = save_format == EVK4_SAVE_FORMAT_CSV

            log_path = ""
            if self.mode == "acquire" and not csv_mode:
                log_path = os.path.join(self.params["output_dir"], self.params["filename"] + ".raw")
                if device.get_i_events_stream():
                    device.get_i_events_stream().log_raw_data(log_path)
                    self.status_update.emit(f"Recording to {log_path}")

            mv_iterator = EventsIterator.from_device(device=device)
            height, width = mv_iterator.get_size()

            csv_writer = None
            if self.mode == "acquire" and csv_mode:
                csv_path = os.path.join(
                    self.params["output_dir"], self.params["filename"] + "_xytp.csv")
                csv_writer = EventCsvWriter(csv_path, width, height).start()
                self.status_update.emit(f"Recording event stream to {csv_path}")

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
                        failures = self._set_biases(device, biases)
                        if failures:
                            # Live tuning: a value the device won't take is user
                            # error, not a fault. Say so and keep streaming.
                            self.status_update.emit(
                                "EVK4 bias not applied: " + "; ".join(failures))
                        else:
                            self.status_update.emit("EVK4 biases updated.")
                    # Hand the chunk to the CSV writer before generating the
                    # display frame: submit() is a memcpy and a queue put, so
                    # doing it first keeps the recorded data as close to the
                    # SDK's delivery timing as possible, and the preview (which
                    # is display-only) absorbs any jitter instead.
                    if csv_writer is not None:
                        csv_writer.submit(evs)
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
                # Always drain the CSV writer, including on the error path: the
                # events already queued are the only copy that exists (there is
                # no .raw to re-read), so they must reach disk even if the run
                # ended badly.
                if csv_writer is not None:
                    csv_writer.close()

            if self.mode == "acquire" and self._is_running:
                if csv_writer is not None:
                    # CSV mode: the image counts exactly the events written, so
                    # there is no second decode pass. No .raw exists — this is
                    # the only record of the run.
                    self._report_csv_result(csv_writer)
                    final_image = csv_writer.image()
                else:
                    self.status_update.emit("Processing final image from RAW data...")
                    final_image = accumulate_event_frame(
                        EventsIterator(input_path=log_path, delta_t=1000000), width, height)
                    # The raw event recording (.raw) is always kept alongside the
                    # .tif/.mat — it is the full event stream the 2D image is
                    # derived from, which downstream analysis re-uses.
                self.process_final_image(final_image)

            self.status_update.emit("Camera stopped.")

        except Exception as e:
            self.error_signal.emit(f"EVK4 Hardware Error: {str(e)}")
        finally:
            # Drop every reference so the USB link is released (mirrors
            # fov_registration._grab_event_frame). Without this an error path left
            # the device handle claimed and the *next* Live View could not open the
            # camera — the failure looked like broken hardware but was this leak.
            del mv_iterator
            del device
            gc.collect()
            # Always emit, including on the error path: MainWindow resets the live
            # UI (Apply-biases button, Start/Stop state) in this handler, so a run
            # that ended by exception must still hand control back or the window is
            # left believing a dead feed is running.
            self.finished_signal.emit()

    def _report_csv_result(self, writer):
        """Surface the CSV writer's outcome — especially any dropped events.

        CSV mode has no complete file to re-read, so a chunk the writer could not
        keep up with is permanently lost. Saying so is the whole point: a run
        that silently recorded 80 % of its events would otherwise look identical
        to a genuinely dim sample.
        """
        if writer.error is not None:
            self.status_update.emit(f"CSV writer error: {writer.error}")
        if writer.events_dropped:
            total = writer.events_written + writer.events_dropped
            pct = 100.0 * writer.events_dropped / total if total else 0.0
            self.status_update.emit(
                f"WARNING: {writer.events_dropped:,} of {total:,} events ({pct:.1f} %) "
                f"were dropped — the CSV writer could not keep up with the event "
                f"rate. The image counts only the events written. Lower the event "
                f"rate (bias/ROI/ERC), raise DSI_EVK4_CSV_QUEUE, or record EVT3 "
                f"(.raw) instead, which cannot drop events.")
        else:
            self.status_update.emit(
                f"Event stream written: {writer.events_written:,} events -> {writer.path}")

    def process_final_image(self, final_image):
        """Post-process and save the accumulated 2D event image.

        ``final_image`` is the full-sensor event-count image, produced either by
        decoding the complete ``.raw`` (EVT3 mode) or by the CSV writer (CSV
        mode). Filtering, smoothing and saving are delegated to the pure core
        layer; only the event *sourcing* differs between the two formats.
        """
        # Crop to the same ROI the live view used, so the saved image matches the
        # framing the user selected (the event record keeps the full window).
        final_image = crop_to_roi(final_image, self._roi)

        # In EVT3 mode the decoded (x, y, p, t) list is deliberately NOT written
        # here — decode is *not* cheap at high event rates, and the gzip of the
        # per-event list dominated acquisition time. The .raw log is the
        # authoritative record, so the stream is generated offline afterwards:
        #     python tools/backfill_event_streams.py
        # In CSV mode the stream has already been written during acquisition and
        # there is no .raw, so that tool has nothing to backfill from.

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
