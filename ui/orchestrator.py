"""Automated 3D DSI Z-stack orchestrator.

This worker is the single integration point that links the hardware layer (PI stage
+ ORCA DCAM or Prophesee EVK4) to the core math. It lives in ui/ rather than
hardware/ because it composes the instruments and the processing layer; keeping it
here avoids a hardware->core->hardware import tangle.

At each focal plane it acquires with the selected camera:
  * Scientific (ORCA): a speckle frame stack -> average (widefield) + standard
    deviation (DSI) images; every plane's raw 16-bit frames are written to their
    own multi-page TIFF (``<filename>_raw_stack_zNNN.tif``), one file per plane,
    because the downstream MATLAB algorithms (e.g. RIM) consume the planes as
    separate files rather than one combined volume.
  * Event (EVK4): an event recording for a fixed duration, accumulated into an
    event image; each plane's raw event stream is also saved to its own
    ``<filename>_events_zNNN.raw`` (one file per plane).

The per-plane sectioned images are assembled into a depth volume and saved as a 3D
TIFF — a single consolidated output per stack for either camera.
"""

import os
import queue
import threading
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    DCAM_EXPOSURE_PROP, EVK4_ERC_RATE, EVK4_MAX_RECONNECT_ATTEMPTS, EVK4_RECONNECT_DELAY_S,
    EVK4_SAVE_FORMAT_CSV, EVK4_SAVE_FORMAT_DEFAULT,
)
from core import (
    EventCsvWriter, accumulate_event_frame, apply_smoothing, compute_dsi_images,
    crop_to_roi, filter_crazy_pixels, normalize_to_8bit, save_axial_average_plot,
    save_axial_sectioning_plot, save_parameter_log,
    save_raw_stack_tiff, save_volume_tiff,
)
from hardware.event_camera import (
    apply_event_roi, EventsIterator, METAVISION_AVAILABLE, initiate_device,
)
from hardware.orca_camera import _ORCA_BUSY_HINT, DCAM_AVAILABLE, Dcam, Dcamapi
from hardware.stage_control import pitools


# Sentinel returned by a plane capture when processing is deferred: the plane's
# raw data was saved but its image was intentionally not reconstructed (that runs
# afterwards on a background thread). Distinct from None, which means "aborted".
_DEFERRED = object()

# Sentinel pushed onto the pipeline queue to tell the consumer thread the run is
# over and it should drain and exit.
_PIPE_SENTINEL = object()


class AutomatedZStackWorker(QThread):
    """Master orchestrator for the combined PI motor + camera acquisition loop."""

    image_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    z_profile_update = pyqtSignal(float, int)
    position_update = pyqtSignal(float)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    awaiting_reconnect = pyqtSignal(bool)  # True -> paused, waiting for the user to replug + Resume
    deferred_ready = pyqtSignal(dict)  # deferred mode: emit the rebuild job when capture finishes

    def __init__(self, pidevice, axis, motor_params, orca_params, save_params=None,
                 camera="orca", evk4_params=None):
        super().__init__()
        self.pidevice = pidevice
        self.axis = axis
        self.motor_params = motor_params
        self.orca_params = orca_params
        self.evk4_params = evk4_params or {}
        self.save_params = save_params or {}
        self.camera = camera  # "orca" or "event"
        self._is_running = True
        self._resume_requested = False  # set by resume() to continue a paused run
        # Processing mode — how per-plane reconstruction relates to capture:
        #   "per_plane" (default): capture then reconstruct inline, one plane at a
        #       time (live feedback, slowest).
        #   "pipelined": reconstruct plane N on a background thread while the stage
        #       captures plane N+1 (live feedback, faster — the compute is hidden
        #       behind the next capture).
        #   "deferred": capture + save raw only; reconstruct everything after the
        #       run (fastest on-instrument, no live feedback). See `_finish_deferred`.
        mode = self.save_params.get("processing_mode")
        if mode is None:  # legacy callers only set the boolean
            mode = "deferred" if self.save_params.get("defer_processing", False) else "per_plane"
        self._defer = (mode == "deferred")
        self._pipeline = (mode == "pipelined")
        # CSV event format has no raw stream to reconstruct from afterwards — the
        # writer builds the image inline as events arrive — so it is incompatible
        # with BOTH deferral and pipelining (both split capture from reconstruction).
        # Deferral silently forcing EVT3 (.raw) instead was a real bug (the user's
        # chosen CSV format was overridden without warning), so the format wins: CSV
        # falls the run back to inline per-plane processing.
        if (self._defer or self._pipeline) and self.camera == "event" and self.evk4_params.get(
                "evk4_save_format", EVK4_SAVE_FORMAT_DEFAULT) == EVK4_SAVE_FORMAT_CSV:
            self._defer = self._pipeline = False
            self._defer_disabled_for_csv = True
        else:
            self._defer_disabled_for_csv = False
        # Pipeline plumbing (created in run() only when pipelined); declared here so
        # helpers can reference them unconditionally.
        self._proc_queue = None
        self._proc_error = None
        self._proc_sem = None
        self._dcam = None  # open ORCA handle (kept on self so recovery can reopen it)
        # Measured ORCA capture throughput, accumulated across planes so the
        # parameter log can record the *real* framerate (not just the estimate).
        # Frame-gaps and wall-clock are summed so the reported fps is the
        # time-weighted average over every plane's capture window.
        self._orca_capture_gaps = 0
        self._orca_capture_time = 0.0

    def _prepare_output_dir(self):
        """Create a per-acquisition subfolder named after the filename base.

        A single Z-stack writes many files. To keep each acquisition
        self-contained and tidy, they are organised as::

            <output_dir>/<filename>/
                <filename>_params.txt          (parameter log)
                <filename>_zstack_*.tif        (depth-volume TIFFs)
                <filename>_*_profile.csv        (event / axial CSV)
                <filename>_*_fit.png            (Gaussian-fit plots)
                raw_files/
                    <filename>_raw_stack_zNNN.tif   (one per plane)
                    <filename>_events_zNNN.raw      (one per plane)

        ``output_dir`` is repointed to ``<output_dir>/<filename>/`` so the
        summary save sites use it, and the bulky per-plane raw data is collected
        under ``raw_dir`` (``.../raw_files``) so it no longer clutters the main
        folder.
        """
        out_dir = self.save_params.get("output_dir", "")
        filename = self.save_params.get("filename", "zstack")
        if out_dir and not self.save_params.get("_dir_prepared", False):
            acq_dir = os.path.join(out_dir, filename)
            raw_dir = os.path.join(acq_dir, "raw_files")
            os.makedirs(raw_dir, exist_ok=True)
            self.save_params["output_dir"] = acq_dir
            self.save_params["raw_dir"] = raw_dir
            self.save_params["_dir_prepared"] = True

    def run(self):
        if not self.pidevice:
            self.error_signal.emit("PI Stage is not connected.")
            return
        if self.camera == "orca" and not DCAM_AVAILABLE:
            self.error_signal.emit("DCAM API not found.")
            return
        if self.camera == "event" and not METAVISION_AVAILABLE:
            self.error_signal.emit("Metavision SDK not found.")
            return

        # Put every file from this acquisition into its own <filename> subfolder.
        self._prepare_output_dir()

        if self._defer_disabled_for_csv:
            self.status_update.emit(
                "Note: pipelined/deferred processing has no effect with the CSV "
                "event format (there is no raw stream to reconstruct from later) — "
                "this run processes each plane inline, as CSV always has.")

        focus = self.motor_params["focus"]
        step_size = self.motor_params["step_size"]
        steps = self.motor_params["steps"]
        # Anchoring: "start" makes ``focus`` the bottom plane and scans upward
        # (for a 3D sample scanned from the bottom of the volume up); "center"
        # (default) makes ``focus`` the middle plane, the original behaviour.
        if self.motor_params.get("start_mode") == "start":
            init_pos = focus
        else:
            init_pos = focus - (step_size * steps / 2)

        # Per-plane sectioned images, accumulated into depth volumes.
        std_volume, avg_volume, event_volume, z_positions = [], [], [], []
        self._dcam = None
        consumer = None  # pipelined-mode background reconstruction thread

        try:
            if self.camera == "orca":
                self._dcam = self._open_orca()

            # Pipelined mode: a background consumer reconstructs plane N while the
            # stage captures plane N+1. It owns all appends/emits so the volumes are
            # only touched from one thread (the producer reads them after join).
            if self._pipeline:
                consumer = self._start_pipeline_consumer(
                    std_volume, avg_volume, event_volume, z_positions)

            # Move to the bottom of the stack (the first plane).
            self.status_update.emit(f"Moving to start position {init_pos:.4f} µm...")
            self.pidevice.MOV(self.axis, init_pos)
            pitools.waitontarget(self.pidevice, axes=self.axis)
            self._emit_position()

            for step in range(steps):
                if not self._is_running:
                    break

                target_pos = init_pos + (step * step_size)
                self.status_update.emit(f"Z-Stack Step {step+1}/{steps} - Moving to {target_pos:.4f} µm...")
                self.pidevice.MOV(self.axis, target_pos)
                pitools.waitontarget(self.pidevice, axes=self.axis)
                self._emit_position()
                time.sleep(0.5)  # Wait for motor mechanical settlement

                z_now = float(self.pidevice.qPOS(self.axis)[self.axis])

                if self._pipeline:
                    # Bound the number of raw stacks in flight (ORCA memory); the
                    # consumer releases a slot when it finishes a plane. Poll the
                    # acquire so a user Stop stays responsive even if the consumer
                    # stalls on a plane. Give the slot back if the capture aborts.
                    if self._proc_sem is not None:
                        while not self._proc_sem.acquire(timeout=0.2):
                            if not self._is_running:
                                break
                        if not self._is_running:
                            break
                    raw_material = self._capture_plane(step)
                    if raw_material is None:  # aborted (user Stop / unrecoverable)
                        if self._proc_sem is not None:
                            self._proc_sem.release()
                        break
                    # A processing fault on an earlier plane surfaces here.
                    if self._proc_error is not None:
                        raise self._proc_error
                    self._proc_queue.put((raw_material, step, z_now))
                    self.status_update.emit(
                        f"Plane {step+1}/{steps} captured — reconstructing in parallel.")
                    continue

                # Both cameras go through the same recover-and-pause wrapper, so a
                # transient fault on either retries the plane and, if needed, pauses
                # the run for a manual Resume instead of aborting.
                result = self._capture_plane(step)
                if result is _DEFERRED:
                    # Raw saved; reconstruction happens after the run. Record the
                    # real plane position so the offline rebuild is exact.
                    z_positions.append(z_now)
                    self.status_update.emit(
                        f"Plane {step+1}/{steps} captured — processing deferred.")
                elif result is not None:
                    if self.camera == "orca":
                        avg_img, std_img = result
                        avg_volume.append(avg_img)
                        std_volume.append(std_img)
                        z_positions.append(z_now)
                        self.image_ready.emit(normalize_to_8bit(std_img))
                        self.z_profile_update.emit(float(np.sum(std_img)), step)
                    else:
                        event_img = result
                        event_volume.append(event_img)
                        z_positions.append(z_now)
                        self.image_ready.emit(normalize_to_8bit(event_img))
                        self.z_profile_update.emit(float(np.sum(event_img)), step)

            # Drain the pipeline before saving: the consumer is still reconstructing
            # the last plane(s), and it owns the volumes until it exits.
            if self._pipeline:
                self._drain_pipeline(consumer)
                consumer = None
                if self._proc_error is not None:
                    raise self._proc_error

            self._close_orca()

            # Recenter the objective on the focus: the scan loop otherwise leaves
            # it parked at the last (top-most) plane, but the user expects it back
            # at the centre of the stack whether the run completed or was stopped.
            self._return_to_focus()

            if self._defer:
                saved_msg = self._finish_deferred(z_positions)
            else:
                saved_msg = self._save_outputs(std_volume, avg_volume, event_volume, z_positions)
            if self._is_running:
                self.status_update.emit(f"Automated Z-Stack Complete.{saved_msg}")
            else:
                self.status_update.emit(f"Z-Stack stopped by user.{saved_msg}")
            self.finished_signal.emit()

        except Exception as e:
            # Try to preserve whatever planes were captured before failing.
            try:
                if consumer is not None:  # stop the pipeline consumer first
                    self._drain_pipeline(consumer)
                if self._defer:
                    # In deferred mode the raw planes are already on disk; queue the
                    # partial rebuild so nothing captured before the fault is lost.
                    self._finish_deferred(z_positions)
                else:
                    self._save_outputs(std_volume, avg_volume, event_volume, z_positions)
            except Exception:
                pass
            self.error_signal.emit(f"Z-Stack Orchestrator Error: {str(e)}")
            self._notify(
                f"{self._camera_label()} acquisition error",
                f"Your DSI {self._camera_label()} acquisition stopped with an error: {e}.")
            self._close_orca()
            # Still try to bring the objective back to focus after a failure.
            self._return_to_focus()

    # ------------------------------------------------------------------ ORCA
    def _open_orca(self):
        self.status_update.emit("Initializing ORCA DCAM API for Z-Stack...")
        if not Dcamapi.init():
            raise RuntimeError(
                f"Dcamapi.init() failed with error {Dcamapi.lasterr()}. {_ORCA_BUSY_HINT}")
        dcam = Dcam(0)
        if not dcam.dev_open():
            raise RuntimeError(
                f"dev_open() failed with error {dcam.lasterr()}. {_ORCA_BUSY_HINT}")
        dcam.prop_setvalue(DCAM_EXPOSURE_PROP, self.orca_params["orca_exposure"] / 1000.0)
        return dcam

    def _close_orca(self):
        """Close the DCAM device + uninit the API. Safe to call repeatedly / when
        the camera was never opened (a no-op for the event camera)."""
        if self._dcam is not None:
            try:
                self._dcam.dev_close()
            except Exception:
                pass
            self._dcam = None
            try:
                Dcamapi.uninit()
            except Exception:
                pass

    def _recover_orca(self):
        """Close and reopen the DCAM device so a transient camera fault can clear.

        Never raises: if the reopen fails the handle is left as ``None`` and the
        next capture attempt fails cleanly, so the retry/pause loop keeps control.
        """
        self._close_orca()
        import gc
        gc.collect()
        try:
            self._dcam = self._open_orca()
        except Exception as e:
            self._dcam = None
            self.status_update.emit(f"ORCA reopen failed: {e}")

    def _capture_plane_orca_once(self, step):
        """Acquire one frame stack at the current plane; return (avg, std) images.

        When raw saving is enabled, this plane's raw frames are written to their
        own multi-page TIFF (``<filename>_raw_stack_zNNN.tif``) — one file per
        plane, so the downstream MATLAB algorithms can consume the planes
        individually rather than as one combined volume. Raises on any DCAM fault
        so the recover-and-pause wrapper (``_capture_plane``) can retry the plane.
        """
        dcam = self._dcam
        if dcam is None:
            raise RuntimeError("ORCA device is not open")
        num_frames = self.orca_params["orca_frames"]
        roi = self.orca_params["orca_roi"]

        if not dcam.buf_alloc(num_frames):
            raise RuntimeError(f"buf_alloc failed: {dcam.lasterr()}")

        # Copy each frame straight into a single preallocated array instead of a
        # growing list of np.copy'd frames + a later np.array() of it (that held
        # two full copies of the stack in RAM at once). The camera's own ring
        # buffer already holds N frames; this keeps the host-side overhead to one.
        raw_stack = None
        count = 0
        capture_start = None  # set on the first frame in hand (matches single-Z)
        capture_elapsed = 0.0
        try:
            if dcam.cap_start():
                for i in range(num_frames):
                    if not self._is_running:
                        break
                    if dcam.wait_capevent_frameready(2000):
                        if i == 0:
                            capture_start = time.perf_counter()  # first frame in hand
                        frame = dcam.buf_getframedata(i)
                        if raw_stack is None:
                            raw_stack = np.empty((num_frames,) + frame.shape, dtype=frame.dtype)
                        raw_stack[i] = frame  # copies out of the SDK ring buffer
                        count += 1
                    else:
                        raise RuntimeError(f"Frame timeout: {dcam.lasterr()}")
                if capture_start is not None:
                    capture_elapsed = time.perf_counter() - capture_start
                dcam.cap_stop()
        finally:
            dcam.buf_release()

        if not self._is_running or count != num_frames:
            return None

        # Accumulate the real capture throughput for this plane. The window spans
        # (count-1) inter-frame gaps, matching the single-Z acquire's fps formula.
        if capture_elapsed > 0 and count > 1:
            self._orca_capture_gaps += count - 1
            self._orca_capture_time += capture_elapsed

        if self._defer:
            # Skip the (expensive) per-plane DSI reconstruction; it runs after the
            # run, rebuilt from the raw stack. Raw must be written now (forced).
            self._save_orca_raw(raw_stack, step)
            return _DEFERRED
        if self._pipeline:
            # Hand the raw frames off to the consumer thread, which saves the raw
            # and runs compute_dsi_images while the stage captures the next plane.
            return raw_stack
        return self._finish_orca_plane(raw_stack, step)

    def _save_orca_raw(self, raw_stack, step):
        """Write one plane's raw 16-bit speckle stack (per the save_raw toggle, or
        always in deferred mode which needs it to rebuild from). Cheap no-op when
        no output/raw dir or raw saving is off."""
        raw_dir = self.save_params.get("raw_dir") or self.save_params.get("output_dir", "")
        if raw_dir and (self._defer or self.save_params.get("save_raw", True)):
            filename = self.save_params.get("filename", "zstack")
            save_raw_stack_tiff(raw_stack, raw_dir, filename,
                                self.orca_params["orca_roi"], plane=step)

    def _finish_orca_plane(self, raw_stack, step):
        """Save the plane's raw (if enabled) and reconstruct its DSI images. Runs
        inline in per-plane mode, or on the consumer thread in pipelined mode —
        both the (expensive) parts that pipelining hides behind the next capture."""
        self._save_orca_raw(raw_stack, step)
        return compute_dsi_images(raw_stack, self.orca_params["orca_roi"])

    # -------------------------------------------------- capture with recovery
    def _camera_label(self):
        """Human-readable name of the active camera for status / notifications."""
        return "ORCA camera" if self.camera == "orca" else "event camera"

    def _capture_plane_once(self, step):
        """Single capture attempt for the active camera; raises on any fault."""
        if self.camera == "orca":
            return self._capture_plane_orca_once(step)
        return self._capture_plane_event_once(step)

    def _recover_device(self):
        """Best-effort recovery of the active camera between retry attempts.

        ORCA closes and reopens its DCAM handle; the event camera reopens per
        attempt inside ``_capture_plane_event_once``, so here it just drops the
        dropped device's refs so the USB link can be reacquired.
        """
        if self.camera == "orca":
            self._recover_orca()
        else:
            import gc
            gc.collect()

    def _capture_plane(self, step):
        """Capture one plane for either camera, surviving a transient fault.

        A brief glitch is absorbed automatically: the device is recovered and the
        same plane retried up to ``EVK4_MAX_RECONNECT_ATTEMPTS`` times. If it still
        fails, the acquisition does **not** abort — it *pauses*: it emails the user,
        emits ``awaiting_reconnect(True)`` so the UI shows a Resume button, and waits
        (no time limit) until the user fixes the camera and clicks Resume — or Stops.
        Because the worker stays alive, every plane already captured is kept and the
        stack continues from this exact plane. Returns None only if the user aborts.

        The return value is the active camera's plane result: an ``(avg, std)`` tuple
        for the ORCA or an event image for the EVK4.
        """
        attempts = max(1, EVK4_MAX_RECONNECT_ATTEMPTS)
        cam = self._camera_label()
        troubled = False
        while self._is_running:
            last_err = None
            for attempt in range(1, attempts + 1):
                try:
                    result = self._capture_plane_once(step)
                    if troubled:  # recovered from a fault
                        self.status_update.emit(
                            f"{cam} recovered — resumed at plane "
                            f"{step + 1}/{self.motor_params['steps']}.")
                        self._notify(
                            f"{cam} recovered",
                            f"Your DSI acquisition recovered the {cam} and resumed at "
                            f"plane {step + 1}.")
                    return result
                except Exception as e:
                    if not self._is_running:
                        return None
                    troubled = True
                    last_err = e
                    self._recover_device()
                    self.status_update.emit(
                        f"{cam} error at plane {step + 1} "
                        f"(attempt {attempt}/{attempts}): {e}")
                    if attempt < attempts:
                        self._sleep_interruptible(EVK4_RECONNECT_DELAY_S)
            if self._is_running:
                self._await_manual_resume(step, last_err)
        return None  # user aborted while retrying / waiting

    def _await_manual_resume(self, step, err):
        """Pause until the user fixes the camera and clicks Resume (or Stops).

        The worker — and every plane captured so far — stays alive, so the run
        continues from this plane whenever the user is ready. There is no timeout."""
        cam = self._camera_label()
        self._resume_requested = False
        self.awaiting_reconnect.emit(True)
        self.status_update.emit(
            f"{cam} lost at plane {step + 1}. Restore the camera, then click "
            f"'Resume Acquisition' (or Stop to abort).")
        self._notify(
            f"acquisition paused — {cam} lost",
            f"Your DSI acquisition paused at plane {step + 1} because the {cam} "
            f"stopped responding; restore the camera and click Resume to continue.")
        while self._is_running and not self._resume_requested:
            time.sleep(0.2)
        self.awaiting_reconnect.emit(False)
        if self._is_running and self._resume_requested:
            self.status_update.emit(f"Resuming acquisition at plane {step + 1}...")

    def resume(self):
        """Request that a paused (awaiting-reconnect) acquisition continue."""
        self._resume_requested = True

    # ----------------------------------------------------------------- EVENT
    def _capture_plane_event_once(self, step):
        """Single capture attempt: (re)initialise the EVK4, record for the fixed
        duration, reconstruct the event image, and save this plane's raw stream.

        The device is (re)initialised per plane for a clean state; its local handle
        is dropped on return/exception so the next plane (or a reconnect) can reopen
        the USB link. Raw logging is always stopped in a ``finally`` — stopping
        without a prior ``log_raw_data`` can crash the native library.

        The saved image is reconstructed by re-reading the plane's raw file (the
        complete, authoritative event record), not by accumulating the live device
        iteration, which drops events (and occasionally whole planes) because it
        competes with the concurrent raw logger. When no raw file is written
        (unsaved run) it falls back to live accumulation so a run still yields an
        image.
        """
        p = self.evk4_params
        device = initiate_device("")

        biases = device.get_i_ll_biases()
        if biases:
            biases.set("bias_fo", p["bias_fo"])
            biases.set("bias_hpf", p["bias_hpf"])
            biases.set("bias_diff_on", p["bias_on"])
            biases.set("bias_diff_off", p["bias_off"])

        erc = device.get_i_erc_module()
        if erc:
            erc.enable(True)
            erc.set_cd_event_rate(EVK4_ERC_RATE)

        # Fixed ROI per plane: set the hardware ROI best-effort (event-rate cut),
        # then software-crop the accumulated image to the same window below.
        roi = p.get("evk4_roi")
        apply_event_roi(device, roi)

        # Save this plane's event record, in whichever format the run selected.
        # EVT3 logging must start before the iterator and be stopped in a finally
        # — stopping without a prior log_raw_data can crash the native library.
        # CSV needs the sensor geometry, so its writer starts after the iterator.
        # (Deferred mode and CSV are mutually exclusive — __init__ already forces
        # self._defer False whenever CSV is selected — so the format alone decides.)
        csv_mode = p.get("evk4_save_format", EVK4_SAVE_FORMAT_DEFAULT) == EVK4_SAVE_FORMAT_CSV
        raw_dir = self.save_params.get("raw_dir") or self.save_params.get("output_dir", "")
        filename = self.save_params.get("filename", "zstack")
        events_stream = device.get_i_events_stream()
        raw_path = None
        if raw_dir and events_stream and not csv_mode:
            raw_path = os.path.join(raw_dir, f"{filename}_events_z{step:03d}.raw")
            events_stream.log_raw_data(raw_path)

        mv_iterator = EventsIterator.from_device(device=device)
        height, width = mv_iterator.get_size()

        csv_writer = None
        if raw_dir and csv_mode:
            csv_writer = EventCsvWriter(
                os.path.join(raw_dir, f"{filename}_events_z{step:03d}_xytp.csv"),
                width, height).start()

        acqu_time = p["acqu_time"]
        self.status_update.emit(f"Recording events at plane {step+1} for {acqu_time} s...")

        # Stop the raw logger exactly once, from whichever of the watchdog or the
        # finally gets there first: stopping it twice — or without a prior
        # log_raw_data — can crash the Metavision native library.
        log_lock = threading.Lock()
        log_stopped = {"done": raw_path is None}

        def stop_logging_once():
            with log_lock:
                if not log_stopped["done"]:
                    log_stopped["done"] = True
                    try:
                        events_stream.stop_log_raw_data()
                    except Exception:
                        pass

        # The recording should last ~acqu_time of wall-clock, then stop. The
        # elapsed-time check in events_for_duration does that whenever event chunks
        # flow, but it can only run once the SDK yields a chunk — so if the camera
        # delivers *nothing* (a bias_hpf at its ceiling suppresses all events, or a
        # USB3/connection drop) ``for evs in mv_iterator`` blocks with nothing to
        # yield and never stops. This watchdog force-stops the event stream a few
        # seconds past acqu_time (grace for stream start-up / first-chunk latency),
        # or immediately on a user Stop, which unblocks the iterator — so a plane
        # that records nothing still finishes in ~acqu_time instead of hanging, and
        # Stop stays responsive. (It also caps a runaway high-rate plane whose
        # delivery stalls: a "record for acqu_time" plane is that long either way.)
        recording_over = threading.Event()
        ceiling_s = acqu_time + 5.0
        timed_out = {"flag": False}

        def watchdog():
            deadline = time.time() + ceiling_s
            while (not recording_over.is_set() and self._is_running
                   and time.time() < deadline):
                time.sleep(0.2)
            if recording_over.is_set():
                return
            timed_out["flag"] = self._is_running  # True: ceiling hit; False: user Stop
            stop_logging_once()  # flush the partial raw before tearing the stream down
            for stopper in (
                lambda: device.get_i_events_stream().stop(),
                lambda: device.get_i_device_control().stop(),
            ):
                try:
                    stopper()
                except Exception:
                    pass

        def events_for_duration():
            """Yield event chunks until the acquisition time elapses or the user
            aborts — so accumulation streams with flat memory."""
            start = time.time()
            for evs in mv_iterator:
                if not self._is_running:
                    return
                yield evs
                if time.time() - start >= acqu_time:
                    return

        wd = threading.Thread(target=watchdog, name="evk4-record-watchdog", daemon=True)
        wd.start()
        event_img = None
        try:
            if raw_path is not None:
                # Raw stream is the authoritative record: just drive the device
                # for the acquisition window, then reconstruct from the complete
                # file below (the live iteration is lossy and must not be trusted
                # for the saved image).
                for _ in events_for_duration():
                    pass
            elif csv_writer is not None:
                # CSV mode: there is no file to re-read afterwards, so this loop
                # is the only chance to capture the events. submit() is a memcpy
                # plus a queue put — formatting and disk I/O happen on the
                # writer thread — so the loop stays about as cheap as the raw
                # path's empty one.
                for evs in events_for_duration():
                    csv_writer.submit(evs)
            else:
                # No event file at all (unsaved run) — accumulate the live
                # stream as a best-effort fallback.
                event_img = accumulate_event_frame(events_for_duration(), width, height)
        except Exception:
            # A watchdog force-stop surfaces as an iterator error — expected; only
            # re-raise a genuine fault so the recover/pause logic can act on it.
            if self._is_running and not timed_out["flag"]:
                raise
        finally:
            recording_over.set()
            stop_logging_once()
            # Drain the CSV writer before the device is torn down: the events it
            # still holds exist nowhere else.
            if csv_writer is not None:
                csv_writer.close()
            wd.join(timeout=2.0)
        if raw_path is not None:
            event_img = None  # reconstructed from the (possibly partial) raw below

        if not self._is_running:
            return None

        # A stalled recording is a capture-side fact known now, whatever the mode.
        if timed_out["flag"]:
            self.status_update.emit(
                f"Plane {step+1}: event delivery stalled — recording capped at "
                f"{ceiling_s:.0f} s and continuing.")

        if self._defer:
            # The plane's raw event stream is on disk; the accumulate + filter +
            # smooth reconstruction runs later on the background thread.
            return _DEFERRED

        if self._pipeline and raw_path is not None:
            # Reconstruct on the consumer thread while the next plane records; the
            # complete .raw is the authoritative record it rebuilds from.
            return (raw_path, width, height)

        if event_img is None:
            if raw_path is not None:
                event_img = self._reconstruct_event_from_raw(raw_path, width, height)
            elif csv_writer is not None:
                # The writer counted exactly what it wrote, so the image and the
                # CSV agree by construction — no second pass needed.
                event_img = csv_writer.image()
                self._report_csv_plane(csv_writer, step)
            else:  # unsaved run force-stopped before any accumulation
                event_img = np.zeros((height, width), dtype=np.float32)

        return self._finish_event_plane(event_img, step, roi)

    def _reconstruct_event_from_raw(self, raw_path, width, height):
        """Accumulate a plane's 2D event image from its saved .raw — the complete,
        authoritative record — matching event_camera.process_final_image(). Used
        inline (per-plane) and on the consumer thread (pipelined).

        The decoded (x, y, p, t) list is deliberately NOT written here: it cost a
        second full decode pass plus a gzip of ~13 bytes/event, which at high event
        rates dominated the per-plane time. The .raw loses nothing, so that stream
        is generated offline afterwards (tools/backfill_event_streams.py).
        """
        reader = EventsIterator(input_path=raw_path, delta_t=1000000)
        return accumulate_event_frame(reader, width, height)

    def _finish_event_plane(self, event_img, step, roi):
        """Crop → filter → smooth an accumulated event image (and report an empty
        plane). Runs inline in per-plane mode, or on the consumer thread when
        pipelined — the reconstruction cost pipelining hides behind the next
        recording."""
        # Report an empty plane so the user sees *why* a run underperforms (usually
        # a bias_hpf near its maximum, or the camera falling back off USB3).
        if float(np.max(event_img)) == 0:
            self.status_update.emit(
                f"Plane {step+1}: 0 events recorded — the camera delivered nothing. "
                f"Check the USB3 connection and lower bias_hpf (a value near its "
                f"maximum suppresses all events).")
        event_img = crop_to_roi(event_img, roi)  # match the live crop framing
        if self.evk4_params.get("filter_crazy_pixels", True):
            event_img = filter_crazy_pixels(event_img)
        if self.evk4_params.get("apply_smoothing", True):
            event_img = apply_smoothing(event_img)
        return event_img

    def _report_csv_plane(self, writer, step):
        """Report a CSV plane's outcome, loudly if any events were dropped.

        In a long unattended stack this is the only signal that a plane recorded
        less than the camera delivered: CSV mode keeps no complete file, so a
        dropped chunk is gone and an under-recorded plane is otherwise
        indistinguishable from a dim one.
        """
        if writer.error is not None:
            self.status_update.emit(f"Plane {step+1}: CSV writer error: {writer.error}")
        if writer.events_dropped:
            total = writer.events_written + writer.events_dropped
            pct = 100.0 * writer.events_dropped / total if total else 0.0
            self.status_update.emit(
                f"Plane {step+1}: WARNING — {writer.events_dropped:,} of {total:,} events "
                f"({pct:.1f} %) dropped; the CSV writer could not keep up. Lower the "
                f"event rate, raise DSI_EVK4_CSV_QUEUE, or record EVT3 (.raw) instead.")

    def _sleep_interruptible(self, seconds):
        """Sleep, but wake early if the user aborts (keeps Stop responsive)."""
        end = time.time() + max(0.0, seconds)
        while time.time() < end and self._is_running:
            time.sleep(0.1)

    def _notify(self, subject, body):
        """Best-effort email notification (a no-op unless SMTP is configured)."""
        try:
            from hardware.notifier import send_email
            send_email(f"[DSI Microscope] {subject}", body)
        except Exception:
            pass

    # ------------------------------------------------ pipelined processing
    def _start_pipeline_consumer(self, std_volume, avg_volume, event_volume, z_positions):
        """Start the background reconstruction consumer for pipelined mode.

        The producer (the run loop) captures each plane and enqueues its raw
        material; this consumer reconstructs it — the expensive part — while the
        next plane is captured, then appends to the depth volumes and emits the live
        preview. It owns the volumes for the run's duration, so the producer only
        reads them after joining the consumer (`_drain_pipeline`).
        """
        self._proc_queue = queue.Queue()
        self._proc_error = None
        # Cap ORCA raw stacks in flight (each is large); event images are small and
        # reconstruction is far faster than a recording, so the event pipeline is
        # left unbounded (its queue never backs up).
        self._proc_sem = threading.Semaphore(2) if self.camera == "orca" else None
        consumer = threading.Thread(
            target=self._pipeline_consumer,
            args=(std_volume, avg_volume, event_volume, z_positions),
            name="zstack-pipeline", daemon=True)
        consumer.start()
        return consumer

    def _pipeline_consumer(self, std_volume, avg_volume, event_volume, z_positions):
        """Reconstruct captured planes off the acquisition thread (pipelined mode).

        Runs the same `_finish_orca_plane` / event reconstruction the inline path
        uses, so the output is identical — only *when* it runs differs. The first
        processing fault is stored and surfaced to the producer, which raises it.
        """
        while True:
            item = self._proc_queue.get()
            try:
                if item is _PIPE_SENTINEL:
                    return
                raw_material, step, z_now = item
                try:
                    if self.camera == "orca":
                        avg_img, std_img = self._finish_orca_plane(raw_material, step)
                        avg_volume.append(avg_img)
                        std_volume.append(std_img)
                        z_positions.append(z_now)
                        self.image_ready.emit(normalize_to_8bit(std_img))
                        self.z_profile_update.emit(float(np.sum(std_img)), step)
                    else:
                        if isinstance(raw_material, tuple):
                            raw_path, width, height = raw_material
                            event_img = self._reconstruct_event_from_raw(raw_path, width, height)
                            event_img = self._finish_event_plane(
                                event_img, step, self.evk4_params.get("evk4_roi"))
                        else:
                            # Unsaved run: no .raw exists, so the capture already
                            # accumulated + finished the image inline (pipelining
                            # degrades gracefully); just record and show it.
                            event_img = raw_material
                        event_volume.append(event_img)
                        z_positions.append(z_now)
                        self.image_ready.emit(normalize_to_8bit(event_img))
                        self.z_profile_update.emit(float(np.sum(event_img)), step)
                except Exception as e:  # noqa: BLE001 — surfaced to the producer
                    if self._proc_error is None:
                        self._proc_error = e
            finally:
                if item is not _PIPE_SENTINEL and self._proc_sem is not None:
                    self._proc_sem.release()  # free a raw-stack slot
                self._proc_queue.task_done()

    def _drain_pipeline(self, consumer):
        """Signal the consumer to finish and wait for it. Every already-captured
        plane is reconstructed before it exits, so an aborted run still keeps the
        planes it managed to capture."""
        if consumer is None:
            return
        try:
            self._proc_queue.put(_PIPE_SENTINEL)
        except Exception:
            pass
        consumer.join()

    # ----------------------------------------------------------------- save
    def _finish_deferred(self, z_positions):
        """Deferred mode: capture is done and every plane's raw is archived on disk.

        Writes an immediate parameter log (so the acquisition folder has a record
        straight away, and the elapsed-timing section can be appended to it), then
        emits the rebuild job that the background processor picks up — so the
        reconstruction runs off-instrument while the microscope is free for the
        next acquisition. Returns a short status message.
        """
        out_dir = self.save_params.get("output_dir", "")
        filename = self.save_params.get("filename", "zstack")
        n = len(z_positions)
        if not out_dir or n == 0:
            return " No planes captured — nothing to process."

        raw_dir = self.save_params.get("raw_dir") or out_dir

        # Immediate parameter log: the settings + measured framerate, marked as
        # pending. The background rebuild writes only the volumes/profiles (it is
        # passed no metadata), so it never overwrites this file and the appended
        # elapsed-timing section survives.
        metadata = self.save_params.get("metadata")
        if metadata:
            log_meta = dict(metadata)
            log_meta["Z-Stack planes"] = {
                "camera": self.camera,
                "num_planes_captured": n,
                "processing": "DEFERRED — reconstruction runs after acquisition",
                "z_positions": ", ".join(f"{z:.4f}" for z in z_positions),
            }
            if self.camera == "orca" and self._orca_capture_time > 0:
                measured_fps = self._orca_capture_gaps / self._orca_capture_time
                log_meta["Measured performance (ORCA)"] = {
                    "measured_framerate_fps": f"{measured_fps:.1f}",
                    "total_capture_time_s": f"{self._orca_capture_time:.3f}",
                    "frames_timed": self._orca_capture_gaps + n,
                }
            save_parameter_log(out_dir, filename, log_meta)

        job = {
            "camera": self.camera,
            "raw_dir": raw_dir,
            "out_dir": out_dir,
            "filename": filename,
            "z_positions": list(z_positions),
        }
        if self.camera == "orca":
            job["expected_frames"] = self.orca_params.get("orca_frames")
            job["save_average"] = True
        else:
            job["roi"] = self.evk4_params.get("evk4_roi")
            job["do_filter"] = self.evk4_params.get("filter_crazy_pixels", True)
            job["do_smooth"] = self.evk4_params.get("apply_smoothing", True)
        self.deferred_ready.emit(job)
        return (f" Captured {n} planes; DSI processing queued in the background — "
                f"the microscope is free for the next acquisition.")

    def _save_outputs(self, std_volume, avg_volume, event_volume, z_positions):
        """Save the depth volume(s) (3D TIFF) and the parameter log. Returns a
        short message for the status bar (empty if nothing was saved)."""
        out_dir = self.save_params.get("output_dir", "")
        filename = self.save_params.get("filename", "zstack")
        if not out_dir:
            return ""

        n = 0
        sectioned, kind = None, None
        if self.camera == "orca" and std_volume:
            save_volume_tiff(np.array(std_volume, dtype=np.float32), out_dir, filename, "zstack_dsi")
            save_volume_tiff(np.array(avg_volume, dtype=np.float32), out_dir, filename, "zstack_average")
            n = len(std_volume)
            sectioned, kind = std_volume, "dsi"
        elif self.camera == "event" and event_volume:
            save_volume_tiff(np.array(event_volume, dtype=np.float32), out_dir, filename, "zstack_event")
            n = len(event_volume)
            sectioned, kind = event_volume, "event"

        if n == 0:
            return ""

        # Axial-sectioning profile (paper Fig. 3a): the mean intensity of each
        # plane's optically-sectioned image vs axial position, Gaussian-fitted to
        # extract the FWHM (= axial sectioning). The profile data is always saved
        # as CSV; the PNG figure needs matplotlib.
        profile_msg = ""
        intensities = [float(np.mean(img)) for img in sectioned]
        fwhm, _, png_path = save_axial_sectioning_plot(
            z_positions[:n], intensities, out_dir, filename, kind
        )
        if fwhm is not None:
            profile_msg = f" Axial FWHM ≈ {fwhm:.2f} µm."
        if png_path is None:
            profile_msg += " (install matplotlib for the profile PNG)"

        # Companion profile of the *average* (widefield) image vs axial position.
        # Unlike the sectioned image it has no peak — its mean intensity is flat
        # across z — so it is fitted with a straight line, not a Gaussian. The
        # average image only exists for the ORCA (the event camera produces none).
        if self.camera == "orca" and avg_volume:
            avg_intensities = [float(np.mean(img)) for img in avg_volume]
            save_axial_average_plot(z_positions[:n], avg_intensities, out_dir, filename)

        metadata = self.save_params.get("metadata")
        if metadata:
            metadata = dict(metadata)
            metadata["Z-Stack planes"] = {
                "camera": self.camera,
                "num_planes_saved": n,
                "z_positions": ", ".join(f"{z:.4f}" for z in z_positions),
            }
            # Record the *measured* ORCA capture framerate (averaged over every
            # plane) next to the estimated settings, so the log carries the real
            # rate the frames were recorded at.
            if self.camera == "orca" and self._orca_capture_time > 0:
                measured_fps = self._orca_capture_gaps / self._orca_capture_time
                metadata["Measured performance (ORCA)"] = {
                    "measured_framerate_fps": f"{measured_fps:.1f}",
                    "total_capture_time_s": f"{self._orca_capture_time:.3f}",
                    "frames_timed": self._orca_capture_gaps + n,
                }
            save_parameter_log(out_dir, filename, metadata)

        return f" Saved {n} planes (3D TIFF) to {out_dir}.{profile_msg}"

    def _emit_position(self):
        """Report the real axis position so the UI readout tracks the stack."""
        try:
            self.position_update.emit(float(self.pidevice.qPOS(self.axis)[self.axis]))
        except Exception:
            pass

    def _return_to_focus(self):
        """Move the objective back to the user focus (centre of the scan).

        Called when the stack ends — completed, stopped, or errored — so the
        stage never sits at the top-most plane afterwards. Best-effort: a move
        failure here must not mask the run's real outcome.
        """
        if not self.pidevice:
            return
        try:
            focus = self.motor_params["focus"]
            self.status_update.emit(f"Returning objective to focus {focus:.4f} µm...")
            self.pidevice.MOV(self.axis, focus)
            pitools.waitontarget(self.pidevice, axes=self.axis)
            self._emit_position()
        except Exception:
            pass

    def stop(self):
        self._is_running = False
