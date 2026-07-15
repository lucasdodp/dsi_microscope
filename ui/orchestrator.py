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
import threading
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    DCAM_EXPOSURE_PROP, EVK4_ERC_RATE, EVK4_MAX_RECONNECT_ATTEMPTS, EVK4_RECONNECT_DELAY_S,
)
from core import (
    accumulate_event_frame, apply_smoothing, compute_dsi_images, crop_to_roi,
    filter_crazy_pixels, normalize_to_8bit, save_axial_average_plot,
    save_axial_sectioning_plot, save_event_stream, save_parameter_log,
    save_raw_stack_tiff, save_volume_tiff,
)
from hardware.event_camera import (
    apply_event_roi, EventsIterator, METAVISION_AVAILABLE, initiate_device,
)
from hardware.orca_camera import DCAM_AVAILABLE, Dcam, Dcamapi
from hardware.stage_control import pitools


class AutomatedZStackWorker(QThread):
    """Master orchestrator for the combined PI motor + camera acquisition loop."""

    image_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    z_profile_update = pyqtSignal(float, int)
    position_update = pyqtSignal(float)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    awaiting_reconnect = pyqtSignal(bool)  # True -> paused, waiting for the user to replug + Resume

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
        self._dcam = None  # open ORCA handle (kept on self so recovery can reopen it)

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

        focus = self.motor_params["focus"]
        step_size = self.motor_params["step_size"]
        steps = self.motor_params["steps"]
        init_pos = focus - (step_size * steps / 2)

        # Per-plane sectioned images, accumulated into depth volumes.
        std_volume, avg_volume, event_volume, z_positions = [], [], [], []
        self._dcam = None

        try:
            if self.camera == "orca":
                self._dcam = self._open_orca()

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

                # Both cameras go through the same recover-and-pause wrapper, so a
                # transient fault on either retries the plane and, if needed, pauses
                # the run for a manual Resume instead of aborting.
                result = self._capture_plane(step)
                if result is not None:
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

            self._close_orca()

            # Recenter the objective on the focus: the scan loop otherwise leaves
            # it parked at the last (top-most) plane, but the user expects it back
            # at the centre of the stack whether the run completed or was stopped.
            self._return_to_focus()

            saved_msg = self._save_outputs(std_volume, avg_volume, event_volume, z_positions)
            if self._is_running:
                self.status_update.emit(f"Automated Z-Stack Complete.{saved_msg}")
            else:
                self.status_update.emit(f"Z-Stack stopped by user.{saved_msg}")
            self.finished_signal.emit()

        except Exception as e:
            # Try to preserve whatever planes were captured before failing.
            try:
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
            raise RuntimeError(f"Dcamapi.init() failed with error {Dcamapi.lasterr()}")
        dcam = Dcam(0)
        if not dcam.dev_open():
            raise RuntimeError(f"dev_open() failed with error {dcam.lasterr()}")
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
        try:
            if dcam.cap_start():
                for i in range(num_frames):
                    if not self._is_running:
                        break
                    if dcam.wait_capevent_frameready(2000):
                        frame = dcam.buf_getframedata(i)
                        if raw_stack is None:
                            raw_stack = np.empty((num_frames,) + frame.shape, dtype=frame.dtype)
                        raw_stack[i] = frame  # copies out of the SDK ring buffer
                        count += 1
                    else:
                        raise RuntimeError(f"Frame timeout: {dcam.lasterr()}")
                dcam.cap_stop()
        finally:
            dcam.buf_release()

        if not self._is_running or count != num_frames:
            return None

        raw_dir = self.save_params.get("raw_dir") or self.save_params.get("output_dir", "")
        if raw_dir and self.save_params.get("save_raw", True):
            filename = self.save_params.get("filename", "zstack")
            save_raw_stack_tiff(raw_stack, raw_dir, filename, roi, plane=step)
        return compute_dsi_images(raw_stack, roi)

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

        # Always save this plane's raw event stream (one .raw per plane). Logging
        # must start before the iterator and be stopped in a finally — stopping
        # without a prior log_raw_data can crash the native library.
        raw_dir = self.save_params.get("raw_dir") or self.save_params.get("output_dir", "")
        filename = self.save_params.get("filename", "zstack")
        events_stream = device.get_i_events_stream()
        raw_path = None
        if raw_dir and events_stream:
            raw_path = os.path.join(raw_dir, f"{filename}_events_z{step:03d}.raw")
            events_stream.log_raw_data(raw_path)

        mv_iterator = EventsIterator.from_device(device=device)
        height, width = mv_iterator.get_size()

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
            else:
                # No raw file to re-read (unsaved run) — accumulate the live
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
            wd.join(timeout=2.0)
        if raw_path is not None:
            event_img = None  # reconstructed from the (possibly partial) raw below

        if not self._is_running:
            return None

        if event_img is None:
            if raw_path is not None:
                # Reconstruct from the plane's saved raw stream — the complete event
                # record — matching event_camera.process_final_image().
                reader = EventsIterator(input_path=raw_path, delta_t=1000000)
                event_img = accumulate_event_frame(reader, width, height)

                # Also save this plane's decoded event stream (x, y, p, t) next to
                # its .raw, mirroring the single-Z acquire — an explicit per-event
                # list for downstream analysis.
                save_event_stream(
                    EventsIterator(input_path=raw_path, delta_t=1000000),
                    raw_dir, f"{filename}_events_z{step:03d}",
                )
            else:  # unsaved run force-stopped before any accumulation
                event_img = np.zeros((height, width), dtype=np.float32)

        # Report empty / stalled planes so the user sees *why* a run underperforms
        # (the common cause is a bias_hpf near its maximum suppressing all events,
        # or the camera falling back off USB3) rather than just a slow, blank stack.
        if float(np.max(event_img)) == 0:
            self.status_update.emit(
                f"Plane {step+1}: 0 events recorded — the camera delivered nothing. "
                f"Check the USB3 connection and lower bias_hpf (a value near its "
                f"maximum suppresses all events).")
        elif timed_out["flag"]:
            self.status_update.emit(
                f"Plane {step+1}: event delivery stalled — recording capped at "
                f"{ceiling_s:.0f} s and continuing.")

        event_img = crop_to_roi(event_img, roi)  # match the live crop framing

        if p.get("filter_crazy_pixels", True):
            event_img = filter_crazy_pixels(event_img)
        if p.get("apply_smoothing", True):
            event_img = apply_smoothing(event_img)
        return event_img

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

    # ----------------------------------------------------------------- save
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
