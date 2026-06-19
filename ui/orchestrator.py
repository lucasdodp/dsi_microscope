"""Automated 3D DSI Z-stack orchestrator.

This worker is the single integration point that links the hardware layer (PI stage
+ ORCA DCAM or Prophesee EVK4) to the core math. It lives in ui/ rather than
hardware/ because it composes the instruments and the processing layer; keeping it
here avoids a hardware->core->hardware import tangle.

At each focal plane it acquires with the selected camera:
  * Scientific (ORCA): a speckle frame stack -> average (widefield) + standard
    deviation (DSI) images; every plane's raw 16-bit frames are appended into a
    single multi-page TIFF for the whole stack (the data the MATLAB RIM algorithm
    consumes).
  * Event (EVK4): an event recording for a fixed duration -> accumulated event
    image; the raw .raw event file is saved per plane.

The per-plane sectioned images are assembled into a depth volume and saved as a 3D
TIFF.
"""

import os
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import DCAM_EXPOSURE_PROP, EVK4_ERC_RATE
from core import (
    accumulate_event_frame, apply_smoothing, compute_dsi_images, filter_crazy_pixels,
    normalize_to_8bit, RawStackTiffWriter, save_parameter_log, save_volume_tiff,
)
from hardware.event_camera import EventsIterator, METAVISION_AVAILABLE, initiate_device
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

        focus = self.motor_params["focus"]
        step_size = self.motor_params["step_size"]
        steps = self.motor_params["steps"]
        init_pos = focus - (step_size * steps / 2)

        out_dir = self.save_params.get("output_dir", "")
        filename = self.save_params.get("filename", "zstack")

        # Per-plane sectioned images, accumulated into depth volumes.
        std_volume, avg_volume, event_volume, z_positions = [], [], [], []
        dcam = None
        # Single archival raw-data file for the whole ORCA stack: every plane's
        # speckle frames are appended into one multi-page TIFF (instead of one
        # file per plane).
        raw_writer = None

        try:
            if self.camera == "orca":
                dcam = self._open_orca()
                if out_dir and self.save_params.get("save_raw", True):
                    raw_writer = RawStackTiffWriter(
                        os.path.join(out_dir, f"raw_stack_{filename}.tif")
                    )

            # Move to the start of the stack.
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

                if self.camera == "orca":
                    result = self._capture_plane_orca(dcam, raw_writer)
                    if result is not None:
                        avg_img, std_img = result
                        avg_volume.append(avg_img)
                        std_volume.append(std_img)
                        z_positions.append(z_now)
                        self.image_ready.emit(normalize_to_8bit(std_img))
                        self.z_profile_update.emit(float(np.sum(std_img)), step)
                else:
                    event_img = self._capture_plane_event(out_dir, filename, step)
                    if event_img is not None:
                        event_volume.append(event_img)
                        z_positions.append(z_now)
                        self.image_ready.emit(normalize_to_8bit(event_img))
                        self.z_profile_update.emit(float(np.sum(event_img)), step)

            if dcam is not None:
                try:
                    dcam.dev_close()
                finally:
                    Dcamapi.uninit()

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
            if dcam is not None:
                try:
                    Dcamapi.uninit()
                except Exception:
                    pass
        finally:
            # Finalize the single raw-data TIFF, flushing whatever planes were
            # appended (so a partial/aborted stack still yields a valid file).
            if raw_writer is not None:
                try:
                    raw_writer.close()
                except Exception:
                    pass

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

    def _capture_plane_orca(self, dcam, raw_writer):
        """Acquire a frame stack at the current plane; return (avg, std) images.

        When ``raw_writer`` is provided, this plane's raw frames are appended to
        the single shared multi-page raw TIFF (plane-major order) rather than
        written to a separate per-plane file.
        """
        num_frames = self.orca_params["orca_frames"]
        roi = self.orca_params["orca_roi"]

        if not dcam.buf_alloc(num_frames):
            raise RuntimeError(f"buf_alloc failed: {dcam.lasterr()}")

        acquired_stack = []
        try:
            if dcam.cap_start():
                for i in range(num_frames):
                    if not self._is_running:
                        break
                    if dcam.wait_capevent_frameready(2000):
                        acquired_stack.append(np.copy(dcam.buf_getframedata(i)))
                    else:
                        raise RuntimeError(f"Frame timeout: {dcam.lasterr()}")
                dcam.cap_stop()
        finally:
            dcam.buf_release()

        if not self._is_running or len(acquired_stack) != num_frames:
            return None

        raw_stack = np.array(acquired_stack)
        if raw_writer is not None:
            raw_writer.append(raw_stack, roi)
        return compute_dsi_images(raw_stack, roi)

    # ----------------------------------------------------------------- EVENT
    def _capture_plane_event(self, out_dir, filename, step):
        """Record events for a fixed duration at the current plane and accumulate
        them into an event image. The raw .raw event file is the per-plane raw
        data. The device is (re)initialized per plane for a clean state."""
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

        save_raw = self.save_params.get("save_raw", True)
        raw_path = ""
        if out_dir and device.get_i_events_stream():
            raw_path = os.path.join(out_dir, f"{filename}_z{step:03d}.raw")
            device.get_i_events_stream().log_raw_data(raw_path)

        mv_iterator = EventsIterator.from_device(device=device)
        height, width = mv_iterator.get_size()

        self.status_update.emit(f"Recording events at plane {step+1} for {p['acqu_time']} s...")
        start = time.time()
        try:
            for _ in mv_iterator:
                if not self._is_running:
                    break
                if time.time() - start >= p["acqu_time"]:
                    break
        finally:
            if device.get_i_events_stream():
                device.get_i_events_stream().stop_log_raw_data()

        if not raw_path:
            return None

        # Rebuild the event image from the just-saved raw file.
        event_img = accumulate_event_frame(
            EventsIterator(input_path=raw_path, delta_t=1000000), width, height
        )
        if not save_raw:
            try:
                os.remove(raw_path)
            except OSError:
                pass
        if p.get("filter_crazy_pixels", True):
            event_img = filter_crazy_pixels(event_img)
        if p.get("apply_smoothing", True):
            event_img = apply_smoothing(event_img)
        return event_img

    # ----------------------------------------------------------------- save
    def _save_outputs(self, std_volume, avg_volume, event_volume, z_positions):
        """Save the depth volume(s) (3D TIFF) and the parameter log. Returns a
        short message for the status bar (empty if nothing was saved)."""
        out_dir = self.save_params.get("output_dir", "")
        filename = self.save_params.get("filename", "zstack")
        if not out_dir:
            return ""

        n = 0
        if self.camera == "orca" and std_volume:
            save_volume_tiff(np.array(std_volume, dtype=np.float32), out_dir, filename, "zstack_dsi")
            save_volume_tiff(np.array(avg_volume, dtype=np.float32), out_dir, filename, "zstack_average")
            n = len(std_volume)
        elif self.camera == "event" and event_volume:
            save_volume_tiff(np.array(event_volume, dtype=np.float32), out_dir, filename, "zstack_event")
            n = len(event_volume)

        if n == 0:
            return ""

        metadata = self.save_params.get("metadata")
        if metadata:
            metadata = dict(metadata)
            metadata["Z-Stack planes"] = {
                "camera": self.camera,
                "num_planes_saved": n,
                "z_positions": ", ".join(f"{z:.4f}" for z in z_positions),
            }
            save_parameter_log(out_dir, filename, metadata)

        return f" Saved {n} planes (3D TIFF) to {out_dir}"

    def _emit_position(self):
        """Report the real axis position so the UI readout tracks the stack."""
        try:
            self.position_update.emit(float(self.pidevice.qPOS(self.axis)[self.axis]))
        except Exception:
            pass

    def stop(self):
        self._is_running = False
