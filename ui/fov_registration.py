"""Live EVK4 -> ORCA field-of-view registration worker.

Captures a quick reference from each camera (both at FULL sensor — a hardware
crop would invalidate the coordinate spaces), then registers the EVK4 event
image onto the ORCA frame with ``core.register_evk4_to_orca`` to measure the
current EVK4->ORCA affine. Used by the "Measure & Match" button when the
cameras may have been moved/rotated since the stored calibration.

Like the Z-stack orchestrator, this lives in ui/ because it composes the
hardware layer with the core math. Requirements for a good measurement:

* a structured sample (e.g. beads) in view of both cameras;
* the speckle modulation (AWG) running, so the EVK4 actually fires events;
* both cameras roughly in focus — the ports' focal planes differ slightly,
  which is fine (the registration works on smoothed blob maps), but a grossly
  defocused camera gives a weak match (reported via the NCC score).
"""

import gc
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    DCAM_EXPOSURE_PROP, DCAM_SUBARRAY_MODE_PROP, DCAM_SUBARRAY_MODE_OFF,
    EVK4_ERC_RATE,
)
from core import accumulate_event_frame, register_evk4_to_orca
from hardware.event_camera import EventsIterator, initiate_device
from hardware.orca_camera import Dcam, Dcamapi


class FovRegistrationWorker(QThread):
    """Capture ORCA + EVK4 references and measure the EVK4->ORCA registration."""

    status_update = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)   # {affine, score, params, orca_img, evk4_img}
    error_signal = pyqtSignal(str)

    # Too few total events means the EVK4 saw essentially nothing — registering
    # noise would silently produce garbage, so it is refused instead.
    MIN_EVENTS = 10_000

    def __init__(self, orca_exposure_ms, evk4_biases, evk4_duration_s=3.0,
                 orca_frames=16, seed_affine=None):
        super().__init__()
        self.orca_exposure_ms = float(orca_exposure_ms)
        self.evk4_biases = dict(evk4_biases)
        self.evk4_duration_s = float(evk4_duration_s)
        self.orca_frames = int(orca_frames)
        self.seed_affine = seed_affine
        self._is_running = True

    def run(self):
        try:
            self.status_update.emit(
                "FOV measurement: capturing the ORCA reference (full sensor)...")
            orca_img = self._capture_orca()
            if orca_img is None:
                return  # user aborted
            self.status_update.emit(
                f"FOV measurement: accumulating EVK4 events for "
                f"{self.evk4_duration_s:g} s (full sensor)...")
            evk4_img = self._capture_evk4()
            if evk4_img is None:
                return
            if float(evk4_img.sum()) < self.MIN_EVENTS:
                raise RuntimeError(
                    "the event camera recorded almost no events. Check that the "
                    "speckle modulation (AWG output) is running and the sample "
                    "is in view, then retry.")
            affine, score, params = register_evk4_to_orca(
                orca_img, evk4_img, seed_affine=self.seed_affine,
                status=self.status_update.emit)
            if not self._is_running:
                return
            self.finished_ok.emit({
                "affine": affine, "score": score, "params": params,
                "orca_img": orca_img, "evk4_img": evk4_img,
            })
        except Exception as e:
            self.error_signal.emit(f"FOV measurement failed: {e}")

    # ------------------------------------------------------------------ ORCA
    def _capture_orca(self):
        """Average a short full-sensor ORCA burst (the registration reference).

        Subarray mode is explicitly forced OFF: the affine is defined in
        full-sensor coordinates, so a leftover hardware crop from a previous
        run would silently shift the whole registration.
        """
        if not Dcamapi.init():
            raise RuntimeError(f"Dcamapi.init() failed: {Dcamapi.lasterr()}")
        dcam = Dcam(0)
        try:
            if not dcam.dev_open():
                raise RuntimeError(f"ORCA dev_open() failed: {dcam.lasterr()}")
            dcam.prop_setvalue(DCAM_SUBARRAY_MODE_PROP, DCAM_SUBARRAY_MODE_OFF)
            dcam.prop_setvalue(DCAM_EXPOSURE_PROP, self.orca_exposure_ms / 1000.0)
            if not dcam.buf_alloc(self.orca_frames):
                raise RuntimeError(f"ORCA buf_alloc failed: {dcam.lasterr()}")
            acc, count = None, 0
            try:
                if dcam.cap_start():
                    for i in range(self.orca_frames):
                        if not self._is_running:
                            return None
                        if not dcam.wait_capevent_frameready(5000):
                            raise RuntimeError(f"ORCA frame timeout: {dcam.lasterr()}")
                        frame = dcam.buf_getframedata(i).astype(np.float64)
                        acc = frame if acc is None else acc + frame
                        count += 1
                    dcam.cap_stop()
            finally:
                dcam.buf_release()
            if count == 0:
                raise RuntimeError("no ORCA frames captured")
            return acc / count
        finally:
            try:
                dcam.dev_close()
            except Exception:
                pass
            try:
                Dcamapi.uninit()
            except Exception:
                pass

    # ----------------------------------------------------------------- EVK4
    def _capture_evk4(self):
        """Accumulate a full-sensor event image for the configured duration."""
        device = initiate_device("")
        mv_iterator = None
        try:
            biases = device.get_i_ll_biases()
            if biases:
                biases.set("bias_fo", self.evk4_biases["bias_fo"])
                biases.set("bias_hpf", self.evk4_biases["bias_hpf"])
                biases.set("bias_diff_on", self.evk4_biases["bias_on"])
                biases.set("bias_diff_off", self.evk4_biases["bias_off"])
            erc = device.get_i_erc_module()
            if erc:
                erc.enable(True)
                erc.set_cd_event_rate(EVK4_ERC_RATE)

            mv_iterator = EventsIterator.from_device(device=device)
            height, width = mv_iterator.get_size()

            def events_for_duration():
                start = time.time()
                for evs in mv_iterator:
                    if not self._is_running:
                        return
                    yield evs
                    if time.time() - start >= self.evk4_duration_s:
                        return

            img = accumulate_event_frame(events_for_duration(), width, height)
            return img if self._is_running else None
        finally:
            # Drop every reference so the USB link is released for the live feed.
            del mv_iterator
            del device
            gc.collect()

    def stop(self):
        self._is_running = False
