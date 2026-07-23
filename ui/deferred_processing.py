"""Background reconstruction of a *deferred* Z-stack acquisition.

When the user enables **deferred processing**, the orchestrator captures each
plane and saves its raw data (ORCA speckle stack TIFF / EVK4 event ``.raw``) but
skips the per-plane reconstruction — so the acquisition finishes as soon as the
last plane is recorded and the microscope is free for the next run. This worker
does the skipped work afterwards, off the acquisition thread, by rebuilding the
summary products from the archived raws exactly as the offline ``tools/`` scripts
would (reusing the same ``core`` functions, so the result is byte-identical to a
live run).

It owns no hardware: the ORCA rebuild reads TIFFs (pure ``core``); the EVK4
rebuild only *decodes* ``.raw`` files, for which it injects a Metavision reader
into the SDK-agnostic ``core.rebuild_evk4_zstack_from_raw``. Because it runs on
its own ``QThread``, a new acquisition can proceed while it works — the two
compete only for CPU/RAM/disk, not for the camera.

MainWindow keeps a queue and runs one of these at a time (so several deferred
runs pile up rather than thrash), starting the next when ``finished_signal``
fires.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from config import EVK4_SENSOR_WIDTH, EVK4_SENSOR_HEIGHT
from core import rebuild_zstack_from_raw, rebuild_evk4_zstack_from_raw
from hardware.event_camera import EventsIterator, METAVISION_AVAILABLE


class DeferredProcessingWorker(QThread):
    """Reconstruct one deferred Z-stack job from its archived per-plane raws.

    The ``job`` dict is the one emitted by ``AutomatedZStackWorker.deferred_ready``
    (see :meth:`AutomatedZStackWorker._finish_deferred`). ``finished_signal`` always
    fires — success or failure — carrying the same job dict, so the owner can start
    the next queued job regardless of outcome.
    """

    status_update = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)   # the job dict (whether it succeeded or not)
    error_signal = pyqtSignal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job

    @staticmethod
    def _evk4_reader_factory(raw_path):
        """Open one plane's ``.raw`` and report its sensor size.

        Returns ``(iterator, width, height)`` as ``core.rebuild_evk4_zstack_from_raw``
        expects. ``EventsIterator.get_size()`` returns ``(height, width)`` — the same
        order the live acquisition path uses — so the two are unpacked accordingly.
        """
        reader = EventsIterator(input_path=raw_path, delta_t=1000000)
        try:
            height, width = reader.get_size()
        except Exception:
            width, height = EVK4_SENSOR_WIDTH, EVK4_SENSOR_HEIGHT
        return reader, width, height

    def run(self):
        job = self.job
        camera = job.get("camera")
        filename = job.get("filename", "zstack")
        label = "ORCA" if camera == "orca" else "event"
        try:
            self.status_update.emit(f"Processing deferred {label} acquisition '{filename}'...")
            # metadata is intentionally NOT passed: the orchestrator already wrote
            # the parameter log (with the measured framerate and, appended by
            # MainWindow, the elapsed timing), and re-writing it here would clobber
            # those sections. This worker only produces the volumes/profiles.
            if camera == "orca":
                n, missing = rebuild_zstack_from_raw(
                    job["raw_dir"], job["out_dir"], filename, job["z_positions"],
                    expected_frames=job.get("expected_frames"),
                    save_average=job.get("save_average", True),
                    metadata=None, status=self.status_update.emit,
                )
                skipped = missing
            else:
                if not METAVISION_AVAILABLE:
                    raise RuntimeError(
                        "Metavision SDK not available — cannot decode the event .raw "
                        "streams. Re-run tools/rebuild_evk4_zstack.py on a machine "
                        "with the SDK.")
                n, skipped = rebuild_evk4_zstack_from_raw(
                    job["raw_dir"], job["out_dir"], filename, job["z_positions"],
                    self._evk4_reader_factory,
                    roi=job.get("roi"),
                    do_filter=job.get("do_filter", True),
                    do_smooth=job.get("do_smooth", True),
                    metadata=None, status=self.status_update.emit,
                )
            tail = f" ({len(skipped)} planes could not be rebuilt)" if skipped else ""
            self.status_update.emit(
                f"Deferred {label} processing complete: '{filename}' — "
                f"{n} planes reconstructed{tail}.")
        except Exception as e:
            self.error_signal.emit(
                f"Deferred {label} processing failed for '{filename}': {e}. "
                f"The raw files are intact — rebuild with tools/rebuild_"
                f"{'orca' if camera == 'orca' else 'evk4'}_zstack.py.")
        finally:
            self.finished_signal.emit(job)
