"""Physik Instrumente (PI) Z-stage — pipython wrapper and move worker.

The blocking commands (`MOV`, `pitools.waitontarget`) MUST run off the GUI thread,
so a dedicated `PIMoveWorker(QThread)` is provided. `StageController` owns the raw
`GCSDevice` handle and connection lifecycle.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    PI_AUTOZERO, PI_AXIS, PI_BAUDRATE, PI_CONTROLLER_NAME, PI_RS232_PORT,
    PI_SERIAL_NUM,
)

try:
    from pipython import GCSDevice, pitools
    PI_AVAILABLE = True
except ImportError:
    GCSDevice = None
    pitools = None
    PI_AVAILABLE = False


class StageController:
    """Owns the E-709 device handle, connection and piezo-axis initialization."""

    def __init__(self, serial_num=PI_SERIAL_NUM, axis=PI_AXIS):
        self.serial_num = serial_num
        self.axis = axis
        self.pidevice = None
        self.connection_info = ""
        # Position unit and travel range, read from the controller on connect.
        # Default to µm: the PIFOC piezo stages report in micrometers.
        self.unit = "µm"
        self.travel_min = None
        self.travel_max = None

    @property
    def is_connected(self):
        return self.pidevice is not None

    def connect(self):
        """Open the controller and put the piezo axis into closed-loop control.

        Tries PI native USB first (if a serial number is configured), then falls
        back to RS-232 over a USB-to-serial adapter (e.g. PI US232R). Returns a
        short description of the interface used. Raises on failure, leaving the
        controller in a clean disconnected state.
        """
        try:
            self.pidevice = GCSDevice(PI_CONTROLLER_NAME)
            self._open_interface()
            self._initialize_axis()
            return self.connection_info
        except Exception:
            self.close()
            raise

    def _open_interface(self):
        """Connect over the interface the E-709 actually uses on this bench.

        The unit is wired through a US232R USB-to-serial adapter (RS-232), but we
        also try PI native USB so the same code works if the cabling changes.
        """
        # 1) PI native USB, only if a serial number is configured.
        if self.serial_num:
            try:
                self.pidevice.ConnectUSB(serialnum=self.serial_num)
                self.connection_info = f"USB SN {self.serial_num}"
                return
            except Exception:
                pass

        # 2) RS-232 on a forced port, if the user pinned one.
        if PI_RS232_PORT is not None:
            self.pidevice.ConnectRS232(comport=PI_RS232_PORT, baudrate=PI_BAUDRATE)
            self.connection_info = f"RS-232 COM{PI_RS232_PORT} @ {PI_BAUDRATE}"
            return

        # 3) RS-232 auto-scan: try each serial port, keep the one that answers as
        #    the expected controller.
        last_err = None
        ports = self._candidate_serial_ports()
        for port in ports:
            try:
                self.pidevice.ConnectRS232(comport=port, baudrate=PI_BAUDRATE)
                if PI_CONTROLLER_NAME in self.pidevice.qIDN():
                    self.connection_info = f"RS-232 COM{port} @ {PI_BAUDRATE}"
                    return
                self.pidevice.CloseConnection()
            except Exception as exc:
                last_err = exc

        raise RuntimeError(
            f"Could not connect to the {PI_CONTROLLER_NAME}. Tried native USB "
            f"(SN {self.serial_num}) and serial ports {ports}. "
            f"Last error: {last_err}"
        )

    @staticmethod
    def _candidate_serial_ports():
        """Return COM port numbers to probe, FTDI / US232R adapters first."""
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
        except Exception:
            return list(range(1, 21))

        def looks_like_pi_adapter(p):
            text = " ".join(str(x) for x in (p.description, p.manufacturer, p.product)).lower()
            return any(key in text for key in ("us232r", "ftdi", "usb serial", "physik"))

        ports.sort(key=lambda p: not looks_like_pi_adapter(p))  # adapters first

        numbers = []
        for p in ports:
            dev = (p.device or "").upper()
            if dev.startswith("COM"):
                try:
                    numbers.append(int(dev[3:]))
                except ValueError:
                    pass
        return numbers or list(range(1, 21))

    def _initialize_axis(self):
        """Initialize the piezo axis: detect its id and enable the servo.

        The E-709 has no reference switches, so (unlike a servo-motor stage) there
        is NO FNL/FRF referencing. The capacitive/SGS sensor gives absolute
        position; enabling the servo (SVO) puts the axis in closed-loop, ready for
        MOV. Auto-zero (ATZ) is run only if explicitly enabled.
        """
        # Use the axis id the controller actually reports (more robust than a
        # hard-coded value).
        try:
            axes = self.pidevice.qSAI()
            if axes:
                self.axis = axes[0]
        except Exception:
            pass

        if PI_AUTOZERO:
            self.pidevice.ATZ(self.axis, 0.0)
            pitools.waitontarget(self.pidevice, axes=self.axis)

        self.pidevice.SVO(self.axis, True)
        self._read_axis_metadata()

    def _read_axis_metadata(self):
        """Read the controller's position unit and travel limits (best effort).

        The PIFOC reports in micrometers; querying qPUN/qTMN/qTMX grounds the UI
        in the controller's real configuration instead of a hard-coded guess.
        """
        try:
            unit = self.pidevice.qPUN(self.axis)[self.axis]
            if unit:
                self.unit = self._pretty_unit(unit)
        except Exception:
            pass  # keep the µm default

        try:
            self.travel_min = float(self.pidevice.qTMN(self.axis)[self.axis])
            self.travel_max = float(self.pidevice.qTMX(self.axis)[self.axis])
        except Exception:
            self.travel_min = self.travel_max = None

    @staticmethod
    def _pretty_unit(unit):
        """Normalize a controller unit string to a display label."""
        low = str(unit).strip().lower()
        if low in ("um", "µm", "micron", "microns", "micrometer", "micrometre"):
            return "µm"
        if low in ("mm", "millimeter", "millimetre"):
            return "mm"
        if low in ("nm", "nanometer", "nanometre"):
            return "nm"
        return str(unit).strip() or "µm"

    def position(self):
        """Return the current axis position (controller units)."""
        return self.pidevice.qPOS(self.axis)[self.axis]

    def close(self):
        """Close the connection, swallowing teardown errors."""
        if self.pidevice:
            try:
                self.pidevice.CloseConnection()
            except Exception:
                pass
            finally:
                self.pidevice = None
                self.connection_info = ""


class PIMoveWorker(QThread):
    """Offloads the blocking PI motor movements from the main GUI thread."""

    status_update = pyqtSignal(str)
    finished_signal = pyqtSignal(float)

    def __init__(self, pidevice, axis, target):
        super().__init__()
        self.pidevice = pidevice
        self.axis = axis
        self.target = target

    def run(self):
        try:
            self.status_update.emit(f"Moving PI Stage to {self.target:.4f}...")
            self.pidevice.MOV(self.axis, self.target)
            pitools.waitontarget(self.pidevice, axes=self.axis)
            current_pos = self.pidevice.qPOS(self.axis)[self.axis]
            self.status_update.emit(f"Movement complete. Current position: {current_pos:.4f}")
            self.finished_signal.emit(current_pos)
        except Exception as e:
            self.status_update.emit(f"Motor Movement Error: {str(e)}")
            # Always finish so the UI re-enables controls and resumes polling.
            try:
                current_pos = self.pidevice.qPOS(self.axis)[self.axis]
            except Exception:
                current_pos = float("nan")
            self.finished_signal.emit(current_pos)
