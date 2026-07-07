"""Siglent SDG6022X arbitrary waveform generator — PyVISA SCPI wrapper.

Pure connectivity: no PyQt6 here. The UI widget (ui/widgets.py) owns the controls
and calls into this controller. Methods raise on failure so the caller can surface
errors through the status bar / message boxes.
"""

# Guarded like every other instrument SDK (pipython/dcam/metavision): the GUI must
# build and run on a machine with no VISA backend. A missing pyvisa (or backend)
# then surfaces as a disabled control / clear error, never a startup crash.
try:
    import pyvisa
    PYVISA_AVAILABLE = True
except ImportError:
    pyvisa = None
    PYVISA_AVAILABLE = False


class AWGController:
    """Thin SCPI controller for the Siglent AWG driving the LC speckle modulation."""

    def __init__(self):
        # The ResourceManager is created lazily (not here) so constructing the
        # widget can never crash the app when no VISA implementation is present:
        # pyvisa.ResourceManager() raises ValueError without a backend.
        self.rm = None
        self.awg = None

    def _ensure_rm(self):
        """Create the VISA ResourceManager on first use. Raises if unavailable."""
        if self.rm is None:
            if not PYVISA_AVAILABLE:
                raise RuntimeError(
                    "PyVISA is not installed — AWG control is unavailable. "
                    "Install pyvisa and a VISA backend (e.g. pyvisa-py) to use the AWG."
                )
            self.rm = pyvisa.ResourceManager()
        return self.rm

    # -- discovery -----------------------------------------------------------
    def list_resources(self):
        """Return the list of available VISA resource strings (possibly empty).

        Degrades to an empty list if pyvisa or a VISA backend is missing, so the
        UI can populate the picker without crashing on a machine with no VISA.
        """
        try:
            return list(self._ensure_rm().list_resources())
        except Exception:
            return []

    # -- connection ----------------------------------------------------------
    def connect(self, address):
        """Open the VISA resource and return the instrument's *IDN? string.

        Raises on failure; leaves ``self.awg`` as None if the connection failed.
        """
        try:
            self.awg = self._ensure_rm().open_resource(address)
            idn = self.awg.query("*IDN?")
            return idn.strip()
        except Exception:
            self.awg = None
            raise

    @property
    def is_connected(self):
        return self.awg is not None

    # -- parameters ----------------------------------------------------------
    def set_params(self, freq_hz, amp_vpp, channel=1, waveform="SQUARE", load="HZ"):
        """Configure the given channel's waveform at the freq / amplitude.

        ``waveform`` is the Siglent ``WVTP`` basic-wave type (SQUARE, SINE, RAMP,
        PULSE …); it defaults to SQUARE so existing callers are unchanged. All of
        these accept the FRQ/AMP/OFST fields used here. ``channel`` is 1 or 2 and
        maps to the instrument's C1 / C2 SCPI prefix, so the two outputs can be
        driven completely independently.

        ``load`` is the output-load setting the Siglent uses to interpret the
        requested amplitude. The LC cell is high-impedance, so it defaults to
        ``"HZ"``: a requested 18 Vpp then actually appears as 18 Vpp at the cell.
        Under the instrument's default 50 Ω load the same request would be halved,
        so the load is set explicitly (before the waveform) every time.
        """
        if not self.awg:
            return
        # Set the load first — it changes how the following AMP value is interpreted.
        self.awg.write(f"C{channel}:OUTP LOAD,{load}")
        cmd = f"C{channel}:BSWV WVTP,{waveform},FRQ,{freq_hz},AMP,{amp_vpp},OFST,0"
        self.awg.write(cmd)

    def set_output(self, enabled, channel=1):
        """Toggle the given channel's output ON/OFF (channel is 1 or 2)."""
        if not self.awg:
            return
        self.awg.write(f"C{channel}:OUTP ON" if enabled else f"C{channel}:OUTP OFF")

    # -- teardown ------------------------------------------------------------
    def close(self):
        """Safely turn both outputs off and close the VISA session."""
        if self.awg:
            try:
                self.awg.write("C1:OUTP OFF")
                self.awg.write("C2:OUTP OFF")
                self.awg.close()
            except Exception:
                pass
            finally:
                self.awg = None
