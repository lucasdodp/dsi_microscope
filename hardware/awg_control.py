"""Siglent SDG6022X arbitrary waveform generator — PyVISA SCPI wrapper.

Pure connectivity: no PyQt6 here. The UI widget (ui/widgets.py) owns the controls
and calls into this controller. Methods raise on failure so the caller can surface
errors through the status bar / message boxes.
"""

import pyvisa


class AWGController:
    """Thin SCPI controller for the Siglent AWG driving the LC speckle modulation."""

    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        self.awg = None

    # -- discovery -----------------------------------------------------------
    def list_resources(self):
        """Return the list of available VISA resource strings (possibly empty)."""
        return list(self.rm.list_resources())

    # -- connection ----------------------------------------------------------
    def connect(self, address):
        """Open the VISA resource and return the instrument's *IDN? string.

        Raises on failure; leaves ``self.awg`` as None if the connection failed.
        """
        try:
            self.awg = self.rm.open_resource(address)
            idn = self.awg.query("*IDN?")
            return idn.strip()
        except Exception:
            self.awg = None
            raise

    @property
    def is_connected(self):
        return self.awg is not None

    # -- parameters ----------------------------------------------------------
    def set_params(self, freq_hz, amp_vpp, channel=1, waveform="SQUARE"):
        """Configure the given channel's waveform at the freq / amplitude.

        ``waveform`` is the Siglent ``WVTP`` basic-wave type (SQUARE, SINE, RAMP,
        PULSE …); it defaults to SQUARE so existing callers are unchanged. All of
        these accept the FRQ/AMP/OFST fields used here. ``channel`` is 1 or 2 and
        maps to the instrument's C1 / C2 SCPI prefix, so the two outputs can be
        driven completely independently.
        """
        if not self.awg:
            return
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
