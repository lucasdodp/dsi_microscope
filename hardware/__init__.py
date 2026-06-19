"""Hardware communication layer.

Each module owns one instrument's raw API connectivity plus its low-level QThread
acquisition loop. Availability flags (``*_AVAILABLE``) are exposed so the UI can
degrade gracefully when an SDK or driver is absent on the host.
"""
