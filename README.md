# Event-DSI Microscope Control

PyQt6 GUI to run the Institut Fresnel event-based Dynamic Speckle Illumination
(event-DSI) microscope: Hamamatsu ORCA-Fusion (sCMOS) and Prophesee EVK4 (event)
cameras, a PI E-709 piezo Z-stage, and a Siglent AWG driving the liquid-crystal
speckle generator.

```
main.py            # entry point:  python main.py
config.py          # machine-specific paths + hardware constants (see "Per-machine config")
core/              # pure NumPy/OpenCV/SciPy math + file saving (no Qt, no SDKs)
hardware/          # camera / stage / AWG SDK wrappers + Qt worker threads
ui/                # main window, control widgets, Z-stack orchestrator
requirements.txt   # pip-installable dependencies
```

The app **degrades gracefully**: if a hardware SDK is missing, that instrument is
reported as unavailable instead of crashing the GUI. So you can install on a new
machine and bring up one device at a time.

---

## Installing on a new computer (e.g. the lab PC)

### 1. Get the code

Clone the repo (or copy the `dsi_microscope/` folder). Do **not** copy your
`.venv/` or `__pycache__/` across machines — recreate them locally.

### 2. Python

Use **64-bit Python 3.x on Windows** (developed on 3.12).

> ⚠️ **Check the Prophesee Metavision SDK's supported Python version first** if you
> need the event camera. Metavision only ships bindings for specific Python
> versions (and sometimes pins `numpy<2`). Match your Python version to whatever
> the installed Metavision SDK supports — otherwise `import metavision_core`
> fails and the EVK4 shows up as unavailable. The ORCA, PI stage, and AWG do not
> have this constraint.

### 3. Create a virtual environment and install the PyPI dependencies

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That covers numpy, scipy, opencv-python, PyQt6, tifffile, pyvisa, pyserial, and
pipython. At this point `python main.py` should already launch the GUI (with the
cameras/stage listed as unavailable until their SDKs are installed).

### 4. VISA backend (for the Siglent AWG)

PyVISA needs a VISA backend at runtime. Install **NI-VISA** (recommended for the
Siglent) from National Instruments. Alternatively, a pure-Python backend:

```powershell
pip install pyvisa-py
```

### 5. Vendor hardware SDKs

These are **not** plain pip installs — install each vendor package on the machine,
then point the app at it.

| Device | What to install | How the app finds it |
| --- | --- | --- |
| **Hamamatsu ORCA** | DCAM-API SDK (includes `dcam.py` under `…\dcamsdk4\samples\python`) | `HAMAMATSU_SDK_PATH` (see below) |
| **Prophesee EVK4** | Metavision SDK installer (provides `metavision_core`, `metavision_sdk_core`) | installed into the active Python env |
| **PI E-709 stage** | PI software / GCS DLLs; `pip install pipython` (already in requirements) | native USB serial, or RS-232 auto-scan |

### 6. Per-machine config

The only settings that differ between your laptop and the lab PC are the Hamamatsu
SDK path and the PI controller identifiers. Set them as **environment variables**
so the same code runs on both machines without editing `config.py`:

```powershell
# Path to the folder containing dcam.py (REQUIRED for the ORCA on this machine):
setx HAMAMATSU_SDK_PATH "C:\path\to\dcamsdk4\samples\python"

# Optional PI overrides (only if the lab controller/cabling differs):
setx PI_SERIAL_NUM "0023550769"     # PI native-USB serial number
setx PI_RS232_PORT "5"              # force a COM port instead of auto-scanning
setx PI_CONTROLLER_NAME "E-709"     # GCS controller model
```

`setx` persists the variable for new terminals — open a fresh shell afterwards.
(For a one-off session use `$env:HAMAMATSU_SDK_PATH = "..."` instead.) If you
prefer, you can just edit the defaults directly in [config.py](config.py).

### 7. Run

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

---

## Acquisition outputs

Saved into the output directory chosen per tab (filenames use the `Filename Base`):

| File | Source |
| --- | --- |
| `raw_stack_<name>.tif` | Raw 16-bit speckle frames. Single-Z: one stack. **Z-stack: all planes' frames in one multi-page TIFF** (plane-major). |
| `dsi_<name>.tif/.mat`, `average_<name>.tif/.mat` | Single-Z DSI (std-dev) + widefield (average) images. |
| `zstack_dsi_<name>.tif`, `zstack_average_<name>.tif` | Z-stack depth volumes (one page per plane). |
| `final_image_<name>.tif/.mat` | EVK4 accumulated event image. |
| `*_zNNN.raw` | EVK4 raw event recording, one per plane. |
| `parameters_<name>.txt` | Full instrument-state log for the acquisition. |

> For large 3D ORCA Z-stacks, make sure `tifffile` is installed (it is in
> `requirements.txt`): the raw stack is then streamed to disk as a BigTIFF with
> flat memory use and no 4 GB file-size limit. Without it, the raw frames are
> buffered in RAM and written in one OpenCV call — fine only for small stacks.

---

## Troubleshooting

- **GUI launches but a camera/stage is "unavailable"** — the SDK import failed.
  Confirm the vendor SDK is installed *into the active venv's Python* and, for the
  ORCA, that `HAMAMATSU_SDK_PATH` points at the folder containing `dcam.py`.
- **`import metavision_core` fails** — almost always a Python-version mismatch
  (see step 2). Recreate the venv with a Python version the Metavision SDK supports.
- **AWG won't connect / no resources listed** — no VISA backend (step 4), or the
  instrument isn't on the bus.
- **PI stage won't connect** — set `PI_RS232_PORT` to the actual COM port, and
  confirm the PI GCS DLLs are installed.
