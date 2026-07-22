"""Pure data-processing routines for event-DSI optical sectioning.

NOTHING in this module may import PyQt6 or a hardware SDK. Functions take plain
arrays (or generic iterables of structured event arrays) and return arrays / write
files. This keeps the optical-sectioning math testable and side-effect free apart
from the explicit `save_*` helpers.
"""

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import scipy.io

from config import (
    EVK4_CRAZY_PIXEL_PERCENTILE, EVK4_CSV_QUEUE_CHUNKS, EVK4_CSV_WORKERS,
)


# ---------------------------------------------------------------------------
# Hamamatsu ORCA — Dynamic Speckle Illumination (DSI) sectioning
# ---------------------------------------------------------------------------
def process_dsi(stack, roi):
    """Compute the DSI optically-sectioned image from a frame stack.

    Mathematically rigorous consecutive-difference fluctuation estimator:
    given the cropped stack reshaped to (Y, X, N) frames, the per-pixel sectioning
    strength is sqrt(sum over consecutive frame differences squared). No np.roll
    offsets are used, so there is no spurious wrap-around between the last and
    first frame.

    Parameters
    ----------
    stack : np.ndarray
        Raw frame stack of shape (N, H, W).
    roi : dict
        Keys ``x_min``, ``x_max``, ``y_min``, ``y_max`` (pixel crop bounds).

    Returns
    -------
    (z_val, display_img) : (float, np.ndarray)
        ``z_val`` is the scalar focus/sectioning metric (sum of the std map);
        ``display_img`` is an 8-bit normalized image for live preview.
    """
    y_min, y_max = roi["y_min"], roi["y_max"]
    x_min, x_max = roi["x_min"], roi["x_max"]

    # Clamp the ROI to the actual sensor frame so a generous UI value can't slice
    # out of bounds.
    max_y, max_x = stack.shape[1], stack.shape[2]
    y_max = min(y_max, max_y)
    x_max = min(x_max, max_x)

    cropped_stack = stack[:, y_min:y_max, x_min:x_max]
    images = np.transpose(cropped_stack, (1, 2, 0)).astype(np.float32)

    # Computationally correct consecutive difference (no np.roll offsets).
    diff = images[:, :, 1:] - images[:, :, :-1]
    images_std = np.sqrt(np.sum(np.square(diff), axis=2))

    z_val = np.sum(images_std)
    display_img = cv2.normalize(images_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return z_val, display_img


# Peak working-set budget for the float64 chunk processed at once in
# ``compute_dsi_images``. Bounding this (rather than casting the whole stack to
# float at once) keeps memory flat regardless of frame count, so a large ORCA
# Z-stack can't drive the machine into swap.
_DSI_CHUNK_BYTES = 128 * 1024 * 1024


def _frame_chunk(h, w):
    """How many H×W frames fit in the float64 working-set budget (at least one)."""
    per_frame = max(1, h * w * 8)  # one H×W plane as float64
    return max(1, _DSI_CHUNK_BYTES // per_frame)


def compute_dsi_images(stack, roi=None):
    """Compute the average (widefield) and standard-deviation (DSI) images.

    This is the single-z DSI reconstruction: given a time series of raw speckle
    frames acquired at one focal plane, the optical sectioning comes from the
    per-pixel statistics *across* the stack (Ventalon & Mertz; cf. reference
    papers), not from moving the objective.

    Memory: the mean and standard deviation are accumulated over the frame axis
    in small chunks instead of materializing ``stack.astype(float32)`` (and the
    deviation array ``np.std`` builds internally). Those would each be the full
    size of the stack — gigabytes for a high frame count at full sensor, enough
    to swap the machine to a standstill. Here the working set is only a few H×W
    planes, so peak memory is independent of N.

    Speed: a stack of small integers (the ORCA's uint16) takes an exact
    single-pass integer path, which is ~1.6x faster than the two-pass float one
    and returns bit-identical results. See ``_dsi_stats_integer``.

    Parameters
    ----------
    stack : np.ndarray
        Raw frame stack of shape (N, H, W).
    roi : dict or None
        Optional crop bounds with keys ``x_min``, ``x_max``, ``y_min``, ``y_max``.

    Returns
    -------
    (avg_img, std_img) : (np.ndarray, np.ndarray)
        Float32 average-intensity image (the conventional widefield equivalent)
        and standard-deviation image (the optically-sectioned DSI image).
    """
    images = stack
    if roi is not None:
        # Clamp to the actual frame so a generous UI value can't slice out of bounds.
        max_y, max_x = images.shape[1], images.shape[2]
        y_min, y_max = roi["y_min"], min(roi["y_max"], max_y)
        x_min, x_max = roi["x_min"], min(roi["x_max"], max_x)
        images = images[:, y_min:y_max, x_min:x_max]

    n, h, w = images.shape

    if _dsi_exact_integer_ok(images.dtype):
        mean, var = _dsi_stats_integer(images, n, h, w)
    else:
        mean, var = _dsi_stats_float(images, n, h, w)

    avg_img = mean.astype(np.float32)
    std_img = np.sqrt(var, out=var).astype(np.float32)
    return avg_img, std_img


def _dsi_exact_integer_ok(dtype):
    """True if a stack of this dtype can use the exact integer one-pass path.

    Restricted to integer types of at most 2 bytes (the ORCA delivers uint16).
    The bound is what makes the accumulators safe: the largest possible product
    is 65535**2 ≈ 4.3e9, so int64 does not overflow until ~2e9 frames. A 4-byte
    integer type would overflow after two frames, so it takes the float path.
    """
    return np.issubdtype(dtype, np.integer) and dtype.itemsize <= 2


def _dsi_stats_integer(images, n, h, w):
    """Exact single-pass mean and variance for small-integer frame stacks.

    Accumulates the sum and the sum of squares as **exact int64** — the frames
    are integers, so neither accumulator rounds at all — and forms the variance
    as ``E[x^2] - E[x]^2``. That identity is the one a float implementation must
    avoid, because it subtracts two large nearly-equal numbers; here both terms
    are exact, so the only rounding is the final float64 subtraction, far below
    the float32 the result is returned in. Verified bit-identical to the
    two-pass float path on speckle-like data.

    It is also the faster path, because it reads the stack **once** instead of
    twice and never materialises a float64 copy of a block: ``einsum`` consumes
    the uint16 view directly and accumulates into int64. Measured on a
    2304x2304 x 50 stack: 1.10 s against 1.72 s for the two-pass float version.
    """
    # The block is read in its native dtype rather than cast, so four times as
    # many frames fit the same working-set budget as the float64 path.
    chunk = max(1, _frame_chunk(h, w) * 4)
    s1 = np.zeros((h, w), dtype=np.int64)
    s2 = np.zeros((h, w), dtype=np.int64)
    for start in range(0, n, chunk):
        block = images[start:start + chunk]        # view, no copy
        s1 += block.sum(axis=0, dtype=np.int64)
        s2 += np.einsum("ijk,ijk->jk", block, block, dtype=np.int64)
    mean = s1 / n
    var = s2 / n - mean * mean
    # Rounding in the subtraction can leave a tiny negative variance on pixels
    # that never fluctuated; clamp so the sqrt below cannot produce NaN.
    np.maximum(var, 0, out=var)
    return mean, var


def _dsi_stats_float(images, n, h, w):
    """Two-pass mean and variance for non-integer stacks.

    Population variance (ddof=0) as the summed squared deviation about the mean.
    Two passes avoid the catastrophic cancellation a single-pass sum of squares
    suffers in floating point — the integer path above sidesteps that by being
    exact, but this one cannot.
    """
    chunk = _frame_chunk(h, w)

    # Pass 1 — mean. ``sum(dtype=float64)`` reduces over the frame axis without
    # casting the whole block to float first, so the only big array is the H×W
    # accumulator.
    acc = np.zeros((h, w), dtype=np.float64)
    for start in range(0, n, chunk):
        acc += images[start:start + chunk].sum(axis=0, dtype=np.float64)
    mean = acc / n

    # Pass 2 — squared deviations about that mean.
    sq = np.zeros((h, w), dtype=np.float64)
    for start in range(0, n, chunk):
        block = images[start:start + chunk].astype(np.float64)
        block -= mean
        block *= block
        sq += block.sum(axis=0)
    return mean, sq / n


def crop_to_roi(image, roi):
    """Crop a 2D (H, W) or 3D (H, W, C) image to an ROI window.

    The ROI uses the same ``x_min``/``x_max``/``y_min``/``y_max`` keys as the
    ORCA DSI crop, expressed in the image's own pixel coordinates. Bounds are
    clamped to the frame so a generous selection can't slice out of range, and
    ``roi=None`` (or a window covering the whole frame) returns the image
    unchanged. Works for both grayscale and colour frames, so it serves the EVK4
    live preview (BGR) and the accumulated event image (2D) alike.
    """
    if roi is None:
        return image
    h, w = image.shape[:2]
    y0 = max(0, min(int(roi["y_min"]), h))
    y1 = max(y0, min(int(roi["y_max"]), h))
    x0 = max(0, min(int(roi["x_min"]), w))
    x1 = max(x0, min(int(roi["x_max"]), w))
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return image
    return image[y0:y1, x0:x1]


def normalize_to_8bit(image):
    """Min-max normalize a float image to an 8-bit image for display / preview."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def save_dsi_results(avg_img, std_img, out_dir, filename):
    """Persist the average and DSI (std) images.

    The ``.mat`` files keep the full float precision for downstream analysis
    (e.g. RIM in MATLAB); the ``.tif`` files are 8-bit normalized previews.
    Returns the output directory for status reporting.
    """
    scipy.io.savemat(os.path.join(out_dir, f"{filename}_average.mat"), {"average_image": avg_img})
    scipy.io.savemat(os.path.join(out_dir, f"{filename}_dsi.mat"), {"dsi_image": std_img})
    cv2.imwrite(os.path.join(out_dir, f"{filename}_average.tif"), normalize_to_8bit(avg_img))
    cv2.imwrite(os.path.join(out_dir, f"{filename}_dsi.tif"), normalize_to_8bit(std_img))
    return out_dir


def _write_multipage_tiff(path, frames):
    """Write a (N, H, W) array as a multi-page TIFF.

    Prefers ``tifffile`` (writes clean ImageJ-compatible stacks with axis
    metadata) and falls back to OpenCV's ``imwritemulti`` so no extra dependency
    is required. Both preserve the array's native bit depth (e.g. 16-bit).
    """
    try:
        import tifffile
        tifffile.imwrite(path, frames, imagej=True)
    except ImportError:
        cv2.imwritemulti(path, list(frames))


def _prepare_raw_frames(stack, roi=None):
    """Crop a raw frame stack to the ROI and coerce it to a native integer depth.

    Used by ``save_raw_stack_tiff`` so the on-disk raw data is identical for the
    single-Z acquire and for every per-plane file of a Z-stack. Raw data is never
    normalized; the camera's native bit depth (typically uint16) is preserved.
    """
    frames = stack
    if roi is not None:
        max_y, max_x = frames.shape[1], frames.shape[2]
        y_min, y_max = roi["y_min"], min(roi["y_max"], max_y)
        x_min, x_max = roi["x_min"], min(roi["x_max"], max_x)
        frames = frames[:, y_min:y_max, x_min:x_max]

    if frames.dtype not in (np.uint8, np.uint16):
        frames = frames.astype(np.uint16)
    return frames


def save_raw_stack_tiff(stack, out_dir, filename, roi=None, plane=None):
    """Save the raw speckle frame stack as a multi-page 16-bit TIFF.

    This is the archival *raw data*: every acquired frame at the camera's native
    bit depth, before any averaging / standard-deviation processing. The TIFF
    pages are the individual frames (third axis = frame index), so the file opens
    in ImageJ/Fiji as a scrollable stack and can be re-processed later (e.g. with
    a different sectioning estimator or the RIM algorithm).

    The filename base comes first so the user's name leads the file name:
    ``<filename>_raw_stack.tif`` for a single plane, or
    ``<filename>_raw_stack_zNNN.tif`` for each plane of a Z-stack.

    Parameters
    ----------
    stack : np.ndarray
        Raw frame stack of shape (N, H, W), typically uint16 from the ORCA.
    out_dir, filename : str
        Destination directory and filename base.
    roi : dict or None
        Optional crop bounds, matching the processed region so the raw and
        processed data cover the same field of view.
    plane : int or None
        Z-plane index. When given, it is appended as ``_zNNN`` so a Z-stack
        writes one file per plane; when ``None`` (single-Z acquire) no plane
        suffix is added.

    Returns
    -------
    str
        The path of the file written.
    """
    frames = _prepare_raw_frames(stack, roi)
    suffix = "" if plane is None else f"_z{plane:03d}"
    path = os.path.join(out_dir, f"{filename}_raw_stack{suffix}.tif")
    _write_multipage_tiff(path, frames)
    return path


def save_volume_tiff(volume, out_dir, filename, kind):
    """Save a (Z, H, W) image volume as a multi-page TIFF (a depth stack).

    Used for Z-stacks: the pages are focal planes (third axis = z), so the file
    opens in ImageJ/Fiji as a scrollable 3D volume. A float32 volume is written
    as a 32-bit TIFF, which keeps intensities directly comparable across planes
    (no per-slice normalization).

    Parameters
    ----------
    volume : np.ndarray
        Image volume of shape (Z, H, W).
    out_dir, filename : str
        Destination directory and filename base (written as
        ``<filename>_<kind>.tif``).
    kind : str
        Short descriptor / filename suffix, e.g. ``"zstack_dsi"``.

    Returns
    -------
    str
        The path of the file written.
    """
    arr = np.asarray(volume)
    if arr.dtype not in (np.uint8, np.uint16, np.float32):
        arr = arr.astype(np.float32)
    path = os.path.join(out_dir, f"{filename}_{kind}.tif")
    _write_multipage_tiff(path, arr)
    return path


def _gaussian(z, amp, mu, sigma, offset):
    """Gaussian peak on a constant background, used for the axial-profile fit."""
    return offset + amp * np.exp(-((z - mu) ** 2) / (2.0 * sigma ** 2))


def _fit_axial_gaussian(z, y):
    """Fit ``y(z)`` with a Gaussian-on-offset; return (fwhm, popt) or (None, None).

    ``fwhm = 2*sqrt(2*ln2)*|sigma|`` in the same units as ``z`` (µm). Needs at
    least 4 points and SciPy; any failure (too few points, no SciPy, fit did not
    converge) degrades to (None, None) so the caller can still save the raw data.
    """
    if z.size < 4:
        return None, None
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None, None

    # Initial guess: peak above a baseline, centred at the brightest plane, width
    # a quarter of the scanned range.
    amp0 = float(y.max() - y.min()) or 1.0
    mu0 = float(z[np.argmax(y)])
    sigma0 = max((float(z.max() - z.min())) / 4.0, 1e-6)
    offset0 = float(y.min())
    try:
        popt, _ = curve_fit(_gaussian, z, y, p0=[amp0, mu0, sigma0, offset0], maxfev=10000)
    except Exception:
        return None, None

    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(popt[2])
    return fwhm, popt


def save_axial_sectioning_plot(z_positions, intensities, out_dir, filename, kind):
    """Save the axial-sectioning profile of a Z-stack (paper Fig. 3a) + its data.

    For each focal plane the mean intensity of the optically-sectioned image is
    plotted against axial position z; a Gaussian is fitted and its FWHM is the
    axial sectioning (the axial extent of the detection PSF convolved with the
    sample). Mirrors Benachir et al., "Event-based DSI", Fig. 3a.

    The underlying data is **always** written as ``<name>_axial_profile_<kind>.csv``
    (z, mean intensity, peak-normalized intensity, and the fit parameters). The
    figure ``<name>_axial_profile_<kind>.png`` is written only if matplotlib is
    installed; without it the data + fit are still saved.

    Parameters
    ----------
    z_positions : sequence of float
        Axial position (µm) of each plane, aligned with ``intensities``.
    intensities : sequence of float
        Mean pixel value of each plane's sectioned image (DSI std image for the
        ORCA, accumulated event image for the EVK4).
    out_dir, filename : str
        Destination directory and filename base.
    kind : str
        Short descriptor / filename infix, e.g. ``"dsi"`` or ``"event"``.

    Returns
    -------
    (fwhm, csv_path, png_path) : (float | None, str, str | None)
        ``fwhm`` in µm (None if the fit failed), the CSV path, and the PNG path
        (None if matplotlib was unavailable).
    """
    z = np.asarray(z_positions, dtype=float)
    inten = np.asarray(intensities, dtype=float)

    # Sort by axial position so the profile and fit are monotone in z.
    order = np.argsort(z)
    z, inten = z[order], inten[order]

    peak = float(inten.max()) if inten.size and inten.max() > 0 else 1.0
    inten_norm = inten / peak

    fwhm, popt = _fit_axial_gaussian(z, inten_norm)

    # --- data (always) -----------------------------------------------------
    csv_path = os.path.join(out_dir, f"{filename}_axial_profile_{kind}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("z_position_um,mean_intensity,normalized_intensity\n")
        for zi, raw, nrm in zip(z, inten, inten_norm):
            f.write(f"{zi:.6f},{raw:.6f},{nrm:.6f}\n")
        f.write("\n# Gaussian fit: offset + amp*exp(-(z-mu)^2 / (2*sigma^2))\n")
        if popt is not None:
            amp, mu, sigma, offset = popt
            f.write(f"# amp={amp:.6f}, mu_um={mu:.6f}, sigma_um={abs(sigma):.6f}, offset={offset:.6f}\n")
            f.write(f"# fwhm_um={fwhm:.6f}\n")
        else:
            f.write("# fit_failed=True (need >=4 planes, SciPy, and a converging fit)\n")

    # --- figure (best effort) ---------------------------------------------
    png_path = _save_axial_figure(z, inten_norm, popt, fwhm, out_dir, filename, kind)
    return fwhm, csv_path, png_path


def _save_axial_figure(z, y_norm, popt, fwhm, out_dir, filename, kind):
    """Render the axial-profile figure to PNG. Returns the path, or None if
    matplotlib is not installed (the data CSV is saved regardless)."""
    try:
        # Object-oriented Agg API (no pyplot global state) so it is safe to call
        # from the Z-stack worker thread and never opens a window.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        return None

    # The data CSV is already saved by the caller, so a rendering failure here
    # must never break the acquisition: treat the PNG as purely best-effort.
    try:
        fig = Figure(figsize=(6, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot(z, y_norm, "o", color="#1f77b4", markersize=4, label="Experimental data")

        if popt is not None:
            z_fit = np.linspace(float(z.min()), float(z.max()), 400)
            ax.plot(z_fit, _gaussian(z_fit, *popt), "-", color="#2ca02c", label="Gaussian fit")
            amp, mu, sigma, offset = popt
            half = offset + amp / 2.0
            ax.annotate(
                "", xy=(mu - fwhm / 2.0, half), xytext=(mu + fwhm / 2.0, half),
                arrowprops=dict(arrowstyle="<->", color="red"),
            )
            ax.text(mu, half, f"  FWHM = {fwhm:.3f} µm", color="red", va="bottom", ha="center")

        ax.set_xlabel("Axial position (µm)")
        ax.set_ylabel("Average intensity (normalised)")
        ax.set_title(f"Axial sectioning — {kind}")
        ax.legend()
        fig.tight_layout()

        png_path = os.path.join(out_dir, f"{filename}_axial_profile_{kind}.png")
        fig.savefig(png_path, dpi=150)
        return png_path
    except Exception:
        return None


def _fit_axial_line(z, y):
    """Fit ``y(z)`` with a straight line; return (slope, intercept) or (None, None).

    Unlike the optically-sectioned image, the widefield-equivalent *average* image
    has no depth discrimination: it collects out-of-focus light at every plane, so
    its mean intensity stays essentially flat across the stack. A straight line
    (not a Gaussian peak) is therefore the right model, and a near-zero slope is
    the signature of that absence of sectioning. Needs at least 2 points; any
    failure degrades to (None, None) so the caller can still save the raw data.
    """
    if z.size < 2:
        return None, None
    try:
        slope, intercept = np.polyfit(z, y, 1)
    except Exception:
        return None, None
    return float(slope), float(intercept)


def save_axial_average_plot(z_positions, intensities, out_dir, filename):
    """Save the axial profile of the *average* (widefield) image + its data.

    Companion to :func:`save_axial_sectioning_plot`. Where the sectioned image
    peaks at the focal plane (Gaussian, with a meaningful FWHM), the conventional
    widefield/average image shows **no** optical sectioning: its mean intensity is
    essentially constant across z. We model it with a straight line to make that
    contrast explicit — the (near-flat) slope quantifies the lack of sectioning.

    The data is **always** written as ``<name>_axial_average.csv`` (z, mean
    intensity, peak-normalized intensity, and the line-fit parameters). The figure
    ``<name>_axial_average.png`` is written only if matplotlib is installed.

    Parameters
    ----------
    z_positions : sequence of float
        Axial position (µm) of each plane, aligned with ``intensities``.
    intensities : sequence of float
        Mean pixel value of each plane's average (widefield) image.
    out_dir, filename : str
        Destination directory and filename base.

    Returns
    -------
    (slope, csv_path, png_path) : (float | None, str, str | None)
        Fitted slope (per µm; None if the fit failed), the CSV path, and the PNG
        path (None if matplotlib was unavailable).
    """
    z = np.asarray(z_positions, dtype=float)
    inten = np.asarray(intensities, dtype=float)

    # Sort by axial position so the profile and fit are monotone in z.
    order = np.argsort(z)
    z, inten = z[order], inten[order]

    peak = float(inten.max()) if inten.size and inten.max() > 0 else 1.0
    inten_norm = inten / peak

    slope, intercept = _fit_axial_line(z, inten_norm)

    # --- data (always) -----------------------------------------------------
    csv_path = os.path.join(out_dir, f"{filename}_axial_average.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("z_position_um,mean_intensity,normalized_intensity\n")
        for zi, raw, nrm in zip(z, inten, inten_norm):
            f.write(f"{zi:.6f},{raw:.6f},{nrm:.6f}\n")
        f.write("\n# Linear fit: intercept + slope*z\n")
        if slope is not None:
            f.write(f"# slope_per_um={slope:.6f}, intercept={intercept:.6f}\n")
        else:
            f.write("# fit_failed=True (need >=2 planes)\n")

    # --- figure (best effort) ---------------------------------------------
    png_path = _save_average_figure(z, inten_norm, slope, intercept, out_dir, filename)
    return slope, csv_path, png_path


def _save_average_figure(z, y_norm, slope, intercept, out_dir, filename):
    """Render the axial *average* profile to PNG with its straight-line fit.

    Returns the path, or None if matplotlib is not installed (the data CSV is
    saved regardless). Mirrors :func:`_save_axial_figure` but overlays a line
    instead of a Gaussian.
    """
    try:
        # Object-oriented Agg API (no pyplot global state) so it is safe to call
        # from the Z-stack worker thread and never opens a window.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        return None

    # The data CSV is already saved by the caller, so a rendering failure here
    # must never break the acquisition: treat the PNG as purely best-effort.
    try:
        fig = Figure(figsize=(6, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot(z, y_norm, "o", color="#1f77b4", markersize=4, label="Experimental data")

        if slope is not None:
            z_fit = np.linspace(float(z.min()), float(z.max()), 400)
            ax.plot(z_fit, intercept + slope * z_fit, "-", color="#2ca02c",
                    label=f"Linear fit (slope = {slope:.3g} /µm)")

        ax.set_xlabel("Axial position (µm)")
        ax.set_ylabel("Average intensity (normalised)")
        ax.set_title("Axial average — widefield (no sectioning)")
        ax.legend()
        fig.tight_layout()

        png_path = os.path.join(out_dir, f"{filename}_axial_average.png")
        fig.savefig(png_path, dpi=150)
        return png_path
    except Exception:
        return None


def save_parameter_log(out_dir, filename, sections):
    """Write a human-readable ``.txt`` record of all acquisition parameters.

    Good laboratory practice: every acquisition is accompanied by a log of the
    full instrument state so it can be reviewed or reproduced later.

    Parameters
    ----------
    out_dir, filename : str
        Destination directory and filename base (the log is written as
        ``<filename>_parameters.txt``).
    sections : dict
        Ordered mapping of ``{section title: {parameter: value}}``.

    Returns
    -------
    str
        The path of the file written.
    """
    lines = []
    for title, params in sections.items():
        lines.append(f"[{title}]")
        for key, value in params.items():
            lines.append(f"{key} = {value}")
        lines.append("")

    path = os.path.join(out_dir, f"{filename}_parameters.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Z-stack rebuild — reassemble the depth volumes from the per-plane raw TIFFs
# ---------------------------------------------------------------------------
# Every ORCA Z-stack plane is archived as its own ``<name>_raw_stack_zNNN.tif``.
# That makes the final products (the DSI / average depth volumes and the axial
# profiles) fully reconstructible from disk — so an acquisition that stopped part
# way (or whose summary save was interrupted) can be finished after the fact, and
# a resumed run can stitch its freshly captured tail onto the planes already saved.
def raw_stack_plane_path(raw_dir, filename, plane):
    """Path of the per-plane raw-stack TIFF for a given Z-stack plane index."""
    return os.path.join(raw_dir, f"{filename}_raw_stack_z{plane:03d}.tif")


def read_multipage_tiff(path):
    """Read a multi-page TIFF into an ``(N, H, W)`` array at its native bit depth.

    Prefers ``tifffile`` (which understands the ImageJ contiguous-stack layout the
    writer produces); falls back to OpenCV's ``imreadmulti``. Raises ``OSError`` on
    a missing / truncated / unreadable file so callers can treat that plane as
    absent. A single-page file is returned with a leading length-1 frame axis.
    """
    try:
        import tifffile
        arr = tifffile.imread(path)
    except ImportError:
        ok, frames = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
        if not ok or not frames:
            raise OSError(f"could not read TIFF: {path}")
        arr = np.asarray(frames)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr


def _tiff_num_frames(path):
    """Number of frames in a multi-page TIFF, read from its headers (no pixels).

    Returns 0 for a missing, truncated, or otherwise unparseable file — e.g. a
    plane interrupted mid-write — so it is naturally classified as incomplete.
    """
    if not os.path.exists(path):
        return 0
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            shape = tf.series[0].shape
        return int(shape[0]) if len(shape) >= 3 else 1
    except Exception:
        # No tifffile, or a header we can't parse: fall back to a coarse "does it
        # look like more than a stub?" size test so an 8-byte truncation still
        # reads as incomplete while a real stack reads as present (>=1 frame).
        try:
            return 1 if os.path.getsize(path) > 1024 else 0
        except OSError:
            return 0


def find_complete_planes(raw_dir, filename, steps, expected_frames=None):
    """Classify plane indices ``0..steps-1`` by whether their raw TIFF is complete.

    A plane is *complete* when its ``<name>_raw_stack_zNNN.tif`` exists and (when
    ``expected_frames`` is given) holds at least that many frames. A truncated file
    — such as the plane being written when an acquisition crashed — counts as
    missing so a resume re-captures it.

    Returns ``(complete, missing)``: two sorted lists of plane indices.
    """
    complete, missing = [], []
    for k in range(steps):
        n = _tiff_num_frames(raw_stack_plane_path(raw_dir, filename, k))
        if n > 0 and (expected_frames is None or n >= expected_frames):
            complete.append(k)
        else:
            missing.append(k)
    return complete, missing


def event_raw_plane_path(raw_dir, filename, plane):
    """Path of the per-plane raw event stream (``.raw``) for a Z-stack plane index."""
    return os.path.join(raw_dir, f"{filename}_events_z{plane:03d}.raw")


def find_complete_event_planes(raw_dir, filename, steps):
    """Classify EVK4 plane indices ``0..steps-1`` by whether their capture finished.

    A plane is *complete* when both its raw event stream
    ``<name>_events_zNNN.raw`` and its decoded companion
    ``<name>_events_zNNN_xytp.mat`` exist. The ``.mat`` is written only after a
    plane has been fully recorded and reconstructed, so its presence marks a clean
    finish; a plane whose recording was interrupted (raw present but no ``.mat``, or
    nothing at all) is reported missing so a resume re-captures it.

    This mirrors :func:`find_complete_planes` (the ORCA equivalent) but keys off the
    event-stream file pair instead of a multi-page TIFF's page count.

    Returns ``(complete, missing)``: two sorted lists of plane indices.
    """
    complete, missing = [], []
    for k in range(steps):
        raw = event_raw_plane_path(raw_dir, filename, k)
        mat = os.path.join(raw_dir, f"{filename}_events_z{k:03d}_xytp.mat")
        if os.path.exists(raw) and os.path.exists(mat):
            complete.append(k)
        else:
            missing.append(k)
    return complete, missing


def rebuild_zstack_from_raw(raw_dir, out_dir, filename, z_positions,
                            expected_frames=None, save_average=True,
                            metadata=None, status=None):
    """Reassemble the ORCA Z-stack outputs from the per-plane raw speckle TIFFs.

    Reads every present ``<name>_raw_stack_zNNN.tif`` in plane order, computes the
    average (widefield) and standard-deviation (DSI) image for each, and writes the
    same products a completed run would: the DSI and average depth volumes (3D
    TIFF), the axial-sectioning profile (DSI) and axial-average (widefield)
    CSV/PNG, and — when ``metadata`` is given — the parameter log. Existing summary
    files (e.g. a partial ``_zstack_dsi.tif``) are overwritten with the full set.

    The raw TIFFs are already ROI-cropped on disk, so no further crop is applied.
    Planes are loaded one at a time and released before the next, so peak memory is
    one raw stack plus the accumulating (small) sectioned volumes.

    Parameters
    ----------
    raw_dir : str
        Folder holding the per-plane raw-stack TIFFs (the ``raw_files`` subfolder).
    out_dir : str
        Acquisition folder to write the assembled volumes / profiles into.
    filename : str
        Filename base shared by every file of the acquisition.
    z_positions : sequence of float
        Nominal axial position (µm) of every plane index ``0..len-1`` (from the
        scan geometry). Only planes actually present on disk are used; ``len``
        defines the plane range that is scanned.
    expected_frames : int or None
        Frames per complete plane; a plane with fewer is skipped as incomplete.
    save_average : bool
        Also write the average (widefield) volume + axial-average profile.
    metadata : dict or None
        Parameter-log sections; when given, the log is (re)written.
    status : callable or None
        Optional progress callback invoked with human-readable strings.

    Returns
    -------
    (n_planes, missing) : (int, list[int])
        Number of planes assembled and the sorted list of plane indices skipped
        because their raw file was missing or incomplete.
    """
    def report(msg):
        if status is not None:
            status(msg)

    steps = len(z_positions)
    complete, missing = find_complete_planes(raw_dir, filename, steps, expected_frames)
    if not complete:
        raise RuntimeError(f"No complete per-plane raw stacks found in {raw_dir}")

    std_volume, avg_volume, z_kept = [], [], []
    for idx, k in enumerate(complete, 1):
        report(f"Rebuilding plane {k + 1}/{steps} ({idx}/{len(complete)} present)...")
        stack = read_multipage_tiff(raw_stack_plane_path(raw_dir, filename, k))
        avg_img, std_img = compute_dsi_images(stack, None)
        std_volume.append(std_img)
        if save_average:
            avg_volume.append(avg_img)
        z_kept.append(float(z_positions[k]))
        del stack  # release the ~GB raw stack before loading the next plane

    report(f"Writing DSI depth volume ({len(std_volume)} planes)...")
    save_volume_tiff(np.array(std_volume, dtype=np.float32), out_dir, filename, "zstack_dsi")
    if save_average and avg_volume:
        save_volume_tiff(np.array(avg_volume, dtype=np.float32), out_dir, filename, "zstack_average")

    # Axial-sectioning profile (DSI) + companion widefield-average profile.
    std_intensities = [float(np.mean(img)) for img in std_volume]
    save_axial_sectioning_plot(z_kept, std_intensities, out_dir, filename, "dsi")
    if save_average and avg_volume:
        avg_intensities = [float(np.mean(img)) for img in avg_volume]
        save_axial_average_plot(z_kept, avg_intensities, out_dir, filename)

    if metadata:
        meta = dict(metadata)
        meta["Z-Stack planes"] = {
            "camera": "orca",
            "num_planes_saved": len(std_volume),
            "missing_planes": ", ".join(str(m) for m in missing) or "none",
            "z_positions": ", ".join(f"{z:.4f}" for z in z_kept),
        }
        save_parameter_log(out_dir, filename, meta)

    report(f"Rebuilt {len(std_volume)} planes"
           + (f"; {len(missing)} still missing: {missing}" if missing else " (complete)"))
    return len(std_volume), missing


def scale_16bit_image(data):
    """Scale a 16-bit camera frame to an 8-bit display image.

    Uses a *fixed* linear map from the full 16-bit range to 8 bits (a plain
    8-bit down-shift), so the displayed brightness tracks the real signal level:
    a shorter exposure looks dimmer, as it does on the vendor software.

    The previous version auto-stretched every frame by its own maximum
    (``imul = 65535 // frame_max``). That inverted the relationship — a dimmer
    frame was scaled by a *larger* multiplier, so reducing the exposure made the
    live preview *brighter*. It was a display artifact only: the raw stack and
    the DSI statistics always run on the untouched camera data, so acquisition
    was never affected.
    """
    if data.dtype == np.uint16:
        return (data >> 8).astype(np.uint8)
    return (data / 256).astype(np.uint8)


def downscale_for_display(image, max_edge):
    """Shrink a display image so its longer side is at most ``max_edge`` px.

    Returns the image unchanged when it already fits (so the EVK4's 1280-px frame
    and any cropped ROI are untouched). Repainting a full 2304×2304 ORCA frame as
    a pixmap on the GUI thread is what caps the live preview well below the camera
    rate; the on-screen display area is only ~1–1.5k px wide, so a frame shrunk to
    ``max_edge`` looks identical but builds and repaints several times faster.

    Display-only: never call this on data destined for saving or DSI analysis —
    it is purely for the live preview. ``INTER_AREA`` is the right filter for
    downscaling (clean, no aliasing). Works for grayscale (H, W) and colour
    (H, W, C) frames alike, so it serves both the ORCA and EVK4 previews.
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return image
    scale = max_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def autocontrast_8bit(data, low_pct=0.5, high_pct=99.5):
    """Percentile contrast-stretch a camera frame to 8-bit for display.

    This is the *display-only* equivalent of the "correct contrast automatically"
    button in HCImage Live: the pixel range between the ``low_pct`` and ``high_pct``
    intensity percentiles is stretched across the full 0–255 range, so faint
    signal on a dim sensor becomes visible instead of collapsing into the bottom
    few codes of a fixed 16→8-bit down-shift (:func:`scale_16bit_image`). Clipping
    a small fraction at each end (rather than min–max normalising) keeps a single
    hot pixel or dead pixel from flattening the whole image.

    It is used for the live preview / acquisition preview only — the raw stack and
    all DSI statistics still run on the untouched camera data, so nothing about the
    saved science changes.

    The percentiles are computed on a strided subsample of the frame (every 4th
    pixel in each axis) so this stays cheap enough to run on every live frame at
    full sensor without capping the display rate.
    """
    arr = np.asarray(data)
    sample = arr[::4, ::4].astype(np.float32, copy=False)
    lo, hi = np.percentile(sample, [low_pct, high_pct])
    if hi <= lo:  # flat / degenerate frame — fall back to the true min/max
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Prophesee EVK4 — event accumulation & 2D image post-processing
# ---------------------------------------------------------------------------
def accumulate_event_frame(event_chunks, width, height):
    """Accumulate a 2D event-count image from an iterable of event chunks.

    Each chunk is a structured array exposing ``'x'`` and ``'y'`` fields (as
    produced by a Metavision ``EventsIterator``). This function is SDK-agnostic:
    it only relies on NumPy fancy-indexing, so the iterator is injected by the
    hardware layer.
    """
    final_image = np.zeros((height, width), dtype=np.float32)
    for evs in event_chunks:
        np.add.at(final_image, (evs["y"], evs["x"]), 1)
    return final_image


def save_event_stream(event_chunks, out_dir, filename):
    """Persist the decoded event stream (x, y, p, t) as a compressed ``.mat``.

    The Prophesee ``.raw`` file is the camera's *encoded* record; this writes the
    explicit per-event list for downstream analysis: pixel coordinates ``x``/``y``
    (uint16), polarity ``p`` (int8, 0/1) and timestamp ``t`` (int64, microseconds).
    Each chunk is a structured array exposing the ``'x'``, ``'y'``, ``'p'`` and
    ``'t'`` fields (as produced by a Metavision ``EventsIterator``); this function
    is SDK-agnostic and only concatenates NumPy arrays, so the iterator is injected
    by the hardware layer (and reused by the offline backfill tool).

    Written as ``<filename>_xytp.mat``, MATLAB-native and gzip-compressed so the
    decoded list stays comparable in size to the raw log rather than 4x larger.
    Each column keeps its native dtype to stay compact. An event-free stream still
    writes an (empty-array) file, so every ``.raw`` has a stream counterpart.

    Returns the path of the file written.
    """
    xs, ys, ps, ts = [], [], [], []
    for evs in event_chunks:
        if len(evs):
            xs.append(evs["x"].copy())
            ys.append(evs["y"].copy())
            ps.append(evs["p"].copy())
            ts.append(evs["t"].copy())

    if xs:
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        p = np.concatenate(ps).astype(np.int8)
        t = np.concatenate(ts)
    else:
        x = np.empty(0, dtype=np.uint16)
        y = np.empty(0, dtype=np.uint16)
        p = np.empty(0, dtype=np.int8)
        t = np.empty(0, dtype=np.int64)

    path = os.path.join(out_dir, f"{filename}_xytp.mat")
    scipy.io.savemat(path, {"x": x, "y": y, "p": p, "t": t}, do_compression=True)
    return path


# Four-digit ASCII lookup table ("0000".."9999") used by ``_ascii_digits`` to
# convert four decimal places per division instead of one. Built once: 40 kB.
_CSV_DIGIT_LUT = np.frombuffer(
    "".join(f"{i:04d}" for i in range(10000)).encode("ascii"),
    dtype=np.uint8).reshape(10000, 4)


def _ascii_digits(values, width):
    """Render non-negative ints as an ``(N, width)`` uint8 array of ASCII digits,
    zero-padded on the left (``5`` at width 3 -> ``b"005"``).

    Vectorised over the whole column: four digits are peeled per ``divmod`` via
    ``_CSV_DIGIT_LUT``, with any leftover places done one at a time. This is the
    hot path of CSV serialisation and, unlike Python string formatting, it
    releases the GIL, which is what lets several chunks format in parallel.
    """
    rem = values.astype(np.int64, copy=False)
    out = np.empty((rem.size, width), dtype=np.uint8)
    j = width
    while j >= 4:
        rem, low = np.divmod(rem, 10000)
        out[:, j - 4:j] = _CSV_DIGIT_LUT[low]
        j -= 4
    while j > 0:
        rem, d = np.divmod(rem, 10)
        out[:, j - 1] = d + 48
        j -= 1
    return out


def _decimal_width(values):
    """Number of decimal places needed for the largest value in ``values``."""
    if not values.size:
        return 1
    return max(1, len(str(int(values.max()))))


def event_csv_rows(cols):
    """Serialise one event chunk to CSV row bytes (``x,y,p,t\\n`` per event).

    ``cols`` is the ``(x, y, p, t)`` tuple of equal-length NumPy arrays. The
    output is byte-identical to ``"%d,%d,%d,%d\\n" % ...`` per row — the fields
    are *not* zero-padded, despite the fixed-width intermediate: each column is
    rendered at the width its largest value needs, then the leading zeros are
    masked out in one pass.

    Falls back to plain Python formatting if any value is negative, since the
    digit rendering above assumes non-negative integers (Metavision CD events
    are always non-negative, so this is a guard, not the expected path).
    """
    if any(col.size and int(col.min()) < 0 for col in cols):
        return ("\n".join(map("%d,%d,%d,%d".__mod__, zip(
            *(c.tolist() for c in cols)))) + "\n").encode("ascii")

    n = cols[0].size
    if not n:
        return b""
    parts, masks = [], []
    comma = np.full((n, 1), ord(","), np.uint8)
    keep_sep = np.ones((n, 1), bool)
    for col in cols:
        digits = _ascii_digits(col, _decimal_width(col))
        # Drop leading zeros: keep from the first non-'0' onwards, but always
        # keep the last place so a value of 0 renders as "0" rather than "".
        significant = np.cumsum(digits != 48, axis=1) > 0
        significant[:, -1] = True
        parts.append(digits); masks.append(significant)
        parts.append(comma); masks.append(keep_sep)
    parts[-1] = np.full((n, 1), ord("\n"), np.uint8)  # trailing comma -> newline
    return np.hstack(parts)[np.hstack(masks)].tobytes()


class EventCsvWriter:
    """Stream decoded events to ``<filename>_xytp.csv`` while they are acquired.

    This is the no-``.raw`` acquisition path (``EVK4_SAVE_FORMAT_CSV``). Unlike
    ``save_event_stream``, which decodes a *complete* raw log after the fact,
    this consumes the live iterator: there is no authoritative file to fall back
    on, so every event that the host fails to keep up with is lost for good.
    Throughput is therefore a data-integrity property here, not a convenience,
    and the class is built around it.

    Nothing but a memcpy happens on the acquisition thread. ``submit()`` copies
    the four columns out of the SDK's chunk (which it must do anyway — the SDK
    reuses that buffer) and hands them to a pool of formatter threads; a single
    writer thread then writes the finished blocks **in submission order** and
    accumulates the event image from them.

    The formatter pool is the point of the design. Serialisation is CPU-bound,
    and Python's ``%``-formatting holds the GIL, so it cannot be parallelised;
    ``event_csv_rows`` does the same work in NumPy, which releases it. Measured
    on the development machine: ~2.1 Mev/s for the Python formatter, ~3.5 Mev/s
    for the NumPy one single-threaded, ~7.3 Mev/s across four threads.

    Because the image is accumulated by the writer from exactly the blocks it
    writes, ``image()`` and the CSV can never disagree — including when chunks
    are dropped.

    The pending queue is bounded (``EVK4_CSV_QUEUE_CHUNKS``). If the formatters
    fall behind, ``submit()`` **drops** the chunk rather than blocking: back-
    pressuring the SDK iterator would stall event delivery and corrupt the
    acquisition's timing. Dropped events are counted in ``events_dropped``, and
    anything still queued at ``close()`` is counted too, so a lossy run reports
    itself instead of passing silently — the one guarantee the ``.raw`` path
    gives for free and this path cannot.

    A writer or formatter failure is captured in ``error`` rather than raised,
    so a disk problem cannot take down an unattended run mid-stack.

    ``submit()`` assumes a **single producer thread** (the acquisition loop).

    Use as a context manager, or ``start()`` / ``submit()`` / ``close()``::

        with EventCsvWriter(path, width, height) as w:
            for evs in iterator:
                w.submit(evs)
        img = w.image()
    """

    HEADER = b"x,y,p,t\n"

    def __init__(self, path, width, height, queue_chunks=EVK4_CSV_QUEUE_CHUNKS,
                 workers=EVK4_CSV_WORKERS):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.events_written = 0
        self.events_dropped = 0
        self.error = None
        self._counts = np.zeros(self.width * self.height, dtype=np.int64)
        # Holds (future, n_events) in submission order, which is also the order
        # the writer consumes them — so the CSV stays chronological even though
        # the chunks are formatted concurrently and may finish out of order.
        self._pending = queue.Queue(maxsize=max(1, int(queue_chunks)))
        self._stop = threading.Event()
        self._thread = None
        self._pool = None
        self._workers = max(1, int(workers))

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        """Spin up the formatter pool and the writer thread."""
        self._pool = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="evk4-csv-fmt")
        self._thread = threading.Thread(
            target=self._run, name="evk4-csv-writer", daemon=True)
        self._thread.start()
        return self

    def submit(self, evs):
        """Queue one event chunk. Returns False if it was dropped (queue full).

        Never blocks and never raises: an empty chunk, a full queue and a dead
        writer are all handled here, because the caller is the timing-critical
        acquisition loop.
        """
        n = len(evs)
        if not n:
            return True
        # Checking fullness before submitting is safe only because there is a
        # single producer; it keeps a rejected chunk from ever reaching the pool.
        if self._pool is None or self._pending.full():
            self.events_dropped += n
            return False
        # The SDK reuses its chunk buffer for the next batch, so every column
        # must be copied before it leaves this thread.
        cols = (evs["x"].copy(), evs["y"].copy(),
                evs["p"].copy(), evs["t"].copy())
        try:
            future = self._pool.submit(self._format, cols, self.width)
        except RuntimeError:      # pool already shut down
            self.events_dropped += n
            return False
        self._pending.put_nowait((future, n))
        return True

    def close(self, timeout=120.0):
        """Drain the queue, stop the threads and close the file.

        Returns True if everything was written cleanly. A timeout, a writer
        error or stranded chunks return False (see ``error``); whatever reached
        the file is still valid, so the caller reports rather than discards.
        """
        if self._thread is None:
            return self.error is None
        self._stop.set()
        self._thread.join(timeout=timeout)
        timed_out = self._thread.is_alive()
        self._thread = None
        # Anything still queued was accepted by submit() but never reached the
        # file — because the writer died or the join timed out. Count it as
        # dropped: an event that was taken in and then lost must not be able to
        # disappear from the totals, or a failed run would report full capture.
        stranded = self._drain()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if stranded and self.error is None:
            self.error = RuntimeError(
                f"CSV writer stopped early; {stranded:,} queued events were not written")
        if timed_out and self.error is None:
            self.error = RuntimeError(
                f"CSV writer did not finish within {timeout:.0f} s")
        return self.error is None

    def _drain(self):
        """Discard any unwritten chunks, adding them to the dropped count."""
        stranded = 0
        while True:
            try:
                future, n = self._pending.get_nowait()
            except queue.Empty:
                break
            future.cancel()
            stranded += n
        self.events_dropped += stranded
        return stranded

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.close()
        return False  # never swallow an acquisition-loop exception

    # -- results -----------------------------------------------------------
    def image(self):
        """The 2D event-count image for the events actually written."""
        return self._counts.reshape(self.height, self.width).astype(np.float32)

    # -- worker threads ----------------------------------------------------
    @staticmethod
    def _format(cols, width):
        """Formatter-pool task: CSV bytes plus the flat pixel indices.

        The indices ride along so the writer can accumulate the image without
        holding on to the full chunk, and without repeating the multiply.
        """
        x, y = cols[0], cols[1]
        idx = y.astype(np.int64) * width + x.astype(np.int64)
        return event_csv_rows(cols), idx

    def _run(self):
        try:
            # Binary + large buffer: one write() per chunk, not per row.
            with open(self.path, "wb", buffering=1 << 20) as fh:
                fh.write(self.HEADER)
                while True:
                    try:
                        future, _n = self._pending.get(timeout=0.1)
                    except queue.Empty:
                        # Only stop once the queue has actually drained, so a
                        # close() during a burst still flushes what was queued.
                        if self._stop.is_set():
                            break
                        continue
                    data, idx = future.result()
                    fh.write(data)
                    self._counts += np.bincount(idx, minlength=self._counts.size)
                    self.events_written += idx.size
        except Exception as exc:  # noqa: BLE001 — reported, never raised into the loop
            self.error = exc


def filter_crazy_pixels(image, percentile=EVK4_CRAZY_PIXEL_PERCENTILE):
    """Zero out hot ('crazy') pixels above the given intensity percentile.

    Modifies and returns ``image`` (in place), matching original behaviour.
    """
    threshold = np.percentile(image, percentile)
    image[image > threshold] = 0
    return image


def apply_smoothing(image):
    """Apply a 5x5 Gaussian spatial smoothing kernel."""
    return cv2.GaussianBlur(image, (5, 5), 0)


def save_mat_tif(image, out_dir, filename):
    """Persist the final EVK4 image as both a MATLAB ``.mat`` and a ``.tif``.

    Returns the output directory for status reporting.
    """
    mat_path = os.path.join(out_dir, f"{filename}_final_image.mat")
    scipy.io.savemat(mat_path, {"final_image": image})

    tif_path = os.path.join(out_dir, f"{filename}_final_image.tif")
    cv2.imwrite(tif_path, image)
    return out_dir


# ---------------------------------------------------------------------------
# EVK4 -> ORCA field-of-view matching
# ---------------------------------------------------------------------------
def evk4_footprint_in_orca(affine, evk4_roi):
    """The EVK4 window's four corners in full-sensor ORCA pixels.

    Returns a 4x2 float array of (x, y) in EVK4-window order TL, TR, BR, BL.
    The footprint is *rotated* (~43 deg) in the ORCA field, so those four points
    are a quadrilateral, not an axis-aligned box — which is exactly why the
    event camera's field is impossible to guess from the ORCA image alone.

    Pure geometry: no clamping, no alignment, and it never raises for a
    footprint that falls off the sensor. Overlays want it raw like this;
    :func:`map_evk4_window_to_orca` builds the aligned camera crop on top.
    """
    A = np.asarray(affine, dtype=np.float64)
    if A.shape != (2, 3):
        raise ValueError(f"affine must be 2x3, got {A.shape}")
    x0, x1 = float(evk4_roi["x_min"]), float(evk4_roi["x_max"])
    y0, y1 = float(evk4_roi["y_min"]), float(evk4_roi["y_max"])
    pts = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]])
    return pts @ A.T


def map_evk4_window_to_orca(affine, evk4_roi, orca_sensor=(2304, 2304), align=4):
    """Map an EVK4 crop window into ORCA coordinates and derive the matching crop.

    ``affine`` is the calibrated 2x3 EVK4->ORCA map (see
    ``config.EVK4_TO_ORCA_AFFINE``): ``[orca_x, orca_y] = A @ [evk_x, evk_y, 1]``.
    Because the EVK4 footprint is *rotated* (~43 deg) in the ORCA field, the
    matching camera crop is the axis-aligned **bounding box** of the mapped
    window: it contains the entire EVK4 view plus four corner triangles the
    EVK4 does not see. The returned corners let the UI draw the true rotated
    footprint inside the crop so the user can validate the overlap.

    ``evk4_roi`` is the EVK4 window as ``{x_min, x_max, y_min, y_max}`` in
    full-sensor IMX636 pixels (the widget's native format). The bounding box is
    expanded outward to the DCAM ``align`` grid (position and size must be
    multiples of 4 for the hardware subarray) and clamped to the sensor.

    Returns ``(crop, corners, clipped)``:
      * ``crop`` — ``{x_min, x_max, y_min, y_max}`` in full-sensor unbinned
        ORCA pixels, aligned and clamped;
      * ``corners`` — 4x2 float array of the mapped EVK4 window corners
        (x, y), in EVK4-window order TL, TR, BR, BL;
      * ``clipped`` — True if the footprint ran off the ORCA sensor and the
        crop had to be cut back (the fields are then not fully shared).
    """
    corners = evk4_footprint_in_orca(affine, evk4_roi)  # 4x2 (x, y) in ORCA px

    sw, sh = int(orca_sensor[0]), int(orca_sensor[1])
    # Expand outward to the alignment grid so the whole footprint stays inside.
    bx0 = int(np.floor(corners[:, 0].min() / align) * align)
    by0 = int(np.floor(corners[:, 1].min() / align) * align)
    bx1 = int(np.ceil(corners[:, 0].max() / align) * align)
    by1 = int(np.ceil(corners[:, 1].max() / align) * align)
    clipped = bx0 < 0 or by0 < 0 or bx1 > sw or by1 > sh
    bx0, by0 = max(0, bx0), max(0, by0)
    bx1, by1 = min(sw, bx1), min(sh, by1)
    if bx1 - bx0 < align or by1 - by0 < align:
        raise ValueError(
            "The mapped EVK4 window lies (almost) entirely off the ORCA sensor — "
            "the registration is likely stale; re-calibrate it.")
    crop = {"x_min": bx0, "x_max": bx1, "y_min": by0, "y_max": by1}
    return crop, corners, clipped


# --- live EVK4 -> ORCA registration (masked-NCC, from the 2026-07-10 analysis) ---
def _blobmap(img, sigma=3.0):
    """Smoothed, robustly-normalised blob-emphasis map in [0, 1].

    The Gaussian smoothing is what makes the registration tolerant to the small
    focal-plane offset between the two camera ports: mildly defocused beads
    still produce overlapping blobs.
    """
    img = np.asarray(img, dtype=np.float64)
    sm = cv2.GaussianBlur(img, (0, 0), sigma)
    lo, hi = np.percentile(sm, [50.0, 99.5])
    return np.clip((sm - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)


def _rotate_full(img, flip, theta):
    """Rotate (optionally x-flipped) ``img`` by ``theta`` deg onto a canvas large
    enough to hold it, returning (canvas, validity mask, 2x3 original->canvas map)."""
    F = np.array([[1.0, 0, 0], [0, 1, 0]])
    if flip:
        img = np.ascontiguousarray(np.fliplr(img))
        F = np.array([[-1.0, 0, img.shape[1] - 1], [0, 1, 0]])
    h, w = img.shape
    s, c = abs(np.sin(np.deg2rad(theta))), abs(np.cos(np.deg2rad(theta)))
    bw, bh = int(np.ceil(w * c + h * s)), int(np.ceil(w * s + h * c))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    M[0, 2] += bw / 2 - w / 2
    M[1, 2] += bh / 2 - h / 2
    rot = cv2.warpAffine(img, M, (bw, bh), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones_like(img), M, (bw, bh), flags=cv2.INTER_NEAREST)
    A3 = np.vstack([M, [0, 0, 1]]) @ np.vstack([F, [0, 0, 1]])
    return rot, mask, A3[:2]


def _scaled(canvas, mask, scale):
    th, tw = int(round(canvas.shape[0] * scale)), int(round(canvas.shape[1] * scale))
    tr = cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_AREA)
    mr = (cv2.resize(mask, (tw, th), interpolation=cv2.INTER_AREA) > 0.999).astype(np.float32)
    return tr, mr


def _masked_ncc(big, tmpl, mask):
    """Masked normalised cross-correlation; returns (best score, (x, y) location).
    Score -2 marks an invalid geometry (template not smaller than the image)."""
    mv = tmpl[mask > 0]
    if mv.size == 0:
        return -2.0, (0, 0)
    t = (tmpl - mv.mean()) * mask
    b = big - big.mean()
    if t.shape[0] >= b.shape[0] or t.shape[1] >= b.shape[1]:
        return -2.0, (0, 0)
    res = np.nan_to_num(cv2.matchTemplate(b, t, cv2.TM_CCORR_NORMED, mask=mask), nan=-1)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), loc


def decompose_fov_affine(affine):
    """Extract (flip, theta_deg, scale) from a 2x3 EVK4->ORCA affine.

    ``theta`` follows the cv2.getRotationMatrix2D convention used by the
    registration search (the 2026-07-10 calibration reads theta = 317 deg,
    scale = 0.745). A negative determinant marks an x-flip.
    """
    A2 = np.asarray(affine, dtype=np.float64)[:, :2]
    flip = bool(np.linalg.det(A2) < 0)
    M = A2 @ np.array([[-1.0, 0], [0, 1.0]]) if flip else A2
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    theta = float((-np.degrees(np.arctan2(M[1, 0], M[0, 0]))) % 360.0)
    return flip, theta, scale


def register_evk4_to_orca(orca_img, evk4_img, seed_affine=None, status=None):
    """Register a live EVK4 image onto a live ORCA image; return the new affine.

    The measurement counterpart of the offline 2026-07-10 ``step2_register.py``:
    both images are reduced to smoothed blob maps and matched by masked NCC over
    flip / rotation / scale / translation, coarse-to-fine. ``orca_img`` must be a
    full-sensor ORCA frame (e.g. an average of a few frames) and ``evk4_img`` a
    full-sensor accumulated event image of the *same* (structured) sample.

    When ``seed_affine`` is given (the current calibration), the search is
    bounded around its rotation/scale — fast, and robust for a bumped/drifted
    setup. If the seeded search matches poorly (NCC < 0.35), it automatically
    widens to a full-circle search (slower) to handle a remounted/rotated
    camera. ``status`` is an optional ``str -> None`` progress callback.

    Returns ``(affine, score, params)``: the 2x3 EVK4->ORCA map (list of lists),
    its NCC score in [0, 1] (values ≳ 0.5 were reliable in the analysis), and
    ``{"flip", "theta", "scale"}``.
    """
    def say(msg):
        if status is not None:
            status(msg)

    cov_o = _blobmap(orca_img, 4.0)
    cov_e = _blobmap(evk4_img, 3.0)
    o4 = cv2.resize(cov_o, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    e4 = cv2.resize(cov_e, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    o2 = cv2.resize(cov_o, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    e2 = cv2.resize(cov_e, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    def coarse(flips, thetas, scales):
        best = None
        for flip in flips:
            for theta in thetas:
                canv, mask, _ = _rotate_full(e4, flip, theta)
                for scale in scales:
                    tr, mr = _scaled(canv, mask, scale)
                    s, _ = _masked_ncc(o4, tr, mr)
                    if best is None or s > best[0]:
                        best = (s, flip, theta, scale)
        return best

    full_flips = (False, True)
    full_thetas = np.arange(0.0, 360.0, 2.0)
    full_scales = np.arange(0.55, 0.96, 0.03)
    if seed_affine is not None:
        flip0, th0, sc0 = decompose_fov_affine(seed_affine)
        say(f"Coarse search around the current calibration (θ ≈ {th0:.0f}°)...")
        best = coarse([flip0],
                      np.arange(th0 - 15.0, th0 + 15.01, 1.5),
                      np.arange(max(0.1, sc0 - 0.06), sc0 + 0.0601, 0.015))
        if best[0] < 0.35:
            say("Weak match near the calibration — widening to a full search "
                "(the camera may have been remounted; this takes longer)...")
            best = coarse(full_flips, full_thetas, full_scales)
    else:
        say("Full coarse registration search (no calibration seed)...")
        best = coarse(full_flips, full_thetas, full_scales)
    s_c, flip, theta_c, scale_c = best

    say(f"Refining registration (coarse NCC {s_c:.2f}, θ = {theta_c:.1f}°, "
        f"scale = {scale_c:.3f})...")
    best2 = None
    for theta in np.arange(theta_c - 3.0, theta_c + 3.01, 0.25):
        canv, mask, _ = _rotate_full(e2, flip, theta)
        for scale in np.arange(scale_c - 0.03, scale_c + 0.0301, 0.005):
            tr, mr = _scaled(canv, mask, scale)
            s, _ = _masked_ncc(o2, tr, mr)
            if best2 is None or s > best2[0]:
                best2 = (s, theta, scale)
    _, theta_r, scale_r = best2

    # Translation at half res, then a cheap local full-resolution polish: the
    # full-res template is only matched inside a small window around the
    # upscaled half-res location, instead of over the whole sensor.
    canv2, mask2, _ = _rotate_full(e2, flip, theta_r)
    tr2, mr2 = _scaled(canv2, mask2, scale_r)
    s2, (x2, y2) = _masked_ncc(o2, tr2, mr2)

    canvas, mask, A_rot = _rotate_full(cov_e, flip, theta_r)
    tr, mr = _scaled(canvas, mask, scale_r)
    x0g, y0g = 2 * x2, 2 * y2
    pad = 12
    x0w, y0w = max(0, x0g - pad), max(0, y0g - pad)
    x1w = min(cov_o.shape[1], x0g + tr.shape[1] + pad)
    y1w = min(cov_o.shape[0], y0g + tr.shape[0] + pad)
    s, (dx, dy) = _masked_ncc(cov_o[y0w:y1w, x0w:x1w], tr, mr)
    if s <= -2.0:  # window degenerate near the sensor edge — keep half-res result
        s, (x0, y0) = s2, (x0g, y0g)
    else:
        x0, y0 = x0w + dx, y0w + dy

    S3 = np.array([[scale_r, 0, 0], [0, scale_r, 0], [0, 0, 1.0]])
    T3 = np.array([[1.0, 0, x0], [0, 1, y0], [0, 0, 1.0]])
    A3 = T3 @ S3 @ np.vstack([A_rot, [0, 0, 1.0]])
    say(f"Registration done: NCC = {s:.2f}, θ = {theta_r:.2f}°, scale = {scale_r:.4f}.")
    return A3[:2].tolist(), float(s), {
        "flip": bool(flip), "theta": float(theta_r), "scale": float(scale_r),
    }


def compose_registration_overlay(orca_img, evk4_img, affine):
    """Green-ORCA / magenta-EVK4 overlay of the registered pair (HxWx3 uint8).

    The EVK4 blob map is warped into the ORCA frame through ``affine`` — where
    the two cameras see the same beads, green + magenta ≈ white. Used as the
    preview-dialog background so the user validates the *measured* registration,
    not just the resulting rectangle.
    """
    g = _blobmap(orca_img, 4.0)
    m = _blobmap(evk4_img, 3.0)
    h, w = g.shape
    A = np.asarray(affine, dtype=np.float64)
    warp = cv2.warpAffine(m, A, (w, h))
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 1] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    mag = (np.clip(warp, 0, 1) * 255).astype(np.uint8)
    rgb[..., 0] = mag
    rgb[..., 2] = mag
    return rgb
