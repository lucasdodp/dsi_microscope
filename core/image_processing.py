"""Pure data-processing routines for event-DSI optical sectioning.

NOTHING in this module may import PyQt6 or a hardware SDK. Functions take plain
arrays (or generic iterables of structured event arrays) and return arrays / write
files. This keeps the optical-sectioning math testable and side-effect free apart
from the explicit `save_*` helpers.
"""

import os

import cv2
import numpy as np
import scipy.io

from config import EVK4_CRAZY_PIXEL_PERCENTILE


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
    in small chunks, in float64, instead of materializing ``stack.astype(float32)``
    (and the deviation array ``np.std`` builds internally). Those would each be
    the full size of the stack — gigabytes for a high frame count at full sensor,
    enough to swap the machine to a standstill. Here the working set is only a few
    H×W planes, so peak memory is independent of N.

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
    chunk = _frame_chunk(h, w)

    # Pass 1 — mean. ``sum(dtype=float64)`` reduces over the frame axis without
    # casting the whole block to float first, so the only big array is the H×W
    # accumulator.
    acc = np.zeros((h, w), dtype=np.float64)
    for start in range(0, n, chunk):
        acc += images[start:start + chunk].sum(axis=0, dtype=np.float64)
    mean = acc / n

    # Pass 2 — population variance as the summed squared deviation about the mean
    # (ddof=0, matching the previous ``np.std``). Two passes avoid the
    # catastrophic cancellation a single-pass sum-of-squares can suffer.
    sq = np.zeros((h, w), dtype=np.float64)
    for start in range(0, n, chunk):
        block = images[start:start + chunk].astype(np.float64)
        block -= mean
        block *= block
        sq += block.sum(axis=0)
    var = sq / n

    avg_img = mean.astype(np.float32)
    std_img = np.sqrt(var, out=var).astype(np.float32)
    return avg_img, std_img


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
    scipy.io.savemat(os.path.join(out_dir, f"average_{filename}.mat"), {"average_image": avg_img})
    scipy.io.savemat(os.path.join(out_dir, f"dsi_{filename}.mat"), {"dsi_image": std_img})
    cv2.imwrite(os.path.join(out_dir, f"average_{filename}.tif"), normalize_to_8bit(avg_img))
    cv2.imwrite(os.path.join(out_dir, f"dsi_{filename}.tif"), normalize_to_8bit(std_img))
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
        ``<kind>_<filename>.tif``).
    kind : str
        Short descriptor / filename prefix, e.g. ``"zstack_dsi"``.

    Returns
    -------
    str
        The path of the file written.
    """
    arr = np.asarray(volume)
    if arr.dtype not in (np.uint8, np.uint16, np.float32):
        arr = arr.astype(np.float32)
    path = os.path.join(out_dir, f"{kind}_{filename}.tif")
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

    The underlying data is **always** written as ``axial_profile_<kind>_<name>.csv``
    (z, mean intensity, peak-normalized intensity, and the fit parameters). The
    figure ``axial_profile_<kind>_<name>.png`` is written only if matplotlib is
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
    csv_path = os.path.join(out_dir, f"axial_profile_{kind}_{filename}.csv")
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

        png_path = os.path.join(out_dir, f"axial_profile_{kind}_{filename}.png")
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

    The data is **always** written as ``axial_average_<name>.csv`` (z, mean
    intensity, peak-normalized intensity, and the line-fit parameters). The figure
    ``axial_average_<name>.png`` is written only if matplotlib is installed.

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
    csv_path = os.path.join(out_dir, f"axial_average_{filename}.csv")
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

        png_path = os.path.join(out_dir, f"axial_average_{filename}.png")
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
        ``parameters_<filename>.txt``).
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

    path = os.path.join(out_dir, f"parameters_{filename}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


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
    mat_path = os.path.join(out_dir, f"final_image_{filename}.mat")
    scipy.io.savemat(mat_path, {"final_image": image})

    tif_path = os.path.join(out_dir, f"final_image_{filename}.tif")
    cv2.imwrite(tif_path, image)
    return out_dir
