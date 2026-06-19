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


def compute_dsi_images(stack, roi=None):
    """Compute the average (widefield) and standard-deviation (DSI) images.

    This is the single-z DSI reconstruction: given a time series of raw speckle
    frames acquired at one focal plane, the optical sectioning comes from the
    per-pixel statistics *across* the stack (Ventalon & Mertz; cf. reference
    papers), not from moving the objective.

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
    images = stack.astype(np.float32)

    if roi is not None:
        # Clamp to the actual frame so a generous UI value can't slice out of bounds.
        max_y, max_x = images.shape[1], images.shape[2]
        y_min, y_max = roi["y_min"], min(roi["y_max"], max_y)
        x_min, x_max = roi["x_min"], min(roi["x_max"], max_x)
        images = images[:, y_min:y_max, x_min:x_max]

    avg_img = np.mean(images, axis=0)
    std_img = np.std(images, axis=0)
    return avg_img, std_img


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

    Shared by ``save_raw_stack_tiff`` and ``RawStackTiffWriter`` so the on-disk
    raw data is identical whether it is written one stack at a time or appended
    plane by plane. Raw data is never normalized; the camera's native bit depth
    (typically uint16) is preserved.
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


def save_raw_stack_tiff(stack, out_dir, filename, roi=None):
    """Save the raw speckle frame stack as a multi-page 16-bit TIFF.

    This is the archival *raw data*: every acquired frame at the camera's native
    bit depth, before any averaging / standard-deviation processing. The TIFF
    pages are the individual frames (third axis = frame index), so the file opens
    in ImageJ/Fiji as a scrollable stack and can be re-processed later (e.g. with
    a different sectioning estimator or the RIM algorithm).

    Parameters
    ----------
    stack : np.ndarray
        Raw frame stack of shape (N, H, W), typically uint16 from the ORCA.
    out_dir, filename : str
        Destination directory and filename base (written as
        ``raw_stack_<filename>.tif``).
    roi : dict or None
        Optional crop bounds, matching the processed region so the raw and
        processed data cover the same field of view.

    Returns
    -------
    str
        The path of the file written.
    """
    frames = _prepare_raw_frames(stack, roi)
    path = os.path.join(out_dir, f"raw_stack_{filename}.tif")
    _write_multipage_tiff(path, frames)
    return path


class RawStackTiffWriter:
    """Append raw speckle frames from many planes into a *single* multi-page TIFF.

    The Z-stack acquires a frame stack at every focal plane. Rather than emitting
    one ``raw_stack_..._zNNN.tif`` per plane, this writer concatenates every
    plane's frames (plane-major order: all of plane 0's frames, then plane 1's,
    ...) into one ``raw_stack_<filename>.tif`` — the single archival raw-data
    file requested for downstream re-processing.

    Implementation: when ``tifffile`` is available, frames are streamed to disk
    incrementally as a BigTIFF, so memory stays flat and the file can exceed the
    4 GB classic-TIFF limit. Without ``tifffile`` it falls back to buffering all
    frames in memory and writing them with a single ``cv2.imwritemulti`` call on
    ``close()`` (fine for modest stacks; install ``tifffile`` for large 3D runs).
    """

    def __init__(self, path):
        self.path = path
        self._tif = None       # tifffile.TiffWriter when streaming
        self._buffer = None     # list of frames when buffering for OpenCV
        try:
            import tifffile
            # BigTIFF + contiguous pages: a plain multi-page stack Fiji opens as a
            # scrollable volume, with no per-file size ceiling.
            self._tif = tifffile.TiffWriter(path, bigtiff=True)
        except ImportError:
            self._buffer = []

    def append(self, stack, roi=None):
        """Add one plane's frame stack (N, H, W) to the file."""
        frames = _prepare_raw_frames(stack, roi)
        if self._tif is not None:
            self._tif.write(frames, contiguous=True)
        else:
            self._buffer.extend(frames)

    def close(self):
        """Flush and finalize the file. Safe to call more than once."""
        if self._tif is not None:
            self._tif.close()
            self._tif = None
        elif self._buffer:
            cv2.imwritemulti(self.path, self._buffer)
            self._buffer = []


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

    Mirrors the original live-view scaling: stretch by an integer multiple of the
    current frame maximum, then down-shift to 8-bit.
    """
    if data.dtype == np.uint16:
        imax = np.amax(data)
        if imax > 0:
            imul = int(65535 / imax)
            data = data * imul
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
