"""Post-processing operations for DSI / event-DSI images — pure array math.

The processing half of ``tools/image_lab.py``. Like ``core/``, this module
imports **no PyQt6 and no hardware SDK**: every function takes plain arrays and
returns arrays, so the image-quality math stays testable in isolation and can be
scripted without opening the GUI.

--------------------------------------------------------------------------
The pipeline
--------------------------------------------------------------------------
Stages run in this order; the ordering is deliberate, not incidental.

*Quantitative* (physics-correct — safe under a measurement):

  1. **Hot-pixel removal** — sensor artifacts must go first, before any
     statistic is computed from neighbourhoods. Outliers are detected against a
     *local* median (> median + k·MAD in a window) and replaced by that median,
     not zeroed: zeroing punches holes into genuinely bright structure.
  2. **Noise-floor subtraction** — a DSI std image is
     ``sqrt(Var_speckle + Var_noise)``. Removing ``Var_noise`` in the variance
     domain flattens the background to true zero, which is what lets faint
     in-focus structure survive the contrast stretch. Either from a measured
     reference (a frozen-speckle std image, AWG off) or analytically from the
     companion average image via ``Var_shot = k·(mean − offset)``.
  3. **Normalization / flat-fielding** — ``std/mean`` is the classic
     Ventalon–Mertz speckle-contrast metric: it divides out the illumination
     envelope and the fluorophore-density weighting. ``std/sqrt(mean)`` keeps the
     result shot-noise-weighted. Self flat-field (divide by a heavily blurred
     copy) removes the beam profile without a companion image.
  4. **Background subtraction** — rolling-ball-equivalent (grayscale opening)
     for residual out-of-focus haze.
  5. **Denoising** — edge-preserving options (bilateral, NLM) instead of a
     Gaussian that rounds beads off. The Anscombe variant applies a
     variance-stabilizing transform first, which is the correct order for the
     Poisson-ish statistics of event counts.
  6. **Deconvolution** — Richardson–Lucy with a Gaussian PSF derived from
     NA / λ / pixel size. Runs *after* denoising because RL amplifies exactly
     the noise the previous stages remove. NOTE: the std image is not strictly a
     linear convolution of the object, so treat this as contrast enhancement,
     not quantitative restoration.
  7. **Unsharp mask** — the cheapest perceived-sharpness gain.

*Display-only* (cosmetic — never quantitative):

  8. **Contrast stretch** — ImageJ's "Auto" B&C (reproducing
     ``ContrastAdjuster.autoAdjust()``), a percentile clip, or manual limits.
  9. **Gamma** — reveals faint structure without clipping the bright end.
 10. **CLAHE** — local histogram equalization; dramatic on uneven fluorescence.
 11. **LUT** — a perceptually-uniform colormap resolves low-contrast detail that
     grayscale hides.
 12. **Scale bar** — burnt into the exported image.

*Stack-level tools:* lateral drift correction (phase correlation, so orthogonal
views stop smearing), and projections through z — MIP, mean, std, extended
depth of field, and depth-colour-coding (hue = z, intensity = signal), which
turns a whole volume into one figure that shows off the optical sectioning.

*3-D rendering:* an isotropic-voxel working volume (``build_view_volume``) and a
shear-warp projector (``render_volume``) that turns a z-stack into a rotatable
volume with no GPU and no extra dependency — see the section comment above
``plan_view_volume`` for why it is factored that way.

--------------------------------------------------------------------------
The ImageJ auto-B&C algorithm (stage 8, "Auto")
--------------------------------------------------------------------------
  1. Build a 256-bin histogram over the slice's [min, max] data range.
  2. Zero out bins that hold more than 10% of the pixels (the dominant
     background peak) so they don't skew the result.
  3. From each end, find the first bin whose count exceeds pixelCount / 5000.
     Those bins define the display min / max.
  4. Linearly map [display_min, display_max] -> [0, 255] for viewing.
"""

from pathlib import Path

import cv2
import numpy as np
import scipy.io
import tifffile
from scipy.signal import fftconvolve

# ImageJ's default auto-threshold divisor (first press of "Auto").
AUTO_THRESHOLD = 5000

EPS = 1e-12


# ===========================================================================
# Pure processing — no Qt in this section.
# ===========================================================================

# --- contrast --------------------------------------------------------------
def imagej_auto_minmax(pixels: np.ndarray, auto_threshold: int = AUTO_THRESHOLD):
    """Return the (display_min, display_max) ImageJ's "Auto" B&C would choose."""
    flat = np.asarray(pixels).ravel()
    pmin = float(flat.min())
    pmax = float(flat.max())
    if pmax <= pmin:
        return pmin, pmax

    n_bins = 256
    bin_size = (pmax - pmin) / n_bins
    hist, _ = np.histogram(flat, bins=n_bins, range=(pmin, pmax))

    pixel_count = flat.size
    threshold = pixel_count / auto_threshold
    limit = pixel_count / 10

    # Ignore over-full bins (e.g. the big dark background peak).
    hist = hist.copy()
    hist[hist > limit] = 0

    over = np.nonzero(hist > threshold)[0]
    if over.size == 0:
        return pmin, pmax

    hmin, hmax = int(over[0]), int(over[-1])
    disp_min = pmin + hmin * bin_size
    disp_max = pmin + hmax * bin_size
    if disp_max <= disp_min:
        return pmin, pmax
    return disp_min, disp_max


def percentile_minmax(pixels: np.ndarray, low_pct: float, high_pct: float):
    """Contrast limits from intensity percentiles (robust to hot/dead pixels).

    Percentiles are computed on a strided subsample so this stays cheap on a
    full-sensor 2304x2304 frame.
    """
    arr = np.asarray(pixels)
    sample = arr[::4, ::4] if arr.size > 512 * 512 else arr
    lo, hi = np.percentile(sample, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    return float(lo), float(hi)


def apply_display_range(pixels: np.ndarray, disp_min: float, disp_max: float) -> np.ndarray:
    """Map [disp_min, disp_max] -> [0, 255] uint8, clipping outside the range."""
    if disp_max <= disp_min:
        return np.zeros(np.shape(pixels), dtype=np.uint8)
    scaled = (np.asarray(pixels, dtype=np.float32) - disp_min) * (255.0 / (disp_max - disp_min))
    return np.clip(scaled, 0, 255).astype(np.uint8)


# --- 1. hot / crazy pixels -------------------------------------------------
def _median_filter(arr, w):
    """Median filter that works for any window size on float32 data.

    ``cv2.medianBlur`` only accepts ksize 3 or 5 for float32, so a larger window
    is approximated by iterating the 5x5 filter. That is enough for the outlier
    statistics here (we need a robust local level, not an exact median).
    """
    w = int(w) | 1
    if w <= 5:
        return cv2.medianBlur(arr, w)
    out = arr
    for _ in range(max(1, w // 4)):
        out = cv2.medianBlur(out, 5)
    return out


def remove_hot_pixels(img, k=6.0, window=5, mask=None):
    """Replace local outliers with the local median (never with zero).

    A pixel is an outlier when it exceeds ``median + k * 1.4826 * MAD`` of its
    ``window x window`` neighbourhood. This is the corrected form of the blunt
    global-percentile filter (``core.filter_crazy_pixels``), which zeroes every
    pixel above a percentile and therefore blanks real bright structure too.

    ``mask`` optionally forces a set of known-bad pixels (a persistent hot-pixel
    map measured once with the illumination blocked) to be replaced as well —
    far more accurate than re-deriving them from each image.
    """
    arr = np.ascontiguousarray(img, dtype=np.float32)
    w = int(window) | 1  # force odd
    med = _median_filter(arr, w)
    dev = np.abs(arr - med)
    mad = _median_filter(dev, w)
    bad = dev > (k * 1.4826 * mad + EPS)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)
    out = arr.copy()
    out[bad] = med[bad]
    return out


def hot_pixel_mask_from_dark(dark_img, k=6.0):
    """Build a persistent hot-pixel mask from a dark / blocked-illumination image."""
    arr = np.asarray(dark_img, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    return arr > (med + k * max(mad, EPS))


# --- 2. noise floor --------------------------------------------------------
def subtract_noise_reference(std_img, noise_std_img, strength=1.0):
    """Remove a measured noise floor from a std image, in the variance domain.

    ``noise_std_img`` is the per-pixel standard deviation of a *frozen-speckle*
    stack (AWG output off, everything else identical): with the speckle static,
    that variance **is** the shot + read noise floor — measured per pixel, with
    no need to know the camera's conversion gain.

        std_corrected = sqrt(max(std^2 - strength * noise^2, 0))
    """
    s2 = np.asarray(std_img, dtype=np.float32) ** 2
    n2 = np.asarray(noise_std_img, dtype=np.float32) ** 2
    return np.sqrt(np.clip(s2 - float(strength) * n2, 0.0, None))


def subtract_noise_analytic(std_img, avg_img, gain_adu_per_e=0.23, offset=100.0,
                            read_noise_adu=4.0, strength=1.0):
    """Remove the shot + read noise floor using the camera noise model.

    For a signal of S electrons digitised at k ADU/e-, the shot-noise variance in
    ADU is ``k * (mean_ADU - offset)``; read noise adds a constant. So::

        var_corrected = var_measured - k*(mean - offset) - sigma_read^2

    Use this when no frozen-speckle reference was acquired. The measured
    reference (:func:`subtract_noise_reference`) is strictly better — it is
    per-pixel and free of any assumption about k.
    """
    var = np.asarray(std_img, dtype=np.float32) ** 2
    mean = np.asarray(avg_img, dtype=np.float32) - float(offset)
    noise_var = float(gain_adu_per_e) * np.clip(mean, 0, None) + float(read_noise_adu) ** 2
    return np.sqrt(np.clip(var - float(strength) * noise_var, 0.0, None))


# --- 3. normalization / flat field ----------------------------------------
def normalize_by_average(std_img, avg_img, offset=0.0, mode="ratio", floor_pct=5.0):
    """Divide out the illumination envelope using the companion average image.

    ``mode="ratio"`` gives ``std/mean`` — the speckle contrast, the classic
    Ventalon–Mertz sectioning metric, which removes both the Gaussian beam
    profile and the fluorophore-density weighting. ``mode="sqrt"`` gives
    ``std/sqrt(mean)``, which keeps the result shot-noise-weighted (a
    shot-noise-limited region then reads flat, so departures from flat are real
    signal).

    The denominator is floored at its ``floor_pct`` percentile so dark corners
    can't explode into a bright rim.
    """
    num = np.asarray(std_img, dtype=np.float32)
    den = np.asarray(avg_img, dtype=np.float32) - float(offset)
    floor = max(float(np.percentile(den, floor_pct)), EPS)
    den = np.clip(den, floor, None)
    if mode == "sqrt":
        den = np.sqrt(den)
    return num / den


def self_flat_field(img, sigma=60.0):
    """Divide by a heavily blurred copy of the image itself.

    Removes the low-spatial-frequency illumination envelope when no companion
    average image is available. ``sigma`` must be much larger than the structures
    of interest, or it eats the signal along with the background.
    """
    arr = np.asarray(img, dtype=np.float32)
    bg = cv2.GaussianBlur(arr, (0, 0), float(sigma))
    floor = max(float(np.percentile(bg, 5.0)), EPS)
    return arr / np.clip(bg, floor, None)


# --- 4. background ---------------------------------------------------------
def rolling_ball_background(img, radius=50):
    """Subtract a rolling-ball-equivalent background (grayscale opening).

    A morphological opening with a disk of ``radius`` estimates the slowly
    varying background (ImageJ's "Subtract Background"); subtracting it removes
    residual out-of-focus haze while leaving structure smaller than the disk.
    """
    arr = np.asarray(img, dtype=np.float32)
    r = max(1, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bg = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    return arr - bg


# --- 5. denoising ----------------------------------------------------------
def _to_uint8_scaled(img):
    """Scale a float image to uint8 for the 8-bit-only OpenCV denoisers."""
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8), lo, 1.0
    scale = 255.0 / (hi - lo)
    return np.clip((arr - lo) * scale, 0, 255).astype(np.uint8), lo, scale


def denoise(img, method="none", strength=1.0):
    """Apply the selected denoiser.

    ``gaussian``  — fast, but rounds off point-like structure (beads).
    ``median``    — good against residual salt-and-pepper.
    ``bilateral`` — edge-preserving; the safe default for beads.
    ``nlm``       — non-local means; best detail retention, slowest.
    ``anscombe``  — Anscombe variance-stabilizing transform -> NLM -> inverse.
                    The correct order for Poisson-ish data such as EVK4 event
                    counts, where the noise level scales with the signal.
    """
    arr = np.asarray(img, dtype=np.float32)
    s = float(strength)
    if method == "none" or s <= 0:
        return arr
    if method == "gaussian":
        return cv2.GaussianBlur(arr, (0, 0), s)
    if method == "median":
        return _median_filter(arr, int(max(3, round(s) * 2 + 1)))
    if method == "bilateral":
        rng = float(arr.max() - arr.min()) or 1.0
        return cv2.bilateralFilter(arr, -1, sigmaColor=0.05 * rng * s, sigmaSpace=3.0 * s)
    if method == "nlm":
        u8, lo, scale = _to_uint8_scaled(arr)
        out = cv2.fastNlMeansDenoising(u8, None, h=float(3.0 * s),
                                       templateWindowSize=7, searchWindowSize=21)
        return out.astype(np.float32) / scale + lo
    if method == "anscombe":
        # z = 2*sqrt(x + 3/8) turns Poisson noise into ~unit-variance Gaussian
        # noise, which is what NLM assumes; invert exactly afterwards.
        shift = float(min(0.0, arr.min()))
        z = 2.0 * np.sqrt(np.clip(arr - shift, 0, None) + 0.375)
        u8, lo, scale = _to_uint8_scaled(z)
        d = cv2.fastNlMeansDenoising(u8, None, h=float(3.0 * s),
                                     templateWindowSize=7, searchWindowSize=21)
        zd = d.astype(np.float32) / scale + lo
        inv = (zd / 2.0) ** 2 - 0.125
        return np.clip(inv, 0, None) + shift
    return arr


# --- 6. deconvolution ------------------------------------------------------
def psf_sigma_px(wavelength_nm=580.0, na=0.65, pixel_um=0.1625):
    """Gaussian-PSF sigma in pixels from the optics.

    Uses the lateral resolution ``FWHM = 0.51*lambda/NA`` converted to pixels,
    then ``sigma = FWHM / 2.355``. ``pixel_um`` is the *sample-side* pixel size
    (camera pitch / magnification) — e.g. 6.5 µm / 40x = 0.1625 µm.
    """
    fwhm_um = 0.51 * (float(wavelength_nm) * 1e-3) / max(float(na), EPS)
    return float(fwhm_um / max(float(pixel_um), EPS) / 2.3548)


def gaussian_psf(sigma):
    """Normalised 2-D Gaussian PSF kernel sized to ~3 sigma each side."""
    r = max(1, int(round(3.0 * float(sigma))))
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(ax ** 2) / (2.0 * float(sigma) ** 2))
    psf = np.outer(g, g)
    return psf / psf.sum()


def richardson_lucy(img, psf, iterations=10):
    """Richardson–Lucy deconvolution (float, non-negativity preserved).

    RL assumes a non-negative image formed by a linear convolution, with Poisson
    noise. A DSI std image does not strictly satisfy the linearity assumption, so
    the result is a *contrast enhancement*, not a quantitative restoration —
    say so in any caption. Keep the iteration count low (5–15); RL amplifies
    noise without bound as it converges.
    """
    arr = np.asarray(img, dtype=np.float64)
    shift = float(min(0.0, arr.min()))
    arr = arr - shift  # RL requires non-negative input
    psf = np.asarray(psf, dtype=np.float64)
    psf_mirror = psf[::-1, ::-1]

    est = np.full(arr.shape, max(float(arr.mean()), EPS), dtype=np.float64)
    for _ in range(int(iterations)):
        conv = fftconvolve(est, psf, mode="same")
        relative = arr / np.clip(conv, EPS, None)
        est = est * fftconvolve(relative, psf_mirror, mode="same")
        np.clip(est, 0, None, out=est)
    return (est + shift).astype(np.float32)


# --- 7. sharpening ---------------------------------------------------------
def unsharp_mask(img, amount=0.6, radius=2.0):
    """``img + amount * (img - blur(img))`` — classic unsharp mask."""
    arr = np.asarray(img, dtype=np.float32)
    if amount <= 0:
        return arr
    blur = cv2.GaussianBlur(arr, (0, 0), float(radius))
    return arr + float(amount) * (arr - blur)


# --- 8-12. display-only ----------------------------------------------------
def apply_gamma(view8, gamma=1.0):
    """Gamma-correct an 8-bit view. gamma < 1 lifts faint structure."""
    g = float(gamma)
    if abs(g - 1.0) < 1e-3:
        return view8
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(view8, lut)


def apply_clahe(view8, clip_limit=2.0, tiles=8):
    """Contrast-limited adaptive histogram equalization.

    Display only: it is a spatially varying, non-monotonic remap, so intensities
    in the result are no longer comparable between regions.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(view8)


# cv2's viridis/magma/inferno are perceptually uniform; the single-hue
# fluorescence ramps are built here as (B, G, R) weights.
_CV_COLORMAPS = {
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Magma": cv2.COLORMAP_MAGMA,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Turbo": cv2.COLORMAP_TURBO,
    "Hot": cv2.COLORMAP_HOT,
    "Jet (not perceptual)": cv2.COLORMAP_JET,
}

_HUE_LUTS = {
    "Green (fluo)": (0.0, 1.0, 0.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Cyan": (1.0, 1.0, 0.0),
    "Red": (0.0, 0.0, 1.0),
}

LUT_NAMES = ["Grayscale"] + list(_HUE_LUTS) + list(_CV_COLORMAPS)


def apply_lut(view8, name="Grayscale"):
    """Colour an 8-bit view. Returns (H,W) gray or (H,W,3) BGR."""
    if name == "Grayscale":
        return view8
    if name in _HUE_LUTS:
        b, g, r = _HUE_LUTS[name]
        out = np.zeros((*view8.shape, 3), np.uint8)
        out[..., 0] = (view8 * b).astype(np.uint8)
        out[..., 1] = (view8 * g).astype(np.uint8)
        out[..., 2] = (view8 * r).astype(np.uint8)
        return out
    return cv2.applyColorMap(view8, _CV_COLORMAPS[name])


def draw_scale_bar(view, um_per_px, bar_um, margin_frac=0.04, thickness_frac=0.012):
    """Burn a labelled scale bar into the bottom-right of a display image.

    Returns the image unchanged if the bar would not fit — drawing a truncated
    bar would misrepresent the scale.
    """
    img = view if view.ndim == 3 else cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    h, w = img.shape[:2]
    length_px = int(round(float(bar_um) / max(float(um_per_px), EPS)))
    if length_px < 2 or length_px > w * (1 - 2 * margin_frac):
        return img

    t = max(2, int(round(h * thickness_frac)))
    mx, my = int(w * margin_frac), int(h * margin_frac)
    x1, y1 = w - mx, h - my
    x0, y0 = x1 - length_px, y1 - t
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)

    label = f"{bar_um:g} um"
    scale = max(0.4, h / 900.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(img, label, (x1 - length_px // 2 - tw // 2, y0 - max(4, th // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                max(1, int(scale * 1.6)), cv2.LINE_AA)
    return img


# --- stack-level tools -----------------------------------------------------
def register_stack(slices, status=None):
    """Correct lateral drift between planes by phase correlation.

    Each plane is aligned to the previous one and the shifts are accumulated, so
    the whole volume ends up in a common frame. Without this, stage/sample drift
    between planes smears every orthogonal (XZ / YZ) view of a Z-stack.

    Returns ``(aligned_slices, shifts)`` with cumulative (dx, dy) per plane.
    """
    first = np.ascontiguousarray(slices[0], dtype=np.float32)
    out = [first]
    shifts = [(0.0, 0.0)]
    cum = np.zeros(2)
    prev = first
    win = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_32F)

    for i in range(1, len(slices)):
        if status:
            status(f"Registering plane {i + 1}/{len(slices)}…")
        cur = np.ascontiguousarray(slices[i], dtype=np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
        cum += (dx, dy)
        M = np.array([[1.0, 0.0, -cum[0]], [0.0, 1.0, -cum[1]]], dtype=np.float32)
        out.append(cv2.warpAffine(cur, M, (cur.shape[1], cur.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
        shifts.append((float(cum[0]), float(cum[1])))
        prev = cur
    return out, shifts


def project_stack(slices, mode="mip"):
    """Project a processed stack along z into a single 2-D image.

    ``mip``  — maximum intensity: the "all-in-focus" companion image, which is
               only meaningful *because* DSI sections (a widefield MIP is mush).
    ``mean`` / ``std`` — average and standard deviation through depth.
    ``edf``  — extended depth of field: per pixel, take the plane where a local
               focus measure (Laplacian energy) peaks. Sharper than a MIP when
               planes differ in noise level.
    """
    vol = np.stack([np.asarray(s, dtype=np.float32) for s in slices], axis=0)
    if mode == "mip":
        return vol.max(axis=0)
    if mode == "mean":
        return vol.mean(axis=0)
    if mode == "std":
        return vol.std(axis=0)
    if mode == "edf":
        focus = np.stack([cv2.GaussianBlur(cv2.Laplacian(s, cv2.CV_32F) ** 2, (0, 0), 3.0)
                          for s in vol], axis=0)
        best = focus.argmax(axis=0)
        return np.take_along_axis(vol, best[None, ...], axis=0)[0]
    raise ValueError(f"unknown projection mode: {mode}")


def depth_colour_code(slices, low_pct=1.0, high_pct=99.8):
    """Depth-colour-coded projection: hue = z plane, intensity = signal.

    Each plane gets a hue across the spectrum and its contrast-stretched
    intensity drives value; the brightest plane wins per pixel. One image that
    carries the whole volume — and it makes the optical sectioning immediately
    visible, since a non-sectioning modality renders as uniform mud.

    Returns an (H, W, 3) BGR uint8 image.
    """
    n = len(slices)
    vol = np.stack([np.asarray(s, dtype=np.float32) for s in slices], axis=0)
    lo, hi = np.percentile(vol[:, ::4, ::4], [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((vol - lo) / (hi - lo), 0, 1)

    h, w = vol.shape[1:]
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = (norm.argmax(axis=0) * (179.0 / max(1, n - 1))).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (norm.max(axis=0) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def depth_colour_legend(n_planes, width=320, height=44):
    """Colour-bar legend matching :func:`depth_colour_code` (BGR uint8)."""
    hsv = np.zeros((height, width, 3), np.uint8)
    hsv[..., 0] = np.linspace(0, 179, width).astype(np.uint8)[None, :]
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    bar = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.putText(bar, "z 0", (4, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, f"z {n_planes - 1}", (width - 58, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def orthogonal_views(slices, z_step_um=None, um_per_px=None):
    """XZ and YZ cuts through the middle of the volume (float32 each).

    Axially scaled to the lateral pixel size when both spacings are given, so
    the cut is geometrically true rather than stretched by the z step.
    """
    vol = np.stack([np.asarray(s, dtype=np.float32) for s in slices], axis=0)
    nz, h, w = vol.shape
    xz = vol[:, h // 2, :]
    yz = vol[:, :, w // 2].T
    if z_step_um and um_per_px:
        factor = float(z_step_um) / float(um_per_px)
        if factor > 1.05 or factor < 0.95:
            xz = cv2.resize(xz, (w, max(1, int(round(nz * factor)))),
                            interpolation=cv2.INTER_LINEAR)
            yz = cv2.resize(yz, (max(1, int(round(nz * factor))), h),
                            interpolation=cv2.INTER_LINEAR)
    return xz, yz


# --- 3-D volume rendering ---------------------------------------------------
# A DSI z-stack is already a 3-D image: the sectioning is what makes the volume
# meaningful, so nothing has to be *reconstructed* — only resampled onto an
# isotropic grid and projected from an arbitrary angle.
#
# The renderer is a **shear-warp** maximum-intensity projector, chosen so that
# rotating stays interactive with no GPU and no new dependency. Resampling the
# whole volume into camera space for every frame (``ndimage.affine_transform``)
# costs one 3-D interpolation per rendered pixel; shear-warp instead factors the
# view into
#
#     (a) a per-slice **translation** — because rays advance by a constant
#         in-plane step per slice once the volume is permuted so the axis most
#         parallel to the view direction comes first, and
#     (b) one final 2-D **affine warp** of the accumulated image.
#
# So the per-frame cost is N 2-D ``cv2.warpAffine`` calls (SIMD, milliseconds)
# instead of a 3-D resampling. The factorization is exact for MIP and for
# front-to-back alpha compositing alike, because both are evaluated along the
# rays, and MIP additionally does not care about the foreshortening the warp
# corrects (``max`` is scale-free, unlike a sum).
#
# Depth is carried alongside the intensity and coloured with the *same* hue ramp
# as :func:`depth_colour_code`, so ``depth_colour_legend`` is a valid legend for
# a 3-D view too, and a bead's colour means the same thing in both. Hue is keyed
# to the bead's position in the **sample**, not its distance from the camera, so
# the colours do not change as the volume is rotated.
def plan_view_volume(n_planes, plane_shape, z_step_um, um_per_px, max_dim=256):
    """Geometry of the isotropic working volume: ``(voxel_um, (nz, ny, nx))``.

    Rendering needs cubic voxels, but the acquired ones are wildly anisotropic
    (0.1625 um laterally against a 0.2-1 um z step, and an axial *resolution* of
    2-4 um). The grid is therefore sized by physical extent, at whatever voxel
    size keeps the longest axis within ``max_dim`` — which bounds the volume at
    ``max_dim**3`` voxels regardless of how big the stack is.

    Over a full 2304-px field that voxel lands near 1.5 um, which throws away
    lateral detail but is still finer than the axial resolution: the *axial*
    sampling, the one that costs sectioning, survives. Crop the field first when
    lateral detail matters.
    """
    h, w = int(plane_shape[0]), int(plane_shape[1])
    um_per_px = float(um_per_px) if um_per_px and um_per_px > 0 else 1.0
    z_step_um = float(z_step_um) if z_step_um and z_step_um > 0 else um_per_px
    n_planes = max(1, int(n_planes))
    # World order here is (x, y, z); the array stays (z, y, x).
    extent = (n_planes * z_step_um, h * um_per_px, w * um_per_px)
    voxel = max(max(extent) / float(max(8, int(max_dim))), 1e-9)
    shape = tuple(max(2, int(round(e / voxel))) for e in extent)
    return voxel, shape


def _resample_axis0(stack, nz):
    """Resample a (Z, H, W) stack to ``nz`` planes.

    Downsampling takes the **maximum** over each output plane's source range,
    not the mean: beads are sparse, and averaging a bead in with the empty
    planes around it dims exactly the structure the view exists to show. (The
    projection downstream is a max anyway, so this is consistent.) Upsampling is
    linear.
    """
    n = int(stack.shape[0])
    nz = max(1, int(nz))
    if nz == n:
        return stack
    if nz < n:
        edges = np.linspace(0, n, nz + 1)
        out = np.empty((nz,) + stack.shape[1:], np.float32)
        for k in range(nz):
            lo = int(np.floor(edges[k]))
            hi = max(lo + 1, int(np.ceil(edges[k + 1])))
            out[k] = stack[lo:hi].max(axis=0)
        return out
    pos = np.linspace(0, n - 1, nz)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, n - 1)
    frac = (pos - lo).astype(np.float32)[:, None, None]
    return (stack[lo] * (1.0 - frac) + stack[hi] * frac).astype(np.float32)


def build_view_volume(get_slice, n_planes, z_step_um, um_per_px,
                      max_dim=256, status=None):
    """Build the isotropic render volume. Returns ``(vol, voxel_um)``.

    ``get_slice(i)`` returns processed plane ``i`` as a 2-D array. Each plane is
    shrunk to the working grid **as it arrives** and then dropped, so peak
    memory is the working volume plus one full-size plane — never the whole
    processed stack (which is gigabytes at full sensor).
    """
    first = np.asarray(get_slice(0), dtype=np.float32)
    voxel, (nz, ny, nx) = plan_view_volume(n_planes, first.shape,
                                           z_step_um, um_per_px, max_dim)
    stack = np.empty((n_planes, ny, nx), np.float32)
    shrinking = nx < first.shape[1] or ny < first.shape[0]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    for i in range(n_planes):
        plane = first if i == 0 else np.asarray(get_slice(i), dtype=np.float32)
        stack[i] = cv2.resize(plane, (nx, ny), interpolation=interp)
        if status is not None and (i % 4 == 0 or i == n_planes - 1):
            status(f"Building 3-D volume… plane {i + 1}/{n_planes}")
    return _resample_axis0(stack, nz), voxel


def volume_display_range(vol, low_pct=1.0, high_pct=99.9):
    """Auto black/white points for a render volume, from a subsample."""
    step = max(1, vol.shape[0] // 64)
    lo, hi = np.percentile(vol[::step, ::4, ::4], [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def view_rotation(azimuth_deg, elevation_deg):
    """World->camera rotation for the volume view (world order x, y, z).

    Camera axes are (u right, v down, w into the screen), matching image
    convention, so azimuth 0 / elevation 0 looks straight down z and reproduces
    the familiar top-down MIP.
    """
    az, el = np.radians(float(azimuth_deg)), np.radians(float(elevation_deg))
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    r_y = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return r_x @ r_y


def project_volume_points(points_xyz, rotation, vol_shape, zoom, out_shape):
    """Project world points (x, y, z voxel indices) to (col, row) pixels.

    The projection is orthographic and isotropic, so this is the exact inverse
    of what :func:`render_volume` draws — overlays land on the structure.
    """
    nz, ny, nx = vol_shape
    centre = (np.array([nx, ny, nz], dtype=np.float64) - 1.0) / 2.0
    uv = (np.asarray(points_xyz, dtype=np.float64) - centre) @ np.asarray(rotation)[:2].T
    uv *= float(zoom)
    uv[:, 0] += out_shape[1] / 2.0
    uv[:, 1] += out_shape[0] / 2.0
    return uv


_BOX_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def _box_corners(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)
                     for z in (z0, z1)], dtype=np.float64)


def draw_volume_box(view, rotation, vol_shape, zoom, z_range=None,
                    labels=True):
    """Draw the volume's wireframe (and the active slab) onto a BGR view.

    A projected box is the cheapest strong depth cue there is: without it an
    orthographic MIP of sparse beads gives the eye nothing to read the rotation
    against, and the structure looks flat no matter how far it is turned.
    """
    nz, ny, nx = vol_shape
    out = view

    def draw(corners, colour, thickness):
        pts = project_volume_points(corners, rotation, vol_shape, zoom, out.shape)
        for i, j in _BOX_EDGES:
            p0 = (int(round(pts[i, 0])), int(round(pts[i, 1])))
            p1 = (int(round(pts[j, 0])), int(round(pts[j, 1])))
            cv2.line(out, p0, p1, colour, thickness, cv2.LINE_AA)
        return pts

    corners = draw(_box_corners(0, nx - 1, 0, ny - 1, 0, nz - 1), (70, 70, 70), 1)
    if z_range is not None:
        z0, z1 = int(z_range[0]), int(z_range[1]) - 1
        if not (z0 <= 0 and z1 >= nz - 1):
            draw(_box_corners(0, nx - 1, 0, ny - 1, z0, z1), (0, 190, 190), 1)
    if labels:
        # Axis names at the far end of the three edges leaving corner (0,0,0).
        ends = project_volume_points(
            np.array([[nx - 1, 0, 0], [0, ny - 1, 0], [0, 0, nz - 1]], float),
            rotation, vol_shape, zoom, out.shape)
        for (name, pt) in zip(("x", "y", "z"), ends):
            cv2.putText(out, name, (int(pt[0]) + 4, int(pt[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                        cv2.LINE_AA)
    return out


def render_volume(vol, rotation, z_range=None, mode="mip", stride=1,
                  display_range=(0.0, 1.0), threshold=0.0, opacity=0.5,
                  colour_by_depth=True, zoom=1.0, margin=1.08):
    """Project an isotropic volume from an arbitrary angle. Returns BGR uint8.

    ``mode``
        ``mip``   — maximum intensity along each ray. The honest choice for
                    sparse beads: nothing occludes anything, so no bead can be
                    hidden by one in front of it.
        ``solid`` — front-to-back alpha compositing. Beads *do* occlude each
                    other, which is the strongest depth cue available, at the
                    cost of hiding faint structure behind bright structure.
    ``z_range``
        ``(z0, z1)`` slab in volume planes — the "scroll through the stack"
        control. The slab is cropped from the volume but is still projected
        about the **whole** volume's centre, so it stays at its true position in
        the frame instead of re-centring as it is scrolled.
    ``stride``
        Render from every n-th voxel and scale the result back up. The output
        is the same size and geometry either way, which is what lets a fast
        draft be swapped for a full-quality frame without the view jumping.
    """
    vol = np.asarray(vol, dtype=np.float32)
    nz, ny, nx = vol.shape
    stride = max(1, int(stride))
    rotation = np.asarray(rotation, dtype=np.float64)

    z0, z1 = (0, nz) if z_range is None else (int(z_range[0]), int(z_range[1]))
    z0 = int(np.clip(z0, 0, nz - 1))
    z1 = int(np.clip(z1, z0 + 1, nz))

    # Strided grid: indices 0, stride, 2*stride, ... of the original volume.
    zs = np.arange(0, nz, stride)
    k0 = int(np.searchsorted(zs, z0, "left"))
    k1 = max(int(np.searchsorted(zs, z1, "left")), k0 + 1)
    sub = vol[zs[k0]:zs[k1 - 1] + 1:stride, ::stride, ::stride]

    # World sizes / origin / centre, all in strided voxels, world order (x,y,z).
    # The centre is the *original* volume's centre expressed in strided units,
    # not the strided array's own middle: the two differ by up to half a stride,
    # which would shift the draft frame against the full-quality one.
    extent = (np.array([nx, ny, nz], dtype=np.float64) - 1.0) / stride
    centre = extent / 2.0
    origin = np.array([0.0, 0.0, float(k0)])
    off = origin - centre

    # --- shear: permute so the axis most parallel to the view comes first ---
    view_dir = rotation[2]
    c = int(np.argmax(np.abs(view_dir)))
    a, b = [i for i in (0, 1, 2) if i != c]
    perm = np.transpose(sub, (2 - c, 2 - a, 2 - b))   # array axis of world i is 2-i
    n_c, n_a, n_b = perm.shape

    ks = np.arange(n_c, dtype=np.float64)
    t_a = off[a] - (off[c] + ks) * (view_dir[a] / view_dir[c])
    t_b = off[b] - (off[c] + ks) * (view_dir[b] / view_dir[c])
    p0, q0 = -t_a.min(), -t_b.min()
    h_i = int(np.ceil(t_a.max() - t_a.min())) + n_a + 2
    w_i = int(np.ceil(t_b.max() - t_b.min())) + n_b + 2

    lo, hi = float(display_range[0]), float(display_range[1])
    span = max(hi - lo, EPS)
    thr = float(np.clip(threshold, 0.0, 0.999))
    cut = lo + thr * span
    solid = (mode == "solid")

    acc = np.zeros((h_i, w_i), np.float32)
    dep = np.zeros((h_i, w_i), np.float32) if colour_by_depth else None
    alpha = np.zeros((h_i, w_i), np.float32) if solid else None
    rows = np.arange(h_i, dtype=np.float32)[:, None]
    cols = np.arange(w_i, dtype=np.float32)[None, :]

    # Front-to-back: w increases with k when the view direction and the
    # principal axis point the same way. Only alpha compositing cares, but the
    # order is free, so both modes use it.
    order = range(n_c) if view_dir[c] > 0 else range(n_c - 1, -1, -1)
    # Sparse volumes are mostly empty: skipping planes with nothing above the
    # black point is what keeps a big slab interactive.
    plane_max = perm.max(axis=(1, 2))

    for k in order:
        if plane_max[k] <= cut:
            continue
        shift = np.array([[1.0, 0.0, t_b[k] + q0], [0.0, 1.0, t_a[k] + p0]],
                         dtype=np.float32)
        warped = cv2.warpAffine(perm[k], shift, (w_i, h_i),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        if dep is not None:
            # The source z of every pixel is analytic — the slice was only
            # translated, so a z ramp stays a ramp and needs no second warp.
            if c == 2:
                z_src = np.float32(k0 + k)
            elif a == 2:
                z_src = rows - np.float32(t_a[k] + p0 - k0)
            else:
                z_src = cols - np.float32(t_b[k] + q0 - k0)
        if solid:
            val = np.clip((warped - cut) / (span * (1.0 - thr)), 0.0, 1.0)
            contrib = (1.0 - alpha) * val * float(opacity)
            acc += contrib * val
            alpha += contrib
            if dep is not None:
                dep += contrib * z_src
        else:
            upd = warped > acc
            if upd.any():
                acc[upd] = warped[upd]
                if dep is not None:
                    dep[upd] = np.broadcast_to(z_src, acc.shape)[upd]

    if solid:
        intensity = np.clip(acc / max(float(acc.max()), EPS), 0.0, 1.0)
        if dep is not None:
            dep = dep / np.maximum(alpha, EPS)
    else:
        intensity = np.clip((acc - cut) / (span * (1.0 - thr)), 0.0, 1.0)

    # --- warp: one 2-D affine takes the sheared image to the screen ---------
    mat2 = float(zoom) * np.array([[rotation[0, a], rotation[0, b]],
                                   [rotation[1, a], rotation[1, b]]])
    # Frame the *whole* volume (not the slab) so scrolling the slab does not
    # make the picture jump. Sized in final pixels first, then divided down by
    # the stride, so every stride yields the identical output size.
    box = (_box_corners(0, extent[0], 0, extent[1], 0, extent[2])
           - centre) @ rotation[:2].T * float(zoom) * stride
    w_out = int(np.ceil((box[:, 0].max() - box[:, 0].min()) * margin)) + 8
    h_out = int(np.ceil((box[:, 1].max() - box[:, 1].min()) * margin)) + 8
    w_o, h_o = -(-w_out // stride), -(-h_out // stride)
    cx = w_o / 2.0 - (mat2[0, 0] * p0 + mat2[0, 1] * q0)
    cy = h_o / 2.0 - (mat2[1, 0] * p0 + mat2[1, 1] * q0)
    # warpAffine reads (x=col=q, y=row=p), hence the swapped columns.
    warp = np.array([[mat2[0, 1], mat2[0, 0], cx],
                     [mat2[1, 1], mat2[1, 0], cy]], dtype=np.float32)
    intensity = cv2.warpAffine(intensity, warp, (w_o, h_o),
                               flags=cv2.INTER_LINEAR)
    if dep is not None:
        dep = cv2.warpAffine(dep, warp, (w_o, h_o), flags=cv2.INTER_NEAREST)

    if dep is not None:
        # Same hue ramp as depth_colour_code, so depth_colour_legend applies.
        hue = np.clip(dep * stride * (179.0 / max(1, nz - 1)), 0, 179)
        hsv = np.zeros((h_o, w_o, 3), np.uint8)
        hsv[..., 0] = hue.astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = (intensity * 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        out = cv2.cvtColor((intensity * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if (w_o, h_o) != (w_out, h_out):
        out = cv2.resize(out, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
    return out


# --- axial (Z) matching between two cameras --------------------------------
# The ORCA and EVK4 ports sit at slightly different optical path lengths, so the
# same physical plane is in focus at different stage positions on the two
# detectors. Plane k of one stack therefore does NOT correspond to plane k of the
# other. These helpers measure that offset and resample one stack onto the
# other's z grid.
def axial_profile(slices, metric="mean"):
    """Per-plane scalar vs depth — the curve whose peak marks the focal plane.

    ``mean``  — mean intensity of the sectioned image. This is exactly the
                quantity ``core.save_axial_sectioning_plot`` fits a Gaussian to
                (paper Fig. 3a): it peaks at focus because DSI sections.
    ``focus`` — Laplacian energy, a pure sharpness measure. Use it when the
                sectioned intensity is too flat to give a clean peak (it
                responds to structure rather than to brightness).
    ``std``   — spatial standard deviation, a middle ground.
    """
    out = []
    for s in slices:
        img = np.asarray(s, dtype=np.float32)
        if metric == "focus":
            out.append(float(np.mean(cv2.Laplacian(img, cv2.CV_32F) ** 2)))
        elif metric == "std":
            out.append(float(img.std()))
        else:
            out.append(float(img.mean()))
    return np.asarray(out, dtype=np.float64)


def _parabolic_refine(y, i):
    """Sub-sample peak position by fitting a parabola through y[i-1:i+2]."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < EPS:
        return float(i)
    return float(i) + 0.5 * (a - c) / denom


def plane_positions(n, z_step, z_start=0.0):
    """Nominal axial position (um) of each plane index."""
    return np.arange(int(n), dtype=np.float64) * float(z_step) + float(z_start)


def axial_profile_peakedness(prof):
    """How concentrated an axial profile is — the guard for profile-based matching.

    Returns the fraction of planes sitting above half of the profile's range. A
    thin (2-D) bead layer gives a sharp Gaussian, so only a small fraction clears
    half-max; a **3D sample** with beads through its whole thickness gives a broad,
    structured curve where most planes do.

    This matters because a correlation score cannot detect the failure: two flat
    profiles correlate *well* with each other at zero shift, so profile matching
    silently reports "no offset" with a healthy-looking score. Peakedness catches
    it; the correlation score does not.

    Rule of thumb: > 0.5 means there is no single focal peak, so
    :func:`find_axial_offset` is not trustworthy — use
    :func:`find_axial_offset_by_images`.
    """
    p = np.asarray(prof, dtype=np.float64)
    lo, hi = float(p.min()), float(p.max())
    if hi <= lo:
        return 1.0
    return float(np.mean((p - lo) / (hi - lo) > 0.5))


def find_axial_offset(z_a, prof_a, z_b, prof_b, oversample=8):
    """Measure the axial offset between two stacks from their axial profiles.

    Both profiles are interpolated onto a common fine grid, mean-subtracted and
    cross-correlated; the lag that maximises the correlation is the offset.
    Sub-plane precision comes from a parabolic fit around the correlation peak,
    so the result is not quantised to the z step.

    Returns ``(dz_um, score)`` where **dz = z_B - z_A for the same physical
    plane**: the plane of B matching A's position ``z`` is at ``z + dz``.
    ``score`` is the normalised correlation peak in [-1, 1]; below ~0.5 treat the
    result as unreliable (typically a profile with no clear focal peak).

    NOTE: the offset absorbs *any* difference between the two stacks — the
    optical focal-plane offset between the camera ports **and** any difference in
    where the two scans started. Feed true z positions (e.g. from the
    acquisition's ``_axial_profile_*.csv``) if you want to separate the two.
    """
    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)
    pa = np.asarray(prof_a, dtype=np.float64)
    pb = np.asarray(prof_b, dtype=np.float64)
    if z_a.size < 2 or z_b.size < 2:
        return 0.0, 0.0

    # Normalise each profile to [0, 1] so a brightness difference between the two
    # detectors cannot bias the correlation.
    def norm(p):
        lo, hi = float(p.min()), float(p.max())
        return (p - lo) / (hi - lo) if hi > lo else np.zeros_like(p)

    pa, pb = norm(pa), norm(pb)

    step_a = float(np.median(np.diff(z_a))) if z_a.size > 1 else 1.0
    step_b = float(np.median(np.diff(z_b))) if z_b.size > 1 else 1.0
    step = max(min(abs(step_a), abs(step_b)) / float(oversample), 1e-6)

    lo = min(z_a.min(), z_b.min())
    hi = max(z_a.max(), z_b.max())
    grid = np.arange(lo, hi + step, step)
    fa = np.interp(grid, z_a, pa, left=0.0, right=0.0)
    fb = np.interp(grid, z_b, pb, left=0.0, right=0.0)
    fa = fa - fa.mean()
    fb = fb - fb.mean()

    denom = np.linalg.norm(fa) * np.linalg.norm(fb)
    if denom < EPS:
        return 0.0, 0.0

    # correlate(fb, fa)[k] = sum_n fb[n + lag] * fa[n]  with lag = k - (len(fa)-1),
    # which peaks where fb(z + lag) ~ fa(z) -- i.e. lag = z_B - z_A. See the
    # docstring's sign convention.
    corr = np.correlate(fb, fa, mode="full") / denom
    k = int(np.argmax(corr))
    k_ref = _parabolic_refine(corr, k)
    dz = (k_ref - (len(fa) - 1)) * step
    return float(dz), float(corr[k])


def find_axial_offset_by_images(slices_a, slices_b, affine=None, max_shift=None,
                                downscale=4, status=None):
    """Measure the axial offset by directly comparing images plane against plane.

    For every candidate integer plane shift, B's planes are warped into A's frame
    (through the lateral field-match ``affine``) and scored against A's by
    zero-mean normalised cross-correlation; the best shift wins, refined to
    sub-plane precision by a parabola.

    Slower than :func:`find_axial_offset` but far more robust when the axial
    intensity profile has no clean peak — it keys off image *content* rather than
    a single number per plane. Returns ``(shift_planes, score)`` in units of B
    planes.
    """
    a = [cv2.resize(np.asarray(s, np.float32), None, fx=1.0 / downscale,
                    fy=1.0 / downscale, interpolation=cv2.INTER_AREA)
         for s in slices_a]
    b_full = [np.asarray(s, np.float32) for s in slices_b]
    if affine is not None:
        h, w = np.asarray(slices_a[0]).shape[:2]
        b_full = [cv2.warpAffine(s, np.asarray(affine, np.float64)[:2], (w, h),
                                 flags=cv2.INTER_LINEAR) for s in b_full]
    b = [cv2.resize(s, None, fx=1.0 / downscale, fy=1.0 / downscale,
                    interpolation=cv2.INTER_AREA) for s in b_full]

    na, nb = len(a), len(b)
    limit = max(na, nb) - 1 if max_shift is None else int(max_shift)
    shifts = np.arange(-limit, limit + 1)
    scores = []
    for si, shift in enumerate(shifts):
        if status is not None and si % 5 == 0:
            status(f"Axial search {si + 1}/{len(shifts)}…")
        vals = []
        for i in range(na):
            j = i + int(shift)
            if 0 <= j < nb:
                vals.append(_zncc(a[i], b[j]))
        # Require a decent overlap, or a one-plane overlap could win by luck.
        scores.append(float(np.mean(vals)) if len(vals) >= max(3, na // 4) else -1.0)

    scores = np.asarray(scores)
    k = int(np.argmax(scores))
    k_ref = _parabolic_refine(scores, k)
    return float(shifts[0] + k_ref), float(scores[k])


def _zncc(x, y):
    """Zero-mean normalised cross-correlation of two equal-shape images."""
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float((x * y).sum() / denom) if denom > EPS else 0.0


def resample_stack_z(slices, z_src, z_target):
    """Linearly interpolate a stack along z onto a new set of axial positions.

    Used to put channel B on channel A's z grid once the offset is known, so
    plane k of both stacks is the same physical plane. Positions outside the
    source range clamp to the end planes rather than fading to black, which would
    otherwise put spurious dark slabs at the ends of the volume.
    """
    z_src = np.asarray(z_src, dtype=np.float64)
    vol = np.stack([np.asarray(s, dtype=np.float32) for s in slices], axis=0)
    order = np.argsort(z_src)
    z_src, vol = z_src[order], vol[order]

    out = []
    for z in np.asarray(z_target, dtype=np.float64):
        if z <= z_src[0]:
            out.append(vol[0].copy())
            continue
        if z >= z_src[-1]:
            out.append(vol[-1].copy())
            continue
        j = int(np.searchsorted(z_src, z) - 1)
        j = max(0, min(j, len(z_src) - 2))
        span = z_src[j + 1] - z_src[j]
        t = 0.0 if span < EPS else (z - z_src[j]) / span
        out.append(((1.0 - t) * vol[j] + t * vol[j + 1]).astype(np.float32))
    return out


def read_axial_profile_csv(path):
    """Read ``z_position_um`` / ``mean_intensity`` from an acquisition's axial CSV.

    The acquisition writes ``<name>_axial_profile_<kind>.csv`` next to every
    Z-stack (see ``core.save_axial_sectioning_plot``). Reading it gives the *true*
    stage positions of each plane, which lets the measured offset be split into
    the genuine focal-plane difference and any difference in scan start.

    Returns ``(z, intensity)`` as float arrays, or ``(None, None)`` if the file
    cannot be parsed.
    """
    try:
        z, inten = [], []
        with open(path, encoding="utf-8") as f:
            header = None
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if header is None:
                    header = parts
                    continue
                if len(parts) < 2:
                    continue
                z.append(float(parts[0]))
                inten.append(float(parts[1]))
        if len(z) < 2:
            return None, None
        return np.asarray(z, dtype=np.float64), np.asarray(inten, dtype=np.float64)
    except Exception:  # noqa: BLE001 — a missing/odd CSV is not an error
        return None, None


def save_axial_comparison(z_a, prof_a, z_b, prof_b, dz, out_dir, filename):
    """Save the two axial profiles + the measured offset as CSV (and PNG if possible).

    The CSV is always written so the measurement is recorded even without
    matplotlib, mirroring the acquisition's own axial-profile contract.
    """
    out_dir = Path(out_dir)
    csv_path = out_dir / f"{filename}_axial_match.csv"

    def norm(p):
        p = np.asarray(p, dtype=np.float64)
        lo, hi = float(p.min()), float(p.max())
        return (p - lo) / (hi - lo) if hi > lo else np.zeros_like(p)

    na, nb = norm(prof_a), norm(prof_b)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(f"# axial offset dz_um = {dz:.6f}  (z_B - z_A for the same plane)\n")
        f.write("channel,z_position_um,profile,profile_normalized\n")
        for z, raw, nrm in zip(np.asarray(z_a), np.asarray(prof_a), na):
            f.write(f"A,{z:.6f},{raw:.6f},{nrm:.6f}\n")
        for z, raw, nrm in zip(np.asarray(z_b), np.asarray(prof_b), nb):
            f.write(f"B,{z:.6f},{raw:.6f},{nrm:.6f}\n")

    png_path = None
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(6.5, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot(z_a, na, "o-", color="#1f77b4", markersize=3, label="A (ORCA)")
        ax.plot(z_b, nb, "s-", color="#d62728", markersize=3, label="B (EVK4)")
        ax.plot(np.asarray(z_b) - dz, nb, "--", color="#2ca02c",
                label=f"B shifted by {-dz:+.3f} µm")
        ax.set_xlabel("Axial position (µm)")
        ax.set_ylabel("Axial profile (normalised)")
        ax.set_title(f"Axial plane matching — offset = {dz:+.3f} µm")
        ax.legend()
        fig.tight_layout()
        png_path = out_dir / f"{filename}_axial_match.png"
        fig.savefig(png_path, dpi=150)
    except Exception:  # noqa: BLE001 — the figure is best effort; the CSV is not
        png_path = None
    return csv_path, png_path


# --- the pipeline ----------------------------------------------------------
def process_slice(img, p, refs=None):
    """Run the quantitative pipeline (stages 1-7) on one slice.

    ``p`` is the parameter dict from the UI; ``refs`` carries the reference
    images (``avg``, ``noise``, ``hot_mask``) already matched to this slice.
    Returns float32 — the display stages are applied separately so the
    quantitative result can be exported at full precision.
    """
    refs = refs or {}
    out = np.asarray(img, dtype=np.float32)

    # 1. sensor artifacts
    if p["hot_on"]:
        out = remove_hot_pixels(out, k=p["hot_k"], window=p["hot_win"],
                                mask=refs.get("hot_mask"))

    # 2. noise floor (variance domain)
    if p["noise_mode"] == "reference" and refs.get("noise") is not None:
        out = subtract_noise_reference(out, refs["noise"], strength=p["noise_strength"])
    elif p["noise_mode"] == "analytic" and refs.get("avg") is not None:
        out = subtract_noise_analytic(out, refs["avg"], gain_adu_per_e=p["gain"],
                                      offset=p["offset"], read_noise_adu=p["read_noise"],
                                      strength=p["noise_strength"])

    # 3. normalization / flat field
    if p["norm_mode"] in ("ratio", "sqrt") and refs.get("avg") is not None:
        out = normalize_by_average(out, refs["avg"], offset=p["offset"], mode=p["norm_mode"])
    elif p["norm_mode"] == "self":
        out = self_flat_field(out, sigma=p["flat_sigma"])

    # 4. background
    if p["bg_radius"] > 0:
        out = rolling_ball_background(out, radius=p["bg_radius"])

    # 5. denoise
    if p["denoise_method"] != "none":
        out = denoise(out, method=p["denoise_method"], strength=p["denoise_strength"])

    # 6. deconvolution
    if p["decon_on"] and p["decon_iters"] > 0:
        out = richardson_lucy(out, gaussian_psf(p["decon_sigma"]), p["decon_iters"])

    # 7. sharpen
    if p["unsharp_amount"] > 0:
        out = unsharp_mask(out, amount=p["unsharp_amount"], radius=p["unsharp_radius"])

    return out


def render_display(processed, p, disp_range=None):
    """Run the display stages (8-12). Returns ``(view, (disp_min, disp_max))``."""
    if disp_range is not None:
        lo, hi = disp_range
    elif p["contrast_mode"] == "auto":
        lo, hi = imagej_auto_minmax(processed)
    elif p["contrast_mode"] == "percentile":
        lo, hi = percentile_minmax(processed, p["pct_low"], p["pct_high"])
    else:
        lo, hi = p["manual_min"], p["manual_max"]

    view = apply_display_range(processed, lo, hi)
    view = apply_gamma(view, p["gamma"])
    if p["clahe_on"]:
        view = apply_clahe(view, p["clahe_clip"], p["clahe_tiles"])
    view = apply_lut(view, p["lut"])
    if p["scalebar_on"]:
        view = draw_scale_bar(view, p["um_per_px"], p["bar_um"])
    return view, (lo, hi)


# ===========================================================================
# Data loading
# ===========================================================================
def load_mat_array(path):
    """Return the first real 2-D/3-D array in a ``.mat`` file (full precision).

    The acquisition writes ``_dsi.mat`` / ``_average.mat`` / ``_final_image.mat``,
    which keep the float precision the 8-bit preview TIFFs throw away — always
    prefer these as the processing input.
    """
    data = scipy.io.loadmat(str(path))
    for key, val in data.items():
        if key.startswith("__"):
            continue
        arr = np.squeeze(np.asarray(val))
        if arr.ndim in (2, 3) and arr.size > 16:
            return arr
    raise ValueError(f"no 2-D/3-D array found in {Path(path).name}")


def load_image_file(path):
    """Load a TIFF or ``.mat`` as an (N, H, W) float32 stack."""
    path = Path(path)
    if path.suffix.lower() == ".mat":
        arr = load_mat_array(path)
    else:
        arr = np.asarray(tifffile.imread(str(path)))
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
        arr = arr[..., 0][None, ...]  # collapse an RGB(A) page to one channel
    return arr.astype(np.float32)


class Frame:
    """One slice, loaded lazily from its source file."""

    def __init__(self, path, page, label):
        self.path = Path(path)
        self.page = page
        self.label = label
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = load_image_file(self.path)[self.page]
        return self._data

    def release(self):
        """Drop the cached pixels so a stack-wide pass stays memory-flat."""
        self._data = None


def scan_paths(paths):
    """Build the Frame list for a set of files, without loading pixel data."""
    frames = []
    for path in paths:
        path = Path(path)
        try:
            if path.suffix.lower() == ".mat":
                n = load_image_file(path).shape[0]
            else:
                with tifffile.TiffFile(str(path)) as tf:
                    shape = tf.series[0].shape
                    n = int(shape[0]) if len(shape) >= 3 and shape[-1] not in (3, 4) else 1
        except Exception as exc:  # noqa: BLE001 — report and skip unreadable files
            print(f"Skipping {path.name}: {exc}")
            continue
        for pg in range(n):
            label = path.name if n == 1 else f"{path.name} [{pg + 1}/{n}]"
            frames.append(Frame(path, pg, label))
    return frames


