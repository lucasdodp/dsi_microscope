"""Check the EVK4<->ORCA field-of-view matching against synthetic ground truth.

Runs offline, needs no hardware and no acquisition data: it renders the same
synthetic beads into a simulated ORCA frame and a simulated EVK4 frame through a
*known* affine, then asks the production code to recover it.

The case that matters is a **tightly cropped ORCA**. ``_masked_ncc`` slides the
EVK4 template inside the ORCA image and rejects any geometry where the template
is not strictly smaller, so an ORCA cropped to the bare footprint bounds the
scale search from above and the measured field of view comes out a few percent
too small (the 2026-07-24 pair capped at 0.710 against a true 0.745). This script
is the regression guard for that: it reproduces the 2026-07-24 geometry exactly.

    python tools/check_fov_matching.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EVK4_TO_ORCA_AFFINE, EVK4_ORCA_CROP_MARGIN  # noqa: E402
from core.image_processing import (  # noqa: E402
    evk4_footprint_in_orca, map_evk4_window_to_orca, register_evk4_to_orca,
    _registration_pad,
)

EVK4_W, EVK4_H = 1280, 720
TRUE_THETA, TRUE_SCALE = 317.0, 0.745
# The 2026-07-24 acquisition's ORCA crop — tighter than the EVK4 footprint needs.
TIGHT_ORCA = (996, 1052)


def true_affine(theta_deg=TRUE_THETA, scale=TRUE_SCALE, origin=(0.0, 0.0)):
    """A 2x3 EVK4->ORCA map placing the footprint's bounding box at ``origin``."""
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) * scale
    corners = np.array([[0, 0], [EVK4_W, 0], [EVK4_W, EVK4_H], [0, EVK4_H]]) @ R.T
    t = np.array([[origin[0] - corners[:, 0].min()],
                  [origin[1] - corners[:, 1].min()]])
    return np.hstack([R, t])


def _render(points, shape, sigma):
    img = np.zeros(shape, np.float32)
    h, w = shape
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            img[yi, xi] = 1.0
    return cv2.GaussianBlur(img, (0, 0), sigma)


def synth_pair(orca_shape, affine, n_beads=180, seed=3):
    """Simulated (orca, evk4) blob images of one bead field seen through ``affine``."""
    rng = np.random.default_rng(seed)
    h, w = orca_shape
    beads = np.column_stack([rng.uniform(60, w - 60, n_beads),
                             rng.uniform(60, h - 60, n_beads)])
    orca = _render(beads, orca_shape, 3.0)
    inv = cv2.invertAffineTransform(np.asarray(affine, np.float64))
    in_evk4 = np.column_stack([beads, np.ones(len(beads))]) @ inv.T
    return orca, _render(in_evk4, (EVK4_H, EVK4_W), 2.5)


def corner_error(measured, truth):
    """Worst-case error, in ORCA pixels, of the four mapped EVK4 corners."""
    roi = {"x_min": 0, "x_max": EVK4_W, "y_min": 0, "y_max": EVK4_H}
    return float(np.abs(evk4_footprint_in_orca(measured, roi)
                        - evk4_footprint_in_orca(truth, roi)).max())


def check_registration(orca_shape, label, tol_px=15.0):
    A = true_affine()
    orca, evk4 = synth_pair(orca_shape, A)
    border = _registration_pad(orca_shape, (EVK4_H, EVK4_W))
    affine, score, info = register_evk4_to_orca(orca, evk4,
                                                seed_affine=EVK4_TO_ORCA_AFFINE)
    err = corner_error(affine, A)
    scale_err = 100.0 * abs(info["scale"] - TRUE_SCALE) / TRUE_SCALE
    ok = err <= tol_px and scale_err <= 2.0
    print(f"  {label:<28} ORCA {orca_shape[1]}x{orca_shape[0]}  border={border:>4} px  "
          f"scale={info['scale']:.4f} ({scale_err:+.1f}%)  "
          f"corner err={err:6.1f} px  NCC={score:.2f}   "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_crop_margin():
    """The matcher's ORCA crop must contain the footprint with room to spare."""
    roi = {"x_min": 0, "x_max": EVK4_W, "y_min": 0, "y_max": EVK4_H}
    crop, corners, clipped = map_evk4_window_to_orca(EVK4_TO_ORCA_AFFINE, roi)
    fw = corners[:, 0].max() - corners[:, 0].min()
    fh = corners[:, 1].max() - corners[:, 1].min()
    cw = crop["x_max"] - crop["x_min"]
    ch = crop["y_max"] - crop["y_min"]
    covers = (crop["x_min"] <= corners[:, 0].min() and crop["y_min"] <= corners[:, 1].min()
              and crop["x_max"] >= corners[:, 0].max() and crop["y_max"] >= corners[:, 1].max())
    # The scale search must be able to overshoot the true scale, or it can only
    # ever settle low — this is the property the margin exists to guarantee.
    headroom = min(cw / fw, ch / fh)
    ok = covers and not clipped and headroom > 1.05
    print(f"  {'matcher crop vs footprint':<28} footprint {fw:.0f}x{fh:.0f} -> "
          f"crop {cw}x{ch}  headroom={headroom:.3f}  covers={covers}  "
          f"clipped={clipped}   {'OK' if ok else 'FAIL'}")
    return ok


def check_margin_not_reported_as_clipped():
    """Margin trimmed by the sensor edge is slack, not a clipped field."""
    # Push the footprint hard against the sensor corner: the margin cannot fit,
    # the footprint itself still does.
    A = true_affine(origin=(1.0, 1.0))
    roi = {"x_min": 0, "x_max": EVK4_W, "y_min": 0, "y_max": EVK4_H}
    _crop, _corners, clipped = map_evk4_window_to_orca(A, roi)
    ok = not clipped
    print(f"  {'margin clipped at edge':<28} footprint fully on sensor, margin "
          f"trimmed -> clipped={clipped}   {'OK' if ok else 'FAIL'}")
    return ok


def main():
    print(f"FOV matching check (crop margin = {EVK4_ORCA_CROP_MARGIN:.0%} per side)\n")
    results = [
        check_crop_margin(),
        check_margin_not_reported_as_clipped(),
        # The regression case: an ORCA cropped tighter than the EVK4 footprint.
        check_registration(TIGHT_ORCA, "tight ORCA crop"),
        # And the ordinary case, which must not regress.
        check_registration((2304, 2304), "full ORCA sensor"),
    ]
    print()
    if all(results):
        print("All checks passed.")
        return 0
    print("FAILED — see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
