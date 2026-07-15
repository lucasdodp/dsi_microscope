"""Rebuild an EVK4 event Z-stack's depth volume + axial profile from the per-plane .raw streams.

The event counterpart of ``tools/rebuild_orca_zstack.py``. Every plane of an EVK4
Z-stack archives its raw event stream as ``<name>_events_zNNN.raw``, so the event
depth volume is fully reconstructible offline — though, unlike the ORCA rebuild,
this one needs the Prophesee Metavision SDK to decode the ``.raw`` files.

Use it to salvage a run whose summary save was interrupted, or to regenerate the
event volume after a resume added the missing tail planes. It reads each present
plane, accumulates its events into a 2D image (cropped to the ROI and optionally
cleaned), and writes ``_zstack_event.tif`` + the axial-sectioning CSV/PNG + the
parameter log. Missing planes are skipped and reported; existing summary files are
overwritten. Nothing under ``raw_files`` is modified.

Plane axial positions are reconstructed from the scan geometry, so pass the same
focus / step / steps used for the acquisition::

    python tools/rebuild_evk4_zstack.py "D:\\2026-07-10\\myscan" \\
        --focus 200 --step 0.2 --steps 71

By default the accumulated images are hot-pixel filtered and smoothed (matching the
acquisition defaults); pass --no-filter / --no-smooth to disable. Pass an ROI with
--roi-x/--roi-y/--roi-w/--roi-h to crop each plane to the same window the run used.
"""

import argparse
import os
import sys

# Allow ``import core`` / ``import hardware`` when run as a plain script from tools/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from config import EVK4_SENSOR_WIDTH, EVK4_SENSOR_HEIGHT  # noqa: E402
from core import (  # noqa: E402
    accumulate_event_frame, apply_smoothing, crop_to_roi, filter_crazy_pixels,
    save_axial_sectioning_plot, save_parameter_log, save_volume_tiff,
)


def _reconstruct_plane(raw_path, roi, do_filter, do_smooth):
    """Accumulate one plane's event image from its .raw (Metavision SDK required)."""
    from metavision_core.event_io import EventsIterator
    reader = EventsIterator(input_path=raw_path, delta_t=1000000)
    try:
        width, height = reader.get_size()
    except Exception:
        width, height = EVK4_SENSOR_WIDTH, EVK4_SENSOR_HEIGHT
    img = accumulate_event_frame(reader, width, height)
    img = crop_to_roi(img, roi)
    if do_filter:
        img = filter_crazy_pixels(img)
    if do_smooth:
        img = apply_smoothing(img)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="The acquisition folder (holds the raw_files subfolder).")
    ap.add_argument("--focus", type=float, required=True,
                    help="Objective focus / stack centre (µm), as used in the run.")
    ap.add_argument("--step", type=float, required=True, help="Z step size (µm).")
    ap.add_argument("--steps", type=int, required=True, help="Total number of planes in the stack.")
    ap.add_argument("--roi-x", type=int, default=None, help="ROI left (px).")
    ap.add_argument("--roi-y", type=int, default=None, help="ROI top (px).")
    ap.add_argument("--roi-w", type=int, default=None, help="ROI width (px).")
    ap.add_argument("--roi-h", type=int, default=None, help="ROI height (px).")
    ap.add_argument("--no-filter", action="store_true", help="Do not hot-pixel filter each plane.")
    ap.add_argument("--no-smooth", action="store_true", help="Do not smooth each plane.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report which planes are present / missing, but write nothing.")
    args = ap.parse_args(argv)

    folder = os.path.normpath(args.folder)
    filename = os.path.basename(folder)
    raw_dir = os.path.join(folder, "raw_files")
    if not os.path.isdir(raw_dir):
        print(f"ERROR: no 'raw_files' subfolder in {folder}")
        return 2

    roi = None
    if None not in (args.roi_x, args.roi_y, args.roi_w, args.roi_h):
        roi = {"x_min": args.roi_x, "y_min": args.roi_y,
               "x_max": args.roi_x + args.roi_w, "y_max": args.roi_y + args.roi_h}

    from core import find_complete_event_planes
    complete, missing = find_complete_event_planes(raw_dir, filename, args.steps)
    print(f"Folder:   {folder}")
    print(f"Filename: {filename}")
    print(f"Planes:   {len(complete)}/{args.steps} complete"
          + (f"; missing {len(missing)}: {missing}" if missing else " (all present)"))

    init_pos = args.focus - (args.step * args.steps / 2.0)
    # A plane counts for the rebuild if its .raw exists (even if the _xytp.mat that
    # marks a *clean finish* is absent — the raw is still the authoritative record).
    present = [k for k in range(args.steps)
               if os.path.exists(os.path.join(raw_dir, f"{filename}_events_z{k:03d}.raw"))]
    if not present:
        print("ERROR: no per-plane .raw event streams found — nothing to rebuild.")
        return 2
    if args.dry_run:
        print(f"Would rebuild {len(present)} planes: {present}")
        print("Dry run: no files written.")
        return 0

    event_volume, z_kept, skipped = [], [], []
    for k in present:
        raw_path = os.path.join(raw_dir, f"{filename}_events_z{k:03d}.raw")
        try:
            img = _reconstruct_plane(raw_path, roi, not args.no_filter, not args.no_smooth)
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"  plane {k}: skipped ({exc})")
            skipped.append(k)
            continue
        event_volume.append(img)
        z_kept.append(init_pos + k * args.step)
        print(f"  rebuilt plane {k + 1}/{args.steps} ({len(event_volume)} present)")

    if not event_volume:
        print("ERROR: no planes could be reconstructed.")
        return 2

    save_volume_tiff(np.array(event_volume, dtype=np.float32), folder, filename, "zstack_event")
    intensities = [float(np.mean(img)) for img in event_volume]
    save_axial_sectioning_plot(z_kept, intensities, folder, filename, "event")
    save_parameter_log(folder, filename, {
        "Rebuild": {
            "tool": "tools/rebuild_evk4_zstack.py",
            "focus_um": args.focus, "step_um": args.step, "steps": args.steps,
            "num_planes_saved": len(event_volume),
            "skipped_planes": ", ".join(str(s) for s in skipped) or "none",
            "roi": roi or "full frame",
        }
    })
    print(f"\nDone: rebuilt {len(event_volume)} planes into {folder}"
          + (f"; skipped {len(skipped)}: {skipped}" if skipped else " (complete)."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
