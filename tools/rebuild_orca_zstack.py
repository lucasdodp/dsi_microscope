"""Rebuild an ORCA Z-stack's depth volumes + axial profiles from the per-plane raw TIFFs.

Every plane of an ORCA Z-stack is archived as its own ``<name>_raw_stack_zNNN.tif``,
so the final products are fully reconstructible offline — no camera or stage needed.
Use this to:

  * salvage a run whose summary save was interrupted (e.g. a crash mid-write left a
    partial ``_zstack_dsi.tif`` and no parameter log), or
  * regenerate the depth volume after a resume added the missing tail planes.

It reads each present plane, computes the average (widefield) and standard-deviation
(DSI) image, and writes the same files a completed run would: ``_zstack_dsi.tif``,
``_zstack_average.tif``, the axial-sectioning / axial-average CSV+PNG, and the
parameter log. Missing / truncated planes are skipped and reported; existing summary
files are overwritten with the full set. Nothing under ``raw_files`` is modified.

The plane axial positions are reconstructed from the scan geometry (the closed-loop
PI stage reproduces nominal positions to sub-nm), so pass the same focus / step / steps
used for the acquisition::

    python tools/rebuild_orca_zstack.py "D:\\2026-07-13\\differentsizes_orca_port1_day2" \\
        --focus 200 --step 0.2 --steps 71 --frames 200

``--frames`` is the frames-per-plane count; a plane TIFF with fewer pages is treated
as incomplete (so a truncated plane is skipped rather than silently shrinking the
volume). Omit it to accept any readable plane file.
"""

import argparse
import os
import sys

# Allow ``import core`` when run as a plain script from the repo tools/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import find_complete_planes, rebuild_zstack_from_raw  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="The acquisition folder (holds the raw_files subfolder).")
    ap.add_argument("--focus", type=float, required=True,
                    help="Objective focus / stack centre (µm), as used in the run.")
    ap.add_argument("--step", type=float, required=True, help="Z step size (µm).")
    ap.add_argument("--steps", type=int, required=True, help="Total number of planes in the stack.")
    ap.add_argument("--frames", type=int, default=None,
                    help="Frames per complete plane; planes with fewer are skipped as truncated.")
    ap.add_argument("--no-average", action="store_true",
                    help="Skip the average (widefield) volume + axial-average profile.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report which planes are present / missing, but write nothing.")
    args = ap.parse_args(argv)

    folder = os.path.normpath(args.folder)
    filename = os.path.basename(folder)
    raw_dir = os.path.join(folder, "raw_files")
    if not os.path.isdir(raw_dir):
        print(f"ERROR: no 'raw_files' subfolder in {folder}")
        return 2

    init_pos = args.focus - (args.step * args.steps / 2.0)
    z_positions = [init_pos + k * args.step for k in range(args.steps)]

    complete, missing = find_complete_planes(raw_dir, filename, args.steps, args.frames)
    print(f"Folder:   {folder}")
    print(f"Filename: {filename}")
    print(f"Planes:   {len(complete)}/{args.steps} complete"
          + (f"; missing {len(missing)}: {missing}" if missing else " (all present)"))
    if not complete:
        print("ERROR: no complete per-plane raw stacks found — nothing to rebuild.")
        return 2
    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    metadata = {
        "Rebuild": {
            "tool": "tools/rebuild_orca_zstack.py",
            "focus_um": args.focus,
            "step_um": args.step,
            "steps": args.steps,
            "frames_per_plane": args.frames if args.frames is not None else "any",
        }
    }

    n, missing = rebuild_zstack_from_raw(
        raw_dir, folder, filename, z_positions,
        expected_frames=args.frames,
        save_average=not args.no_average,
        metadata=metadata,
        status=lambda msg: print(f"  {msg}"),
    )
    print(f"\nDone: rebuilt {n} planes into {folder}"
          + (f"; {len(missing)} still missing: {missing}" if missing else " (complete)."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
