"""Backfill decoded event-stream files for already-acquired EVK4 ``.raw`` data.

The acquisition pipeline now saves an explicit per-event list ``<name>_xytp.mat``
(x, y, p, t) next to every Prophesee ``.raw`` log. This standalone tool creates
that same file for data acquired *before* the feature existed, by walking one or
more root folders, decoding each ``.raw`` with the Metavision SDK and writing the
compressed ``.mat`` via the shared ``core.save_event_stream`` (so the on-disk
result is byte-for-byte the format a fresh acquisition produces).

It is **resumable and idempotent**: a ``.raw`` whose ``_xytp.mat`` already exists
is skipped, so the run can be interrupted and restarted, and re-running it is a
no-op. Nothing is ever deleted or overwritten unless ``--overwrite`` is given.

Run (defaults to the two lab data drives)::

    python tools/backfill_event_streams.py                 # everything, both drives
    python tools/backfill_event_streams.py --limit 5       # first 5 only (smoke test)
    python tools/backfill_event_streams.py --dry-run       # list work, write nothing
    python tools/backfill_event_streams.py --workers 6     # parallelise across cores
    python tools/backfill_event_streams.py "D:\\2026-07-03"  # a specific subtree
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allow ``import core`` when run as a plain script from the repo tools/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_ROOTS = ["D:\\", r"C:\DSI Microscope Data"]
STREAM_SUFFIX = "_xytp.mat"


def stream_path_for(raw_path):
    """The ``_xytp.mat`` path that pairs with a given ``.raw`` file."""
    base = raw_path[:-4] if raw_path.lower().endswith(".raw") else raw_path
    return base + STREAM_SUFFIX


def find_raw_files(roots):
    """Yield every ``*.raw`` under the given roots (recursive, case-insensitive)."""
    for root in roots:
        root = os.path.normpath(root)
        if not os.path.isdir(root):
            print(f"  ! skipping missing path: {root}")
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith(".raw"):
                    yield os.path.join(dirpath, name)


def convert_one(raw_path, overwrite=False):
    """Decode one ``.raw`` and write its ``_xytp.mat``. Returns (status, info).

    status is one of ``"done"``, ``"skip"`` or ``"error"``. Runs in a worker
    process, so it imports the SDK/core lazily and never raises to the pool.
    """
    out_path = stream_path_for(raw_path)
    if os.path.exists(out_path) and not overwrite:
        return "skip", out_path
    try:
        from metavision_core.event_io import EventsIterator
        from core import save_event_stream

        out_dir = os.path.dirname(raw_path)
        # save_event_stream appends the suffix itself, so hand it the bare stem.
        stem = os.path.basename(raw_path)[:-4]
        iterator = EventsIterator(input_path=raw_path, delta_t=1000000)
        written = save_event_stream(iterator, out_dir, stem)
        return "done", written
    except Exception as exc:  # noqa: BLE001 — report, never crash the batch
        return "error", f"{raw_path}: {exc}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=DEFAULT_ROOTS,
                    help="Folders to scan (default: the two lab data drives).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N files that still need conversion.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (default 1 = sequential).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-decode even if the _xytp.mat already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be converted, but write nothing.")
    args = ap.parse_args(argv)

    roots = args.roots or DEFAULT_ROOTS
    print(f"Scanning: {roots}")
    all_raw = list(find_raw_files(roots))
    pending = [p for p in all_raw if args.overwrite or not os.path.exists(stream_path_for(p))]
    todo = pending[:args.limit] if args.limit is not None else pending

    print(f"Found {len(all_raw)} .raw files; {len(all_raw) - len(pending)} already have "
          f"a stream file; {len(todo)} to process"
          + (f" (limited to {args.limit} of {len(pending)})" if args.limit is not None else "")
          + ".")

    if args.dry_run:
        for p in todo[:50]:
            print(f"  would write: {stream_path_for(p)}")
        if len(todo) > 50:
            print(f"  ... and {len(todo) - 50} more")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    done = skipped = errors = 0
    bytes_written = 0
    t0 = time.time()

    def record(status, info):
        nonlocal done, skipped, errors, bytes_written
        if status == "done":
            done += 1
            try:
                bytes_written += os.path.getsize(info)
            except OSError:
                pass
        elif status == "skip":
            skipped += 1
        else:
            errors += 1
            print(f"  ERROR {info}")

    n = len(todo)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(convert_one, p, args.overwrite): p for p in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                status, info = fut.result()
                record(status, info)
                if i % 25 == 0 or i == n:
                    _progress(i, n, done, skipped, errors, bytes_written, t0)
    else:
        for i, p in enumerate(todo, 1):
            record(*convert_one(p, args.overwrite))
            if i % 10 == 0 or i == n:
                _progress(i, n, done, skipped, errors, bytes_written, t0)

    dt = time.time() - t0
    print(f"\nFinished in {dt/60:.1f} min: {done} written, {skipped} skipped, "
          f"{errors} errors, {bytes_written/1e9:.2f} GB written.")
    return 1 if errors else 0


def _progress(i, n, done, skipped, errors, bytes_written, t0):
    dt = time.time() - t0
    rate = i / dt if dt > 0 else 0
    eta = (n - i) / rate if rate > 0 else 0
    print(f"  [{i}/{n}] done={done} skip={skipped} err={errors} "
          f"{bytes_written/1e9:.2f}GB  {rate:.1f} files/s  ETA {eta/60:.1f} min")


if __name__ == "__main__":
    raise SystemExit(main())
