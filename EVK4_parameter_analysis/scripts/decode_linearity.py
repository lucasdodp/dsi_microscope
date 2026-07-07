"""Decode the 2026-07-01 bias sweep -> total ON/OFF events + spans per bias.
Resumable: caches after each bias, skips biases already in the cache."""
import numpy as np, evt3_chunked as evt3, glob, os, re, time, json

ROOT = r'C:\Stage Institut Fresnel Local Files\DSI Microscope Data\2026-07-01\bias_tests_orange_singleline_18v_2500hz'
CACHE = os.path.join(os.path.dirname(__file__), 'bias_counts.json')

results = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

dirs = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))],
              key=lambda d: os.path.getsize(os.path.join(ROOT, d,
                  sorted(glob.glob(os.path.join(ROOT, d, '*_events_z000.raw')))[0]))
                  if glob.glob(os.path.join(ROOT, d, '*_events_z000.raw')) else 0)

t0 = time.time()
for d in dirs:
    if d in results:
        print(f"skip {d} (cached)"); continue
    bias = int(re.match(r'(-?\d+)_bias', d).group(1))
    files = sorted(glob.glob(os.path.join(ROOT, d, '*_events_z*.raw')))
    files = [f for f in files if not f.endswith('.tmp_index')]
    on = off = 0
    spans = []
    for fp in files:
        w = evt3.read_evt3_words(fp)
        no, nf, tmn, tmx = evt3.decode_counts(w)
        on += no; off += nf; spans.append((tmx - tmn) / 1e6)
        del w
    results[d] = dict(bias=bias, n_planes=len(files), on=on, off=off,
                      span_med=float(np.median(spans)), span_max=float(np.max(spans)))
    json.dump(results, open(CACHE, 'w'), indent=1)
    tot = on + off
    print(f"[{time.time()-t0:6.0f}s] {d:16s} bias={bias:3d} planes={len(files)} "
          f"tot={tot/1e6:9.2f}M ON/OFF={on/max(off,1):.2f} span_med={np.median(spans):.2f}s",
          flush=True)

print("done in", round(time.time()-t0), "s")
