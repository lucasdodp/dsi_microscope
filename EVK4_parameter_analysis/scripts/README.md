# Analysis scripts

Python scripts that generate the figures. Each script has its data path and output
path as constants near the **top of the file** — edit those to point at a different
acquisition, then re-run.

## Linearity + overview (2026-07-01 bias sweep)

Run order (only `make_figures.py` is needed for the figures):

| Script | What it does | Reads | Writes |
|--------|--------------|-------|--------|
| `make_figures.py` | **Makes the two figures.** Reads event counts + FWHM/SNR/R² from the axial-profile CSVs and fits linear vs exponential. | the bias-sweep folder's `*_axial_profile_event.csv` | `linearity.png`, `SIMPLE_overview.png` |
| `proxy_check.py` | Sanity check: confirms the CSV event counts match a true raw-decode (to ~0.4%). | CSVs + `bias_counts.json` | prints a table |
| `decode_linearity.py` | (Optional, heavy) Decodes the raw `.raw` files for true ON/OFF event counts + time spans. Resumable — caches per bias. **Memory-lean/chunked**, so it survives the big saturated files. | the `.raw` files | `bias_counts.json` |
| `evt3_chunked.py` | The EVT3 raw-event decoder used by `decode_linearity.py` (chunked, low memory). Not run directly. | — | — |

**To reuse on a new bias sweep:** edit just `ROOT` (the sweep folder) at the top of
`make_figures.py`, then `python make_figures.py`. **Figures are always written into the
data folder** (`OUT = os.path.dirname(ROOT)`) so they stay with the data — you don't need
to set an output path. If you want true decoded counts too, edit `ROOT` in
`decode_linearity.py` and run it first.

Notes:
- `make_figures.py` needs **no** raw decoding — it uses the per-plane event counts the
  acquisition program already wrote into the CSVs (validated == true counts).
- The event count per bias = `sum(mean_intensity) x pixels`; pixel count `NPIX` is set
  for the full 1280x720 sensor — change it if you crop the ROI.
- The SNR panel uses the per-acquisition axial peak/noise (one acquisition per bias).
  The 20-repeat mean/std SNR needs repeated acquisitions (see `analyze.py`).

## General toolkit
| Script | What it does |
|--------|--------------|
| `analyze.py` | Turn-key `overview` / `duration` analysis over a folder of acquisitions (groups repeats by bias). Run `python analyze.py overview "<folder>"`. |
| `evt3.py` | General EVT3 decoder (full/polarity/rate/counts) used by `analyze.py`. `read_evt3_words` here returns `(words, header)`. |
