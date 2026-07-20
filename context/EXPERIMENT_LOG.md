# Experiment & Data-Treatment Log

A simple running log of acquisitions and how their data was treated. **Newest entry on top.**
Update this file every time we do a data treatment. Fields marked `_(fill in)_` are to be
completed by hand from real-life measurements.

<!--
TEMPLATE — copy this block for a new entry:

## YYYY-MM-DD — <short title>

**Conditions**
- Sample: _(fill in)_
- Light power at sample: _(fill in)_
- Illumination (AWG): _(fill in — e.g. CH1 2500 Hz, 18 Vpp, square)_
- Camera settings: _(bias, duration, ROI, # planes, step)_
- Notes / what happened during acquisition: _(fill in)_
- Anything that failed: _(fill in)_

**Data treatment**
- <what we computed / how, in a sentence or two>

**Main files generated**
- <paths>
-->

---

## 2026-07-17 — bias_on × bias_off sweep: **USB2 corrupted the data**, and what the biases actually do

**Conditions**
- Sample: _(fill in)_
- Light power at sample: _(fill in)_
- Illumination (AWG): CH1 square, **1500 Hz**, 18 Vpp. CH2 OFF (2000 Hz, 18 Vpp, not output).
- Camera settings: EVK4 `bias_fo = 40`, `bias_hpf = 0`, **0.5 s/plane**, full sensor (1280×720). PI stage: target focus 200 µm, 0.5 µm step, 21 planes (25 for `test6`).
- Exploratory runs in `C:\DSI Microscope Data\2026-07-17\bias_on_off_sweep\`: `test1`–`test4` (`on=0, off=0`, four repeats), `test5` (`on=40, off=40`), `test6` (`on=-50, off=0`).
- Notes: the first `0_bias_hpf_set1` batch attempt ran with the **laser shutter closed** (discovered after ~20 planes) — those numbers are a dark measurement, not signal. The event camera was **running on USB2 all day** (`[HAL][WARNING] Your EVK camera isn't connected in USB3`), and also dropped off the bus twice mid-batch.
- Anything that failed: see Result 1 — most of the day's data is compromised by the USB link, not by the settings.

**Data treatment**
- Per-plane `.raw` decoded with the Metavision SDK; events counted directly and split by polarity (`p`). Crucially, event rate was recomputed against **live recording time** (span between first and last timestamp, minus dead air) rather than nominal `acqu_time` — this is what separated a link artefact from a physical result.
- Dead air per plane measured by histogramming event timestamps in 5 ms bins and summing empty bins.
- **Colormaps of the 30-cell batch sweep** (`threshold (bias_on = bias_off)` × `bias_hpf`), same FWHM + event-count pipeline as 07-15/07-16: per-acquisition `*_axial_profile_event.csv` → FWHM from the Gaussian-fit comment, total events = `sum(mean_intensity) × 1280×720`; data-driven dead-cell zeroing (4 cells: 90/100, 115/100, 140/75, 140/100 read exactly zero events → zeroed, FWHM there is a degenerate fit). Bias values read from each folder's `*_parameters.txt` (all 30 confirm `bias_on == bias_off`; the script asserts it). USB2-suspect cells red-hatched from a byte-rate screen (>30 MB/s) plus measured dead air; only the four `threshold = 10` cells breach, and the single measured stall (`10/50`) loses 0.25 % — negligible for the totals, so the colormaps stand as measured.
- **Result — clean monotonic 2-D surface.** Event count peaks at `threshold = 10, hpf = 0` (**183.7 M**) and falls monotonically along BOTH axes (raising the threshold divides events by ~10 per ~55 units; raising `bias_hpf` collapses them as on the previous days), bottoming to zero by `hpf = 100` for `thr ≥ 90` and by `hpf = 75` for `thr = 140`. FWHM narrows from ~3.3 µm (thr 10) toward the sampling limit as either knob rises — but this tracks the event collapse, so it is **signal-starvation, not better sectioning** (same caveat as 07-15/07-16). Usable window: low-to-mid threshold with `hpf ≤ 50` on a healthy event count.

**Results**
- **Result 1 — the "double-peaked Gaussian" is USB2, not the sample.** `test1`–`test4` all show a single Gaussian split by a dip *at focus*. The dip planes are the only ones with **stalls**: `test1` z011/z012/z013 have **85 / 330 / 255 ms of complete dead air** (zero events — impossible for a real sensor at these biases); all other 18 planes have exactly 0 ms. z012 recorded for only 170 of its 500 ms. **Dividing by live time collapses the double peak into one clean Gaussian peaking at z = 200.74 µm — exactly where the dip was.** The apparent peaks wandered ~2 µm between test1–test4 because stall onset is a chaotic buffer-overflow threshold, not a physical position. The `Evt3 InvalidVectBase` warnings are the same root cause.
- **Result 2 — the rule is the byte rate, not the warning.** USB2 practical ceiling ≈ 38–45 MB/s. `test1` peaks at 21.8 MB / 0.5 s = **43.6 MB/s → stalls**. `test5` peaks at 27 MB/s → no stalls. The 2 s/plane batch rows ran ~18 MB/s and were unaffected. **Under ~30 MB/s is safe; near 38+ MB/s loses planes silently.** The manual (§3.1) states USB 3.0 SuperSpeed is *required* for the necessary bandwidth and power.
- **Result 3 — `bias_on` and `bias_off` should be set EQUAL.** The EVK4 manual (p.13, `metavision_platform_info`) shows **every default bias = 0** — they are relative offsets from a factory-balanced calibration, *not* absolute values with differing zero points. Measured ON:OFF ratio **at the peak plane** confirms it: `0/0` → **1.04**, `40/40` → **0.98** (both balanced), but `-50/0` → **7.29** (badly imbalanced, only ON moved off its default). The spin-box limits (`bias_on -85..140`, `bias_off -35..190`) span 225 each and are offset by 50, which *looks* like it implies a 50 offset between them — **it does not**; that arithmetic coincidence is a trap.
- **Result 4 — higher equal biases dramatically improve sectioning.** peak/floor contrast: `0/0` → **2.0×**, `-50/0` → **1.2×**, `40/40` → **93.5×** (peak 6.58 Me/s vs floor 0.07 Me/s). High biases suppress the out-of-focus background while the in-focus signal survives — and they cut the data rate below the USB2 ceiling as a side effect. This retrospectively supports the `bias_on = bias_off = 40` used on 07-15/07-16.
- **Result 5 — ON:OFF ratio is a signal-vs-noise diagnostic.** At `0/0` the *off-focus* plane reads ON:OFF = 0.09 (noise is OFF-dominated) while the *in-focus* plane reads 1.04 (speckle is symmetric). **ON:OFF ≈ 1 means you are looking at speckle; far from 1 means you are looking at noise.** Measuring the wrong plane inverts the conclusion — beware.
- **Result 6 — direction of the knobs (measured):** lower = more sensitive, for both. `bias_on` 0 → -50 raised ON events 4.4×; 0 → +40 cut them 19×. The two are **coupled**: `test1` and `test6` both have `off=0`, yet OFF fell 8.13 → 0.43 Me/s when `bias_on` changed — a sensitive ON fires first and resets the pixel, starving OFF.
- **Caveat:** the ERC cap is 20 Me/s (`config.py`), and the live-time-corrected true peak at `0/0` is **21.6 Me/s** — i.e. at low biases the peak is clipped by the ERC *before* USB2 touches it. Low-bias rows are doubly unreliable.

**Software changes made the same day** (see git log)
- `_xytp.mat` no longer written during acquisition (2 full decode passes + gzip of ~13 B/event dominated per-plane time); generate offline with `tools/backfill_event_streams.py`.
- Batch queue now persists in the session/preset (`evk4_queue`) — a 30-row sweep survives a restart.
- EVK4 reconnect budget 3×5 s → **24×5 s = 2 min**, so a USB re-enumeration (10–30 s on Windows) recovers unattended instead of pausing for a manual Resume.
- Bias application isolated per-bias and made non-fatal: a value the SDK rejects no longer tears down the live feed and orphans the USB handle (which presented as "Live View won't connect").

**Main files generated**
- `D:\2026-07-17\bias_on_off_hpf_analysis\` (copied to `D:\Files for the report\07c_2026-07-17_bias_on_off_hpf`): `analyze_bias_on_off_hpf_sweep.py`, `combined_heatmap_bias_on_off_hpf.png` (primary), `fwhm_heatmap_bias_on_off_hpf.png`, `event_count_heatmap_bias_on_off_hpf.png`, `combined_3d_bias_on_off_hpf.png`, `results_bias_on_off_hpf.csv` (full grid, `source` marks 26 `measured` / 4 `no_events`), `check_usb2_stalls.py` + `usb2_stall_check.csv`
- `C:\DSI Microscope Data\2026-07-17\bias_on_off_sweep\test1..test6\` (raw + axial profiles; **test1–test4 profiles are stall-corrupted — recompute against live time before use**)
- `presets/bias_hpf_sweep_6sets.json` — 30-row queue, `bias_on = bias_off ∈ {10,35,60,90,115,140}` × `bias_hpf ∈ {0,25,50,75,100}`, highest-bias set first.

**Open / next**
- **USB2 is permanent — the lab computer has no USB3 ports**, so the camera runs on USB2 every day (this warning is not a fault). The only mitigation is to keep the peak per-plane byte rate under ~30 MB/s: prefer higher biases (fewer events) and/or longer acquisition time per plane (same events over more seconds → lower MB/s). Treat high-event-rate cells (low threshold, low `bias_hpf`, short exposure) as lower bounds, and stop framing any earlier day as a "USB3 baseline" — there is none.
- Consider a stall check in `orchestrator._capture_plane_event`: compare each plane's live time to `acqu_time` and raise on significant dead air, so the existing retry loop rejects corrupted planes automatically instead of silently keeping them.
- A dark noise floor is **Z-independent**, so it needs no Z-stack: `num_steps = 1`, shutter closed, run the same 30-row queue ≈ 5 min → per-bias noise floor for SNR.

---

## 2026-07-16 — bias_fo × bias_hpf 2-D sweep at **500 Hz** (completes the 07-15 pair)

**Conditions**
- Sample: **fluorescent marker (highlighter pen) on a glass substrate** — not a biological sample. **Same sample and same stage position as 2026-07-15**, never moved between the three frequency runs, so 500 / 1500 / 2500 Hz differ *only* in AWG frequency. (Neither day's `Context.txt` recorded the sample; noted here.)
- Light power at sample: **0.30 mW** (same as both 07-15 runs).
- Illumination (AWG): CH1 square, **500 Hz**, 18 Vpp. CH2 OFF (2000 Hz, 18 Vpp, not output).
- Camera settings: EVK4 bias_on = bias_off = 40, **2 s/plane**, full sensor (1280×720). PI stage: target focus 200 µm, **35 planes, 0.5 µm step** (z = 191.2 → 208.3 µm). ORCA connected (200 ms × 200) but this analysis uses the event channel only.
- Grid: `bias_fo` ∈ [-35, -10, 15, 35, 55] × `bias_hpf` ∈ [0, 25, 50, 75, 100, 120] = **30 cells, all 30 physically acquired** (unlike 1500 Hz).
- Anything that failed: nothing. A stray `teste/` folder in the sweep directory is not part of the grid and is skipped by the script.

**Data treatment**
- Same pipeline as 2026-07-15: per-acquisition `*_axial_profile_event.csv` → FWHM from the Gaussian-fit comment, total events = `sum(mean_intensity) × 1280×720`.
- **Correction to the 07-15 script (matters here, not there):** it hardcoded "`bias_hpf = 120` has no events" and forced that one column to zero. At 500 Hz the signal dies a step earlier — **`bias_hpf = 100` is also exactly zero** (`sum(mean_intensity)` literally `0.0` in all 5 cells). The rule was made **data-driven** (any cell measuring zero events is zeroed), which is the rule the 07-15 script's own docstring already justified; the old `hpf = 120` forcing is kept for cells that were never acquired. **Verified backward-compatible: re-running it on the 07-15 data reproduces `results_1500hz.csv` and `results_2500hz.csv` identically.**
- Why it mattered: on an all-zero profile the fit returns **`amp = -0.000000`** (zero-amplitude Gaussian — no peak) with `mu` pinned to 191.24, the first plane of the scan, versus ≈200 µm (the focus) for cells with signal. Its reported `fwhm_um ≈ 10 µm` is meaningless; unguarded it would have been plotted as a real 10 µm slice and would have crushed the colour scale (true range 1.54–1.88 µm). Note 10 µm *does* fit inside the 17.0 µm z-range — the fit is meaningless because there is no peak, not because the number is impossible.
- **Result 1 — 500 Hz yield collapses, confirming 1500 Hz directly:** peak **3.0 M** events (hpf = 0) vs **126 M** at 1500 Hz and **92 M** at 2500 Hz → **≈ 42× below 1500 Hz**. Because sample/position/power were held fixed, this is a controlled comparison, and it independently reproduces the 2026-07-07 decorrelation result (~40×) from a separate measurement.
- **Result 2 — `bias_hpf` is NOT independent of AWG frequency (new):** as a fraction of each frequency's own peak, `hpf = 75` keeps ~40 % at 1500 Hz, ~31 % at 2500 Hz, but only **~6 % at 500 Hz**; `hpf = 100` is alive at 1500 (~1.2 M) and 2500 Hz (~0.17 M) but **exactly zero** at 500 Hz. Physically sensible: slower speckle → fluctuations at lower temporal frequencies → the same high-pass removes more of them. **`bias_hpf ≲ 60` is a 1500 Hz statement, not a universal one.** At 500 Hz the usable window is `hpf ≤ 50`. Re-tune `hpf` whenever the AWG frequency changes.
- **Result 3 — the thin FWHM at 500 Hz is starvation, not resolution:** cells with signal read **1.54–1.88 µm**, thinner than 1500 Hz (2.5–3.4) and 2500 Hz (2.3–3.3), but on ~40× fewer events; within the sweep FWHM falls 1.88 → 1.54 µm exactly as events fall 3.0 M → 0.19 M. Also, at 0.5 µm z-step a 1.54 µm FWHM spans only ~3 planes — near the sampling limit, so the differences between thin cells are not meaningful. **Do not read 500 Hz as the sharpest setting.**
- **Result 4 — `bias_fo` stays a weak knob:** ≤ ~20 % effect on event count, `bias_fo = -35` consistently poorest (2.42 M vs ~3.03 M at hpf = 0) — same ranking as 1500 Hz. `hpf` dominates.
- **Caveat:** the FWHM here is measured on a marker layer, not on a point-like object, so it is a relative sectioning metric across settings rather than an absolute PSF width. That is fine for this purpose — every cell is compared on the same sample.

**Main files generated** (in `D:\2026-07-16\bias_fo_hpf_500hz_analysis\`, copied to `07b_2026-07-16_biasfo_hpf_500hz`)
- `analyze_bias_fo_hpf_sweep.py` (now takes frequency **and** base folder: `python analyze_bias_fo_hpf_sweep.py 1500 D:/2026-07-15` reproduces 07-15), `README.md`
- `combined_heatmap_500hz.png` (primary), `fwhm_heatmap_500hz.png`, `event_count_heatmap_500hz.png`, `combined_3d_500hz.png`
- `results_500hz.csv` — full grid, `source` column marks each cell `measured` (20) / `no_events` (10)

---

## 2026-07-15 — bias_fo × bias_hpf 2-D sweep at 1500 and 2500 Hz

**Conditions**
- Sample: **fluorescent marker (highlighter pen) on a glass substrate**. Same sample and stage position as the 07-16 500 Hz run — the stage was not moved across any of the three frequencies.
- Light power at sample: **0.30 mW** at 1500 Hz and **0.30 mW** at 2500 Hz (`Power.txt`).
- Illumination (AWG): CH1 square, **1500 Hz** and **2500 Hz**, 18 Vpp.
- Camera settings: EVK4 bias_on = bias_off = 40, 2 s/plane, full sensor. PI stage: target focus 200 µm, **35 planes, 0.5 µm step**.
- Grid: `bias_fo` ∈ [-35, -10, 15, 35, 55] × `bias_hpf` ∈ [0, 25, 50, 75, 100, 120].
- Anything that failed: at **1500 Hz only `bias_fo = -35` was acquired at `hpf = 120`** (26 of 30 cells); the rest of that column is filled by the forced-zero rule. 2500 Hz has all 30.

**Data treatment**
- Per-acquisition `*_axial_profile_event.csv` → FWHM (Gaussian-fit comment) + total events (`sum(mean_intensity) × NPIX`, `NPIX = 1280×720`). The event proxy was validated against a true raw EVT3 decode to ~0.4 % in the 2026-07-01 study, so no raw decoding.
- `bias_hpf = 120` yields zero events at both frequencies → that column zeroed (FWHM undefined there). See the 07-16 entry for the generalisation of this rule.
- **Result — `hpf` dominates, `fo` barely matters.** Events highest at hpf = 0–25 (~100–126 M at 1500 Hz; ~68–92 M at 2500 Hz), falling to ~50 M / ~28 M at hpf = 75 and ~1.2 M / ~0.17 M at hpf = 100. The heat maps are essentially horizontally striped. `bias_fo = -35` is consistently the poorest.
- **Result — FWHM narrows as hpf rises** (≈3.0–3.4 → 1.5–1.7 µm at 1500 Hz) but tracks the collapse in events → signal-starvation artefact, not better sectioning. Usable window: low-to-mid hpf (0–50), FWHM ≈2.5–3 µm on a healthy event count.

**Main files generated** (in `D:\2026-07-15\`, copied to `07a_2026-07-15_biasfo_hpf_2Dsweep`)
- `bias_fo_hpf_1500hz_analysis/` and `bias_fo_hpf_2500hz_analysis/`, each with `analyze_bias_fo_hpf_sweep.py`, `README.md`, `results_<freq>hz.csv`, `combined_heatmap_<freq>hz.png`, `fwhm_heatmap_<freq>hz.png`, `event_count_heatmap_<freq>hz.png`, `combined_3d_<freq>hz.png`

---

## 2026-07-10 — Is the EVK4 linear in **log(intensity)**? (sparse different-size beads, two ports)

**Conditions**
- Sample: fluorescent beads of clearly **different sizes**, sparse and individually
  resolvable (+ one dense close-packed clump, excluded). The size spread is the brightness
  axis: within the matched field the beads span **40× (port 2) / 95× (ports 1&3)**.
- Light power at sample: **1.98 mW** (`Power.txt`).
- Illumination (AWG): CH1 square **1500 Hz**, 18 Vpp (CH2 off).
- Camera settings: ORCA 20 ms × 200 frames/plane, full sensor; EVK4 bias fo=5/hpf=40/on=off=40,
  5 s/plane, full sensor; both **71 planes, 0.2 µm step**.
- Two configs (`Notes.txt`): **A = port 2** (80 % EVK4 / 20 % ORCA beamsplitter:
  `differentsizes_orca` ↔ `differentsizes_evk4`); **B = ports 1&3** (100 % each, the "best":
  `differentsizes_orca_port1` ↔ `differentsizes_evk4_port3`).

**Data treatment**
- Question (tutor): is the event-camera response linear in **log(intensity)**? Refines the
  2026-07-08/09 finding (EVK4 sub-linear in intensity) by testing the logarithmic model directly.
- Per-camera z-MIP → register EVK4→ORCA (masked-NCC). Both configs **θ=317°, scale=0.745**,
  matching pixel-pitch ratio 4.86/6.5=0.748 to <0.4 % (magnification equal on both ports); NCC
  0.71 (A) / 0.86 (B). Segmented each bead on the ORCA average MIP; matched per-bead integrated
  photometry (ORCA avg = true brightness, ORCA DSI std, EVK4 events) through mapped apertures.
  Bead selection verified against a registered overlay (`overlay_check.png`): every resolvable
  bead inside the EVK4 field is used (incl. the brightest — cap raised to 15k px after that check;
  the honeycomb monolayer stays outside the matched window).
- **Reference OK — ORCA linear:** DSI std vs widefield average per bead **R²=1.00** (both configs),
  slope~1; brightness axis unsaturated (raw peak ≤8500/65535).
- **Result — EVK4 compressive/log-like, NOT linear in intensity:** integrated events vs brightness:
  config A (n=9, 91×) linear R²=0.77, log R²=0.72, **power g=0.44** (r_log=0.85); config B (n=10, 182×)
  linear R²=0.83 < **log R²=0.92** (r=0.96, power g=0.53); pooled (z-scored, n=19) linear R²=0.80,
  log R²=0.82 (r=0.91). A **sub-linear power law brightness^0.44–0.53** is the most consistent
  descriptor in both configs; in the clean 100 % config the compression is explicitly **logarithmic**
  (linear-axes plot visibly saturates: two brightest beads give ~equal events). Config A is noisier —
  one bright bead fires anomalously **few** events (1.1e6 a.u.→2900 ev; likely EVK4 rate saturation) —
  so there log≈linear. Not separable log vs mild power law over ~2 decades, n≈9, but both = strongly
  compressive, never proportional (90–180× brighter → only ~5–15× more events).
- **Mechanism — per-pixel flat:** peak-vs-peak EVK4 vs ORCA **R²≈0**; the compression is an **area
  effect** (brighter beads are larger → more fluctuating pixels), i.e. the sensor fires on
  log-intensity *changes* (contrast), not absolute level. Reproduces 2026-07-02.
- **Verdict:** firmly **non-linear in intensity** (compressive, exponent ≈0.5); in the clean data
  well described as **linear in log(intensity)** (r=0.96) — so **yes, approximately**, as expected
  from the log front-end — but the effective log transfer is realised spatially (in-focus area), not
  per-pixel.
- **Caveats:** small n (EVK4 FOV is a small crop of the ORCA field → only overlapping beads match);
  the port-2 anomalous-few-events bead makes config A ambiguous; config B (100 %, wider range) is the
  more reliable.

**Main files generated** (in `D:\2026-07-10\linearity_analysis\`)
- `EVK4_log_linearity_report.pdf` (standalone report); `README.md`
- `figures/fig1_fields.png`, `fig2_transfer.png` (the result), `fig3_models.png`,
  `reg_overlay_*.png`, `overlay_check.png` (bead-selection diagnostic)
- `beads.csv` (19 beads), `registration.json`, `fit_summary.json`, `scripts/step1…step7`

---

## 2026-07-09 — EVK4 intensity linearity vs ORCA (repeat, single 1 µm field)

**Conditions**
- Sample: one "1 µm" fluorescent-bead field (dual-camera via beam-splitter). Only one bead size this time.
- Light power at sample: **0.76 mW**.
- Illumination (AWG): CH1 square **1500 Hz**, 18 Vpp (same optimum). CH2 also ON (2000 Hz, 18 Vpp).
- Camera settings: ORCA 20 ms × 200 frames/plane, full sensor; EVK4 bias fo=5/hpf=40/on=off=40, **2 s/plane**; both 51 planes, 0.2 µm step.
- Notes: per-camera z-MIP used (ports not perfectly co-focal).

**Data treatment** (same pipeline as 2026-07-08)
- Registered the single EVK4↔ORCA pair (coverage-map + void pattern; **point-ICP avoided** — the periodic bead lattice aliases it). Fine transform maximised bead-match fraction.
- **Registration changed vs 07-08: mirror + 81° rotation, scale 0.727** (07-08 was 317°, no mirror) → the EVK4 was physically re-mounted between sessions; each session must be registered independently. Bead-match 80%.
- Matched per-bead photometry on **298 beads**; brightness axis = natural bead-to-bead variation (8.5× spread, 5th–95th pct).
- **Result 1 — ORCA linear (reference):** DSI std vs widefield average per bead, **r = 1.00, R² = 0.997**, slope 0.79. Validates method + brightness axis.
- **Result 2 — EVK4 NOT linear (compressed):** events vs ORCA true brightness, **slope 0.33, r = 0.51**. Event signal rises with brightness but strongly sub-linearly (a decade of brightness → ~2× events). Reproduces & sharpens 07-08 (cleaner single field; per-field slopes across both days all <1).
- **Result 3 — bead size:** this "1 µm" sample measures **5.85 µm** (lattice autocorrelation), same ~6 µm population as 07-08 → label still wrong.
- **Caveat:** exact exponent (0.33) is indicative (residual focus/registration scatter); direction (sub-linear) is firm. Definitive test still needs a controlled brightness sweep on sparse isolated beads.

**Main files generated** (in `D:\2026-07-09\linearity_analysis\`)
- `linearity_report.tex`, `README.md`
- `figures/fig1_fields.png`, `fig2_linearity.png`, `reg_overlay.png`
- `beads.csv` (298 beads), `registration.json`, `bead_size.json`, `scripts/` (d0709_step1→step6)

---

## 2026-07-08 — EVK4 intensity linearity vs ORCA ground truth (1 / 2 / 3.7 µm beads)

**Conditions**
- Sample: fluorescent beads, three nominal diameters (1, 2, 3.7 µm), close-packed layers.
- Light power at sample: **0.88 mW** (from `Power.txt`).
- Illumination (AWG): CH1 = square, **1500 Hz**, 18 Vpp (the 2026-07-07 event-yield optimum). Identical for all six stacks.
- Camera settings: **dual-camera, same field via beam-splitter.** ORCA 50 ms × 200 frames/plane, full sensor. EVK4 bias fo=5/hpf=40/on=off=40, 5 s/plane, full sensor. Both: 41 planes, 0.2 µm step. Six z-stacks (orca + evk4 × three sizes).
- Notes: the two optical ports are not exactly co-focal → used a z-MIP per camera so each bead is measured at its own best-focus plane.
- Anything that failed: 2 µm cross-camera registration was weak (NCC 0.30) — its per-bead correlation is unreliable.

**Data treatment**
- Question (tutor): is the event camera linear in intensity? Use the ORCA (linear sCMOS) as ground truth on the *same* beads.
- **Registration ("cropping" done computationally):** matched the two cameras via coverage-map masked NCC over mirror/rotation/scale. One rigid transform fits all three fields: EVK4→ORCA = **rot 317°, scale 0.743**. Scale matches the pixel-pitch ratio 4.86/6.5 = 0.748 to <1% (independent physics check that magnification is equal on both ports).
- Matched per-bead photometry on **569 beads**: background-subtracted integrated signal in ORCA average (true brightness), ORCA DSI std, EVK4 events, through scale-matched apertures.
- **Result 1 — ORCA is linear (reference):** DSI std vs widefield average per bead, Pearson **r = 0.90** (per-sample R² 0.91–0.98), slope ~0.8. Valid ground truth.
- **Result 2 — EVK4 is NOT linear in intensity:** events vs ORCA true brightness pooled **r = −0.12, slope ≈ 0** (a 10× brighter bead gives ~the same events). Per sample vs ORCA-DSI: r = 0.27 / 0.11 / 0.73 (1/2/3.7 µm); only the largest, best-registered field shows a real but **sub-linear** relation (slope 0.73). Confirms the 2026-07-02 "compressed/log-like" finding, now against a proper linear reference.
- **Result 3 — the nominal bead sizes are WRONG (all three samples are the same size):** measured the close-packed lattice spacing (nearest-neighbour = diameter, radial autocorrelation over ~16 patches/sample). Diameters: 1µm→35.0±1.0px, 2µm→37.0±1.5px, 3.7µm→36.5±0.5px = ratio **1.00 : 1.06 : 1.04** (identical within ~6%), apparent ~5.9µm at nominal 40× (0.1625µm/px). So the "1/2/3.7µm" labels do not reflect a 1:2:3.7 ratio — likely mislabelled vials / same stock, or true magnification ≠ 40×. Consequence: the dataset does not span a range of sizes, so median per-bead signal is (correctly) non-monotonic in nominal size; the per-bead brightness test is the well-posed one.
- **Caveat:** some scatter in Result 2 is experimental (focal-plane offset, ~5 px registration residual, poor 2 µm match); correlation tracks registration quality, so the intrinsic relation may be a bit tighter — but still clearly compressive, never proportional.
- **Follow-up recommended:** sparse isolated beads; one-time grid/graticule cross-camera calibration; sweep true brightness directly (laser power / ORCA exposure) to trace the transfer curve and its saturation knee.

**Main files generated** (in the data folder `D:\2026-07-08\linearity_analysis\`)
- `linearity_report.tex` — full LaTeX report; `README.md` — summary
- `figures/fig1_fields.png` … `fig4_aggregate.png`, `fig5_beadsize.png`/`fig6_beadsize_zoom.png` (bead size), plus `joint_verify.png`, `match_check.png`
- `beads.csv` (569 beads), `joint_registration.json`, `bead_size.json`, `scripts/` (step1→step9)

---

## 2026-07-07 — LC decorrelation × high-pass: event yield vs AWG frequency

**Conditions**
- Sample: fluorescent marker _(confirm)_
- Light power at sample: **0.51 mW**
- Illumination (AWG): CH1 = 18 Vpp, **frequency swept 500 / 1000 / 1500 / 2000 / 2500 Hz**
- Camera settings: EVK4, bias_fo = 5, bias_on/off = 40, 5 s/plane; **bias_hpf swept 20→110** at each frequency. (Plane count inconsistent: 61 for 500/1500 Hz, 25 for the others — focus confirmed centered in all.)
- Notes / what happened during acquisition: _(fill in)_
- Anything that failed: _(fill in)_

**Data treatment**
- For each AWG frequency, plotted the in-focus event rate (peak of the axial mean event count — normalises out plane count) vs bias_hpf.
- **Result 1 — event yield peaks at ~1500 Hz** (at hpf=20: 500 Hz → 0.40, 1000 → 6.2, **1500 → 16.4**, 2000 → 10.3, 2500 → 7.5 ev/px). There is an optimal LC drive frequency for event generation.
- **Result 2 — the high-pass cut-off barely shifts** with frequency (half-max ≈ 62–69, full collapse by hpf ≈ 100 at every frequency).
- **Interpretation:** over 500–2500 Hz the AWG frequency mainly tunes the *strength* of the speckle decorrelation (peaked ~1500 Hz), not its *rate* (cut-off ≈ constant). To confirm, measure the decorrelation time directly (ORCA autocorrelation).

**Main files generated** (in the data folder `D:\2026-07-07\`)
- `decorrelation_bandpass.png` — event rate vs hpf (per frequency) + yield vs AWG frequency
- `make_decorrelation_figures.py` — the analysis script

---

## 2026-07-03 — Band-pass biases: events vs high-pass / low-pass (fluorescent marker)

**Conditions**
- Sample: fluorescent marker (uniform) _(confirm)_
- Light power at sample: **0.43 mW** (from `Power.txt`)
- Illumination (AWG): CH1 = 2500 Hz, 18 Vpp (single decorrelation — AWG frequency NOT yet swept)
- Camera settings: EVK4, bias_on/off = 40, 5 s, 71 planes. Two 1-D sweeps: **bias_hpf** 0→120 (fo=0), **bias_fo** −35→55 (hpf=40).
- Notes / what happened during acquisition: _(fill in)_
- Anything that failed: _(fill in)_

**Data treatment**
- Question (tutor): how does the event count depend on the high-pass / low-pass biases (temporal band-pass)?
- Counted total events per setting (axial CSV proxy). Not saturated (~2.4 Mev/s ≪ ceiling).
- **Result — clean band-pass behaviour:** raising **bias_hpf** (high-pass) is flat to ~40 then collapses (866 M → ~0 by 110–120) as the rising low-freq cutoff filters out the speckle; raising **bias_fo** (low-pass) rises then plateaus (372 M → ~860 M above fo≈5) once the front-end bandwidth clears the speckle. Useful band ≈ **bias_fo ≳ 5 and bias_hpf ≲ 60**. Bonus: higher hpf also thins the section (FWHM 4.8 → 3.3 µm) until signal starves; fo barely changes FWHM.
- **Not done yet:** repeat vs LC decorrelation (AWG frequency) — the band-pass edges should shift with it.

**Main files generated** (in the data folder `D:\2026-07-03\`)
- `bandpass_events.png` — events + FWHM vs bias_hpf and bias_fo
- `make_bandpass_figures.py` — the analysis script

---

## 2026-07-02 — Intensity linearity (lots-of-dots + stained samples)

**Conditions**
- Sample: "lots of dots" (many fluorescent dots of varied sizes) + a stained sample _(add specifics — fill in)_
- Light power at sample: _(fill in)_
- Illumination (AWG): CH1 = 2500 Hz, 18 Vpp
- Camera settings: EVK4. lots-of-dots: bias 40, 5 s, 61 planes. stained: biases 20 / 60 / 80, 5–10 s.
- Notes / what happened during acquisition: _(fill in)_
- Anything that failed: _(fill in)_

**Data treatment**
- Question (tutor): is the event signal linear in intensity? (e.g. a 2× bigger/brighter bead → 2× signal?)
- Segmented every dot in the lots-of-dots event image; per dot measured area, total events, events/pixel.
- **Result: the camera is ≈linear in AREA but compressed in per-pixel brightness.** Total signal ∝ area^1.1 (2× area → ~2× signal), while the per-pixel event rate barely changes (~2.4 → 4 ev/px across a 1000× size range). → the event-DSI signal measures the *extent* of in-focus contrast, not per-pixel brightness; its intensity response is log-like (compressed) = non-linear in intensity. A definitive test needs the same dots imaged with the linear ORCA (not in this data).
- Stained sample (side note): focal signal at bias 20 (~2 ev/px) ≈ 3–4× that at bias 60–80, consistent with the steep bias response (but only 3 biases, one non-monotonic — not a clean sweep).

**Main files generated** (in the data folder `D:\2026-07-02\`)
- `intensity_linearity.png` — total-events-vs-area and per-pixel-response-vs-area
- `intensity_linearity.py` — the analysis script

---

## 2026-07-02 — Duration test (single-line sample)

**Conditions**
- Sample: single line _(add details — fill in)_
- Light power at sample: _(fill in)_
- Illumination (AWG): CH1 = 2500 Hz, 18 Vpp _(confirm)_
- Camera settings: EVK4, **bias 40** (fixed), 51 planes; acquisition time swept **0.1, 1, 2, 5, 10 s**. All spans full (no saturation).
- Notes / what happened during acquisition: _(fill in)_
- Anything that failed: _(fill in)_

**Data treatment**
- Question (tutor): does a longer acquisition make the result better?
- From the accumulated event images (`*_zstack_event.tif`) + axial CSVs, measured per duration at the in-focus plane: SNR = (signal−background)/noise, contrast = signal/background, FWHM, and total events.
- **Result: longer acquisition does NOT improve quality.** SNR is flat-to-decreasing and contrast drops (≈12 → 1.8) because the out-of-focus background fills in; both signal and background grow ∝ time; FWHM stays ~3.3–3.8 µm (resolution is time-independent). → duration is not the quality lever (the bias is); a short acquisition (~1 s) is enough.

**Main files generated** (in the data folder `D:\2026-07-02\`)
- `snr_vs_duration.png` — SNR / contrast / FWHM / signal&background vs duration
- `duration_focal_images.png` — focal-plane event image at each duration (background fills in)
- `make_duration_figures.py` — the analysis script

---

## 2026-07-01 — Bias linearity test (orange, single line)

**Conditions**
- Sample: orange sample, single line _(add details: dye, prep, thickness — fill in)_
- Light power at sample: _(fill in)_
- Illumination (AWG): CH1 = 2500 Hz, 18 Vpp _(confirm / add CH2 if used)_
- Camera settings: EVK4, full sensor 1280×720, **5 s/plane, 71 planes, 0.2 µm step**; bias_on = bias_off swept over **0, 20, 40, 60, 80, 100**.
- Notes / what happened during acquisition: _(fill in)_
- Anything that failed: _(fill in — e.g. an earlier batch-queue run gave suspicious data; still to confirm)_

**Data treatment**
- Goal: test whether tuning the bias (threshold) is linear (tutor's question).
- Counted total events per bias. Event counts were taken from the axial-profile CSVs and
  cross-checked against a direct decode of the raw event files (matched to 0.4 %). All 6
  biases recorded the full 5 s (none saturated), so all were used.
- Fitted events-vs-bias with a straight line and with an exponential.
- **Result: NON-LINEAR (exponential).** Straight-line fit R² = 0.79 vs exponential fit
  R² = 0.998; events drop ÷10 per ~68 bias units (halve per ~20). So the bias is a
  logarithmic knob — confirms the tutor's suspicion.
- Also produced the standard overview: events, FWHM (3.9 → 2.3 µm, thinner at higher bias),
  SNR, and Gaussian-fit R² per bias.

**Main files generated** (in the data folder `…/DSI Microscope Data/2026-07-01/`)
- `linearity.png` — the linearity result (linear vs exponential fit + verdict)
- `SIMPLE_overview.png` — 4-panel overview (events / FWHM / SNR / fit-R² vs bias)
- Analysis scripts kept in the scratchpad (`lin/`: `evt3.py`, `decode_linearity.py`, `make_figures.py`)

---

## 2026-06-26 — EVK4 parameter investigation (bias / duration / crop)

**Conditions**
- Sample: _(fill in)_
- Light power at sample: _(fill in)_
- Illumination (AWG): CH1 = 2500 Hz, 18 Vpp
- Camera settings: EVK4, full sensor 1280×720. Three campaigns: **bias sweep** (bias_on=bias_off
  −30 → +120, 1 s/plane, 41 planes), **duration** (0.1–10 s), **hardware crop ON/OFF**.
- Notes / what happened during acquisition: bias 0 and below saturated; the −20 and +20 folders
  were empty (runs not saved). _(add anything else)_
- Anything that failed: _(fill in)_

**Data treatment**
- Wrote a custom EVT3 decoder (no vendor SDK) to read the raw event streams.
- Measured, vs the dimensionless bias: total events (exponential, ÷10 per ~34 units), ON/OFF
  balance, and optical-sectioning quality (FWHM, contrast, SNR).
- Found the camera **saturates below ~bias 5** (rate ceiling ~9–20 Mev/s, and only ~half the
  requested time is recorded). Best sectioning ≈ bias 30–50 for that sample.
- Duration: events accumulate linearly with time but, at the saturating bias used, longer
  recording did not improve sectioning. Crop: hardware ROI cuts the event rate ∝ area.
- Conclusion: the bias is a logarithmic contrast-sensitivity knob.

**Main files generated** (in `dsi_microscope/EVK4_parameter_analysis/`)
- `SIMPLE_overview.png`, `SIMPLE_duration.png` — presentation figures
- `fig1…fig8.png`, `figA…figB.png` — detailed figures
- `EVK4_analysis_report.tex` — LaTeX report; `SUMMARY.md`, `TUTOR_BRIEF.md` — write-ups
- `scripts/` — the decoder (`evt3.py`) and all analysis/plot scripts
