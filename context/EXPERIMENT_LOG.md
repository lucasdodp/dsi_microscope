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
