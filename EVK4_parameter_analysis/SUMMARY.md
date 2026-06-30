# EVK4 Event-Camera Parameter Analysis — 2026-06-26 datasets

**Goal (from your professor):** the Prophesee bias values are dimensionless company numbers
with no physical unit. This analysis recovers what they *mean* by measuring the actual event
data, and quantifies the camera's operating limits.

All numbers below come from decoding the raw Prophesee `.raw` event streams directly
(EVT3.0, IMX636, 1280×720) — the Metavision SDK was **not** needed; a small custom decoder
(`evt3.py`, included here) reads every event's polarity and timestamp. 12 bias settings × 41
planes, plus the duration and crop campaigns, were fully decoded (~25 GB of events).

---

## TL;DR — the headline insights

1. **`bias_diff_on/off` is a logarithmic contrast-threshold (sensitivity) knob.**
   Event yield falls **×10 for every +34 bias units** (halves every ~10 units), almost
   perfectly exponential (R² = 0.999) over the usable range. → *Fig 1.*

2. **Below ~bias 5 the camera saturates and the data is corrupted, not "stronger".**
   The Event-Rate Controller is capped at **20 Mevents/s** (hit at bias −30) and the
   USB/host link sustains only **~9 Mevents/s**. Above that the per-plane recording is
   **truncated — only ~half of the requested time is actually captured** — and the stream
   becomes OFF-event flooded. → *Fig 3.*

3. **The bias directly sets optical-sectioning quality.** Raising the threshold rejects
   low-contrast out-of-focus light, so the axial section sharpens from **14 µm → 7.5 µm**
   and background contrast improves **~250×**, peaking around **bias 30–50**, then the
   signal collapses into noise by ~bias 75. → *Figs 4, 5, 6.*

4. **ON and OFF thresholds are not symmetric at the same number.** Polarity is most balanced
   at bias ~10–40; ON events die off faster than OFF as the bias rises. → *Fig 2.*

5. **Acquisition time** accumulates events linearly but does **not** fix saturation: at a
   saturating bias, longer recording ≠ better sectioning. → *Fig 7.*

6. **Hardware ROI (crop) is a genuine sensor-side data cut**: event rate scales with ROI
   area while per-pixel density is conserved — an effective way to stay under the bandwidth
   ceiling. → *Fig 8.*

### ➜ Recommended operating point
**bias_diff_on = bias_diff_off ≈ 30–50**, full sensor, ~1 s/plane. This sits *below* the
saturation/truncation regime, gives the **thinnest optical section and best background
rejection**, and keeps the event rate (~0.5–1.5 Mev/s) well within the link budget so the
full requested time is recorded. Use a lower bias (10–20) only if you need more raw signal
and can tolerate a thicker section; crop the ROI if you must go lower without truncating.

---

## What each figure shows

| File | Insight |
|------|---------|
| `fig1_event_yield_vs_bias.png` | **Calibration curve.** Total/ON/OFF events vs bias (log). Exponential fit: ×10 per 34 bias units. Saturated points (bias ≤ 0) sit clamped on the ~195 M/stack ceiling. |
| `fig2_onoff_balance_vs_bias.png` | ON fraction vs bias. Balanced near bias 10–40; ON vanishes faster than OFF at high bias → the "0" of ON and OFF are not the same physical threshold. |
| `fig3_erc_saturation.png` | (a) Event rate vs time: bias −30 pins the 20 Mev/s ERC cap; bias 0 the ~9 Mev/s USB ceiling, both cut short. (b) At a saturating bias only ~50 % of the requested time is recorded (from the duration sweep). |
| `fig4_sectioning_vs_bias.png` | Axial FWHM ↓ (14→7.5 µm) and peak/background contrast ↑ (~250×) as bias rises to ~50, then the fit fails (signal starved). Optimal band shaded. |
| `fig5_axial_profiles.png` | The per-plane axial profiles (peak-normalised): higher bias = a narrower peak on a near-zero background. |
| `fig6_spatial_focal.png` | Focal-plane event images: flooded/noisy at low bias → clean speckle at bias 30–50 → starved at bias 75. |
| `fig7_duration.png` | Events accumulate ∝ time (pinned at the saturated ~9 Mev/s, half-captured); contrast/FWHM barely improve → time can't fix saturation. |
| `fig8_crop.png` | Hardware ROI cuts event rate ∝ area while ev/px/s stays constant (within a scene) — a true on-sensor reduction that relieves truncation. |

---

## Key measured numbers

| Quantity | Value | Source |
|---|---|---|
| Event-yield slope | −0.0292 decades per bias unit (×10 per **34.2** units; ÷2 per **10.3** units) | Fig 1 fit, R²=0.999 |
| ERC hard cap | **20 Mev/s** (configured `EVK4_ERC_RATE`; observed at bias −30) | Fig 3a |
| Sustained USB/host throughput | **~9 Mev/s** (bias-0 plateau) | Fig 3a |
| Truncation onset | between bias 0 (truncates) and bias 10 (full capture); ≲ 2 Mev/s is safe | Fig 3 |
| Captured fraction when saturated | **47–52 %** of requested time, at any duration | Fig 3b / duration |
| Axial section, bias 0 → 50 | **13.9 µm → 7.5 µm** | Fig 4 |
| Background contrast, bias 0 → 50 | **3.2 → 269** | Fig 4 |
| Best SNR | bias 50 (SNR ≈ 1525 from the pipeline's axial fit) | csv |

---

## Important caveats / how to read this

- **Saturated runs (bias ≤ 0) are not directly comparable** to the rest: their event totals
  are clamped at the link ceiling and their time axis is truncated, so their "sectioning"
  numbers reflect a flooded, background-dominated image. They are kept only to show *where*
  the usable range ends.
- The **contrast / SNR metrics explode at high bias** partly because the out-of-focus
  baseline falls toward zero (a division). The **FWHM** and the **spatial images** are the
  more robust quality indicators there; both agree that ~30–50 is the sweet spot and that
  ≥75 is signal-starved.
- The **duration sweep was taken at bias ≈ 0 (saturating)**, so it characterises accumulation
  under saturation. To study integration time cleanly, repeat it at bias 30–50.
- The **`_highres` runs (bias 100/110/120)** differ only in Z-step (0.4 µm vs 1.0 µm) and sit
  in the dead, signal-starved range; the **`0 … BLEACHED`** run is a photobleaching repeat of
  bias 0 (≈12 % lower peak, same shape) and confirms reproducibility.
- The `-20` and `+20` folders are **empty** (runs not saved).

---

## Reproduce

```
python evt3.py <file.raw>     # decode one stream (sanity check)
python decode_bias.py         # full bias sweep -> bias_decoded.npz
python parse_csv.py           # axial-sectioning CSVs -> csv_parsed.npz
python extract_extras.py      # temporal traces + focal images
python decode_dur.py / decode_crop.py
python plots.py ; python plots2.py   # all 8 figures
```
The decoder and scripts are copied into the `scripts/` subfolder next to this report.
