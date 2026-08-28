# Experiment log

## 2026-08-25 — Is the 2×f_AWG event-rate peak global intensity or speckle?

**Question (from the tutor, on the 2026-07-20 temporal-response analysis):** are
the peaks at 2× the LC drive frequency due to global intensity fluctuations or
something else?

**Data:** the same decoded `*_xytp.mat` streams the 07-20 analysis used —
`D:\2026-07-15 hp and lp 1500hz 2500hz\awg_frequency_{1500,2500}hz` and
`D:\2026-07-16 hp and lp 500hz\awg_frequency_500hz`, focal plane of the
`fo=15, hpf=0` cell of each.
**Analysis:** `D:\2026-07-20 agarose sample evk4\temporal_spectrum_analysis\`
→ `analyze_2f_global_vs_speckle.py`, `global_vs_speckle_results.csv`, figs 5–8.

**Why it needed a new analysis:** `analyze_temporal_spectrum.py` histogrammed all
timestamps over the whole sensor and **discarded the polarity bit**, so its 2f
line is a *total-activity* rate. Both a global brightness swing and speckle
reconfiguration produce a 2f line in that quantity — it cannot separate them.
The polarity bit is the discriminator: a global swing puts ON and OFF in
**antiphase**, speckle boiling makes them fire **together**.

**Result: the 2f line is a global (common-mode) intensity fluctuation.** At
1500 Hz the net (ON−OFF) rate carries a *larger* 2f line than the total rate
(43.6 vs 18.3 events per 25 µs bin — an 18% common-mode swing of the whole field
vs 7.7% of total activity), 220× above a polarity-shuffled null; ON and OFF are
antiphase at the line (−168°; −153° at 500 Hz, −122° at 2500 Hz); the 2f phase is
one near-uniform value across the sensor (16×16-block resultant R = 0.98 vs 0.03
chance); and the swing is the same relative size on dim background pixels as on
the brightest 1%. Speckle boiling is still there — the per-pixel resultant is
0.46, so about half the per-pixel modulation amplitude has scattered phase — but
the **line** at 2f is the common mode.

**Consequences:** (1) the 07-20 "coherent fraction" / `line_floor` metrics rank
the *global* modulation, not speckle decorrelation, so the "1500 Hz is the most
temporally coherent drive" conclusion needs re-reading in that light; (2) a
common-mode swing fires events on out-of-focus background as readily as on
beads, which directly costs sectioning contrast — worth chasing at the source
(LC transmission into the collection NA varying with |E|). Incidental finding:
the 2500 Hz streams carry strong ~40 kHz instrumental spikes, present
identically in ON and OFF, which corrupt any broadband ON/OFF correlation —
metrics must be evaluated at the 2f line itself.

## 2026-07-24 — EVK4 sensor crop ON vs OFF evaluation (2026-06-26 data)

**Data:** `E:\DSI Microscope data\2026-06-26 crop on vs crop off\EVK4 Crop OFF vs ON`
(3 acquisitions; Crop OFF = full sensor 1280×720, Crop ON = hardware ROI on the
EVK4, "Left" ≈481×720 and "Top Left" ≈464×314). Same stage/illumination; T1 & T2 at
1 s/plane (31/41 planes), T3 at 5 s/plane (41 planes). Question: is cropping the
event sensor better than not cropping?

**Analysis:** compared the per-plane DSI axial-sectioning profiles (existing
`*_axial_profile_event.csv`, Gaussian-fit FWHM + amp/offset contrast) and total
event data volume (sum of `*_events_z*.raw` file sizes) OFF vs ON. Figures saved in
the data folder: `FIG1_axial_profile_OFF_vs_ON.png` (matched T3 overlay),
`FIG2_summary_bars.png` (FWHM / contrast / MB-per-plane across all 3).

**Result:** cropping is better on every axis and shows no signal penalty. Axial FWHM
sharper with crop in all 3 (T1 14.6→12.3, T2 17.0→13.0, T3 13.5→12.5 µm); peak
contrast higher (T1 2.5→3.0, T2 2.0→3.1, T3 3.8→4.6×); event data cut 1.8–3.0×
(T3 54→23 MB/plane) easing the USB2 ceiling. Caveat: part of the FWHM/contrast gain
is that the ROI excludes off-target background, not purely a bandwidth effect — but
that still favours cropping in practice. Only cost is reduced field of view (the
intended trade-off).

## 2026-07-24 — EVK4 log-linearity test on the 3D agarose sample (2026-07-21 data)

**Data:** `D:\2026-07-21 agarose sample evk4 orca` (acq1/2/3, ORCA + EVK4 z-stacks,
beads dispersed through an agarose gel). Repeat of the 2026-07-10 linearity
analysis, adapted for beads at many depths.
**Analysis:** `D:\2026-07-21 agarose sample evk4 orca\linearity_analysis_3d`
(scripts, `beads.csv`, `fit_summary.json`, `figures/`, `README.md`).

**Method (3D adaptations):** beads detected as axially-isolated peaks on the
ORCA-DSI stack with per-bead focal depth z0 (per_bead_axial); ORCA brightness read
on each bead's own in-focus slice via per-bead segmentation (not a stack
projection or fixed aperture, which fail in 3D); EVK4↔ORCA registration by
pose-clustering on bead centroids — a ~43° rotation + 0.7477 scale (pitch ratio
4.86/6.5), consistent across all three acquisitions. Same three models as before
(linear / log-linear / power) + per-pixel control.

**Result:** method validated (ORCA DSI-vs-brightness reference linear, R²≥0.94;
correspondence visually confirmed). Event response is compressive (power g≈0.2–0.56,
<1) but log-linearity only weakly supported vs the 2D sample — pooled (n=50,
per-config z-scored) log R²=0.41 > linear R²=0.35, r=0.64. Per-pixel control flat
(R²≈0), reproducing the 2D finding. Noisier than 2D because the agarose beads have
a smaller/noisier brightness spread, event blobs are large/partly merged, and depth
is a mild confound (deeper beads → fewer events per unit brightness).
