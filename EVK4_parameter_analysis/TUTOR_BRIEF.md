# EVK4 bias parameter — quick brief (5 min)

**Question:** the Prophesee `bias_diff_on/off` values have no physical unit. What do they mean,
and what setting should we use?

**What I did:** swept the bias from −30 to +100 and measured the recorded events (one figure:
`SIMPLE_overview.png`).

## Two things to say

**1. The bias is a sensitivity dial (panel a).**
Higher bias → fewer events, dropping ×10 for every ~34 units. So it sets how big a brightness
change is needed to make an event — i.e. the contrast threshold.

**2. The bias sets the optical-section thickness (panel b).**
Raising it removes out-of-focus light, so the slice gets thinner (better sectioning):
~14 µm at bias 0 → 7.5 µm at bias 50.

## Two limits

- **Too low (bias ≤ ~5, red):** the camera saturates — it hits its data-rate ceiling, only
  ~half the recording time is kept, and the image floods. **Unusable.**
- **Too high (bias ≥ ~75):** almost no events — signal lost in noise.

## Conclusion / recommendation

**Use bias ≈ 30–50 (green "sweet spot"):** thinnest, cleanest optical section while staying
safely below saturation. Lower (10–20) only if you need more raw signal.

---
*One figure to show: `SIMPLE_overview.png`. Backup detail in `SUMMARY.md` if asked.*