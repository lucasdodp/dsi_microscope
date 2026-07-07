"""Linearity analysis + SIMPLE_overview for the 2026-07-01 bias sweep.
Event counts come from the axial-profile CSVs (validated == decoded counts to 0.4%)."""
import numpy as np, glob, re, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Point ROOT at the bias-sweep folder to analyse. Figures are always written into
# the data folder (the folder that contains the sweep), so each run's outputs stay
# organised with the data they came from. Override OUT only if you want them elsewhere.
ROOT = r'C:\Stage Institut Fresnel Local Files\DSI Microscope Data\2026-07-01\bias_tests_orange_singleline_18v_2500hz'
OUT = os.path.dirname(ROOT)   # -> the data folder; edit only to override
NPIX = 1280 * 720
plt.rcParams.update({'figure.dpi': 130, 'font.size': 14, 'axes.grid': True,
                     'grid.alpha': 0.35, 'axes.axisbelow': True, 'savefig.bbox': 'tight'})

def parse(path):
    z, mi, ni, fit = [], [], [], {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            for m in re.finditer(r'(\w+)=([-\d.eE]+)', line):
                fit[m.group(1)] = float(m.group(2))
        elif line[0].isdigit() or line[0] == '-':
            p = line.split(',')
            if len(p) == 3:
                try:
                    z.append(float(p[0])); mi.append(float(p[1])); ni.append(float(p[2]))
                except ValueError:
                    pass
    return np.array(z), np.array(mi), np.array(ni), fit

rows = []
for d in sorted(os.listdir(ROOT)):
    c = glob.glob(os.path.join(ROOT, d, '*_axial_profile_event.csv'))
    if not c:
        continue
    bias = int(re.match(r'(-?\d+)_bias', d).group(1))
    z, mi, ni, fit = parse(c[0])
    events = mi.sum() * NPIX
    lo = np.sort(mi)[:8]
    peak, base, noise = mi.max(), np.median(lo), np.std(lo)
    snr = (peak - base) / noise if noise > 0 else np.nan
    amp, mu, sig, off = fit.get('amp'), fit.get('mu_um'), fit.get('sigma_um'), fit.get('offset')
    if None not in (amp, mu, sig, off) and sig:
        model = off + amp * np.exp(-(z - mu) ** 2 / (2 * sig ** 2))
        r2 = 1 - np.sum((ni - model) ** 2) / np.sum((ni - ni.mean()) ** 2)
    else:
        r2 = np.nan
    rows.append(dict(bias=bias, events=events, fwhm=fit.get('fwhm_um', np.nan),
                     snr=snr, r2=r2))
rows.sort(key=lambda r: r['bias'])
b = np.array([r['bias'] for r in rows], float)
ev = np.array([r['events'] for r in rows])
fw = np.array([r['fwhm'] for r in rows])
snr = np.array([r['snr'] for r in rows])
r2 = np.array([r['r2'] for r in rows])

# ---- fits (all points; none saturated) ----
lin = np.polyfit(b, ev, 1)
ev_lin = np.polyval(lin, b)
r2_lin = 1 - np.sum((ev - ev_lin) ** 2) / np.sum((ev - ev.mean()) ** 2)
logfit = np.polyfit(b, np.log10(ev), 1)
r2_exp = 1 - np.sum((np.log10(ev) - np.polyval(logfit, b)) ** 2) / np.sum((np.log10(ev) - np.log10(ev).mean()) ** 2)
units_per_decade = -1.0 / logfit[0]
halve = np.log10(2) / -logfit[0]
print("events:", (ev/1e6).round(1).tolist())
print(f"linear R2={r2_lin:.4f} | exponential R2={r2_exp:.4f} | x10 per {units_per_decade:.0f} units, halve per {halve:.1f}")

# ================= FIG 1: LINEARITY =================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
fig.subplots_adjust(wspace=0.28)
xs = np.linspace(b.min(), b.max(), 100)

a1.scatter(b, ev / 1e6, s=90, color='#1f4e79', zorder=3, label='measured events')
a1.plot(xs, np.polyval(lin, xs) / 1e6, '--', color='crimson', lw=2,
        label=f'straight-line fit (R²={r2_lin:.3f})')
a1.set_xlabel('bias setting (bias_on = bias_off)')
a1.set_ylabel('events recorded (millions, per Z-stack)')
a1.set_title('')
a1.legend(fontsize=10)

a2.scatter(b, ev, s=90, color='#1f4e79', zorder=3, label='measured events')
a2.plot(xs, 10 ** np.polyval(logfit, xs), '-', color='#2e7d32', lw=2,
        label=f'exponential fit (R²={r2_exp:.3f})')
a2.set_yscale('log')
a2.set_xlabel('bias setting')
a2.set_ylabel('events recorded (log scale)')
a2.set_title('')
a2.legend(fontsize=10)

verdict = ("")
fig.suptitle(verdict, fontsize=13, y=1.03, weight='bold')
fig.savefig(os.path.join(OUT, 'linearity.png'))
plt.close(fig)

# ================= FIG 2: SIMPLE_overview =================
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
ax1, ax2, ax3, ax4 = axs.ravel()
fig.subplots_adjust(wspace=0.26, hspace=0.28)

ax1.plot(b, ev, 'o-', color='#1f4e79', ms=9, lw=2)
ax1.set_yscale('log'); ax1.set_xlabel('bias setting'); ax1.set_ylabel('events recorded (per Z-stack)')
ax1.set_title('Events per bias')

ax2.plot(b, fw, 'o-', color='#1f4e79', ms=9, lw=2)
ax2.set_xlabel('bias setting'); ax2.set_ylabel('optical-slice thickness FWHM (µm)')
ax2.set_title('FWHM per bias')

ax3.plot(b, snr, 'o-', color='#c1121f', ms=9, lw=2)
ax3.set_yscale('log'); ax3.set_xlabel('bias setting'); ax3.set_ylabel('signal-to-noise ratio')
ax3.set_title('SNR per bias')

ax4.plot(b, r2, 'o-', color='#1f4e79', ms=9, lw=2)
ax4.axhline(1.0, color='0.6', ls='--', lw=1)
ax4.set_ylim(min(0.6, np.nanmin(r2) - 0.05), 1.02)
ax4.set_xlabel('bias setting'); ax4.set_ylabel('Gaussian-fit accuracy  $R^2$')
ax4.set_title('Fit quality per bias')

fig.savefig(os.path.join(OUT, 'SIMPLE_overview.png'))
plt.close(fig)
print("saved linearity.png and SIMPLE_overview.png to", OUT)
