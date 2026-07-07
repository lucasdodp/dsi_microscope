"""EVK4 parameter analysis -- turn-key re-plotting for repeated test campaigns.

Produces the two figures the tutor asked for:
  * SIMPLE_overview.png  -- 4 panels vs bias  (events, FWHM, SNR, fit-R^2)
  * SIMPLE_duration.png  -- 2 panels vs acquisition time (recorded time, SNR)

NEW SNR definition (tutor's method): acquire N>=2 repeated z-stacks ("packets")
with the SAME parameters, take the Gaussian-fit FWHM of each packet, and define
    SNR = mean(FWHM) / std(FWHM)        (across the N packets)
i.e. how *repeatable* the sectioning thickness is (high = stable/reliable).

USAGE
-----
    python analyze.py overview  "PATH\\to\\bias sweep root"   [out.png]
    python analyze.py duration  "PATH\\to\\duration root"     [out.png]

The script scans the root recursively for every acquisition (any folder that
contains a *parameters.txt). It groups packets by the parameter that was varied
(bias for 'overview', acquisition time for 'duration'), reading the value from
each parameters.txt -- so folder naming does not matter, only that each repeat
is its own acquisition folder (which the microscope now creates automatically).
"""
import sys, os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(HERE)   # the EVK4_parameter_analysis folder

plt.rcParams.update({'figure.dpi': 130, 'font.size': 15, 'axes.grid': True,
                     'grid.alpha': 0.35, 'axes.axisbelow': True, 'savefig.bbox': 'tight'})

SAT_BAND = (-35, 5)     # red guide band (saturation risk at low bias)
BEST_BAND = (28, 55)    # green guide band (recommended range)


# ---------------------------------------------------------------- parsing
def read_params(folder):
    """Return the parameters.txt of an acquisition as a key->string dict."""
    p = glob.glob(os.path.join(folder, '*parameters.txt'))
    if not p:
        return {}
    out = {}
    for line in open(p[0], encoding='utf-8', errors='replace'):
        m = re.match(r'\s*([\w ]+?)\s*=\s*(.+)', line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def parse_axial_csv(path):
    """Return (z, mean_i, norm_i, fit_dict) from an *_axial_profile_event.csv."""
    z, mean_i, norm_i, fit = [], [], [], {}
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            for m in re.finditer(r'(\w+)=([-\d.eE]+)', line):
                fit[m.group(1)] = float(m.group(2))
        elif line[0].isdigit() or line[0] == '-':
            parts = line.split(',')
            if len(parts) == 3:
                try:
                    z.append(float(parts[0])); mean_i.append(float(parts[1])); norm_i.append(float(parts[2]))
                except ValueError:
                    pass
    return np.array(z), np.array(mean_i), np.array(norm_i), fit


def _r2(y, model):
    ss_res = np.sum((y - model) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def packet_metrics(folder):
    """All per-acquisition numbers we need, computed from the CSV + params."""
    csv = glob.glob(os.path.join(folder, '*_axial_profile_event.csv'))
    if not csv:
        return None
    z, mean_i, norm_i, fit = parse_axial_csv(csv[0])
    if len(mean_i) < 3:
        return None
    pr = read_params(folder)

    def pf(key, default=np.nan):
        try:
            return float(pr.get(key, default))
        except (TypeError, ValueError):
            return default

    # pixel count from ROI (for an absolute event signal); default full sensor
    nx = pf('roi_x_max', 1280) - pf('roi_x_min', 0)
    ny = pf('roi_y_max', 720) - pf('roi_y_min', 0)
    n_pix = nx * ny if (nx > 0 and ny > 0) else 1280 * 720

    fwhm = fit.get('fwhm_um', np.nan)
    amp, mu, sig, off = fit.get('amp'), fit.get('mu_um'), fit.get('sigma_um'), fit.get('offset')
    r2 = _r2(norm_i, off + amp * np.exp(-(z - mu) ** 2 / (2 * sig ** 2))) \
        if None not in (amp, mu, sig, off) and sig else np.nan
    peak = float(mean_i.max())
    good = (peak > 0.4) and (2 < fwhm < 30)

    return dict(folder=folder,
                bias_on=pf('bias_on'), bias_off=pf('bias_off'),
                req_time=pf('acquisition_time_s'),
                fwhm=fwhm, r2=r2, peak=peak, good=good,
                event_signal=float(mean_i.sum()) * n_pix)


def find_packets(root):
    """Every acquisition folder (one that contains a parameters.txt) under root."""
    seen, packets = set(), []
    for p in glob.glob(os.path.join(root, '**', '*parameters.txt'), recursive=True):
        d = os.path.dirname(p)
        if d in seen:
            continue
        seen.add(d)
        m = packet_metrics(d)
        if m:
            packets.append(m)
    return packets


def group_by(packets, key, ndigits=3):
    """Group packets by a rounded numeric key -> {value: [packets]}."""
    groups = {}
    for pk in packets:
        v = pk[key]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        groups.setdefault(round(float(v), ndigits), []).append(pk)
    return dict(sorted(groups.items()))


def fwhm_snr(packets):
    """Tutor's SNR = mean(FWHM)/std(FWHM) over the good packets in a group."""
    fw = np.array([p['fwhm'] for p in packets if p['good'] and np.isfinite(p['fwhm'])])
    if len(fw) < 2:
        return np.nan, fw.mean() if len(fw) else np.nan, np.nan, len(fw)
    sd = fw.std(ddof=1)
    return (fw.mean() / sd if sd > 0 else np.nan), fw.mean(), sd, len(fw)


def _zones(ax):
    ax.axvspan(*SAT_BAND, color='crimson', alpha=0.10)
    ax.axvspan(*BEST_BAND, color='green', alpha=0.12)


# ---------------------------------------------------------------- overview
def cmd_overview(root, out_png):
    packets = find_packets(root)
    groups = group_by(packets, 'bias_on')
    if not groups:
        sys.exit("No acquisitions with a readable bias_on found under: " + root)

    bias, ev, ev_sd, fw, fw_sd, snr, r2, nrep = [], [], [], [], [], [], [], []
    for b, pks in groups.items():
        gp = [p for p in pks if p['good']]
        s, fmean, fsd, n = fwhm_snr(pks)
        bias.append(b)
        ev.append(np.mean([p['event_signal'] for p in pks]))
        ev_sd.append(np.std([p['event_signal'] for p in pks]))
        fw.append(fmean); fw_sd.append(fsd if np.isfinite(fsd) else 0.0)
        snr.append(s); r2.append(np.nanmean([p['r2'] for p in gp]) if gp else np.nan)
        nrep.append(len(pks))
    bias = np.array(bias); ev = np.array(ev); ev_sd = np.array(ev_sd)
    fw = np.array(fw); fw_sd = np.array(fw_sd); snr = np.array(snr); r2 = np.array(r2)
    print(f"{'bias':>6} {'reps':>4} {'events':>12} {'FWHM':>7} {'SNR':>7} {'R2':>6}")
    for i in range(len(bias)):
        print(f"{bias[i]:6.0f} {nrep[i]:4d} {ev[i]:12.3e} {fw[i]:7.2f} {snr[i]:7.1f} {r2[i]:6.3f}")

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 10.5))
    ax1, ax2, ax3, ax4 = axs.ravel()
    fig.subplots_adjust(wspace=0.28, hspace=0.28)
    gfw = np.isfinite(fw)

    # (a) events
    _zones(ax1)
    ax1.errorbar(bias, ev, yerr=ev_sd, fmt='o-', color='#1f4e79', ms=9, lw=1.8, capsize=3)
    ax1.set_yscale('log'); ax1.set_xlabel('bias setting')
    ax1.set_ylabel('events recorded (per Z-stack)'); ax1.set_title('Events per bias')

    # (b) FWHM (mean +/- std over repeats)
    _zones(ax2)
    ax2.errorbar(bias[gfw], fw[gfw], yerr=fw_sd[gfw], fmt='o-', color='#1f4e79', ms=9, lw=2, capsize=3)
    ax2.set_xlabel('bias setting')
    ax2.set_ylabel('optical-slice thickness FWHM (µm)'); ax2.set_title('FWHM per bias')

    # (c) NEW SNR = mean(FWHM)/std(FWHM)
    _zones(ax3)
    gsnr = np.isfinite(snr)
    ax3.plot(bias[gsnr], snr[gsnr], 'o-', color='#1f4e79', ms=10, lw=2.2)
    ax3.set_xlabel('bias setting')
    ax3.set_ylabel('SNR  =  mean(FWHM) / std(FWHM)')
    ax3.set_title('FWHM repeatability (SNR) per bias')
    if not gsnr.any():
        ax3.text(0.5, 0.5, 'needs >= 2 repeats\nper bias', transform=ax3.transAxes,
                 ha='center', va='center', color='0.5', fontsize=14)

    # (d) fit quality
    _zones(ax4)
    gr = np.isfinite(r2)
    ax4.plot(bias[gr], r2[gr], 'o-', color='#1f4e79', ms=10, lw=2.2)
    ax4.axhline(1.0, color='0.6', ls='--', lw=1)
    ax4.set_ylim(min(0.6, np.nanmin(r2) - 0.05) if gr.any() else 0.6, 1.03)
    ax4.set_xlabel('bias setting')
    ax4.set_ylabel('Gaussian-fit accuracy  $R^2$'); ax4.set_title('Fit quality per bias')

    for ax in (ax1, ax2, ax3, ax4):
        ax.text(np.mean(BEST_BAND), ax.get_ylim()[0], 'best\nrange', color='green',
                ha='center', va='bottom', fontsize=11, weight='bold')

    fig.savefig(out_png); plt.close(fig)
    print("saved", out_png)


# ---------------------------------------------------------------- duration
def cmd_duration(root, out_png):
    import evt3
    packets = find_packets(root)
    groups = group_by(packets, 'req_time')
    if not groups:
        sys.exit("No acquisitions with a readable acquisition_time_s found under: " + root)

    def recorded_time(folder):
        # Raw event streams may sit directly in the acquisition folder (old
        # layout) or under its ``raw_files/`` subfolder (new layout) — search
        # recursively so both are found.
        files = sorted(glob.glob(os.path.join(folder, '**', '*_events_z*.raw'), recursive=True))
        if not files:
            return np.nan
        w, _ = evt3.read_evt3_words(files[len(files) // 2])
        _, _, tmin, tmax = evt3.decode_counts(w)
        return (tmax - tmin) / 1e6

    req, rec, snr, nrep = [], [], [], []
    for t, pks in groups.items():
        recs = [recorded_time(p['folder']) for p in pks]
        s, _, _, _ = fwhm_snr(pks)
        req.append(t); rec.append(np.nanmean(recs)); snr.append(s); nrep.append(len(pks))
    req = np.array(req); rec = np.array(rec); snr = np.array(snr)
    print(f"{'req(s)':>7} {'reps':>4} {'recorded(s)':>12} {'%kept':>6} {'SNR':>7}")
    for i in range(len(req)):
        print(f"{req[i]:7.2f} {nrep[i]:4d} {rec[i]:12.3f} {rec[i]/req[i]*100:6.0f} {snr[i]:7.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    fig.subplots_adjust(wspace=0.30)
    lim = max(np.nanmax(req), np.nanmax(rec)) * 1.08
    cap = np.nanmedian(rec / req) * 100

    ax1.plot([0, lim], [0, lim], '--', color='0.6', lw=1.5, label='ideal (all time kept)')
    ax1.plot(req, rec, 'o-', color='#1f4e79', ms=9, lw=2)
    for r, s in zip(req, rec):
        if np.isfinite(s):
            ax1.annotate(f'{s/r*100:.0f}%', (r, s), textcoords='offset points', xytext=(7, -12), fontsize=11)
    ax1.set_xlabel('requested time per plane (s)'); ax1.set_ylabel('actually recorded time (s)')
    ax1.set_title('Recorded vs requested time\n(median kept: %.0f%%)' % cap)
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim); ax1.legend(fontsize=11)

    gsnr = np.isfinite(snr)
    ax2.plot(req[gsnr], snr[gsnr], 'o-', color='#c1121f', ms=9, lw=2)
    ax2.set_xscale('log'); ax2.set_xlabel('requested time per plane (s)')
    ax2.set_ylabel('SNR  =  mean(FWHM) / std(FWHM)')
    ax2.set_title('FWHM repeatability (SNR) vs time')
    if not gsnr.any():
        ax2.text(0.5, 0.5, 'needs >= 2 repeats\nper duration', transform=ax2.transAxes,
                 ha='center', va='center', color='0.5', fontsize=14)
    else:
        ax2.set_ylim(0, np.nanmax(snr[gsnr]) * 1.3)

    fig.savefig(out_png); plt.close(fig)
    print("saved", out_png)


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('overview', 'duration'):
        sys.exit(__doc__)
    mode, root = sys.argv[1], sys.argv[2]
    default = 'SIMPLE_overview.png' if mode == 'overview' else 'SIMPLE_duration.png'
    out_png = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ANALYSIS_DIR, default)
    (cmd_overview if mode == 'overview' else cmd_duration)(root, out_png)
