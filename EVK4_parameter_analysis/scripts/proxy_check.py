import numpy as np, glob, re, os, json
ROOT = r'C:\Stage Institut Fresnel Local Files\DSI Microscope Data\2026-07-01\bias_tests_orange_singleline_18v_2500hz'
NPIX = 1280*720
def parse(path):
    z,mi,ni,fit=[],[],[],{}
    for line in open(path):
        line=line.strip()
        if not line: continue
        if line.startswith('#'):
            for m in re.finditer(r'(\w+)=([-\d.eE]+)',line): fit[m.group(1)]=float(m.group(2))
        elif line[0].isdigit() or line[0]=='-':
            p=line.split(',')
            if len(p)==3:
                try: z.append(float(p[0]));mi.append(float(p[1]));ni.append(float(p[2]))
                except: pass
    return np.array(z),np.array(mi),np.array(ni),fit
dec = json.load(open('bias_counts.json'))
print(f"{'bias':>4} {'proxy_events':>14} {'decoded':>12} {'ratio dec/proxy':>16}  fwhm  peak")
for d in sorted(os.listdir(ROOT)):
    c=glob.glob(os.path.join(ROOT,d,'*_axial_profile_event.csv'))
    if not c: continue
    bias=int(re.match(r'(-?\d+)_bias',d).group(1))
    z,mi,ni,fit=parse(c[0])
    proxy=mi.sum()*NPIX
    decoded=dec.get(d,{}).get('on',0)+dec.get(d,{}).get('off',0) if d in dec else None
    r = f"{decoded/proxy:.3f}" if decoded else "  —"
    print(f"{bias:4d} {proxy:14.3e} {(decoded or 0)/1e6:10.1f}M {r:>16}  {fit.get('fwhm_um',float('nan')):.2f}  {mi.max():.2f}")
