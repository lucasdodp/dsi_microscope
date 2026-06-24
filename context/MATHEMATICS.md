# DSI Microscope — Mathematical Reference

Every equation, formula, and relation implemented in the codebase, for both **data gathering** (acquisition) and **data processing** (reconstruction). Each block names the function/file that implements it, so the math and the code stay traceable.

> **Rendering note.** The equations are written in LaTeX inside `$ … $` / `$$ … $$` blocks. They render as real symbols (Σ, σ, √, …) in the VS Code Markdown preview (and on GitHub). The companion physics narrative is in [ARCHITECTURE.md](ARCHITECTURE.md); this file is the formal counterpart.

---

## 0. Notation

| Symbol | Meaning |
|--------|---------|
| $I_i(x,y)$ | raw intensity of frame $i$ at pixel $(x,y)$ |
| $N$ | number of frames in a speckle stack (ORCA) |
| $H, W$ | image height, width in pixels |
| $(x,y)$ | pixel coordinates; sums $\sum_{x,y}$ run over all $H\times W$ pixels |
| $z_k$ | axial (focus) position of plane $k$ in µm |
| $K$ | number of Z-stack planes (`steps`) |
| $\bar I, \sigma$ | per-pixel average and standard-deviation images |
| $V(x,y)$ | accumulated event-count image (EVK4) |
| $\lfloor\cdot\rfloor$ | floor (integer division) |

---

## 1. Data gathering (acquisition)

### 1.1 Z-stack focal positions — `orchestrator.AutomatedZStackWorker.run`

The stack is centred on the user focus $z_f$ with $K$ planes spaced by $\Delta z$ (`step_size`). The objective is driven to

$$
z_{\text{start}} = z_f - \frac{\Delta z \, K}{2},
\qquad
z^{\text{target}}_k = z_{\text{start}} + k\,\Delta z,
\quad k = 0,1,\dots,K-1 .
$$

The **commanded** target $z^{\text{target}}_k$ is set with `MOV`; the **recorded** position $z_k$ used in all downstream profiles is read back from the controller (`qPOS`), so encoder error is captured rather than assumed:

$$
z_k = \texttt{qPOS}(\text{axis}) .
$$

### 1.2 Hardware ROI (subarray) geometry — `OrcaParamsWidget._compute_roi`

DCAM requires every subarray edge to be a multiple of 4 px. Requested width/height $w_r,h_r$ are floored to that grid, the region is centred, shifted by the user offsets $(o_x,o_y)$, clamped to the sensor $W_s\times H_s$ (=2304²), then the origin is 4-aligned:

$$
w = 4\left\lfloor \tfrac{w_r}{4}\right\rfloor,\qquad
h = 4\left\lfloor \tfrac{h_r}{4}\right\rfloor,
$$

$$
x_{\min} = 4\left\lfloor \tfrac{1}{4}\,\mathrm{clamp}\!\Big(\tfrac{W_s-w}{2}+o_x,\;0,\;W_s-w\Big)\right\rfloor,
$$

and symmetrically for $y_{\min}$, with $x_{\max}=x_{\min}+w$, $y_{\max}=y_{\min}+h$. (Misalignment silently degrades the hardware subarray to a slow software crop, so the flooring is functional, not cosmetic.)

### 1.3 Frame time and framerate — `OrcaParamsWidget.estimated_frame_time_s`

Let $t_{\exp}$ be the exposure (s) and $\text{1H}$ the per-row readout time for the selected scan speed (`ORCA_ROW_READOUT_US`: Fast 4.86765 µs, Standard 18.64706 µs, Ultra-Quiet 80 µs). For a subarray of $V_n=h$ rows, the free-running readout period is the manual's $(V_n+1)\cdot\text{1H}$. The achievable frame time is whichever of exposure and readout dominates:

$$
t_{\text{readout}} = (V_n + 1)\cdot \text{1H},
\qquad
t_{\text{frame}} = \max\!\big(t_{\exp},\, t_{\text{readout}}\big),
\qquad
\text{fps} = \frac{1}{t_{\text{frame}}} .
$$

The acquisition is **exposure-limited** when $t_{\text{readout}}\le t_{\exp}$, otherwise **readout-limited**. Lowering the subarray *height* $V_n$ is what raises the framerate.

### 1.4 EVK4 event stream

The sensor emits a list of events, each a tuple $(x,y,t,p)$ with polarity $p=\pm1$ and timestamp $t$ (1 µs resolution). Per plane the events are collected over a fixed duration $t_{\text{rec}}$ (`acqu_time`); the generator stops at the first chunk past the deadline:

$$
\text{collect } \{(x,y,t,p)\} \ \text{ while } \ t - t_0 < t_{\text{rec}} .
$$

---

## 2. ORCA / DSI processing

### 2.1 Per-pixel average — the widefield-equivalent image — `compute_dsi_images`

$$
\boxed{\ \bar I(x,y) = \frac{1}{N}\sum_{i=0}^{N-1} I_i(x,y)\ }
$$

This is the conventional uniform-illumination image. It has **no optical sectioning** (see §4.4).

### 2.2 Per-pixel standard deviation — the DSI sectioned image — `compute_dsi_images`

Population variance (denominator $N$, i.e. `ddof = 0`), then its square root:

$$
\mathrm{Var}(x,y) = \frac{1}{N}\sum_{i=0}^{N-1}\big(I_i(x,y) - \bar I(x,y)\big)^2,
\qquad
\boxed{\ \sigma(x,y) = \sqrt{\mathrm{Var}(x,y)}\ }
$$

The optical sectioning comes from this $\sigma$: in-focus speckle grains fluctuate strongly across the stack (large $\sigma$), out-of-focus signal is spatially averaged and barely fluctuates (small $\sigma$).

**Two-pass / chunked evaluation.** The mean (pass 1) and the squared deviations (pass 2) are accumulated in `float64` over blocks of $c$ frames rather than materialising the whole stack as float. The two-pass form avoids the catastrophic cancellation of a single-pass $\sum I^2 - (\sum I)^2$. The block size is bounded by a memory budget $B = 128\ \text{MiB}$:

$$
c = \max\!\left(1,\ \left\lfloor \frac{B}{8\,H\,W} \right\rfloor\right)
\quad\text{(frames per chunk; 8 = bytes per float64).}
$$

The result is mathematically identical to `np.std(stack, axis=0)`; only the memory profile differs.

### 2.3 Legacy consecutive-difference estimator — `process_dsi`

An alternative fluctuation metric (exported but not on the active path): the root-sum-square of differences between *consecutive* frames,

$$
D(x,y) = \sqrt{\sum_{i=1}^{N-1}\big(I_i(x,y) - I_{i-1}(x,y)\big)^2},
$$

with a scalar focus score obtained by summing over the field of view:

$$
z_{\text{val}} = \sum_{x,y} D(x,y) .
$$

### 2.4 Live Z-profile scalar — `orchestrator` (`z_profile_update`)

During a stack, each plane emits a single focus number. For the ORCA this is the **sum** of the sectioned image; for the EVK4 the sum of the event image:

$$
P_k^{\text{live}} = \sum_{x,y} \sigma_k(x,y)
\qquad\text{(ORCA)},\qquad
P_k^{\text{live}} = \sum_{x,y} V_k(x,y)
\qquad\text{(EVK4)} .
$$

(Note: the *saved* axial profile in §4 uses the per-pixel **mean**, not the sum — see there.)

---

## 3. EVK4 event processing

### 3.1 Event accumulation — `accumulate_event_frame`

The 2D image is the count of events at each pixel, polarity ignored (paper Eq. 3):

$$
\boxed{\ V(x,y) = \sum_{k} \big|e_k(x,y)\big|\ },
\qquad e_k=\pm1,
$$

i.e. $V(x,y)=\#\{\text{events recorded at }(x,y)\}$. Implemented as `np.add.at(V, (y,x), 1)`.

### 3.2 Crazy-pixel rejection — `filter_crazy_pixels`

A few pixels self-trigger and saturate the image. Let $T = \mathcal{P}_{p}(V)$ be the $p$-th percentile of all pixel values ($p = 99.9$, `EVK4_CRAZY_PIXEL_PERCENTILE`). Those above it are zeroed:

$$
V(x,y) \leftarrow
\begin{cases}
0, & V(x,y) > T,\\[2pt]
V(x,y), & \text{otherwise.}
\end{cases}
$$

The percentile $\mathcal{P}_p$ is the value below which $p\%$ of the pixel values fall (NumPy linear interpolation between order statistics).

### 3.3 Spatial smoothing — `apply_smoothing`

A separable $5\times5$ Gaussian convolution corrects sensor inhomogeneity:

$$
V'(x,y) = \sum_{i=-2}^{2}\sum_{j=-2}^{2} G(i,j)\,V(x-i,\,y-j),
\qquad
G(i,j) = g(i)\,g(j),
$$

$$
g(k) = \frac{1}{Z}\exp\!\left(-\frac{k^2}{2 s^2}\right),
\qquad
s = 0.3\big((k_{\text{size}}-1)\cdot 0.5 - 1\big) + 0.8 = 1.1\ \text{px}
$$

for kernel size $k_{\text{size}}=5$ (OpenCV's default $\sigma$ when none is given); $Z$ normalises each 1-D kernel to unit sum.

---

## 4. Axial profiles (3D, per-plane) — `save_axial_sectioning_plot`, `save_axial_average_plot`

After a Z-stack, each plane $k$ contributes one scalar = the **spatial mean** of that plane's image, and these are plotted against $z_k$.

### 4.1 Per-plane mean intensities

$$
\mu_k = \frac{1}{H W}\sum_{x,y} \sigma_k(x,y)
\quad\text{(sectioned)},
\qquad
a_k = \frac{1}{H W}\sum_{x,y} \bar I_k(x,y)
\quad\text{(average)} .
$$

For the EVK4, $\sigma_k$ is replaced by the event image $V_k$.

### 4.2 Peak normalisation

Both profiles are normalised to their peak before fitting/plotting:

$$
\tilde\mu_k = \frac{\mu_k}{\max_j \mu_j},
\qquad
\tilde a_k = \frac{a_k}{\max_j a_j} .
$$

### 4.3 Sectioned profile — Gaussian fit & FWHM — `_fit_axial_gaussian`

The sectioned profile peaks at focus and is modelled by a Gaussian on a constant background:

$$
\boxed{\ f(z) = b + A\,\exp\!\left(-\frac{(z - z_0)^2}{2 s^2}\right)\ }
$$

fitted by non-linear least squares (`scipy.optimize.curve_fit`, Levenberg–Marquardt) minimising

$$
\min_{A,z_0,s,b}\ \sum_k \big(f(z_k) - \tilde\mu_k\big)^2,
$$

with initial guesses $A_0=\max\tilde\mu-\min\tilde\mu$, $z_{0}= z_{\arg\max\tilde\mu}$, $s_0=(z_{\max}-z_{\min})/4$, $b_0=\min\tilde\mu$. The **axial sectioning** is the full width at half maximum:

$$
\boxed{\ \text{FWHM} = 2\sqrt{2\ln 2}\;|s| \approx 2.3548\,|s|\ \ [\mu\text{m}]\ }
$$

This is the number reported in the status bar (paper Fig. 3a / Fig. 3c). Needs $\ge 4$ planes and SciPy, else it degrades to data-only.

### 4.4 Average profile — straight-line fit — `_fit_axial_line`

The widefield/average image collects out-of-focus light at every plane, so $\tilde a_k$ is essentially flat in $z$ — it is modelled by a line, not a peak:

$$
\boxed{\ g(z) = m\,z + c\ }
$$

fitted by ordinary least squares (`np.polyfit`, degree 1) minimising $\sum_k (m z_k + c - \tilde a_k)^2$, closed form:

$$
m = \frac{K\sum_k z_k\tilde a_k - \big(\sum_k z_k\big)\big(\sum_k \tilde a_k\big)}
{K\sum_k z_k^2 - \big(\sum_k z_k\big)^2},
\qquad
c = \frac{\sum_k \tilde a_k - m\sum_k z_k}{K}.
$$

A slope $m \approx 0$ (flat line) is the quantitative signature of **no optical sectioning**, in contrast to the Gaussian peak of §4.3. Needs $\ge 2$ planes. *(ORCA-only — the EVK4 produces no average image.)*

---

## 5. Display scaling

### 5.1 Min–max to 8-bit — `normalize_to_8bit`

Linear stretch of a float image to $[0,255]$ (OpenCV `NORM_MINMAX`):

$$
N(x,y) = \mathrm{round}\!\left(255\cdot\frac{I(x,y) - I_{\min}}{I_{\max} - I_{\min}}\right),
\qquad
I_{\min}=\min_{x,y} I,\ I_{\max}=\max_{x,y} I .
$$

### 5.2 16-bit live-view scaling — `scale_16bit_image`

An integer stretch by the current frame maximum, then a down-shift to 8 bits:

$$
O(x,y) = \left\lfloor \frac{I(x,y)\cdot \big\lfloor 65535 / I_{\max}\big\rfloor}{256} \right\rfloor,
\qquad I_{\max}=\max_{x,y} I .
$$

---

## 6. Acquisition-time model — `config.py`, `main_window`, `OrcaParamsWidget`

These drive only the "≈ … s" estimate labels; the live timer is ground truth.

### 6.1 Per-plane compute and disk terms

$$
T_{\text{compute}}(N) = \frac{N\,w\,h}{R_{\text{px}}},
\qquad
T_{\text{save}}(N) = \frac{N\,w\,h\cdot 2}{R_{\text{disk}}},
$$

with $R_{\text{px}} = 85\times10^6$ px/s (`ORCA_DSI_PROCESS_PIXELS_PER_S`), $R_{\text{disk}} = 150\times10^6$ B/s (`ZSTACK_DISK_BYTES_PER_S`), and the factor 2 = bytes per uint16 pixel. $T_{\text{save}}=0$ when raw saving is off.

### 6.2 Single-plane acquisitions

$$
T_{\text{ORCA}} = T_{\text{init}} + N\,t_{\text{frame}} + T_{\text{compute}}(N) + T_{\text{save}}(N),
\qquad T_{\text{init}}=1.5\ \text{s},
$$

$$
T_{\text{EVK4}} = O_{\text{EVK4}} + t_{\text{rec}},
\qquad O_{\text{EVK4}}=4\ \text{s}.
$$

### 6.3 Z-stacks ($K$ planes)

$$
T^{\text{ORCA}}_{\text{stack}} = T_{\text{init}} + K\Big(O_{\text{plane}} + N\,t_{\text{frame}} + T_{\text{compute}}(N) + T_{\text{save}}(N)\Big),
\quad O_{\text{plane}}=2\ \text{s},
$$

$$
T^{\text{EVK4}}_{\text{stack}} = K\big(O_{\text{EVK4}} + t_{\text{rec}}\big).
$$

### 6.4 Self-calibration — `MainWindow._calibrate`

Each completed run stores the pair $(T_{\text{pred}}, T_{\text{act}})$. The displayed estimate is scaled by the **median** of past actual/predicted ratios for that run type:

$$
\phi = \operatorname{median}_{r}\!\left(\frac{T_{\text{act}}^{(r)}}{T_{\text{pred}}^{(r)}}\right),
\qquad
T_{\text{shown}} = \phi \cdot T_{\text{pred}} .
$$

With no history $\phi = 1$ (raw physics estimate); the median makes it robust to occasional outliers.

### 6.5 Peak-RAM pre-flight (ORCA Z-stack) — `MainWindow._zstack_orca_peak_bytes`

A guard against swapping the machine. With $p = w h$ pixels per plane:

$$
M_{\text{peak}} = \underbrace{2\,N p\cdot 2}_{\text{SDK ring + host copy (uint16)}}
+ \underbrace{3\,K p\cdot 4}_{\text{avg + DSI float32 vols + save copy}}
+ \underbrace{128\,\text{MiB}}_{\text{compute chunk}} .
$$

Compared against free physical RAM before launch.

---

## Appendix A — Sensor physics referenced by the design (not implemented in this code)

These describe the EVK4's internal behaviour (from Benachir et al.). The code consumes the camera's event output directly; it does **not** simulate these, but they justify the parameter choices.

**Pixel low-pass response** (response time $\tau$, sample interval $\Delta t$):

$$
R(t_i) = R(t_{i-1}) + \big(S(t_i) - R(t_{i-1})\big)\left(1 - e^{-\Delta t/\tau}\right).
$$

**Logarithmic signal variation and thresholding** (event fires when $|\Delta S|$ crosses the detection threshold $\theta$):

$$
\Delta S(x,y,t_i) = \log_{10}\!\left(\frac{I(x,y,t_i)}{I(x,y,t_{i-1})}\right),
\qquad
|\Delta S| > \theta \Rightarrow \text{event } (\pm).
$$

Ideal sectioning regime: $\tau \approx \Delta t$ (the speckle decorrelation time, matched here to the 50 ms ORCA exposure).

---

## Appendix B — Comparison metrics (defined in the reference papers; candidates for future in-app implementation)

Used to compare event-DSI against sCMOS DSI; **not yet in the codebase** (computed externally in MATLAB in the papers).

**Mutual information** between two images $A,B$ with marginal/joint intensity distributions $p_A, p_B, p_{AB}$:

$$
H(A) = -\sum_{a} p_A(a)\ln p_A(a),
\qquad
H(A,B) = -\sum_{a,b} p_{AB}(a,b)\ln p_{AB}(a,b),
$$

$$
\boxed{\ \mathrm{MI}(A,B) = H(A) + H(B) - H(A,B)\ } .
$$

**Image contrast** (RIM paper, Fig. 6) — spatial standard deviation over spatial mean of a single image $I$:

$$
C = \frac{\langle \sigma(I)\rangle}{\langle I\rangle},
\qquad
\langle\cdot\rangle = \text{spatial average over } (x,y).
$$

---

*Keep this file in sync with `core/image_processing.py` (processing math), `ui/orchestrator.py` (acquisition geometry), `config.py` + `ui/main_window.py` (timing model). When an equation in the code changes, update the matching block here.*
