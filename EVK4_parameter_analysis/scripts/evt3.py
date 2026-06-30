"""Minimal, dependency-free EVT3.0 decoder for Prophesee EVK4 (IMX636) .raw files.

EVT3 is a stream of little-endian uint16 words. High nibble = event type,
low 12 bits = payload. We decode just what we need to *characterise the biases*:
per-event polarity (ON/OFF), timestamp (us), and optionally (x, y) for images.

Type codes (Prophesee openeb):
  0x0 EVT_ADDR_Y     payload[10:0]=y
  0x2 EVT_ADDR_X     payload[10:0]=x, payload[11]=polarity   -> 1 CD event
  0x3 VECT_BASE_X    payload[10:0]=base_x, payload[11]=polarity (for following vectors)
  0x4 VECT_12        12-bit mask -> events at base_x+i for set bits; base_x += 12
  0x5 VECT_8          8-bit mask -> events at base_x+i for set bits; base_x += 8
  0x6 EVT_TIME_LOW   payload = time low 12 bits (us)
  0x8 EVT_TIME_HIGH  payload = time high 12 bits (us)
  others: triggers / monitoring (ignored)
Polarity convention: bit11 = 1 -> ON (positive contrast), 0 -> OFF.
"""
import numpy as np

_POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 12)], dtype=np.int64)


def read_evt3_words(path):
    """Return (uint16 words, header_dict). Skips the ASCII '% ...' header."""
    with open(path, "rb") as f:
        raw = f.read()
    # Header is ASCII lines starting with '%', terminated by '% end\n'.
    marker = b"% end\n"
    idx = raw.find(marker)
    header = {}
    if idx != -1:
        head_txt = raw[:idx].decode("ascii", "replace")
        for line in head_txt.splitlines():
            if line.startswith("%"):
                parts = line[1:].strip().split(" ", 1)
                if len(parts) == 2:
                    header[parts[0]] = parts[1]
        body = raw[idx + len(marker):]
    else:
        body = raw
    # Ensure even length
    if len(body) % 2:
        body = body[:-1]
    words = np.frombuffer(body, dtype="<u2")
    return words, header


def _ffill_payload(mask, payload):
    """Forward-fill payload values at positions where mask is True."""
    n = len(payload)
    idx = np.where(mask, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    return payload[idx]


def decode_polarity_time(words):
    """Decode every CD event's polarity and timestamp (no x/y).

    Returns dict with:
      n_on, n_off  : total ON / OFF event counts (ints)
      ev_t         : per-'event-word' timestamp (us), int64  (one entry per word
                     that produces >=1 event; weight by ev_mult)
      ev_mult      : number of events that word produced
      ev_pol       : polarity (1=ON,0=OFF) for that word's events
    """
    typ = (words >> 12).astype(np.uint8)
    pay = (words & 0x0FFF).astype(np.int64)

    is_th = typ == 0x8
    is_tl = typ == 0x6
    th = _ffill_payload(is_th, pay)
    tl = _ffill_payload(is_tl, pay)
    ts = (th << 12) | tl  # microseconds

    is_base = typ == 0x3
    base_pol = (pay >> 11) & 1
    vect_pol = _ffill_payload(is_base, base_pol)

    is_x = typ == 0x2
    is_v12 = typ == 0x4
    is_v8 = typ == 0x5

    pol_x = (pay >> 11) & 1
    pc12 = _POPCOUNT16[(pay & 0xFFF).astype(np.intp)]
    pc8 = _POPCOUNT16[(pay & 0x00FF).astype(np.intp)]

    # Build per-event-word arrays
    ev_mask = is_x | is_v12 | is_v8
    mult = np.zeros(len(words), dtype=np.int64)
    mult[is_x] = 1
    mult[is_v12] = pc12[is_v12]
    mult[is_v8] = pc8[is_v8]

    pol = np.zeros(len(words), dtype=np.int64)
    pol[is_x] = pol_x[is_x]
    pol[is_v12] = vect_pol[is_v12]
    pol[is_v8] = vect_pol[is_v8]

    ev_t = ts[ev_mask]
    ev_mult = mult[ev_mask]
    ev_pol = pol[ev_mask]

    on_mask = ev_pol == 1
    n_on = int(ev_mult[on_mask].sum())
    n_off = int(ev_mult[~on_mask].sum())
    return dict(n_on=n_on, n_off=n_off, ev_t=ev_t, ev_mult=ev_mult, ev_pol=ev_pol)


def _ffill_i32(mask, payload):
    """Forward-fill payload at True positions, using int32 indices (lean)."""
    n = len(payload)
    idx = np.where(mask, np.arange(n, dtype=np.int32), np.int32(0))
    np.maximum.accumulate(idx, out=idx)
    return payload[idx]


def decode_counts(words):
    """Memory-lean: return (n_on, n_off, t_min_us, t_max_us) without building
    per-event arrays. Uses int32/uint8 temporaries (peak ~12 B/word)."""
    typ = (words >> 12).astype(np.uint8)
    pay = (words & 0x0FFF)  # uint16

    is_x = typ == 0x2
    is_v12 = typ == 0x4
    is_v8 = typ == 0x5
    is_base = typ == 0x3

    base_pol = ((pay >> 11) & 1).astype(np.uint8)
    vect_pol = _ffill_i32(is_base, base_pol)  # uint8 per word
    pol_x_on = ((pay >> 11) & 1).astype(bool)

    pc12 = _POPCOUNT16[(pay & 0x0FFF).astype(np.intp)]  # int64 lut gather
    pc8 = _POPCOUNT16[(pay & 0x00FF).astype(np.intp)]

    von = vect_pol == 1
    n_on = int(np.count_nonzero(is_x & pol_x_on))
    n_on += int(pc12[is_v12 & von].sum()) + int(pc8[is_v8 & von].sum())
    n_off = int(np.count_nonzero(is_x & ~pol_x_on))
    n_off += int(pc12[is_v12 & ~von].sum()) + int(pc8[is_v8 & ~von].sum())

    # timestamps only at the first/last event-producing word
    is_th = typ == 0x8
    is_tl = typ == 0x6
    ts = (_ffill_i32(is_th, pay).astype(np.int32) << 12) | _ffill_i32(is_tl, pay).astype(np.int32)
    ev_mask = is_x | is_v12 | is_v8
    idx = np.flatnonzero(ev_mask)
    if len(idx):
        t_min, t_max = int(ts[idx[0]]), int(ts[idx[-1]])
    else:
        t_min = t_max = 0
    return n_on, n_off, t_min, t_max


def rate_timeseries(words, bin_us=10000):
    """Return (centers_s, on_per_s, off_per_s) binned event rate. Lean int32."""
    typ = (words >> 12).astype(np.uint8)
    pay = (words & 0x0FFF)
    is_th = typ == 0x8
    is_tl = typ == 0x6
    ts = (_ffill_i32(is_th, pay).astype(np.int32) << 12) | _ffill_i32(is_tl, pay).astype(np.int32)
    is_base = typ == 0x3
    vect_pol = _ffill_i32(is_base, ((pay >> 11) & 1).astype(np.uint8))
    is_x = typ == 0x2
    is_v12 = typ == 0x4
    is_v8 = typ == 0x5
    pc12 = _POPCOUNT16[(pay & 0x0FFF).astype(np.intp)]
    pc8 = _POPCOUNT16[(pay & 0x00FF).astype(np.intp)]
    mult = np.zeros(len(words), dtype=np.int32)
    mult[is_x] = 1
    mult[is_v12] = pc12[is_v12]
    mult[is_v8] = pc8[is_v8]
    pol_on = ((pay >> 11) & 1).astype(bool)
    is_on = (is_x & pol_on) | ((is_v12 | is_v8) & (vect_pol == 1))
    ev = is_x | is_v12 | is_v8
    t = ts[ev]; m = mult[ev]; on = is_on[ev]
    t0, t1 = t.min(), t.max()
    edges = np.arange(t0, t1 + bin_us, bin_us)
    h_on, _ = np.histogram(t[on], bins=edges, weights=m[on])
    h_off, _ = np.histogram(t[~on], bins=edges, weights=m[~on])
    centers = (edges[:-1] + bin_us / 2 - t0) / 1e6
    return centers, h_on / (bin_us / 1e6), h_off / (bin_us / 1e6)


def decode_full(words, width=1280, height=720):
    """Full decode to (x, y, polarity, t) per event. Slower (Python loop over
    vector state) but vectorised where possible. Returns dict of arrays."""
    typ = (words >> 12).astype(np.uint8)
    pay = (words & 0x0FFF).astype(np.int64)

    is_th = typ == 0x8
    is_tl = typ == 0x6
    ts = (_ffill_payload(is_th, pay) << 12) | _ffill_payload(is_tl, pay)
    y_ff = _ffill_payload(typ == 0x0, pay & 0x07FF)

    # Single-pixel CD events (0x2) are easy & vectorised.
    is_x = typ == 0x2
    xs = (pay[is_x] & 0x07FF)
    ys = y_ff[is_x]
    ps = (pay[is_x] >> 11) & 1
    tx = ts[is_x]

    # Vector events need sequential base_x; handle with a compact loop over only
    # the vector-related words (0x3/0x4/0x5). Build an event-count image directly.
    img_on = np.zeros((height, width), dtype=np.int64)
    img_off = np.zeros((height, width), dtype=np.int64)
    # accumulate single events
    np.add.at(img_on, (ys[ps == 1], xs[ps == 1]), 1)
    np.add.at(img_off, (ys[ps == 0], xs[ps == 0]), 1)
    return dict(img_on=img_on, img_off=img_off, n_single=int(is_x.sum()))


if __name__ == "__main__":
    import sys
    w, h = read_evt3_words(sys.argv[1])
    print("header:", h)
    print("n_words:", len(w))
    d = decode_polarity_time(w)
    print("n_on:", d["n_on"], "n_off:", d["n_off"], "total:", d["n_on"] + d["n_off"])
    if len(d["ev_t"]):
        span = (d["ev_t"].max() - d["ev_t"].min()) / 1e6
        print("time span (s):", round(span, 4))
