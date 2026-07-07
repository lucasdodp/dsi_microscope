"""Minimal, memory-lean EVT3.0 decoder (counts + timestamps only)."""
import numpy as np

# popcount LUT as uint8 (max popcount of 12 bits = 12) -> big memory saving on
# huge saturated files vs an int64 LUT.
_POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 12)], dtype=np.uint8)


def read_evt3_words(path):
    with open(path, "rb") as f:
        raw = f.read()
    marker = b"% end\n"
    idx = raw.find(marker)
    body = raw[idx + len(marker):] if idx != -1 else raw
    if len(body) % 2:
        body = body[:-1]
    return np.frombuffer(body, dtype="<u2")


def _ffill_seed(mask, payload, seed):
    """Forward-fill payload at True positions; positions before the first True
    take ``seed`` (the last valid value carried from the previous chunk)."""
    n = len(payload)
    idx = np.where(mask, np.arange(n, dtype=np.int32), np.int32(-1))
    np.maximum.accumulate(idx, out=idx)
    out = payload[np.maximum(idx, 0)].astype(payload.dtype, copy=True)
    out[idx < 0] = seed
    return out


def decode_counts(words, chunk=2_000_000):
    """Return (n_on, n_off, t_min_us, t_max_us), decoding in chunks so peak memory
    is independent of file size (robust on low-RAM machines / huge saturated files).
    Forward-fill state (time high/low, vector polarity) is carried across chunks."""
    n_on = n_off = 0
    last_th = last_tl = 0
    last_vpol = np.uint8(0)
    t_first = None
    t_last = 0
    for s in range(0, len(words), chunk):
        w = words[s:s + chunk]
        typ = (w >> 12).astype(np.uint8)
        pay = (w & 0x0FFF)

        is_th = typ == 0x8
        is_tl = typ == 0x6
        is_base = typ == 0x3
        th = _ffill_seed(is_th, pay, last_th)
        tl = _ffill_seed(is_tl, pay, last_tl)
        vpol = _ffill_seed(is_base, ((pay >> 11) & 1).astype(np.uint8), last_vpol)
        last_th, last_tl, last_vpol = int(th[-1]), int(tl[-1]), int(vpol[-1])

        is_x = typ == 0x2
        is_v12 = typ == 0x4
        is_v8 = typ == 0x5
        pol_x_on = ((pay >> 11) & 1).astype(bool)
        pc12 = _POPCOUNT16[(pay & 0x0FFF).astype(np.intp)]
        pc8 = _POPCOUNT16[(pay & 0x00FF).astype(np.intp)]
        von = vpol == 1
        n_on += int(np.count_nonzero(is_x & pol_x_on)) + int(pc12[is_v12 & von].sum()) + int(pc8[is_v8 & von].sum())
        n_off += int(np.count_nonzero(is_x & ~pol_x_on)) + int(pc12[is_v12 & ~von].sum()) + int(pc8[is_v8 & ~von].sum())

        ts = (th.astype(np.int32) << 12) | tl.astype(np.int32)
        ev = is_x | is_v12 | is_v8
        idx = np.flatnonzero(ev)
        if len(idx):
            if t_first is None:
                t_first = int(ts[idx[0]])
            t_last = int(ts[idx[-1]])
    return n_on, n_off, (t_first or 0), t_last
