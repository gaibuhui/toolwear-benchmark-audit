"""Cache NASA Ames milling features into a portable .npz.

Run with an environment that has pyarrow (e.g. the `fov` conda env):
    python scripts/nasa_feat_cache.py

Produces data/toolwear/nasa_feat.npz with
    Xs  (n,66)  6 channels x 11 time/frequency features
    PH  (n,4)   time, DOC, feed, material  (machine-controller process variables)
    CAS (n,)    cutting-condition id
    RUN (n,)    run index
    VB  (n,)    flank wear label (mm)
    prev(n,)    previous wear value within the same case (NaN for the first run)
    CH  (6,)    channel names
Feature definition identical to scripts/nasa_baselines.py (paper's T-table).
"""
import numpy as np
import pyarrow.parquet as pq

P = "data/toolwear/nasa_milling/data.parquet"
OUT = "data/toolwear/nasa_feat.npz"
CH = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

rows = [r for r in pq.ParquetFile(P).read().to_pylist()
        if r["VB"] is not None and not np.isnan(r["VB"])]


def feats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.zeros(11)
    x = x - x.mean()
    sd = x.std() + 1e-12
    w = x * np.hanning(len(x))
    f = np.abs(np.fft.rfft(w))
    p = f ** 2 + 1e-12
    fr = np.fft.rfftfreq(len(x))
    cen = float((fr * f).sum() / p.sum())
    bands = [float(p[e].sum() / p.sum()) for e in np.array_split(np.arange(len(f)), 4)]
    return np.array([x.mean(), sd, np.sqrt((x ** 2).mean()), np.abs(x).max(),
                     float(((x / sd) ** 4).mean() - 3), float(((x / sd) ** 3).mean()),
                     cen, *bands])


Xs = np.stack([np.concatenate([feats(r[c]) for c in CH]) for r in rows])
PH = np.stack([[r["time"], r["DOC"], r["feed"], r["material"]] for r in rows]).astype(np.float64)
CAS = np.array([r["case"] for r in rows])
RUN = np.array([r["run"] for r in rows])
VB = np.array([r["VB"] for r in rows])

# --- data-quality probe: per-row max |sample| across all six channels ---
# A benchmark audit should surface silently corrupted records; the public NASA mirror
# contains exactly one (values of order 1e34), which alone makes 23 of the 66 derived
# feature columns span 30 orders of magnitude.
ROWMAX = np.array([max(np.abs(np.asarray(r[c], dtype=np.float64)).max() for c in CH)
                   for r in rows])

o = np.lexsort((RUN, CAS))
Xs, PH, CAS, RUN, VB, ROWMAX = Xs[o], PH[o], CAS[o], RUN[o], VB[o], ROWMAX[o]

prev = np.full(len(VB), np.nan)
for c in set(CAS.tolist()):
    i = np.where(CAS == c)[0]
    prev[i[1:]] = VB[i[:-1]]

np.savez_compressed(OUT, Xs=Xs, PH=PH, CAS=CAS, RUN=RUN, VB=VB, prev=prev,
                    ROWMAX=ROWMAX, CH=np.array(CH), FEAT_PER_CH=11)
nbad = int((ROWMAX > 1e6).sum())
print("rows=%d cases=%d sensor_feats=%d  corrupted rows (max|sample|>1e6): %d  max=%.2e -> %s"
      % (len(VB), len(set(CAS.tolist())), Xs.shape[1], nbad, ROWMAX.max(), OUT))
