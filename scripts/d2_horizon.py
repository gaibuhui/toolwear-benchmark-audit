"""D2 (NASA milling) horizon-increment protocol: dW_h = VB(i+h)-VB(i), leave-one-case-out.

Baselines: (a) persistence-on-increments dW=0; (b) linear on dCT (cutting-time increment);
(c) linear on (dCT, dCT*DOC, dCT*feed, dCT*material).  VB in mm -> report um.
Small-sample caveat: n pairs = 146 - 16*h roughly; stated honestly in output.
"""
import numpy as np, pyarrow.parquet as pq
from collections import defaultdict

P = "data/toolwear/nasa_milling/data.parquet"
rows = [r for r in pq.ParquetFile(P).read().to_pylist()
        if r["VB"] is not None and not np.isnan(r["VB"])]
rows.sort(key=lambda r: (r["case"], r["run"]))
seq = defaultdict(list)
for r in rows:
    seq[r["case"]].append(r)
print("n=%d cases=%d, per-case len: %s" % (len(rows), len(seq),
      sorted(len(v) for v in seq.values())))


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


print("\n%3s %5s %9s %10s %12s %12s %10s" %
      ("h", "n", "mean|dW|", "persist", "dCT lin", "dCT x cond", "units um"))
for h in [1, 2, 3, 4]:
    C, dW, dCT, D_, F_, M_ = [], [], [], [], [], []
    for c, rs in seq.items():
        for j in range(len(rs) - h):
            a, b = rs[j], rs[j + h]
            C.append(c)
            dW.append((b["VB"] - a["VB"]) * 1000.0)
            dCT.append(b["time"] - a["time"])
            D_.append(a["DOC"]); F_.append(a["feed"]); M_.append(a["material"])
    gt = np.array(dW); ct = np.array(dCT); C = np.array(C)
    D_, F_, M_ = np.array(D_), np.array(F_), np.array(M_)
    m = ~np.isnan(gt) & ~np.isnan(ct)
    gt, ct, C, D_, F_, M_ = gt[m], ct[m], C[m], D_[m], F_[m], M_[m]
    p2 = np.full(len(gt), np.nan); p3 = np.full(len(gt), np.nan)
    for c in sorted(set(C.tolist())):
        te = C == c; tr = ~te
        if tr.sum() < 20 or te.sum() < 3:
            continue
        A = np.vstack([ct[tr], np.ones(tr.sum())]).T
        w = np.linalg.lstsq(A, gt[tr], rcond=None)[0]
        p2[te] = w[0] * ct[te] + w[1]
        A3 = np.column_stack([ct[tr], ct[tr] * D_[tr], ct[tr] * F_[tr], ct[tr] * M_[tr],
                              np.ones(tr.sum())])
        w3 = np.linalg.lstsq(A3, gt[tr], rcond=None)[0]
        B3 = np.column_stack([ct[te], ct[te] * D_[te], ct[te] * F_[te], ct[te] * M_[te],
                              np.ones(te.sum())])
        p3[te] = B3 @ w3
    mm = ~np.isnan(p2)
    m3 = ~np.isnan(p3)
    print("%3d %5d %9.2f %10.2f %12.2f %12.2f" %
          (h, len(gt), np.abs(gt).mean(), np.abs(gt).mean(),
           np.abs(gt[mm] - p2[mm]).mean(), np.abs(gt[m3] - p3[m3]).mean()))
    print("        dCT lin R2=%.3f   dCT x cond R2=%.3f" %
          (r2(gt[mm], p2[mm]), r2(gt[m3], p3[m3])))
print("\nNOTE: each step = one VB measurement interval (4-64 cuts in the raw logs);")
print("sample size is small (146 labelled cuts) - relative ordering only.")
