"""Recompute the persistence reference under ONE consistent convention.

Convention: for the first run of a tool there is no previous measurement.  Physically a fresh
tool has VB ~= 0, so prev_B := 0 (NOT the training mean).  The earlier scripts used the training
mean, which inflated the persistence error on exactly the 9 (P3) / 3 (P4) first-run samples and
made the deep model look better than it is.

This script prints the corrected persistence row for P1..P4 so the tables stay self-consistent.
"""
import numpy as np

D = np.load("data/toolwear/ab_features.npz")
L = D["L"]
M = L[:, 0].astype(int); T = L[:, 1].astype(int); R = L[:, 2].astype(int); W = L[:, 4]
o = np.lexsort((R, T, M)); M, T, R, W = M[o], T[o], R[o], W[o]
KEYS = sorted(set(zip(M.tolist(), T.tolist())))
SEQ = {k: np.where((M == k[0]) & (T == k[1]))[0] for k in KEYS}
prevW = np.full(len(W), np.nan)
for k in KEYS:
    i = SEQ[k]; prevW[i[1:]] = W[i[:-1]]
prev0 = np.where(np.isnan(prevW), 0.0, prevW)          # correct convention
prevM = prevW.copy()
# old convention: training-set mean, approximated per split below


def met(y, p):
    r2 = 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return float(np.abs(y - p).mean()), float(np.sqrt(((y - p) ** 2).mean())), float(r2)


rows = []
# P1 random 70/30 (5 seeds)
for tag, pv in [("P1 random", prev0)]:
    acc = []
    for s in range(5):
        idx = np.random.RandomState(s).permutation(len(W)); c = int(len(W) * 0.7)
        ytr = W[idx[:c]]
        p_old = np.where(np.isnan(prevW[idx[c:]]), ytr.mean(), prevW[idx[c:]])
        p_new = pv[idx[c:]]
        acc.append((met(W[idx[c:]], p_old), met(W[idx[c:]], p_new)))
    a = np.mean([x[0] for x in acc], 0); b = np.mean([x[1] for x in acc], 0)
    rows.append(("P1 random 70/30", a, b))

# P2 suffix of each tool
i2 = np.concatenate([SEQ[k][int(len(SEQ[k]) * .7):] for k in KEYS])
y2 = W[i2]
rows.append(("P2 within-tool temporal",
             met(y2, np.where(np.isnan(prevW[i2]), W.mean(), prevW[i2])), met(y2, prev0[i2])))

# P3 leave-one-tool-out (37.5% of samples are 'first samples' if we concatenate folds; here: all)
i3 = np.concatenate([SEQ[k] for k in KEYS]); y3 = W[i3]
rows.append(("P3 leave-one-tool-out",
             met(y3, np.where(np.isnan(prevW[i3]), W.mean(), prevW[i3])), met(y3, prev0[i3])))

# P4 leave-one-machine-out
i4 = np.concatenate([SEQ[k] for k in KEYS]); y4 = W[i4]
rows.append(("P4 leave-one-machine-out",
             met(y4, np.where(np.isnan(prevW[i4]), W.mean(), prevW[i4])), met(y4, prev0[i4])))

print("%-26s %18s %18s   %s" % ("protocol", "OLD (train-mean fill)", "NEW (prev=0 fill)", "n_first"))
print("%-26s %8s %10s %8s %10s" % ("", "MAE", "R2", "MAE", "R2"))
for name, a, b in rows:
    n_first = int(sum(np.isnan(prevW[k]).sum() for k in
                      ([i2] if "P2" in name else [i3])))
    print("%-26s %8.3f %10.3f %8.3f %10.3f   %d" % (name, a[0], a[2], b[0], b[2], n_first))

print("\nRMSE 影响（P3）: OLD %.3f  ->  NEW %.3f" % (rows[2][1][1], rows[2][2][1]))
print("首件样本数 = 9（9 把刀），平均真值 wear = %.2f um" % W[[SEQ[k][0] for k in KEYS]].mean())
