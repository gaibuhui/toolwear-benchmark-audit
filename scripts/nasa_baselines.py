"""NASA Ames milling baselines, v2.

Key addition over v1: two "leakage probe" baselines
  * B2b  linear/MLP fit on (time, DOC, feed, material) -- NO sensor signal at all
  * B5   kNN(k=1) in feature space
Both are cheap ways to expose random-split leakage: if the same cutting condition (case)
appears in both train and test, a model can simply recall it.

Units: VB in this dataset is in mm; 1 mm = 1000 um.  VB range 0 - 1.53 mm.
Literature reports MAE 0.0251-0.0311 mm (BRANN, ESWA) and R2 0.91-0.96 (IJAMT).
"""
import numpy as np, torch, torch.nn as nn, pyarrow.parquet as pq

P = "data/toolwear/nasa_milling/data.parquet"
CH = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
rows = [r for r in pq.ParquetFile(P).read().to_pylist()
        if r["VB"] is not None and not np.isnan(r["VB"])]


def feats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8: return np.zeros(11)
    x = x - x.mean(); sd = x.std() + 1e-12
    w = x * np.hanning(len(x)); f = np.abs(np.fft.rfft(w)); p = f ** 2 + 1e-12
    fr = np.fft.rfftfreq(len(x)); cen = float((fr * f).sum() / p.sum())
    bands = [float(p[e].sum() / p.sum()) for e in np.array_split(np.arange(len(f)), 4)]
    return np.array([x.mean(), sd, np.sqrt((x ** 2).mean()), np.abs(x).max(),
                     float(((x / sd) ** 4).mean() - 3), float(((x / sd) ** 3).mean()), cen, *bands])


Xs = np.stack([np.concatenate([feats(r[c]) for c in CH]) for r in rows])
PH = np.stack([[r["time"], r["DOC"], r["feed"], r["material"]] for r in rows]).astype(np.float64)
CAS = np.array([r["case"] for r in rows]); RUN = np.array([r["run"] for r in rows])
VB = np.array([r["VB"] for r in rows])
o = np.lexsort((RUN, CAS)); Xs, PH, CAS, RUN, VB = Xs[o], PH[o], CAS[o], RUN[o], VB[o]

prev = np.full(len(VB), np.nan)
for c in set(CAS.tolist()):
    i = np.where(CAS == c)[0]; prev[i[1:]] = VB[i[:-1]]


def z(A):
    return (A - A.mean(0)) / (A.std(0) + 1e-9)


Xz, Pz = z(Xs), z(PH)
print("n=%d  cases=%d  sensor feats=%d  phys feats=%d" % (len(VB), len(set(CAS.tolist())), Xz.shape[1], Pz.shape[1]))


def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
def met(y, p): return float(np.abs(y - p).mean()), float(np.sqrt(((y - p) ** 2).mean())), float(r2(y, p))


def fit_lin(A, y, lam=1.0):
    return np.linalg.lstsq(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y, rcond=None)[0]


def pred_mlp(A, y, B, epochs=500, h=64, seed=0):
    torch.manual_seed(seed)
    mu, s = y.mean(), y.std() + 1e-9
    net = nn.Sequential(nn.Linear(A.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    Z = torch.tensor(A, dtype=torch.float32); T = torch.tensor((y - mu) / s, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); l = lf(net(Z), T); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(B, dtype=torch.float32)).numpy().ravel() * s + mu


def knn1(Atr, ytr, Ate):
    d = ((Ate[:, None, :] - Atr[None, :, :]) ** 2).sum(-1)
    return ytr[d.argmin(1)]


def block(tr, te):
    ytr, yte = VB[tr], VB[te]
    out = {}
    out["B0 global mean"] = met(yte, np.full_like(yte, ytr.mean()))
    out["B1 persistence"] = met(yte, np.where(np.isnan(prev[te]), ytr.mean(), prev[te]))
    out["B2 run index (lin)"] = met(yte, np.polyval(np.polyfit(RUN[tr], ytr, 1), RUN[te]))
    out["B2b phys only (MLP)"] = met(yte, pred_mlp(Pz[tr], ytr, Pz[te]))
    out["B3a ridge(sensor)"] = met(yte, Xz[te] @ fit_lin(Xz[tr], ytr))
    out["B3b MLP(sensor)"] = met(yte, pred_mlp(Xz[tr], ytr, Xz[te]))
    out["B5 kNN1(sensor)"] = met(yte, knn1(Xz[tr], ytr, Xz[te]))
    out["B5b kNN1(phys)"] = met(yte, knn1(Pz[tr], ytr, Pz[te]))
    return out


def show(title, res):
    print("\n" + "=" * 78); print(title); print("=" * 78)
    print("%-24s %10s %10s %10s %10s" % ("baseline", "MAE mm", "MAE um", "RMSE um", "R2"))
    for k in sorted(res):
        m = res[k]; print("%-24s %10.4f %10.2f %10.2f %10.3f" % (k, m[0], m[0] * 1000, m[1] * 1000, m[2]))


acc = {}
for s in range(5):
    idx = np.random.RandomState(s).permutation(len(VB)); c = int(len(VB) * 0.7)
    for k, v in block(idx[:c], idx[c:]).items(): acc.setdefault(k, []).append(v)
show("P1  random 70/30  <- the protocol of BRANN / IJAMT / ESWA", {k: np.mean(v, 0) for k, v in acc.items()})

tr, te = [], []
for c in sorted(set(CAS.tolist())):
    i = np.where(CAS == c)[0]; k = int(len(i) * 0.7)
    tr.extend(i[:k].tolist()); te.extend(i[k:].tolist())
show("P2  within-condition temporal 70/30", block(np.array(tr), np.array(te)))

acc3 = {}
for c in sorted(set(CAS.tolist())):
    t = np.where(CAS == c)[0]; r = np.where(CAS != c)[0]
    for k, v in block(r, t).items(): acc3.setdefault(k, []).append(v)
show("P3  leave-one-condition-out (16 folds)", {k: np.mean(v, 0) for k, v in acc3.items()})

print("\n" + "=" * 78)
print("reference: reported values on the same dataset")
print("=" * 78)
print("BRANN             arXiv 2311.18620   MAE 0.0251  RMSE 0.0352  (mm)")
print("GLMamba+DA        ESWA 2026          MAE 0.0253  RMSE 0.0311  (mm)")
print("1DCNN-Transformer IJAMT 2026         RMSE <0.0936  R2 0.9069-0.9105")
print("MCL+meta          IJAMT 2025         R2 0.9369 -> 0.9565")
print("LSTM / CNN / MLP / SVR / LR baselines in BRANN: 0.0322 / 0.1835 / 0.183 / 0.17 / 0.1879 (MAE mm)")
