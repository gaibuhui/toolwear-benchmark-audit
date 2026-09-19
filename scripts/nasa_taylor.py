"""N1 强化：Taylor 刀具寿命律的显式形式，以及"传感器信号是否在 Taylor 律之外还有增量"

Taylor's tool-life relation, in the form actually usable per cut:
        VB  =  C * t^a * d^b * f^c * g(material) * [optional vibration/force terms]
with t = cumulative cutting time, d = depth of cut, f = feed (spindle speed is constant in the
NASA milling set, so it is absorbed into C).

The decisive question for the paper is NOT "can we fit VB" but:
    ** given a physics-legal model built only from process variables, does adding the
       force / vibration / acoustic-emission features reduce the error at all? **

Models
  T0 global mean                      | T1 persistence
  T2 Taylor log-linear (5 params)     | T3 Taylor log-quadratic (11 params)
  T4 MLP on (t,d,f,m)                 | T5 MLP on 66 sensor features
  T6 **Taylor + sensor-residual**     -> T2 fitted first, then a sensor MLP predicts its residual
  T7 kNN(1) on sensor features        (leakage probe)

Protocols: P1 random 70/30 (literature standard) | P2 within-condition temporal | P3 leave-one-condition-out
Units: VB in mm (range 0 - 1.53).  Literature on this dataset reports MAE 0.0251-0.0311 mm.
"""
import numpy as np, torch, torch.nn as nn, pyarrow.parquet as pq

P = "data/toolwear/nasa_milling/data.parquet"
CH = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
rows = [r for r in pq.ParquetFile(P).read().to_pylist() if r["VB"] is not None and not np.isnan(r["VB"])]


def sig_feats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8: return np.zeros(11)
    x = x - x.mean(); sd = x.std() + 1e-12
    f = np.abs(np.fft.rfft(x * np.hanning(len(x)))); p = f ** 2 + 1e-12
    fr = np.fft.rfftfreq(len(x)); cen = float((fr * f).sum() / p.sum())
    bands = [float(p[e].sum() / p.sum()) for e in np.array_split(np.arange(len(f)), 4)]
    return np.array([x.mean(), sd, np.sqrt((x ** 2).mean()), np.abs(x).max(),
                     float(((x / sd) ** 4).mean() - 3), float(((x / sd) ** 3).mean()), cen, *bands])


SEN = np.stack([np.concatenate([sig_feats(r[c]) for c in CH]) for r in rows])
T = np.array([r["time"] for r in rows], dtype=np.float64)
DOC = np.array([r["DOC"] for r in rows]); FEED = np.array([r["feed"] for r in rows])
MAT = np.array([r["material"] for r in rows], dtype=np.float64)
CAS = np.array([r["case"] for r in rows]); RUN = np.array([r["run"] for r in rows])
VB = np.array([r["VB"] for r in rows])
o = np.lexsort((RUN, CAS)); SEN, T, DOC, FEED, MAT, CAS, RUN, VB = \
    SEN[o], T[o], DOC[o], FEED[o], MAT[o], CAS[o], RUN[o], VB[o]

prev = np.full(len(VB), np.nan)
for c in set(CAS.tolist()):
    i = np.where(CAS == c)[0]; prev[i[1:]] = VB[i[:-1]]
prev0 = np.where(np.isnan(prev), 0.0, prev)

# sensor features standardised; process features used in *log* space for Taylor
SENz = (SEN - SEN.mean(0)) / (SEN.std(0) + 1e-9)
PROC = np.column_stack([T, DOC, FEED, MAT])
PROCz = (PROC - PROC.mean(0)) / (PROC.std(0) + 1e-9)

EPS = 0.005                     # 5 um offset so that VB = 0 (fresh tool) is representable in log space
LT = np.log(T + 1e-3); LD = np.log(DOC); LF = np.log(FEED)


def taylor_design(order=1):
    cols = [np.ones_like(LT), LT, LD, LF, MAT]
    if order >= 2:
        cols += [LT ** 2, LD ** 2, LF ** 2, LT * LD, LT * LF, LD * LF]
    return np.column_stack(cols)


def fit_lin(X, y, lam=1e-6):
    return np.linalg.lstsq(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y, rcond=None)[0]


def pred_mlp(A, y, B, epochs=500, h=64, seed=0):
    if len(set(np.round(y, 9).tolist())) < 2: return np.full(len(B), y.mean())
    torch.manual_seed(seed)
    mu, s = y.mean(), y.std() + 1e-9
    net = nn.Sequential(nn.Linear(A.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    Z = torch.tensor(A, dtype=torch.float32); Yy = torch.tensor((y - mu) / s, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); l = lf(net(Z), Yy); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(B, dtype=torch.float32)).numpy().ravel() * s + mu


def knn1(Atr, ytr, Ate):
    d = ((Ate ** 2).sum(1)[:, None] + (Atr ** 2).sum(1)[None, :] - 2 * (Ate @ Atr.T))
    return ytr[d.argmin(1)]


def met(y, p):
    p = np.clip(p, 0.0, None)
    r2 = 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return float(np.abs(y - p).mean()) * 1000, float(np.sqrt(((y - p) ** 2).mean())) * 1000, float(r2)


def block(tr, te, seed=0):
    ytr, yte = VB[tr], VB[te]
    out = {}
    out["T0 mean"] = met(yte, np.full_like(yte, ytr.mean()))
    out["T1 persistence"] = met(yte, prev0[te])
    # Taylor, fitted in log space
    for nm, order in [("T2 Taylor(log-lin)", 1), ("T3 Taylor(log-quad)", 2)]:
        Xd = taylor_design(order)
        w = fit_lin(Xd[tr], np.log(ytr + EPS))
        out[nm] = met(yte, np.exp(Xd[te] @ w) - EPS)
    out["T4 MLP(process)"] = met(yte, pred_mlp(PROCz[tr], ytr, PROCz[te], seed=seed))
    out["T5 MLP(66 sensor)"] = met(yte, pred_mlp(SENz[tr], ytr, SENz[te], seed=seed))
    # T6: Taylor then sensor-residual
    Xd = taylor_design(1)
    w = fit_lin(Xd[tr], np.log(ytr + EPS))
    base_tr = np.exp(Xd[tr] @ w) - EPS
    base_te = np.exp(Xd[te] @ w) - EPS
    resid = ytr - base_tr
    out["T6 Taylor+sensor-resid"] = met(yte, base_te + pred_mlp(SENz[tr], resid, SENz[te], seed=seed))
    out["T7 kNN1(sensor)"] = met(yte, knn1(SENz[tr], ytr, SENz[te]))
    return out, w


def show(title, res):
    print("\n" + "=" * 82); print(title); print("=" * 82)
    print("%-24s %10s %10s %9s" % ("model", "MAE um", "RMSE um", "R2"))
    for k in sorted(res):
        m = res[k]; print("%-24s %10.2f %10.2f %9.3f" % (k, m[0], m[1], m[2]))


acc = {}
Ws = []
for s in range(5):
    idx = np.random.RandomState(s).permutation(len(VB)); c = int(len(VB) * .7)
    r, w = block(idx[:c], idx[c:], seed=s)
    Ws.append(w)
    for k, v in r.items(): acc.setdefault(k, []).append(v)
show("P1  random 70/30  (5 seeds)  <- protocol of the published papers on this dataset",
     {k: np.mean(v, 0) for k, v in acc.items()})
w_mean = np.mean(Ws, 0)
print("\nTaylor 拟合系数（log 空间，5 seed 平均）:")
print("  %-28s %8.3f" % ("intercept", w_mean[0]))
print("  %-28s %8.3f" % ("a  (log cumulative time)", w_mean[1]))
print("  %-28s %8.3f" % ("b  (log depth of cut)", w_mean[2]))
print("  %-28s %8.3f" % ("c  (log feed)", w_mean[3]))
print("  %-28s %8.3f" % ("d  (material offset)", w_mean[4]))
print("  -> VB ~= C * t^%.2f * d^%.2f * f^%.2f * exp(%.2f*material)" % (w_mean[1], w_mean[2], w_mean[3], w_mean[4]))

tr, te = [], []
for c in sorted(set(CAS.tolist())):
    i = np.where(CAS == c)[0]; k = int(len(i) * .7)
    tr.extend(i[:k].tolist()); te.extend(i[k:].tolist())
r, _ = block(np.array(tr), np.array(te))
show("P2  within-condition temporal 70/30", r)

acc3 = {}
for c in sorted(set(CAS.tolist())):
    t_ = np.where(CAS == c)[0]; r_ = np.where(CAS != c)[0]
    rr, _ = block(r_, t_)
    for k, v in rr.items(): acc3.setdefault(k, []).append(v)
show("P3  leave-one-condition-out (16 folds)", {k: np.mean(v, 0) for k, v in acc3.items()})
