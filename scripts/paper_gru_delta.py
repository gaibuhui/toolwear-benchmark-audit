"""B4c: increment-anchored GRU.

Formulation:  dW_k = wear_k - prev_measured_wear_k,   prediction = prev_measured + GRU(features)
The previous measurement is a *residual anchor*, so the network only has to learn the increment.
This is the fairest possible test of "can ANY learned model beat the persistence rule?":
it beats persistence if and only if it predicts the increment better than the constant 0.

If the trained network collapses to dW_hat ~= 0, that is not a failure of training - it is the
finding: persistence is the optimal learnable solution for this label.
"""
import numpy as np, torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = np.load("data/toolwear/ab_features.npz")
X = D["X"].astype(np.float64); L = D["L"]
M = L[:, 0].astype(int); T = L[:, 1].astype(int); R = L[:, 2].astype(int)
CT = L[:, 3]; W = L[:, 4]
o = np.lexsort((R, T, M)); X, M, T, R, CT, W = X[o], M[o], T[o], R[o], CT[o], W[o]
sd = X.std(0); keep = sd > 1e-3
Xs = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]

KEYS = sorted(set(zip(M.tolist(), T.tolist())))
SEQ = {k: np.where((M == k[0]) & (T == k[1]))[0] for k in KEYS}
dCT = np.zeros(len(W)); prevW = np.full(len(W), np.nan)
for k in KEYS:
    i = SEQ[k]
    dCT[i[1:]] = CT[i[1:]] - CT[i[:-1]]
    prevW[i[1:]] = W[i[:-1]]
prevWf = np.where(np.isnan(prevW), 0.0, prevW)


def col(v): return ((v - v.mean()) / (v.std() + 1e-9))[:, None]
FEAT = np.column_stack([Xs, col(CT), col(dCT), col(prevWf)])
print("n=%d seqs=%d dim=%d device=%s" % (len(W), len(KEYS), FEAT.shape[1], DEV), flush=True)


def met(y, p):
    r2 = 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return float(np.abs(y - p).mean()), float(np.sqrt(((y - p) ** 2).mean())), float(r2)


class GRU(nn.Module):
    def __init__(self, d, h=128, nl=2, dp=0.1):
        super().__init__()
        self.rnn = nn.GRU(d, h, num_layers=nl, batch_first=True, dropout=dp)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, h0=None):
        o, hn = self.rnn(x, h0)
        return self.head(o).squeeze(-1), hn


def run(train_seqs, test_seqs, epochs=180, h=128, seed=0, warmup=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dy = np.concatenate([s[1] for s in train_seqs])          # increments on the training fold
    dmu, dsd = float(dy.mean()), float(dy.std())
    tb = [(torch.tensor(s[0], dtype=torch.float32, device=DEV),
           torch.tensor((s[1] - dmu) / dsd, dtype=torch.float32, device=DEV)) for s in train_seqs]
    net = GRU(FEAT.shape[1], h, 2).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=6e-5)
    lf = nn.MSELoss(); net.train()
    for ep in range(epochs):
        opt.zero_grad(); tot = 0.0; n = 0
        for x, y in tb:
            p, _ = net(x); tot = tot + lf(p, y) * len(y); n += len(y)
        (tot / n).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    net.eval()
    out, dhats = [], []
    with torch.no_grad():
        for it in test_seqs:
            if warmup:
                Fp, dp_, Ft, dt_, base = it
                _, h0 = net(torch.tensor(Fp, dtype=torch.float32, device=DEV)[None])
                p, _ = net(torch.tensor(Ft, dtype=torch.float32, device=DEV)[None], h0)
            else:
                Ft, dt_, base = it
                p, _ = net(torch.tensor(Ft, dtype=torch.float32, device=DEV)[None])
            dh = p[0].cpu().numpy() * dsd + dmu
            out.append((base + dh, dt_ + base)); dhats.append(dh)
    return out, np.concatenate(dhats)


def seq_k(k): 
    i = SEQ[k]; d = W[i] - prevWf[i]
    return FEAT[i], d, W[i]


# ---------- P2 within-tool temporal ----------
print("\n" + "=" * 78); print("P2  within-tool temporal 70/30   (increment-anchored GRU)"); print("=" * 78)
res = {}
trs, tes = [], []
for k in KEYS:
    F, d, w = seq_k(k); n = int(len(w) * 0.7)
    trs.append((F[:n], d[:n]))
    tes.append((F[:n], d[:n], F[n:], d[n:], prevWf[SEQ[k]][n:]))
_y2 = np.concatenate([W[SEQ[k]][int(len(SEQ[k]) * 0.7):] for k in KEYS])
_p2 = np.concatenate([prevWf[SEQ[k]][int(len(SEQ[k]) * 0.7):] for k in KEYS])
res["B1 persistence (dW=0)"] = met(_y2, _p2)
preds, dhat = run(trs, tes, seed=0, warmup=True)
y = np.concatenate([p[1] for p in preds]); p = np.concatenate([p[0] for p in preds])
res["B4c GRU(delta-anchored)"] = met(y, p)
print("%-26s %9s %9s %9s" % ("method", "MAE", "RMSE", "R2"))
for k in sorted(res): print("%-26s %9.3f %9.3f %9.3f" % (k, *res[k]))
base = np.concatenate([t[4] for t in tes]); true_d = y - base
print("true increment : mean %.3f  std %.3f" % (true_d.mean(), true_d.std()))
print("pred increment : mean %.3f  std %.3f" % (dhat.mean(), dhat.std()))

# ---------- P3 leave-one-tool-out ----------
print("\n" + "=" * 78); print("P3  leave-one-tool-out (9 folds, increment-anchored GRU)"); print("=" * 78)
allp = []
for s in range(1):
    preds, dhat = [], []
    for k in KEYS:
        trs = [seq_k(kk)[:2] for kk in KEYS if kk != k]
        F, d, w = seq_k(k)
        ti = [(F, d, prevWf[SEQ[k]])]
        pr, dh = run(trs, ti, epochs=150, seed=s)
        preds.extend(pr); dhat.append(dh)
    y = np.concatenate([p[1] for p in preds]); p = np.concatenate([p[0] for p in preds])
    allp.append(met(y, p))
m = np.mean(allp, 0)
pw = np.concatenate([prevWf[SEQ[k]] for k in KEYS])
yw = np.concatenate([W[SEQ[k]] for k in KEYS])
print("%-26s %9s %9s %9s" % ("method", "MAE", "RMSE", "R2"))
print("%-26s %9.3f %9.3f %9.3f" % ("B1 persistence (dW=0)", *met(yw, pw)))
print("%-26s %9.3f %9.3f %9.3f" % ("B4c GRU(delta-anchored)", *m))
print("predicted increment: mean %.3f um  std %.3f" % (np.concatenate(dhat).mean(), np.concatenate(dhat).std()))


# ---------- P1 windowed (literature standard) ----------
print("\n" + "=" * 78); print("P1  random 70/30 windowed (increment-anchored)"); print("=" * 78)
WL = 16
win, wy, wb = [], [], []
for k in KEYS:
    i = SEQ[k]
    for p in range(len(i)):
        s = max(0, p - WL + 1)
        blk = FEAT[i[s:p + 1]]
        if len(blk) < WL:
            blk = np.vstack([np.repeat(blk[:1], WL - len(blk), 0), blk])
        win.append(blk); wy.append(W[i[p]] - prevWf[i[p]]); wb.append(prevWf[i[p]])
WIN = np.stack(win); WYv = np.array(wy); WB = np.array(wb)
ACC = []
for s in range(3):
    rng = np.random.RandomState(s); perm = rng.permutation(len(WYv)); c = int(len(WYv) * .7)
    tr, te = perm[:c], perm[c:]
    torch.manual_seed(s)
    dmu, dsd = float(WYv[tr].mean()), float(WYv[tr].std())
    net = GRU(WIN.shape[2], 128, 2).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=6e-5)
    lf = nn.MSELoss(); net.train()
    for ep in range(200):
        opt.zero_grad(); tot = 0.0; n = 0
        for ii in np.array_split(tr, 8):
            p, _ = net(torch.tensor(WIN[ii], dtype=torch.float32, device=DEV))
            yv = torch.tensor((WYv[ii] - dmu) / dsd, dtype=torch.float32, device=DEV)
            tot = tot + lf(p[:, -1], yv) * len(ii); n += len(ii)
        (tot / n).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        pr = []
        for ii in np.array_split(te, 8):
            p, _ = net(torch.tensor(WIN[ii], dtype=torch.float32, device=DEV))
            pr.append((p[:, -1].cpu().numpy() * dsd + dmu) + WB[ii])
    pred = np.concatenate(pr)
    ACC.append(met(WYv[te] + WB[te], pred))
    ACC[-1] = met(WYv[te] + WB[te], pred)
print("%-26s %9.3f %9.3f %9.3f" % ("B4c GRU(delta-anchored)", *np.mean(ACC, 0)))
print("%-26s %9.3f %9.3f %9.3f" % ("B1 persistence (dW=0)",
      *met(WYv + WB, WB)))

# ---------- P4 leave-one-machine-out ----------
print("\n" + "=" * 78); print("P4  leave-one-machine-out (3 folds, increment-anchored)"); print("=" * 78)
preds = []
for m in [1, 2, 3]:
    trs = [seq_k(kk)[:2] for kk in KEYS if kk[0] != m]
    for kk in KEYS:
        if kk[0] != m: continue
        F, d, w = seq_k(kk)
        pr, dh = run(trs, [(F, d, prevWf[SEQ[kk]])], epochs=150, seed=0)
        preds.extend(pr)
yy = np.concatenate([p[1] for p in preds]); pp = np.concatenate([p[0] for p in preds])
print("%-26s %9.3f %9.3f %9.3f" % ("B4c GRU(delta-anchored)", *met(yy, pp)))
print("%-26s %9.3f %9.3f %9.3f" % ("B1 persistence (dW=0)", *met(W, prevWf)))
