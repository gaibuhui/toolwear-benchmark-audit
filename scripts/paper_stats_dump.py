"""Per-sample error dump for paired statistical tests (Wilcoxon / bootstrap).

Reproduces the exact model configurations of paper_protocol_sweep.py / paper_gru_v2.py /
paper_gru_delta.py with ONE seed (seed 0) and saves per-sample absolute errors + test row
indices for every (protocol, model) cell, plus the event-row mask for the forward-fill
sensitivity framing.  Output: data/toolwear/stats_dump.npz
"""
import numpy as np, torch, torch.nn as nn, time, os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, flush=True)
D = np.load("data/toolwear/ab_features.npz")
X = D["X"].astype(np.float64); L = D["L"]
M = L[:, 0].astype(int); T = L[:, 1].astype(int); R = L[:, 2].astype(int)
CT = L[:, 3]; W = L[:, 4]
o = np.lexsort((R, T, M)); X, M, T, R, CT, W = X[o], M[o], T[o], R[o], CT[o], W[o]
sd = X.std(0); keep = sd > 1e-3
Xk = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
# B3a uses the fix_ridge.py configuration: stricter screening + winsorize + centred ridge
keep2 = X.std(0) > 0.05
Xr = np.clip((X[:, keep2] - X[:, keep2].mean(0)) / X[:, keep2].std(0), -5.0, 5.0)

KEYS = sorted(set(zip(M.tolist(), T.tolist())))
SEQ = {k: np.where((M == k[0]) & (T == k[1]))[0] for k in KEYS}
dCT = np.zeros(len(W)); prevW = np.full(len(W), np.nan)
for k in KEYS:
    i = SEQ[k]
    dCT[i[1:]] = CT[i[1:]] - CT[i[:-1]]
    prevW[i[1:]] = W[i[:-1]]
prevW0 = np.where(np.isnan(prevW), 0.0, prevW)

# event rows: a row whose wear differs from the previous row of the same tool
ev_rows = np.zeros(len(W), dtype=bool)
for k in KEYS:
    i = SEQ[k]
    ev_rows[i[1:]] = W[i[1:]] != W[i[:-1]]
print("event rows: %d / %d (%.1f%%)" % (ev_rows.sum(), len(W), 100 * ev_rows.mean()), flush=True)


def col(v): return ((v - v.mean()) / (v.std() + 1e-9))[:, None]
FEAT = np.column_stack([Xk, col(CT), col(dCT)])              # B4a
FEAT_S = np.column_stack([FEAT, col(prevW0)])                # B4b
FEAT_D = np.column_stack([Xk, col(CT), col(dCT), col(prevW0)])  # B4c


def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def ridge(Xtr, ytr, lam=100.0):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    return np.linalg.lstsq(A, Xtr.T @ ytr, rcond=None)[0]


def mlp_pred(Xtr, ytr, Xte, epochs=300, h=96, seed=0):
    torch.manual_seed(seed)
    Z = torch.tensor(Xtr, dtype=torch.float32); t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    net = nn.Sequential(nn.Linear(Z.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); lf = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); l = lf(net(Z), t); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32)).numpy().ravel()


class GRUNet(nn.Module):
    def __init__(self, d, h=128, nl=2, dp=0.1):
        super().__init__()
        self.rnn = nn.GRU(d, h, num_layers=nl, batch_first=True, dropout=dp)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x, h0=None):
        o, hn = self.rnn(x, h0); return self.head(o).squeeze(-1), hn


def _fit(model, batches, epochs, lr, wd, last_only=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)
    lf = nn.MSELoss(); model.train()
    for ep in range(epochs):
        opt.zero_grad(); tot = 0.0; n = 0
        for x, y in batches:
            p, _ = model(x)
            if last_only:
                p = p[:, -1]
            tot = tot + lf(p, y) * len(y); n += len(y)
        (tot / max(n, 1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    return model


def gru_seq(train_seqs, test_seqs, epochs, seed=0, warmup=False):
    torch.manual_seed(seed); np.random.seed(seed)
    ys = np.concatenate([s[1] for s in train_seqs])
    ymu, ysd = float(ys.mean()), float(ys.std())
    tb = [(torch.tensor(s[0], dtype=torch.float32, device=DEV),
           torch.tensor((s[1] - ymu) / ysd, dtype=torch.float32, device=DEV)) for s in train_seqs]
    net = GRUNet(train_seqs[0][0].shape[1]).to(DEV)
    _fit(net, tb, epochs, 3e-3, 1e-4); net.eval()
    out = []
    with torch.no_grad():
        for item in test_seqs:
            if warmup:
                Fp, yp, Ft, yt = item
                _, h0 = net(torch.tensor(Fp, dtype=torch.float32, device=DEV)[None])
                p, _ = net(torch.tensor(Ft, dtype=torch.float32, device=DEV)[None], h0)
            else:
                Ft, yt = item
                p, _ = net(torch.tensor(Ft, dtype=torch.float32, device=DEV)[None])
            out.append((p[0].cpu().numpy() * ysd + ymu, yt))
    return out


def seq_of(k, F):
    i = SEQ[k]; return F[i], W[i]


STORE = {}   # key "P1|B3b" -> abs err array; "P1|rows" -> te rows


def put(proto, model, te_rows, pred):
    STORE["%s|%s" % (proto, model)] = (pred - W[te_rows]).astype(np.float32)  # signed error
    STORE["%s|%s|rows" % (proto, model)] = te_rows.astype(np.int32)


def fast_models(tr, te):
    ytr = W[tr]
    out = {}
    out["B0"] = np.full(len(te), ytr.mean())
    out["B1"] = prevW0[te]
    A = np.vstack([CT[tr], np.ones(len(tr))]).T
    c = np.linalg.lstsq(A, ytr, rcond=None)[0]
    out["B2"] = c[0] * CT[te] + c[1]
    yc = ytr - ytr.mean()
    A2 = Xr[tr].T @ Xr[tr] + 1000.0 * np.eye(Xr.shape[1])
    wr = np.linalg.lstsq(A2, Xr[tr].T @ yc, rcond=None)[0]
    out["B3a"] = Xr[te] @ wr + ytr.mean()
    out["B3b"] = mlp_pred(Xk[tr], ytr, Xk[te])
    return out


def seqs_for_state(sa, test_keys=None, warm_keys=None):
    """P2-style: full-sequence train + (prefix,test) tuples."""
    F = FEAT_S if sa else FEAT
    trs, tes = [], []
    for k in KEYS:
        Fold, y = seq_of(k, F); n = int(len(y) * 0.7)
        trs.append((Fold[:n], y[:n]))
        tes.append((Fold[:n], y[:n], Fold[n:], y[n:]))
    return trs, tes


# ---------------- windows (shared by P1 for B4a/b/c) ----------------
print("building windows ...", flush=True)
WL = 16
win_a, win_b, win_c, win_y, win_rows = [], [], [], [], []
for k in KEYS:
    i = SEQ[k]
    for p in range(len(i)):
        s = max(0, p - WL + 1)
        blkA = FEAT[i[s:p + 1]]
        if len(blkA) < WL:
            blkA = np.vstack([np.repeat(blkA[:1], WL - len(blkA), 0), blkA])
        blkB = FEAT_S[i[s:p + 1]]
        if len(blkB) < WL:
            blkB = np.vstack([np.repeat(blkB[:1], WL - len(blkB), 0), blkB])
        blkC = FEAT_D[i[s:p + 1]]
        if len(blkC) < WL:
            blkC = np.vstack([np.repeat(blkC[:1], WL - len(blkC), 0), blkC])
        win_a.append(blkA); win_b.append(blkB); win_c.append(blkC)
        win_y.append(W[i[p]]); win_rows.append(i[p])
WX, WXS, WIN = np.stack(win_a), np.stack(win_b), np.stack(win_c)
WY, WROWS = np.array(win_y), np.array(win_rows)
print("windows:", WX.shape, flush=True)


def window_net(Xw, tr, yw, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    ymu, ysd = float(yw[tr].mean()), float(yw[tr].std())
    net = GRUNet(Xw.shape[2]).to(DEV)
    b = [(torch.tensor(Xw[i], dtype=torch.float32, device=DEV),
          torch.tensor((yw[i] - ymu) / ysd, dtype=torch.float32, device=DEV))
         for i in np.array_split(tr, 8) if len(i)]
    _fit(net, b, epochs, 3e-3, 1e-4, last_only=True)
    net.eval()
    return net, ymu, ysd


def window_pred(net, ymu, ysd, Xw, te):
    with torch.no_grad():
        out = [net(torch.tensor(Xw[i], dtype=torch.float32, device=DEV))[0][:, -1].cpu().numpy() * ysd + ymu
               for i in np.array_split(te, 8) if len(i)]
    return np.concatenate(out)


# ==================== P1 ====================
t0 = time.time()
print("\n[P1] fast models ...", flush=True)
rng = np.random.RandomState(0); perm = rng.permutation(len(W)); c = int(len(W) * 0.7)
tr, te = perm[:c], perm[c:]
fm = fast_models(tr, te)
for k, v in fm.items():
    put("P1", k, te, v)
# B4c (delta-anchored, windowed, 200 ep): predict dW, add anchor WB = prevW0 of the row
WB = prevW0[WROWS]; WYd = WY - WB
net, dmu, dsd = window_net(WIN, tr, WYd, 200, seed=0)
put("P1", "B4c", te, window_pred(net, dmu, dsd, WIN, te) + WB[te])
del net
net, mu, sd_ = window_net(WX, tr, WY, 250, seed=0)
put("P1", "B4a", te, window_pred(net, mu, sd_, WX, te)); del net
net, mu, sd_ = window_net(WXS, tr, WY, 250, seed=0)
put("P1", "B4b", te, window_pred(net, mu, sd_, WXS, te)); del net
print("P1 done %.1f min" % ((time.time() - t0) / 60), flush=True)

# ==================== P2 ====================
t0 = time.time()
print("\n[P2] ...", flush=True)
trr, tee = [], []
for k in KEYS:
    i = SEQ[k]; n = int(len(i) * 0.7)
    trr.extend(i[:n].tolist()); tee.extend(i[n:].tolist())
trr, tee = np.array(trr), np.array(tee)
fm = fast_models(trr, tee)
for k, v in fm.items():
    put("P2", k, tee, v)
for tag, F, ep in [("B4a", FEAT, 250), ("B4b", FEAT_S, 250)]:
    trs, tes = [], []
    for k in KEYS:
        Fo, y = seq_of(k, F); n = int(len(y) * 0.7)
        trs.append((Fo[:n], y[:n])); tes.append((Fo[:n], y[:n], Fo[n:], y[n:]))
    pr = gru_seq(trs, tes, ep, seed=0, warmup=True)
    put("P2", tag, tee, np.concatenate([p[0] for p in pr]))
# B4c
trs, tes = [], []
for k in KEYS:
    F, y = seq_of(k, FEAT_D); n = int(len(y) * 0.7)
    trs.append((F[:n], (y - prevW0[SEQ[k]])[:n]))
    tes.append((F[:n], y[:n], F[n:], y[n:]))
pr = gru_seq(trs, tes, 180, seed=0, warmup=True)
put("P2", "B4c", tee, np.concatenate([p[0] for p in pr]) + prevW0[tee])
print("P2 done %.1f min" % ((time.time() - t0) / 60), flush=True)

# ==================== P3 ====================
t0 = time.time()
print("\n[P3] ...", flush=True)
acc = {m: [] for m in ["B0", "B1", "B2", "B3a", "B3b"]}
rows3 = []
for k in KEYS:
    t = np.where((M == k[0]) & (T == k[1]))[0]; rn = np.where(~((M == k[0]) & (T == k[1])))[0]
    fm = fast_models(rn, t)
    for k2, v in fm.items():
        acc[k2].append(v)
    rows3.append(t)
te3 = np.concatenate(rows3)
for k2, v in acc.items():
    put("P3", k2, te3, np.concatenate(v))  # v holds predictions; put() stores signed error
for tag, F, ep in [("B4a", FEAT, 150), ("B4b", FEAT_S, 150)]:
    preds = []
    for k in KEYS:
        trs = [seq_of(kk, F) for kk in KEYS if kk != k]
        tes = [seq_of(k, F)]
        pr = gru_seq(trs, tes, ep, seed=0)
        preds.extend(pr)
    put("P3", tag, te3, np.concatenate([p[0] for p in preds]))
preds = []
for k in KEYS:
    trs = [(FEAT_D[SEQ[kk]], W[SEQ[kk]] - prevW0[SEQ[kk]]) for kk in KEYS if kk != k]
    ti = [(FEAT_D[SEQ[k]], prevW0[SEQ[k]])]
    pr = gru_seq(trs, ti, 150, seed=0)
    preds.extend(pr)
put("P3", "B4c", te3, np.concatenate([p[0] for p in preds]) + prevW0[te3])
print("P3 done %.1f min" % ((time.time() - t0) / 60), flush=True)

# ==================== P4 ====================
t0 = time.time()
print("\n[P4] ...", flush=True)
acc = {m: [] for m in ["B0", "B1", "B2", "B3a", "B3b"]}
rows4 = []
for m in [1, 2, 3]:
    t = np.where(M == m)[0]; rn = np.where(M != m)[0]
    fm = fast_models(rn, t)
    for k2, v in fm.items():
        acc[k2].append(v)
    rows4.append(t)
te4 = np.concatenate(rows4)
for k2, v in acc.items():
    put("P4", k2, te4, np.concatenate(v))
for tag, F, ep in [("B4a", FEAT, 150), ("B4b", FEAT_S, 150)]:
    preds = []
    for mm in [1, 2, 3]:
        trs = [seq_of(kk, F) for kk in KEYS if kk[0] != mm]
        tes = [seq_of(kk, F) for kk in KEYS if kk[0] == mm]
        pr = gru_seq(trs, tes, ep, seed=0)
        preds.extend(pr)
    put("P4", tag, te4, np.concatenate([p[0] for p in preds]))
preds = []
for mm in [1, 2, 3]:
    trs = [(FEAT_D[SEQ[kk]], W[SEQ[kk]] - prevW0[SEQ[kk]]) for kk in KEYS if kk[0] != mm]
    tes = [(FEAT_D[SEQ[kk]], prevW0[SEQ[kk]]) for kk in KEYS if kk[0] == mm]
    pr = gru_seq(trs, tes, 150, seed=0)
    preds.extend(pr)
put("P4", "B4c", te4, np.concatenate([p[0] for p in preds]) + prevW0[te4])
print("P4 done %.1f min" % ((time.time() - t0) / 60), flush=True)

# ---------------- save ----------------
STORE["ev_rows"] = ev_rows
STORE["W"] = W.astype(np.float32)
np.savez_compressed("data/toolwear/stats_dump.npz", **STORE)
print("\nSAVED stats_dump.npz  keys=%d" % len(STORE), flush=True)

# quick sanity: MAE table (should reproduce Table I within seed noise)
print("\n%-6s %8s %8s %8s %8s %8s %8s %8s" % ("proto", "B0", "B1", "B2", "B3a", "B3b", "B4a/b/c", ""))
for proto in ["P1", "P2", "P3", "P4"]:
    row = [proto]
    for m in ["B0", "B1", "B2", "B3a", "B3b", "B4a", "B4b", "B4c"]:
        row.append("%8.3f" % STORE["%s|%s" % (proto, m)].mean())
    print("%-6s %s" % (row[0], " ".join(row[1:])))
