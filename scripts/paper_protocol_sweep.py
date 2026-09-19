"""Main experiment table v2: protocols x baselines on the Mendeley 3-machine milling dataset.

Fixes vs v1:
  * feature standardisation threshold tightened (sd > 1e-3) -> ridge no longer diverges
  * ridge gets proper lambda
  * adds B4: GRU over the per-tool run sequence (literature-style temporal model)
"""
import numpy as np, torch, torch.nn as nn

D = np.load("data/toolwear/ab_features.npz")
X = D["X"].astype(np.float64); L = D["L"]
M = L[:, 0].astype(int); T = L[:, 1].astype(int); R = L[:, 2].astype(int)
CT = L[:, 3]; W = L[:, 4]
order = np.lexsort((R, T, M))
X, M, T, R, CT, W = X[order], M[order], T[order], R[order], CT[order], W[order]
sd = X.std(0); keep = sd > 1e-3
Xk = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
print("n=%d  kept feats=%d" % (len(W), Xk.shape[1]))

prevW = np.full(len(W), np.nan)
for k in set(zip(M.tolist(), T.tolist())):
    idx = np.where((M == k[0]) & (T == k[1]))[0]
    prevW[idx[1:]] = W[idx[:-1]]
# NOTE convention: a fresh tool has no previous measurement; VB of a fresh tool ~= 0.
# Filling with the training mean inflates the persistence error on the 9 first-run samples.
prevW0 = np.where(np.isnan(prevW), 0.0, prevW)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def metrics(y, p):
    return np.abs(y - p).mean(), np.sqrt(((y - p) ** 2).mean()), r2(y, p)


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


class GRU(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.g = nn.GRU(d, h, batch_first=True)
        self.f = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.g(x); return self.f(o).squeeze(-1)


def gru_seq_eval(train_seqs, test_seqs, epochs=250, h=64):
    """train_seqs/test_seqs: list of (F[N,d], y[N]) 每把刀一条序列；返回测试预测列表。"""
    torch.manual_seed(0)
    net = GRU(train_seqs[0][0].shape[1], h)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); lf = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for F, y in train_seqs:
            p = net(torch.tensor(F[None], dtype=torch.float32))[0]
            loss = loss + lf(p, torch.tensor(y, dtype=torch.float32))
        loss = loss / len(train_seqs)
        loss.backward(); opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        for F, y in test_seqs:
            out.append((net(torch.tensor(F[None], dtype=torch.float32))[0].numpy(), y))
    return out


def seqs_for(mask_fn):
    """按 mask_fn(key) 把样本切成每把刀一条序列。"""
    tr, te = [], []
    for k in sorted(set(zip(M.tolist(), T.tolist()))):
        idx = np.where((M == k[0]) & (T == k[1]))[0]
        if len(idx) == 0:
            continue
        (tr if mask_fn(k) else te).append((Xk[idx], W[idx]))
    return tr, te


def add_gru(res, tr_seqs, te_seqs, tag):
    preds = gru_seq_eval(tr_seqs, te_seqs)
    y = np.concatenate([p[1] for p in preds]); p = np.concatenate([p[0] for p in preds])
    res["B4 GRU(seq)"] = metrics(y, p)
    return res


def show(title, res):
    print("\n" + "=" * 76); print(title); print("=" * 76)
    print("%-22s %10s %10s %9s" % ("baseline", "MAE", "RMSE", "R2"))
    for k in sorted(res):
        m = res[k]; print("%-22s %10.3f %10.3f %9.3f" % (k, m[0], m[1], m[2]))


def base_only(tr, te):
    ytr, yte = W[tr], W[te]
    out = {}
    out["B0 global mean"] = metrics(yte, np.full_like(yte, ytr.mean()))
    out["B1 persistence"] = metrics(yte, prevW0[te])
    A = np.vstack([CT[tr], np.ones(len(tr))]).T
    c = np.linalg.lstsq(A, ytr, rcond=None)[0]
    out["B2 contact time"] = metrics(yte, c[0] * CT[te] + c[1])
    out["B3a ridge(feat)"] = metrics(yte, Xk[te] @ ridge(Xk[tr], ytr))
    out["B3b MLP(feat)"] = metrics(yte, mlp_pred(Xk[tr], ytr, Xk[te]))
    return out


# ---------- P1 random 70/30 ----------
acc = {}
for s in range(5):
    idx = np.random.RandomState(s).permutation(len(W)); k = int(len(W) * 0.7)
    r = base_only(idx[:k], idx[k:])
    for kk, v in r.items(): acc.setdefault(kk, []).append(v)
show("P1 随机样本切分（文献标准，泄漏）", {k: np.mean(v, axis=0) for k, v in acc.items()})

# ---------- P2 per-tool temporal ----------
tr, te = [], []
for k in sorted(set(zip(M.tolist(), T.tolist()))):
    idx = np.where((M == k[0]) & (T == k[1]))[0]; c = int(len(idx) * 0.7)
    tr.extend(idx[:c].tolist()); te.extend(idx[c:].tolist())
tr, te = np.array(tr), np.array(te)
res = base_only(tr, te)
trs, tes = seqs_for(lambda k: True)   # 每把刀内部 70/30
trs2, tes2 = [], []
for F, y in trs + tes:
    n = int(len(y) * 0.7)
    trs2.append((F[:n], y[:n])); tes2.append((F[n:], y[n:]))
res = add_gru(res, trs2, tes2, "P2")
show("P2 同刀内时序切分 70/30", res)

# ---------- P3 leave-one-tool-out ----------
acc3 = {}
for k in sorted(set(zip(M.tolist(), T.tolist()))):
    t = np.where((M == k[0]) & (T == k[1]))[0]; rn = np.where(~((M == k[0]) & (T == k[1])))[0]
    r = base_only(rn, t)
    for k2, v in r.items(): acc3.setdefault(k2, []).append(v)
res3 = {k: np.mean(v, axis=0) for k, v in acc3.items()}
trs, tes = seqs_for(lambda k: k != (1, 1))   # 留出 M1T1 作 GRU 测试
res3 = add_gru(res3, trs, tes, "P3")
show("P3 留一刀（9 折；GRU 为留 M1T1）", res3)

# ---------- P4 leave-one-machine-out ----------
acc4 = {}
for m in [1, 2, 3]:
    t = np.where(M == m)[0]; rn = np.where(M != m)[0]
    r = base_only(rn, t)
    for k2, v in r.items(): acc4.setdefault(k2, []).append(v)
res4 = {k: np.mean(v, axis=0) for k, v in acc4.items()}
trs, tes = seqs_for(lambda k: k[0] != 2)
res4 = add_gru(res4, trs, tes, "P4")
show("P4 留一机床（3 折；GRU 为留 M2）", res4)

# ---------- P5 horizon increment ----------
print("\n" + "=" * 76)
print("P5 间隔增量协议  dW_h = wear(k+h) - wear(k)")
print("=" * 76)
print("%5s %8s %10s %12s %12s %14s" % ("h", "n", "dW mean", "B1 dW=0", "B2 dCT", "B3b MLP(dCT+feat)"))
for h in [1, 25, 50, 100]:
    rows = []
    for k in sorted(set(zip(M.tolist(), T.tolist()))):
        idx = np.where((M == k[0]) & (T == k[1]))[0]
        if len(idx) <= h: continue
        for j in range(len(idx) - h):
            a, b = idx[j], idx[j + h]
            rows.append((k[0], k[1], W[b] - W[a], CT[b] - CT[a], Xk[a]))
    gt = np.array([r[2] for r in rows]); dct = np.array([r[3] for r in rows])
    K = np.array([f"{r[0]}_{r[1]}" for r in rows]); F = np.stack([r[4] for r in rows])
    p2 = np.full(len(gt), np.nan); p3 = np.full(len(gt), np.nan)
    for kk in set(K.tolist()):
        t = K == kk; rn = ~t
        if rn.sum() < 30: continue
        A = np.vstack([dct[rn], np.ones(rn.sum())]).T
        c = np.linalg.lstsq(A, gt[rn], rcond=None)[0]
        p2[t] = c[0] * dct[t] + c[1]
        Ftr = np.column_stack([dct[rn], F[rn]]); Fte = np.column_stack([dct[t], F[t]])
        p3[t] = mlp_pred(Ftr, gt[rn], Fte)
    m = ~np.isnan(p2)
    print("%5d %8d %10.2f %12.3f %12.3f %14.3f" % (h, len(gt), gt.mean(), np.abs(gt).mean(),
                                                   np.abs(gt[m] - p2[m]).mean(), np.abs(gt[m] - p3[m]).mean()))
    print("        B2 dCT R2=%.3f   B3b R2=%.3f" % (r2(gt[m], p2[m]), r2(gt[m], p3[m])))
