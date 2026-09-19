"""B3a fix: stricter feature screening (sd>0.05) + fixed ridge lambda=1000."""
import numpy as np

D = np.load("data/toolwear/ab_features.npz")
X = D["X"].astype(np.float64); L = D["L"]
M = L[:, 0].astype(int); T = L[:, 1].astype(int); R = L[:, 2].astype(int)
CT = L[:, 3]; W = L[:, 4]
o = np.lexsort((R, T, M)); X, M, T, R, CT, W = X[o], M[o], T[o], R[o], CT[o], W[o]

keep = X.std(0) > 0.05
Xk = (X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0)
# 峰度类特征重尾 -> 标准化后离群值爆炸；winsorize 到 ±5（标准做法）
CLIP = 5.0
n_before = int((np.abs(Xk) > CLIP).sum())
Xk = np.clip(Xk, -CLIP, CLIP)
LAM = 1000.0
print("kept feats=%d  clipped %d values (%.3f%%)  lambda=%g"
      % (Xk.shape[1], n_before, 100 * n_before / Xk.size, LAM))

prevW = np.full(len(W), np.nan)
for k in set(zip(M.tolist(), T.tolist())):
    idx = np.where((M == k[0]) & (T == k[1]))[0]
    prevW[idx[1:]] = W[idx[:-1]]


def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
def met(y, p): return np.abs(y - p).mean(), np.sqrt(((y - p) ** 2).mean()), r2(y, p)


def one(tr, te):
    ytr, yte = W[tr], W[te]
    out = {}
    out["B0 mean"] = met(yte, np.full_like(yte, ytr.mean()))
    out["B1 persistence"] = met(yte, np.where(np.isnan(prevW[te]), 0.0, prevW[te]))
    A = np.vstack([CT[tr], np.ones(len(tr))]).T
    c = np.linalg.lstsq(A, ytr, rcond=None)[0]
    out["B2 contact"] = met(yte, c[0] * CT[te] + c[1])
    # B3a ridge：必须带截距（中心化目标 + 预测时加回训练均值）
    yc = ytr - ytr.mean()
    A2 = Xk[tr].T @ Xk[tr] + LAM * np.eye(Xk.shape[1])
    w = np.linalg.lstsq(A2, Xk[tr].T @ yc, rcond=None)[0]
    out["B3a ridge"] = met(yte, Xk[te] @ w + ytr.mean())
    return out


def show(tag, acc):
    print("\n=== %s ===" % tag)
    print("%-18s %10s %10s %9s" % ("baseline", "MAE", "RMSE", "R2"))
    for k in sorted(acc):
        m = np.mean(acc[k], axis=0)
        print("%-18s %10.3f %10.3f %9.3f" % (k, m[0], m[1], m[2]))


# P1
acc = {}
for s in range(5):
    idx = np.random.RandomState(s).permutation(len(W)); k = int(len(W) * 0.7)
    for kk, v in one(idx[:k], idx[k:]).items(): acc.setdefault(kk, []).append(v)
show("P1 随机切分（5 seeds 均值）", acc)

# P2
tr, te = [], []
for k in sorted(set(zip(M.tolist(), T.tolist()))):
    idx = np.where((M == k[0]) & (T == k[1]))[0]; c = int(len(idx) * 0.7)
    tr.extend(idx[:c].tolist()); te.extend(idx[c:].tolist())
show("P2 同刀内时序 70/30", {k: [v] for k, v in one(np.array(tr), np.array(te)).items()})

# P3
acc3 = {}
for k in sorted(set(zip(M.tolist(), T.tolist()))):
    te = np.where((M == k[0]) & (T == k[1]))[0]; tr = np.where(~((M == k[0]) & (T == k[1])))[0]
    for kk, v in one(tr, te).items(): acc3.setdefault(kk, []).append(v)
show("P3 留一刀（9 折均值）", acc3)

# P4
acc4 = {}
for mm in [1, 2, 3]:
    te = np.where(M == mm)[0]; tr = np.where(M != mm)[0]
    for kk, v in one(tr, te).items(): acc4.setdefault(kk, []).append(v)
show("P4 留一机床（3 折均值）", acc4)
