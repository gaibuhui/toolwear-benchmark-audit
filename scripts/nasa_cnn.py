"""N1 补强：原始波形 1D-CNN vs 过程变量回归 —— 堵住"你的 66 维手工特征太弱"这条质疑

争议点：§V-C 里"传感器特征反而更差"是在**手工统计特征**下得到的。审稿人可以说：
    "你的特征没提取出信息，原始波形端到端学习会好得多。"
本脚本用**原始 6 通道波形**训练 1D-CNN，与过程变量模型在同一协议下对比。

模型
  M1 MLP(process)                 t, d, f, material —— 零传感器（§V-C 的 T4）
  M2 CNN(raw 6ch)                 仅原始波形
  M3 CNN(raw 6ch) + process       在池化层拼接过程变量 -> **传感器是否在过程变量之外还有增量？**
  M4 MLP(66 hand-crafted feats)   手工传感器特征（§V-C 的 T5）
  M5 persistence                  平凡基线

协议 P1 随机 70/30 (5 seeds) | P2 同工况内时序 70/30 | P3 留一工况 (16 folds)
数据：NASA milling，146 条有 VB 的走刀；6 通道，50 kHz 级，每条 9000-15360 点 -> 中心裁剪 2048 点
"""
import numpy as np, torch, torch.nn as nn, pyarrow.parquet as pq

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P = "data/toolwear/nasa_milling/data.parquet"
CH = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
rows = [r for r in pq.ParquetFile(P).read().to_pylist() if r["VB"] is not None and not np.isnan(r["VB"])]
L = 2048
print("rows=%d  device=%s  window=%d" % (len(rows), DEV, L), flush=True)


def crop(x, mode="center", rng=None):
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= L:
        if mode == "center":
            s = (len(x) - L) // 2
        else:
            s = rng.randint(0, len(x) - L + 1)
        return x[s:s + L]
    return np.pad(x, (0, L - len(x)), mode="edge")


SIG = np.stack([[crop(r[c]) for c in CH] for r in rows])          # N, 6, L
MU = SIG.mean((0, 2), keepdims=True); SD = SIG.std((0, 2), keepdims=True) + 1e-9
SIG = (SIG - MU) / SD
# 预生成增广池：每条走刀 K 个随机时间窗（第 0 个为中心裁剪）。避免每 epoch 重做 numpy 裁剪（性能瓶颈）
K = 16
rng0 = np.random.RandomState(0)
AUG = np.empty((len(rows), K, len(CH), L), dtype=np.float32)
for n, r in enumerate(rows):
    for k in range(K):
        AUG[n, k] = [crop(r[c], "rand" if k else "center", rng0) for c in CH]
AUG = (AUG - MU[None]) / SD[None]
print("augmentation pool %s (%.0f MB)" % (AUG.shape, AUG.nbytes / 1e6), flush=True)
PROC = np.array([[r["time"], r["DOC"], r["feed"], r["material"]] for r in rows], dtype=np.float64)
PROCz = ((PROC - PROC.mean(0)) / (PROC.std(0) + 1e-9)).astype(np.float32)
VB = np.array([r["VB"] for r in rows])
CAS = np.array([r["case"] for r in rows]); RUN = np.array([r["run"] for r in rows])
o = np.lexsort((RUN, CAS)); SIG, PROCz, VB, CAS = SIG[o], PROCz[o], VB[o], CAS[o]
prev = np.full(len(VB), np.nan)
for c in set(CAS.tolist()):
    i = np.where(CAS == c)[0]; prev[i[1:]] = VB[i[:-1]]
prev0 = np.where(np.isnan(prev), 0.0, prev)


def hand_feats():
    F = []
    for n in range(len(VB)):
        row = []
        for ch in range(SIG.shape[1]):
            x = SIG[n, ch] - SIG[n, ch].mean(); sd = x.std() + 1e-12
            f = np.abs(np.fft.rfft(x * np.hanning(L))); p = f ** 2 + 1e-12
            fr = np.fft.rfftfreq(L)
            cen = float((fr * f).sum() / (f.sum() + 1e-12))
            bands = [float(p[e].sum() / p.sum()) for e in np.array_split(np.arange(len(f)), 4)]
            row += [sd, np.sqrt((x ** 2).mean()), np.abs(x).max(),
                    float(((x / sd) ** 4).mean() - 3), cen, *bands]
        F.append(row)
    return np.stack(F).astype(np.float32)


HF = hand_feats(); HF = (HF - HF.mean(0)) / (HF.std(0) + 1e-9)
print("signal %s  process %s  handcrafted %s" % (SIG.shape, PROCz.shape, HF.shape), flush=True)


class CNN(nn.Module):
    def __init__(self, nch=6, use_proc=False, nproc=4, h=128, dp=0.2):
        super().__init__()
        self.use_proc = use_proc
        self.body = nn.Sequential(
            nn.Conv1d(nch, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.drop = nn.Dropout(dp)
        self.head = nn.Sequential(nn.Linear(h + (nproc if use_proc else 0), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, p=None):
        z = self.body(x).squeeze(-1)
        z = self.drop(z)
        if self.use_proc:
            z = torch.cat([z, p], 1)
        return self.head(z).squeeze(-1)


def train_cnn(tr, use_proc, seed=0, epochs=300, aug=True):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    y = VB[tr]; mu, s = y.mean(), y.std() + 1e-9
    net = CNN(use_proc=use_proc).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-5)
    lf = nn.MSELoss(); net.train()
    Ptr = torch.tensor(PROCz[tr], dtype=torch.float32, device=DEV)
    Ttr = torch.tensor((y - mu) / s, dtype=torch.float32, device=DEV)
    for ep in range(epochs):
        kk = rng.randint(0, K, len(tr))
        xb = AUG[tr, kk]                                # 直接用预生成增广池
        Xb = torch.tensor(xb, dtype=torch.float32, device=DEV)
        opt.zero_grad()
        out = net(Xb, Ptr)
        loss = lf(out, Ttr)
        loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    net.eval()
    return net, mu, s


def predict_cnn(net, mu, s, te, use_proc):
    with torch.no_grad():
        X = torch.tensor(SIG[te], dtype=torch.float32, device=DEV)
        Pv = torch.tensor(PROCz[te], dtype=torch.float32, device=DEV)
        return net(X, Pv if use_proc else None).cpu().numpy() * s + mu


def mlp(A, y, B, epochs=500, h=64, seed=0):
    torch.manual_seed(seed)
    mu, s = y.mean(), y.std() + 1e-9
    net = nn.Sequential(nn.Linear(A.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)).to(DEV)
    Z = torch.tensor(A, dtype=torch.float32, device=DEV)
    T = torch.tensor((y - mu) / s, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); l = lf(net(Z).squeeze(-1), T); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(B, dtype=torch.float32, device=DEV)).squeeze(-1).cpu().numpy() * s + mu


def met(y, p):
    p = np.clip(p, 0, None)
    r2 = 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return float(np.abs(y - p).mean()) * 1000, float(np.sqrt(((y - p) ** 2).mean())) * 1000, float(r2)


def block(tr, te, seed=0, epochs=300):
    ytr, yte = VB[tr], VB[te]
    out = {}
    out["M5 persistence"] = met(yte, prev0[te])
    out["M1 MLP(process)"] = met(yte, mlp(PROCz[tr], ytr, PROCz[te], seed=seed))
    out["M4 MLP(66 handcrafted)"] = met(yte, mlp(HF[tr], ytr, HF[te], seed=seed))
    for nm, up in [("M2 CNN(raw 6ch)", False), ("M3 CNN(raw)+process", True)]:
        net, mu, s = train_cnn(tr, up, seed=seed, epochs=epochs)
        out[nm] = met(yte, predict_cnn(net, mu, s, te, up))
    return out


def show(title, res):
    print("\n" + "=" * 84); print(title); print("=" * 84)
    print("%-26s %10s %10s %9s" % ("model", "MAE um", "RMSE um", "R2"))
    for k in sorted(res):
        m = res[k]; print("%-26s %10.2f %10.2f %9.3f" % (k, m[0], m[1], m[2]))


acc = {}
for s in range(5):
    idx = np.random.RandomState(s).permutation(len(VB)); c = int(len(VB) * .7)
    for k, v in block(idx[:c], idx[c:], seed=s).items(): acc.setdefault(k, []).append(v)
    print("  P1 seed %d done" % s, flush=True)
show("P1  random 70/30 (5 seeds)  <- the protocol used by the published papers on this dataset",
     {k: np.mean(v, 0) for k, v in acc.items()})

tr, te = [], []
for c in sorted(set(CAS.tolist())):
    i = np.where(CAS == c)[0]; k = int(len(i) * .7)
    tr.extend(i[:k].tolist()); te.extend(i[k:].tolist())
show("P2  within-condition temporal 70/30", block(np.array(tr), np.array(te)))

acc3 = {}
for c in sorted(set(CAS.tolist())):
    t_ = np.where(CAS == c)[0]; r_ = np.where(CAS != c)[0]
    rr = block(r_, t_, epochs=150)
    for k, v in rr.items(): acc3.setdefault(k, []).append(v)
show("P3  leave-one-condition-out (16 folds)", {k: np.mean(v, 0) for k, v in acc3.items()})
