"""C4 — does the protocol bias transfer from tool wear to fault diagnosis?

Dataset: CWRU 12 kHz drive-end (HF `babylon9/cwru_12k_de`), 13,746 windows x 4,096 samples.
  * 12 labels = {Normal} x {B,IR,OR} x {7,14,21,28} mil
  * rpm in {1730,1750,1772,1797} == load {3,2,1,0} HP   <- grouping variable for the honest protocol

Protocols
  P1 random 70/30 over windows        <- what the literature does
  P2 leave-one-load-out (rpm)         <- train on 3 loads, test on the 4th
  P3 leave-one-severity-out           <- train on 7/14/21 mil, test on 28 mil (unseen severity)

Key baseline: B1 kNN(1).  Under P1 it is a pure *leakage probe*: adjacent 4,096-sample windows come
from the same recording and overlap, so the nearest neighbour of a test window is usually its own
neighbour from the training set.  If kNN(1) reaches the published 99-100 %, the reported accuracy
says nothing about the representation or the model.
"""
import os, collections, numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq

ROOT = "data/bearing/cwru12k/data"
files = [os.path.join(ROOT, f) for f in sorted(os.listdir(ROOT)) if f.endswith(".parquet")]
S, Y, R = [], [], []
for p in files:
    t = pq.ParquetFile(p).read()
    S.extend(t.column("signal").to_pylist()); Y.extend(t.column("label").to_pylist()); R.extend(t.column("rpm").to_pylist())
YL = np.array(Y); RPM = np.asarray(R)
FT = np.array([y.split("_")[0] if "_" in y else "N" for y in Y])
SZ = np.array([int(y.split("_")[1]) if "_" in y else 0 for y in Y])
print("windows=%d  labels=%s" % (len(YL), sorted(set(Y))))
print("rpm(load): %s   fault types: %s   sizes: %s"
      % (sorted(collections.Counter(R).items()), sorted(set(FT.tolist())), sorted(set(SZ.tolist()))))


def feats(x):
    x = x - x.mean(); sd = x.std() + 1e-12
    rms = np.sqrt((x ** 2).mean()); peak = np.abs(x).max()
    f = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), d=1.0 / 12000); p = f ** 2 + 1e-12
    cen = float((fr * f).sum() / (f.sum() + 1e-12))
    spread = float(np.sqrt((((fr - cen) ** 2) * f).sum() / (f.sum() + 1e-12)))
    bands = [float(p[e].sum() / p.sum()) for e in np.array_split(np.arange(len(f)), 8)]
    return np.array([x.mean(), sd, rms, peak, peak / (rms + 1e-12), float(((x / sd) ** 4).mean() - 3),
                     float(((x / sd) ** 3).mean()), rms / (np.abs(x).mean() + 1e-12), cen, spread, *bands])


F = np.stack([feats(x) for x in np.asarray(S, dtype=np.float64)])
F = (F - F.mean(0)) / (F.std(0) + 1e-9)
print("features", F.shape, flush=True)
EN = (F ** 2).sum(1)


def acc(y, p): return float((y == p).mean())


def knn(Atr, ytr, Ate, k=1, chunk=1024):
    ntr = (Atr ** 2).sum(1); out = []
    for i in range(0, len(Ate), chunk):
        A = Ate[i:i + chunk]
        d = (A ** 2).sum(1)[:, None] + ntr[None, :] - 2.0 * (A @ Atr.T)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        for r in idx:
            out.append(collections.Counter(ytr[r].tolist()).most_common(1)[0][0])
    return np.array(out)


def mlp(Atr, ytr, Ate, ncls, epochs=300, h=128, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Atr.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ncls))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
    Z = torch.tensor(Atr, dtype=torch.float32); T = torch.tensor(ytr, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad(); l = lf(net(Z), T); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Ate, dtype=torch.float32)).argmax(1).numpy()


def evaluate(tr, te, lab, seed=0, tag=""):
    cls = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(cls)}
    ytr, yte = np.array([c2i[c] for c in lab[tr]]), np.array([c2i[c] for c in lab[te]])
    out = {}
    maj = collections.Counter(ytr.tolist()).most_common(1)[0][0]
    out["B0 majority"] = acc(yte, np.full_like(yte, maj))
    out["B1 kNN(1)"] = acc(yte, knn(F[tr], ytr, F[te], 1))
    out["B1b kNN(5)"] = acc(yte, knn(F[tr], ytr, F[te], 5))
    out["B2 MLP(feat)"] = acc(yte, mlp(F[tr], ytr, F[te], len(cls), seed=seed))
    if tag:
        print("%-22s %8s %8s %8s %8s" % (tag, *["%.3f" % out[k] for k in
              ["B0 majority", "B1 kNN(1)", "B1b kNN(5)", "B2 MLP(feat)"]]), flush=True)
    return out


# ================= TASK A: 4-class fault type =================
print("\n" + "=" * 80)
print("TASK A  4-class fault type (Normal / Ball / InnerRace / OuterRace)")
print("=" * 80)
print("%-22s %8s %8s %8s %8s" % ("protocol", "B0 maj", "kNN(1)", "kNN(5)", "MLP"))

A1 = collections.defaultdict(list)
for s in range(3):
    perm = np.random.RandomState(s).permutation(len(YL)); c = int(len(YL) * .7)
    r = evaluate(perm[:c], perm[c:], FT, seed=s, tag="P1 random (seed %d)" % s)
    for k, v in r.items(): A1[k].append(v)
print("%-22s %8.3f %8.3f %8.3f %8.3f" % ("P1 random  3-seed mean",
      *[np.mean(A1[k]) for k in ["B0 majority", "B1 kNN(1)", "B1b kNN(5)", "B2 MLP(feat)"]]), flush=True)

A2 = collections.defaultdict(list)
for rpm in sorted(set(RPM.tolist())):
    tr = np.where(RPM != rpm)[0]; te = np.where(RPM == rpm)[0]
    r = evaluate(tr, te, FT, tag="P2 held-out load rpm=%d" % rpm)
    for k, v in r.items(): A2[k].append(v)
print("%-22s %8.3f %8.3f %8.3f %8.3f" % ("P2 leave-load 4-fold mean",
      *[np.mean(A2[k]) for k in ["B0 majority", "B1 kNN(1)", "B1b kNN(5)", "B2 MLP(feat)"]]), flush=True)

print("\n--- P3 leave-one-severity-out (train 7/14/21 mil, test 28 mil), 4-class ---")
r = evaluate(np.where(SZ != 28)[0], np.where(SZ == 28)[0], FT, tag="P3 unseen severity")
print("%-22s %8.3f %8.3f %8.3f %8.3f" % ("P3", *[r[k] for k in
      ["B0 majority", "B1 kNN(1)", "B1b kNN(5)", "B2 MLP(feat)"]]))

# ================= TASK B: 12-class =================
print("\n" + "=" * 80)
print("TASK B  12-class (fault type x severity)")
print("=" * 80)
print("%-22s %8s %8s %8s" % ("protocol", "B0 maj", "kNN(1)", "MLP"))
B1_ = collections.defaultdict(list)
for s in range(3):
    perm = np.random.RandomState(100 + s).permutation(len(YL)); c = int(len(YL) * .7)
    r = evaluate(perm[:c], perm[c:], YL, seed=s)
    for k, v in r.items(): B1_[k].append(v)
print("%-22s %8.3f %8.3f %8.3f" % ("P1 random  3-seed mean",
      *[np.mean(B1_[k]) for k in ["B0 majority", "B1 kNN(1)", "B2 MLP(feat)"]]), flush=True)
B2_ = collections.defaultdict(list)
for rpm in sorted(set(RPM.tolist())):
    tr = np.where(RPM != rpm)[0]; te = np.where(RPM == rpm)[0]
    r = evaluate(tr, te, YL)
    for k, v in r.items(): B2_[k].append(v)
print("%-22s %8.3f %8.3f %8.3f" % ("P2 leave-load 4-fold mean",
      *[np.mean(B2_[k]) for k in ["B0 majority", "B1 kNN(1)", "B2 MLP(feat)"]]), flush=True)
