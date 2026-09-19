"""C4 v2 — separate the PROTOCOL effect from the FEATURE effect on CWRU.

v1 result: with 18 generic time/spectral statistics,
    P1 random      kNN(1) = 1.000   MLP = 1.000
    P2 leave-load  kNN(1) = 0.999   MLP = 0.998
    P3 leave-size  kNN(1) = 0.498   MLP = 0.360
The P3 collapse could be either (a) genuine bearing-identity leakage removal, or (b) our features
simply have no physics in them.  v2 settles this by adding classic envelope-spectrum features
(band-pass 2-6 kHz -> Hilbert envelope -> spectrum amplitude at BPFO/BPFI/BSF/FTF harmonics).

If P3 recovers substantially with the physics features, the protocol (not the features) is what
made v1 collapse, and the honest story is "protocol choice dominates".  If P3 stays at chance,
the CWRU 4-class task is a recording-fingerprint task.
"""
import os, collections, numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq

ROOT = "data/bearing/cwru12k/data"
files = [os.path.join(ROOT, f) for f in sorted(os.listdir(ROOT)) if f.endswith(".parquet")]
S, Y, R = [], [], []
for p in files:
    t = pq.ParquetFile(p).read()
    S.extend(t.column("signal").to_pylist()); Y.extend(t.column("label").to_pylist()); R.extend(t.column("rpm").to_pylist())
FS = 12000.0
YL = np.array(Y); RPM = np.asarray(R)
FT = np.array([y.split("_")[0] if "_" in y else "N" for y in Y])
SZ = np.array([int(y.split("_")[1]) if "_" in y else 0 for y in Y])
X = np.asarray(S, dtype=np.float64)
print("windows=%d  X=%s" % (len(YL), X.shape), flush=True)

# CWRU drive-end SKF 6205-2RS JEM multiples
MULT = {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 4.7135, "FTF": 0.3983}


def hilbert_abs(x):
    n = len(x); Xf = np.fft.fft(x)
    h = np.zeros(n); h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0; h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(Xf * h))


def generic(x):
    x = x - x.mean(); sd = x.std() + 1e-12
    rms = np.sqrt((x ** 2).mean()); peak = np.abs(x).max()
    f = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), d=1.0 / FS); p = f ** 2 + 1e-12
    cen = float((fr * f).sum() / (f.sum() + 1e-12))
    spread = float(np.sqrt((((fr - cen) ** 2) * f).sum() / (f.sum() + 1e-12)))
    bands = [float(p[e].sum() / p.sum()) for e in np.array_split(np.arange(len(f)), 8)]
    return np.array([x.mean(), sd, rms, peak, peak / (rms + 1e-12), float(((x / sd) ** 4).mean() - 3),
                     float(((x / sd) ** 3).mean()), rms / (np.abs(x).mean() + 1e-12), cen, spread, *bands])


def envelope(x, rpm):
    n = len(x)
    Xf = np.fft.rfft(x); fr = np.fft.rfftfreq(n, d=1.0 / FS)
    mask = (fr >= 2000) & (fr <= 6000)
    Yf = np.zeros_like(Xf); Yf[mask] = Xf[mask]
    xb = np.fft.irfft(Yf, n)
    env = hilbert_abs(xb) - np.mean(hilbert_abs(xb))
    E = np.abs(np.fft.rfft(env * np.hanning(n)))
    fe = np.fft.rfftfreq(n, d=1.0 / FS)
    fs = rpm / 60.0
    out = []
    for m in MULT.values():
        for h in (1, 2, 3):
            tgt = m * fs * h
            i = int(np.argmin(np.abs(fe - tgt)))
            out.append(float(E[max(0, i - 2):i + 3].max()) / (E.mean() + 1e-12))    # normalised
    out.append(float(((env / (env.std() + 1e-12)) ** 4).mean() - 3))                # envelope kurtosis
    return np.array(out)


G = np.stack([generic(x) for x in X])
E = np.stack([envelope(X[i], RPM[i]) for i in range(len(X))])
print("generic", G.shape, " envelope", E.shape, flush=True)
Gz = (G - G.mean(0)) / (G.std(0) + 1e-9)
Ez = (E - E.mean(0)) / (E.std(0) + 1e-9)
FSETS = {"generic(18)": Gz, "generic+envelope(31)": np.hstack([Gz, Ez])}


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


def evaluate(F, tr, te, lab, seed=0):
    cls = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(cls)}
    ytr = np.array([c2i[c] for c in lab[tr]]); yte = np.array([c2i[c] for c in lab[te]])
    maj = collections.Counter(ytr.tolist()).most_common(1)[0][0]
    return {"B0 maj": acc(yte, np.full_like(yte, maj)),
            "kNN(1)": acc(yte, knn(F[tr], ytr, F[te], 1)),
            "MLP": acc(yte, mlp(F[tr], ytr, F[te], len(cls), seed=seed))}


def table(title, rows):
    print("\n" + "=" * 84); print(title); print("=" * 84)
    print("%-24s %-10s %8s %8s %8s" % ("protocol", "features", "B0 maj", "kNN(1)", "MLP"))
    for name, fd in rows:
        print("%-24s %-10s %8.3f %8.3f %8.3f" % (name[0], name[1], fd["B0 maj"], fd["kNN(1)"], fd["MLP"]))


print("\n\n############ TASK A  4-class fault type ############")
# P1 random
perm = np.random.RandomState(0).permutation(len(YL)); c = int(len(YL) * .7)
rows = []
for fname, F in FSETS.items():
    rows.append((("P1 random 70/30", fname), evaluate(F, perm[:c], perm[c:], FT)))
# P2 leave-one-load-out
for fname, F in FSETS.items():
    a = collections.defaultdict(list)
    for rpm in sorted(set(RPM.tolist())):
        r = evaluate(F, np.where(RPM != rpm)[0], np.where(RPM == rpm)[0], FT)
        for k, v in r.items(): a[k].append(v)
    rows.append((("P2 leave-one-LOAD", fname), {k: np.mean(v) for k, v in a.items()}))
# P3 leave-one-severity-out
for fname, F in FSETS.items():
    rows.append((("P3 leave-one-SIZE(28mil)", fname), evaluate(F, np.where(SZ != 28)[0], np.where(SZ == 28)[0], FT)))
table("TASK A  4-class  (P3 test set contains only B and IR -> chance = 0.500)", rows)

print("\n\n############ TASK B  12-class ############")
rows = []
perm = np.random.RandomState(100).permutation(len(YL)); c = int(len(YL) * .7)
for fname, F in FSETS.items():
    rows.append((("P1 random 70/30", fname), evaluate(F, perm[:c], perm[c:], YL)))
for fname, F in FSETS.items():
    a = collections.defaultdict(list)
    for rpm in sorted(set(RPM.tolist())):
        r = evaluate(F, np.where(RPM != rpm)[0], np.where(RPM == rpm)[0], YL)
        for k, v in r.items(): a[k].append(v)
    rows.append((("P2 leave-one-LOAD", fname), {k: np.mean(v) for k, v in a.items()}))
table("TASK B  12-class", rows)
