"""C4 v3 — one clean, interpretable task so the numbers mean something.

Task: binary Ball vs InnerRace fault.  Test = size 28 mil (unseen severity); train = sizes 7/14/21 mil
      of the SAME two fault types, so the label space is identical in train and test and chance = 0.500.

Protocol ladder
  P1 random 70/30 over windows   - what the literature does
  P2 leave-one-load-out          - the "cross-condition" protocol widely used for domain adaptation
  P3 leave-one-severity-out      - removes bearing identity (Hendriks et al., MSSP 2022, recommend this)

Two feature sets, to separate the protocol effect from the feature effect:
  generic   18 time/spectral statistics
  +envelope 31 = generic + band-pass(2-6 kHz) -> Hilbert envelope -> amplitude at BPFO/BPFI/BSF/FTF
             harmonics (classic bearing-diagnostics features)
"""
import os, collections, numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq

ROOT = "data/bearing/cwru12k/data"
S, Y, R = [], [], []
for f in sorted(os.listdir(ROOT)):
    if not f.endswith(".parquet"): continue
    t = pq.ParquetFile(os.path.join(ROOT, f)).read()
    S.extend(t.column("signal").to_pylist()); Y.extend(t.column("label").to_pylist()); R.extend(t.column("rpm").to_pylist())
FS = 12000.0
Y = np.array(Y); RPM = np.asarray(R)
FT = np.array([y.split("_")[0] if "_" in y else "N" for y in Y])
SZ = np.array([int(y.split("_")[1]) if "_" in y else 0 for y in Y])
X = np.asarray(S, dtype=np.float64)
MASK = (FT == "B") | (FT == "IR")                     # binary task
print("binary subset: %d / %d windows" % (MASK.sum(), len(Y)), flush=True)

MULT = [3.5848, 5.4152, 4.7135, 0.3983]               # BPFO, BPFI, BSF, FTF (SKF 6205-2RS)


def hilbert_abs(x):
    n = len(x); Xf = np.fft.fft(x)
    h = np.zeros(n); h[0] = 1.0
    if n % 2 == 0: h[n // 2] = 1.0; h[1:n // 2] = 2.0
    else: h[1:(n + 1) // 2] = 2.0
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
    m = (fr >= 2000) & (fr <= 6000)
    Yf = np.zeros_like(Xf); Yf[m] = Xf[m]
    env = hilbert_abs(np.fft.irfft(Yf, n)); env = env - env.mean()
    E = np.abs(np.fft.rfft(env * np.hanning(n))); fe = np.fft.rfftfreq(n, d=1.0 / FS)
    fs = rpm / 60.0
    out = []
    for mu in MULT:
        for h in (1, 2, 3):
            i = int(np.argmin(np.abs(fe - mu * fs * h)))
            out.append(float(E[max(0, i - 2):i + 3].max()) / (E.mean() + 1e-12))
    out.append(float(((env / (env.std() + 1e-12)) ** 4).mean() - 3))
    return np.array(out)


idx = np.where(MASK)[0]
G = np.stack([generic(X[i]) for i in idx])
Ev = np.stack([envelope(X[i], RPM[i]) for i in idx])
Gz = (G - G.mean(0)) / (G.std(0) + 1e-9); Ez = (Ev - Ev.mean(0)) / (Ev.std(0) + 1e-9)
FSETS = {"generic(18)": Gz, "gen+envelope(31)": np.hstack([Gz, Ez])}
FTb = FT[idx]; SZb = SZ[idx]; RPMb = RPM[idx]
print("features", {k: v.shape for k, v in FSETS.items()}, flush=True)


def acc(y, p): return float((y == p).mean())


def knn(Atr, ytr, Ate, k=1, chunk=1024):
    ntr = (Atr ** 2).sum(1); out = []
    for i in range(0, len(Ate), chunk):
        A = Ate[i:i + chunk]
        d = (A ** 2).sum(1)[:, None] + ntr[None, :] - 2.0 * (A @ Atr.T)
        ii = np.argpartition(d, k, axis=1)[:, :k]
        for r in ii: out.append(collections.Counter(ytr[r].tolist()).most_common(1)[0][0])
    return np.array(out)


def mlp(Atr, ytr, Ate, epochs=400, h=128, seed=0):
    torch.manual_seed(seed)
    if Atr.shape[0] == 0 or len(set(ytr.tolist())) < 2: return np.zeros(len(Ate), dtype=int)
    net = nn.Sequential(nn.Linear(Atr.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
    Z = torch.tensor(Atr, dtype=torch.float32); T = torch.tensor(ytr, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad(); l = lf(net(Z), T); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Ate, dtype=torch.float32)).argmax(1).numpy()


def ev(F, tr, te, lab, seed=0):
    ytr = (lab[tr] == "IR").astype(int); yte = (lab[te] == "IR").astype(int)
    maj = collections.Counter(ytr.tolist()).most_common(1)[0][0]
    return {"B0 maj": acc(yte, np.full_like(yte, maj)),
            "kNN(1)": acc(yte, knn(F[tr], ytr, F[te], 1)),
            "MLP": acc(yte, mlp(F[tr], ytr, F[te], seed=seed))}


rows = []
for fn, F in FSETS.items():   # P1
    a = collections.defaultdict(list)
    for s in range(3):
        p = np.random.RandomState(s).permutation(len(idx)); c = int(len(idx) * .7)
        r = ev(F, p[:c], p[c:], FTb, seed=s)
        for k, v in r.items(): a[k].append(v)
    rows.append(("P1 random 70/30", fn, {k: np.mean(v) for k, v in a.items()}))
for fn, F in FSETS.items():   # P2
    a = collections.defaultdict(list)
    for rpm in sorted(set(RPMb.tolist())):
        r = ev(F, np.where(RPMb != rpm)[0], np.where(RPMb == rpm)[0], FTb)
        for k, v in r.items(): a[k].append(v)
    rows.append(("P2 leave-one-LOAD", fn, {k: np.mean(v) for k, v in a.items()}))
for fn, F in FSETS.items():   # P3
    tr = np.where(SZb != 28)[0]; te = np.where(SZb == 28)[0]
    rows.append(("P3 leave-one-SIZE", fn, ev(F, tr, te, FTb)))

print("\n" + "=" * 84)
print("CWRU binary Ball-vs-InnerRace  (train size 7/14/21 mil; chance = 0.500)")
print("=" * 84)
print("%-20s %-18s %8s %8s %8s" % ("protocol", "features", "B0 maj", "kNN(1)", "MLP"))
for pname, fn, d in rows:
    print("%-20s %-18s %8.3f %8.3f %8.3f" % (pname, fn, d["B0 maj"], d["kNN(1)"], d["MLP"]))
