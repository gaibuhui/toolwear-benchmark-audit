"""D2 controls for review points 2 and 6.

Run:  python scripts/nasa_feat_ablation.py

(A) Feature-count-matched ablation.
    The paper claims "66 sensor features do not help, 4 process variables do".  A reviewer may
    reply that 66 features on ~150 samples overfits by construction.  We therefore repeat the P1
    protocol with sensor feature sets of the SAME order of magnitude as the process-variable set
    (k = 4, 10, 20 randomly drawn features, 10 draws each) and compare, within one script and one
    model family, against the 4 process variables.

(B) Data-quality probe and its effect.
    The public NASA mirror contains exactly ONE record whose six channels carry samples of order
    1e34; that single record makes 23 of the 66 derived feature columns span 30 orders of
    magnitude, so those columns are numerically dead for the other 145 rows.  We run every arm
    twice: on all 146 records (as published) and on the 145 clean records.

(C) Sensor-feature importance.
    Random-forest importance, |Spearman rho| against the wear label, and LASSO selection counts,
    to separate "the corpus's sensor features are uninformative" from "they are diluted".

Every arm uses the same estimator, the same protocol (P1 random 70/30, 5 seeds), the same
normalisation (train-fold statistics only) and a standardised target.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, Ridge
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

D = np.load("data/toolwear/nasa_feat.npz", allow_pickle=True)
Xs, PH, VB, ROWMAX = D["Xs"], D["PH"], D["VB"], D["ROWMAX"]
CH = list(D["CH"])
FPC = int(D["FEAT_PER_CH"])
FNAME = ["%s:%d" % (c, j) for c in CH for j in range(FPC)]
SEEDS = list(range(5))
ROWS = []


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def mae_um(y, p):
    return float(np.abs(y - p).mean()) * 1000.0


def mlp(Atr, ytr, Ate, seed):
    mu, sd = ytr.mean(), ytr.std() + 1e-12
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=seed,
                     learning_rate_init=3e-3, alpha=1e-3, tol=1e-7)
    m.fit(Atr, (ytr - mu) / sd)
    return m.predict(Ate) * sd + mu


def ridge(Atr, ytr, Ate):
    return Ridge(alpha=1.0).fit(Atr, ytr).predict(Ate)


def evalset(sel, kind, mask):
    """mask: boolean row selector. Returns (mean MAE um, sd, mean R2)."""
    X = sel; y = VB[mask]
    ms, rs = [], []
    for s in SEEDS:
        idx = np.random.RandomState(s).permutation(len(y))
        c = int(len(y) * 0.7)
        tr, te = idx[:c], idx[c:]
        A = X[mask]
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
        Atr, Ate = (A[tr] - mu) / sd, (A[te] - mu) / sd
        p = mlp(Atr, y[tr], Ate, s) if kind == "mlp" else ridge(Atr, y[tr], Ate)
        ms.append(mae_um(y[te], p))
        rs.append(r2(y[te], p))
    return float(np.mean(ms)), float(np.std(ms)), float(np.mean(rs))


def report(name, m, sd, r, tag):
    print("%-52s %8.1f %7.1f %9.3f" % (name, m, sd, r))
    ROWS.append((tag, name, m, sd, r))


MASK = {"all (n=146)": np.ones(len(VB), bool),
        "clean (n=145)": ROWMAX <= 1e6}

for tag, mask in MASK.items():
    print("=" * 84)
    print("D2  %s   p/process=%d  sensor=%d" % (tag, PH.shape[1], Xs.shape[1]))
    print("=" * 84)
    print("%-52s %8s %7s %9s" % ("arm (P1 random 70/30, 5 seeds)", "MAE um", "sd", "R2"))
    print("-" * 84)
    report("4 process variables            MLP", *evalset(PH, "mlp", mask), tag=tag)
    report("66 sensor features             MLP", *evalset(Xs, "mlp", mask), tag=tag)
    report("4 process variables            ridge", *evalset(PH, "ridge", mask), tag=tag)
    report("66 sensor features             ridge", *evalset(Xs, "ridge", mask), tag=tag)
    print("-" * 84)
    print("feature-count-matched sensor subsets (10 random draws each)")
    for k in (4, 10, 20):
        ms, rs = [], []
        for rep in range(10):
            sel = np.random.RandomState(1000 + rep).choice(Xs.shape[1], k, replace=False)
            m, _, r = evalset(Xs[:, sel], "mlp", mask)
            ms.append(m); rs.append(r)
        report("sensor subset k=%d                MLP" % k,
               float(np.mean(ms)), float(np.std(ms)), float(np.mean(rs)), tag=tag)
        print("      best/worst draw: %.1f / %.1f um" % (min(ms), max(ms)))
    print()

# ---------------- importance (clean data) ----------------
print("=" * 84)
print("sensor-feature importance (145 clean records)")
print("=" * 84)
m = ROWMAX <= 1e6
Xc, yc = Xs[m], VB[m]
Xz = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
rf = RandomForestRegressor(n_estimators=500, random_state=0).fit(Xz, yc)
imp = rf.feature_importances_
order = np.argsort(-imp)
rho = np.nan_to_num(np.array([abs(spearmanr(Xc[:, j], yc).statistic) for j in range(Xs.shape[1])]))
print("top-10 by RF importance:")
for j in order[:10]:
    print("   %-14s imp=%.4f   |rho(VB)|=%.3f" % (FNAME[j], imp[j], rho[j]))
print()
for thr in (0.1, 0.2, 0.3):
    print("features with |rho(VB)| < %.1f : %2d / %d" % (thr, int((rho < thr).sum()), Xs.shape[1]))
print("LassoCV nonzero coefficients  : %d / %d"
      % (int((np.abs(LassoCV(cv=5, random_state=0, max_iter=50000).fit(Xz, yc).coef_) > 1e-8).sum()),
         Xs.shape[1]))
print("RF importance of top-5        : %.3f   (uniform = %.4f)"
      % (float(imp[order[:5]].sum()), 1.0 / Xs.shape[1]))

# --- is the sensor-feature signal just a proxy for the process variables? ---
print()
print("partial correlation: sensor features vs the wear RESIDUAL after the 4 process variables")
Ph = (PH[m] - PH[m].mean(0)) / (PH[m].std(0) + 1e-9)
A = np.hstack([np.ones((len(yc), 1)), Ph])
beta = np.linalg.lstsq(A, yc, rcond=None)[0]
res = yc - A @ beta
print("  residual std = %.4f mm   (wear std = %.4f mm, so process vars explain R2=%.3f)"
      % (res.std(), yc.std(), 1 - res.var() / yc.var()))
rho_res = np.nan_to_num(np.array([abs(spearmanr(Xc[:, j], res).statistic)
                                  for j in range(Xs.shape[1])]))
for thr in (0.2, 0.3):
    print("  features with |rho(residual)| > %.1f : %d / %d"
          % (thr, int((rho_res > thr).sum()), Xs.shape[1]))
print("  largest |rho(residual)| = %.3f (%s);  largest |rho(VB)| = %.3f"
      % (rho_res.max(), FNAME[int(rho_res.argmax())], rho.max()))
print("  median |rho(VB)| = %.3f  ->  median |rho(residual)| = %.3f"
      % (float(np.median(rho)), float(np.median(rho_res))))

np.savez("data/toolwear/nasa_ablation.npz",
         rows=np.array(ROWS, dtype=object), imp=imp, rho=rho, rho_res=rho_res,
         fname=np.array(FNAME), allow_pickle=True)
print("\nsaved -> data/toolwear/nasa_ablation.npz")
