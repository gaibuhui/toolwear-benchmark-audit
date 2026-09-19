"""Paired statistical tests for the 28 protocol x model cells vs persistence (B1).

Inputs : data/toolwear/stats_dump.npz (per-sample |err| + test rows, from paper_stats_dump.py)
Tests  : Wilcoxon signed-rank on per-sample |err| differences (scipy if available, else
         normal approximation with tie correction); bootstrap 95% CI of the MAE difference
         (10,000 resamples; clusters = tools for P3, machines for P4, samples for P1/P2).
Also  : framing-B (measurement-event rows only) metrics for every model under P1.
Output : stats_results.md
"""
import numpy as np

D = dict(np.load("data/toolwear/stats_dump.npz"))
W = D["W"].astype(np.float64)
EV = D["ev_rows"].astype(bool)
PROTOS = ["P1", "P2", "P3", "P4"]
MODELS = ["B0", "B2", "B3a", "B3b", "B4a", "B4b", "B4c"]

try:
    from scipy.stats import wilcoxon as _wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def wilcoxon_p(d):
    """d: paired differences. Returns two-sided p."""
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return float("nan")
    r = np.argsort(np.argsort(np.abs(d))) + 1.0          # ranks with average ties
    # average ranks for ties
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    Wp = ranks[d > 0].sum()
    mu = n * (n + 1) / 4.0
    # tie-corrected variance
    _, cnt = np.unique(a, return_counts=True)
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - (cnt * (cnt ** 3 - 1)).sum() / 48.0
    z = (Wp - mu) / np.sqrt(sigma2)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(p)


def boot_ci(d, clusters, B=10000, seed=0):
    rng = np.random.RandomState(seed)
    means = np.empty(B)
    if clusters is None:
        n = len(d)
        for b in range(B):
            means[b] = d[rng.randint(0, n, n)].mean()
    else:
        uc = np.unique(clusters)
        idx_by_c = {c: np.where(clusters == c)[0] for c in uc}
        for b in range(B):
            pick = rng.choice(uc, len(uc), replace=True)
            ii = np.concatenate([idx_by_c[c] for c in pick])
            means[b] = d[ii].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def clusters_for(proto, rows):
    """cluster id per test row: tool key for P3, machine for P4, None else."""
    if proto in ("P1", "P2"):
        return None
    D0 = np.load("data/toolwear/ab_features.npz")["L"]
    MM = D0[:, 0].astype(int); TT = D0[:, 1].astype(int)
    return np.array(["%d_%d" % (MM[r], TT[r]) for r in rows])


lines = ["# 28-cell paired tests vs persistence (B1), seed-0 dump\n",
         "diff = MAE(model) - MAE(B1) on the same test rows; positive = persistence wins.\n",
         "bootstrap: sample-level (P1/P2), tool-cluster (P3), machine-cluster (P4, 3 clusters).\n",
         "| protocol | model | MAE diff | 95% CI | Wilcoxon p | n |",
         "|---|---|---|---|---|---|"]
for proto in PROTOS:
    s1 = D["%s|B1" % proto].astype(np.float64)          # signed errors
    r1 = D["%s|B1|rows" % proto]
    cl = clusters_for(proto, r1)
    a1 = np.abs(s1)
    for m in MODELS:
        sm = D["%s|%s" % (proto, m)].astype(np.float64)
        rm = D["%s|%s|rows" % (proto, m)]
        assert (rm == r1).all(), "row mismatch %s %s" % (proto, m)
        d = np.abs(sm) - a1
        lo, hi = boot_ci(d, cl)
        p = wilcoxon_p(d)
        lines.append("| %s | %s | %+.3f | [%+.3f, %+.3f] | %.2e | %d |" %
                     (proto, m, d.mean(), lo, hi, p, len(d)))
lines.append("\nscipy available: %s" % HAVE_SCIPY)

# ---------------- framing B (event rows only), P1 ----------------
lines.append("\n# Framing B: metrics restricted to measurement-event rows (P1, seed 0)\n")
lines.append("| model | MAE um (event rows) | R2 (event rows) | MAE um (all rows) |")
lines.append("|---|---|---|---|")
s1 = D["P1|B1"].astype(np.float64); r1 = D["P1|B1|rows"]
for m in ["B0", "B1", "B2", "B3a", "B3b", "B4a", "B4b", "B4c"]:
    sm = D["P1|%s" % m].astype(np.float64)
    rm = D["P1|%s|rows" % m]
    y = W[rm]
    sel = EV[rm]
    yb = y[sel]; eb = np.abs(sm[sel])
    r2b = 1 - (sm[sel] ** 2).sum() / (((yb - yb.mean()) ** 2).sum() + 1e-12)
    lines.append("| %s | %.3f | %.3f | %.3f |" % (m, eb.mean(), r2b, np.abs(sm).mean()))
open("stats_results.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines[:12]))
print("... written to stats_results.md")
