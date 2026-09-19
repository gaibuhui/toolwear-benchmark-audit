"""D1 forward-fill sensitivity: persistence under (A) dataset convention (all rows,
forward-filled labels) vs (B) measurement-event rows only (time-to-event framing).

A measurement event is inferred as a row whose wear differs from the previous row of the
same (machine, tool); the first row of each tool is also an event. Under the forward-fill
convention the two framings differ ONLY in which rows enter the metric.
"""
import numpy as np, pyarrow.parquet as pq
from collections import defaultdict

P = "data/toolwear/alphabeta/dataset.parquet"
t = pq.ParquetFile(P).read(columns=["machine", "tool", "run", "cumulated_tool_contact_time", "wear"])
mch = np.asarray(t.column("machine")).astype(int)
tool = np.asarray(t.column("tool")).astype(int)
run = np.asarray(t.column("run")).astype(int)
ct = np.asarray(t.column("cumulated_tool_contact_time")).astype(float)
w = np.asarray(t.column("wear")).astype(float)

groups = defaultdict(list)
for i in range(len(w)):
    groups[(mch[i], tool[i])].append(i)

n_all, n_ev = 0, 0
absdw_all, dw_all = [], []          # all consecutive pairs within a tool
absdw_ev, dw_ev, dct_ev, gap_ev = [], [], [], []   # pairs whose second row is an event
yev_l, pev_l = [], []               # event targets and persistence predictions
first_zero_mae = []                 # fresh-tool convention handled separately
tools = 0
for k in sorted(groups):
    idx = sorted(groups[k], key=lambda i: run[i])
    ws = w[idx]; cs = ct[idx]
    tools += 1
    n_all += len(idx)
    d = np.diff(ws)
    absdw_all.append(np.abs(d)); dw_all.append(d)
    ev = np.where(d != 0)[0]           # j-th pair (idx[j], idx[j+1]) is an event pair
    n_ev += len(ev)
    absdw_ev.append(np.abs(d[ev])); dw_ev.append(d[ev])
    dct_ev.append(np.abs(cs[ev + 1] - cs[ev]))
    yev_l.append(ws[ev + 1]); pev_l.append(ws[ev])   # persistence: predict previous value
    gaps = np.diff(np.concatenate([[-1], ev, [len(idx) - 1]]))
    gap_ev.append(gaps[1:-1])

absdw_all = np.concatenate(absdw_all); dw_all = np.concatenate(dw_all)
absdw_ev = np.concatenate(absdw_ev); dw_ev = np.concatenate(dw_ev)
dct_ev = np.concatenate(dct_ev); gap_ev = np.concatenate(gap_ev)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


# persistence prediction = previous wear value; its error is |dW| per pair by construction
pred_all = np.zeros_like(dw_all)
pred_ev = np.zeros_like(dw_ev)
print("tools=%d  rows=%d  event pairs=%d  (%.1f%% of pairs)" %
      (tools, n_all, n_ev, 100.0 * n_ev / (n_all - tools)))
print("zero-increment fraction (rows): %.1f%%" % (100.0 * (dw_all == 0).mean()))
print()
print("%-46s %10s %10s %10s" % ("framing", "MAE um", "R2", "mean dCT"))
print("%-46s %10.3f %10.4f %10.1f" %
      ("A all rows (forward-fill convention)", absdw_all.mean(),
       r2(w[1:] if False else np.concatenate([w[sorted(groups[k], key=lambda i: run[i])[1:]] for k in sorted(groups)]),
          np.concatenate([w[sorted(groups[k], key=lambda i: run[i])[:-1]] for k in sorted(groups)])),
       np.abs(np.concatenate([np.abs(np.diff(ct[sorted(groups[k], key=lambda i: run[i])])) for k in sorted(groups)])).mean()))
yev = np.concatenate(yev_l); pev = np.concatenate(pev_l)
print("%-46s %10.3f %10.4f %10.1f" %
      ("B event rows only (time-to-event)", absdw_ev.mean(), r2(yev, pev), dct_ev.mean()))
print()
print("median/mean inspection gap (rows between events): %d / %.1f" %
      (np.median(gap_ev), gap_ev.mean()))
print("deflation factor of persistence MAE by forward fill: %.1fx" % (absdw_ev.mean() / absdw_all.mean()))
print("event |dW| distribution: p50=%.2f p90=%.2f p99=%.2f max=%.2f um" %
      tuple(np.percentile(absdw_ev, [50, 90, 99, 100])))
