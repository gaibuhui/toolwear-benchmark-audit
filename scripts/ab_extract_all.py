"""Extract features for ALL 6418 rows of the Mendeley 3-machine milling dataset.

Two synchronized signal blocks per row:
  machine block @ 500 Hz  (2206 samples) : position deviation, tool position, axis torques, spindle torque
  sensor  block @ 25 kHz  (110250 samples): dynamometer force xyz
Streams row-group by row-group and caches to npz so later protocol sweeps are cheap.
"""
import os, time
import numpy as np
import pyarrow.parquet as pq

P = "data/toolwear/alphabeta/dataset.parquet"
OUT = "data/toolwear/ab_features.npz"

SEN = ["force_sensor_x", "force_sensor_y", "force_sensor_z"]
MAC = ["position_control_deviation_axis_x", "position_control_deviation_axis_y",
       "tool_position_x", "tool_position_y", "tool_position_z",
       "torque_axis_x", "torque_axis_y", "torque_axis_z", "torque_spindle"]
LBL = ["machine", "tool", "run", "cumulated_tool_contact_time", "wear"]
COLS = LBL + SEN + MAC


def f_sensor(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return np.zeros(16, dtype=np.float32)
    x = x - x.mean()
    n = len(x); sd = x.std() + 1e-12
    t = [x.std(), np.sqrt((x ** 2).mean()), np.abs(x).mean(), np.abs(x).max(),
         float(np.mean(x ** 3) / sd ** 3), float(np.mean(x ** 4) / sd ** 4 - 3),
         float(np.mean(np.abs(np.diff(x)))), float(((x[:-1] * x[1:]) < 0).mean())]
    X = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    X /= (X.sum() + 1e-12)
    nb = 8
    e = np.linspace(0, len(X), nb + 1).astype(int)
    return np.concatenate([t, [X[e[i]:e[i + 1]].sum() for i in range(nb)]]).astype(np.float32)


def f_machine(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.zeros(10, dtype=np.float32)
    x = x - x.mean()
    sd = x.std() + 1e-12
    return np.array([x.std(), np.abs(x).mean(), np.abs(x).max(),
                     float(np.mean(x ** 4) / sd ** 4 - 3),
                     float(np.mean(np.abs(np.diff(x)))),
                     float(((x[:-1] * x[1:]) < 0).mean()),
                     float(np.percentile(np.abs(x), 99)),
                     float(np.percentile(np.abs(x), 50)),
                     float(np.percentile(np.abs(x), 1)),
                     float(np.abs(np.diff(x)).max())], dtype=np.float32)


f = pq.ParquetFile(P)
n_rg = f.metadata.num_row_groups
print("row groups:", n_rg, flush=True)

CKPT = "data/toolwear/ab_feat_ckpt.npz"
start_rg = 0
feats, labels = [], []
if os.path.exists(CKPT):
    z = np.load(CKPT)
    feats = list(z["X"]); labels = list(z["L"]); start_rg = int(z["next_rg"])
    print("resume from rg", start_rg, "rows", len(feats), flush=True)

t0 = time.time()
for gi in range(start_rg, n_rg):
    t = f.read_row_group(gi, columns=COLS)
    for r in t.to_pylist():
        sen = np.concatenate([f_sensor(r[c]) for c in SEN])
        mac = np.concatenate([f_machine(r[c]) for c in MAC])
        feats.append(np.concatenate([sen, mac]))
        labels.append([r["machine"], r["tool"], r["run"],
                       r["cumulated_tool_contact_time"], r["wear"]])
    if gi % 5 == 0 or gi == n_rg - 1:
        np.savez_compressed(CKPT, X=np.stack(feats).astype(np.float32),
                            L=np.array(labels, dtype=np.float64), next_rg=gi + 1)
        print("  ckpt rg %3d/%d rows=%d  %.1f min" % (gi, n_rg, len(feats), (time.time() - t0) / 60), flush=True)

X = np.stack(feats).astype(np.float32)
L = np.array(labels, dtype=np.float64)
np.savez_compressed(OUT, X=X, L=L,
                    col_names=np.array(["machine", "tool", "run", "contact_time", "wear"]))
print("SAVED", OUT, X.shape, "%.1f min" % ((time.time() - t0) / 60))
