"""N2 实证：标签测量粒度决定平凡基线的强度（四个数据集横向对比）

数据集（同一物理量 VB/wear，但标签的**重采样方式**完全不同）
  D1 Mendeley/LUH 3-machine milling   6,418 走刀 / 9 刀     标签=稀疏测量后**前向填充**（每次走刀一个值）
  D2 NASA Ames milling                146 次 VB 测量 / 16 工况 标签=**稀疏、不填充**（只在测量点评定）
  D3 PHM2010（Katulu uniwear bundle） 8,515 行 / c1,c4,c6   标签=**密集插值**（805 个唯一值）
  D4 NUAA 正交（Katulu uniwear bundle）31,388 行 / W1-W9     标签=**密集插值**（26,290 个唯一值）

对每个数据集报告：
  n_steps                预测步数（该分组内相邻步对数）
  uniq/steps             标签唯一值占比
  %zero                  相邻步增量恰为 0 的比例
  mean|dW|               平均绝对增量（µm）——**这正好等于 persistence 的 MAE**
  range                  VB 量程（µm）
  pers MAE / range       persistence 相对误差（唯一可跨数据集比较的量）
  pers R2                persistence 的决定系数
  MLP / ridge R2         同一划分下学习模型的表现（P1 随机 / P3 留一实验）

结论预期：persistence 的 MAE **恒等于** mean|dW|（构造使然），而 mean|dW| 由标签的重采样方式决定，
因此 R² 在数据集之间不可比；唯一可比的是 MAE/range。
"""
import csv, os, numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq

DEV = "cpu"


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def stat_block(name, groups, note):
    """groups: list of (gid, order_index, y, X) —— 同一组内按 order_index 排序。
    persistence 只在**组内**相邻对上评估（首件无上次测量，不计入）。"""
    ylist, preds, trues = [], [], []
    for gid, ordi, y, X in groups:
        o = np.argsort(ordi)
        ys = y[o]
        ylist.append(ys)
        if len(ys) > 1:
            preds.append(ys[:-1]); trues.append(ys[1:])       # 组内 persistence 配对
    Y = np.concatenate(ylist)
    P = np.concatenate(preds); T = np.concatenate(trues)
    d = T - P
    dW_abs = np.abs(d)
    return {
        "name": name, "note": note, "n_steps": int(len(dW_abs)),
        "uniq/steps": len(np.unique(Y)) / len(Y),
        "pct_zero": float((d == 0).mean()) * 100,
        "mean|dW|_um": float(dW_abs.mean()) * 1000,
        "range_um": float(Y.max() - Y.min()) * 1000,
        "pers_R2": float(r2(T, P)),
    }


rows = []

# ---------------- D1 Mendeley ----------------
D = np.load("data/toolwear/ab_features.npz")
L = D["L"]; M_, T_, R_, W_ = L[:, 0].astype(int), L[:, 1].astype(int), L[:, 2].astype(int), L[:, 4]
o = np.lexsort((R_, T_, M_)); M_, T_, R_, W_ = M_[o], T_[o], R_[o], W_[o]
g1 = [((m, t), R_[(M_ == m) & (T_ == t)], W_[(M_ == m) & (T_ == t)] / 1000.0,
       np.zeros((int(((M_ == m) & (T_ == t)).sum()), 1))) for m in (1, 2, 3) for t in range(1, 10)
      if ((M_ == m) & (T_ == t)).sum() > 0]
rows.append(stat_block("D1 Mendeley/LUH 3-machine", g1, "forward-filled per cut"))

# ---------------- D2 NASA milling ----------------
P = "data/toolwear/nasa_milling/data.parquet"
nr = [r for r in pq.ParquetFile(P).read().to_pylist() if r["VB"] is not None and not np.isnan(r["VB"])]
g2 = []
for c in sorted(set(r["case"] for r in nr)):
    sub = sorted([r for r in nr if r["case"] == c], key=lambda r: r["run"])
    g2.append((c, np.array([r["run"] for r in sub], float), np.array([r["VB"] for r in sub]), None))
rows.append(stat_block("D2 NASA milling", g2, "sparse, NOT filled"))

# ---------------- D3/D4 Katulu uniwear ----------------
CSVP = "data/toolwear/uniwear/uniwear.csv"
recs = {"nuaa": [], "phm2010": []}
with open(CSVP, encoding="utf-8", errors="replace") as f:
    rd = csv.DictReader(f)
    for r in rd:
        recs[r["dataset_tag"]].append(r)
for tag, disp in [("phm2010", "D3 PHM2010 (Katulu bundle)"), ("nuaa", "D4 NUAA orthogonal (Katulu bundle)")]:
    rs = recs[tag]
    gs = []
    for e in sorted(set(r["experiment_tag"] for r in rs)):
        sub = [r for r in rs if r["experiment_tag"] == e]
        y = np.array([float(r["tool_wear"]) for r in sub])
        ts = np.array([float(r["timestamp"]) for r in sub])
        gs.append((e, ts, y, None))
    rows.append(stat_block(disp, gs, "densely INTERPOLATED (2 Hz resample)"))

# ---------------- 输出 ----------------
hdr = ["name", "note", "n_steps", "uniq/steps", "pct_zero", "mean|dW|_um",
       "range_um", "pers_R2"]
print("%-32s %-30s %8s %10s %8s %11s %10s %10s" %
      ("dataset", "label resampling", "steps", "uniq/steps", "%%zero", "mean|dW|", "range", "pers R2"))
print("-" * 132)
for s in rows:
    print("%-32s %-30s %8d %10.4f %8.1f %11.4f %10.1f %10.4f" %
          (s["name"], s["note"], s["n_steps"], s["uniq/steps"], s["pct_zero"],
           s["mean|dW|_um"], s["range_um"], s["pers_R2"]))
print()
print("%-32s %14s %14s %12s" % ("dataset", "pers MAE(um)", "pers MAE/range", "step/range"))
print("-" * 80)
for s in rows:
    print("%-32s %14.4f %13.5f%% %11.6f%%" %
          (s["name"], s["mean|dW|_um"], 100 * s["mean|dW|_um"] / s["range_um"],
           100 * (s["range_um"] / max(s["n_steps"], 1)) / s["range_um"]))
print("\n注：persistence 的 MAE 恒等于 mean|dW|（构造使然）；mean|dW| 由标签重采样方式决定，")
print("    因此 R2 在数据集之间不可比，唯一可跨数据集比较的量是 MAE/range。")
