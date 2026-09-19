import pyarrow.parquet as pq
import numpy as np, collections

P = "data/toolwear/alphabeta/dataset.parquet"
f = pq.ParquetFile(P)

# 只读标签列（列裁剪，很快）
t = f.read(columns=["machine", "tool", "run", "cumulated_tool_contact_time", "wear"])
m = np.asarray(t.column("machine")); tl = np.asarray(t.column("tool"))
rn = np.asarray(t.column("run")); ct = np.asarray(t.column("cumulated_tool_contact_time"))
wr = np.asarray(t.column("wear"))

print("n=%d" % len(wr))
print("machine 分布:", dict(collections.Counter(m.tolist())))
print("tool 分布:", dict(sorted(collections.Counter(tl.tolist()).items())))
print("wear(µm): min=%d max=%d mean=%.1f std=%.1f  唯一值数=%d"
      % (wr.min(), wr.max(), wr.mean(), wr.std(), len(set(wr.tolist()))))
print("contact_time(s): min=%.2f max=%.2f" % (ct.min(), ct.max()))

print("\n=== (machine, tool) 组合 → 走刀数 / wear 范围 ===")
combo = collections.defaultdict(list)
for i in range(len(wr)):
    combo[(int(m[i]), int(tl[i]))].append((int(rn[i]), int(wr[i]), float(ct[i])))
tot = 0
for k in sorted(combo):
    v = sorted(combo[k])
    tot += len(v)
    ws = [x[1] for x in v]
    print("  M%d T%-2d  runs=%-4d  run[%d..%d]  wear[%d..%d]  Δwear=%d"
          % (k[0], k[1], len(v), v[0][0], v[-1][0], min(ws), max(ws), max(ws) - min(ws)))
print("合计:", tot)

print("\n=== 每把刀的 wear 轨迹（抽样打印第 1 把） ===")
k0 = sorted(combo)[0]
v = sorted(combo[k0])
print("  %s: %s" % (k0, [(r, w) for r, w, c in v][:40]))

print("\n=== wear 单调性检查（每把刀内是否非降） ===")
bad = 0
for k in sorted(combo):
    v = sorted(combo[k])
    ws = [x[1] for x in v]
    d = np.diff(ws)
    if (d < 0).sum() > 0:
        bad += 1
        print("  M%d T%d 有 %d 处下降 (min Δ=%d)" % (k[0], k[1], int((d < 0).sum()), int(d.min())))
print("非单调的刀数: %d / %d" % (bad, len(combo)))
