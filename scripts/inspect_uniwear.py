import csv, collections, os

P = "data/toolwear/uniwear/uniwear.csv"
print("size %.2f MB" % (os.path.getsize(P) / 1e6))
with open(P, encoding="utf-8", errors="replace") as f:
    r = csv.reader(f)
    hdr = next(r)
    print("cols(%d): %s" % (len(hdr), hdr))
    rows = list(r)
print("rows:", len(rows))
idx = {h: i for i, h in enumerate(hdr)}
print("\n=== dataset_tag 分布 ===", collections.Counter(x[idx["dataset_tag"]] for x in rows))
print("=== experiment_tag 分布 ===", collections.Counter(x[idx["experiment_tag"]] for x in rows))
# tool_wear 范围与唯一值数
for tag in sorted(set(x[idx["dataset_tag"]] for x in rows)):
    sub = [x for x in rows if x[idx["dataset_tag"]] == tag]
    w = sorted(set(float(x[idx["tool_wear"]]) for x in sub))
    print("\n--- %s : %d rows, tool_wear %d unique, range %.4f .. %.4f mm" %
          (tag, len(sub), len(w), w[0], w[-1]))
    print("    前 12 个唯一值:", [round(v, 4) for v in w[:12]])
    # 每个 experiment 的采样数
    ec = collections.Counter(x[idx["experiment_tag"]] for x in sub)
    print("    每个 experiment 行数:", dict(ec))
print("\n前 2 行示例:")
for x in rows[:2]:
    print("   ", {h: x[idx[h]] for h in hdr})
