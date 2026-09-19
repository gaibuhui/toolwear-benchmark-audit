import numpy as np, pyarrow.parquet as pq
P = 'data/toolwear/nasa_milling/data.parquet'
CH = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
rows = [r for r in pq.ParquetFile(P).read().to_pylist() if r["VB"] is not None and not np.isnan(r["VB"])]
out = []
out.append("rows=%d" % len(rows))
PROC = np.column_stack([[r["time"], r["DOC"], r["feed"], r["material"]] for r in rows])
out.append("PROC shape=%s" % (PROC.shape,))
VB = np.array([r["VB"] for r in rows]); CAS = np.array([r["case"] for r in rows])
RUN = np.array([r["run"] for r in rows])
out.append("VB %s CAS %s RUN %s" % (VB.shape, CAS.shape, RUN.shape))
o = np.lexsort((RUN, CAS))
out.append("o shape=%s min=%d max=%d" % (o.shape, o.min(), o.max()))
# 检查各通道信号长度
lens = {c: set(len(r[c]) for r in rows) for c in CH}
out.append("signal lengths (min,max): " + ", ".join("%s:%d-%d" % (c, min(v), max(v)) for c, v in lens.items()))
open("diag.txt", "w", encoding="utf-8").write("\n".join(out))
print("written")
