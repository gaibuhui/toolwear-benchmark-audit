"""下载 Katulu Uniwear 打包的 PHM2010 与 NUAA 刀具磨损数据（CC-BY-4.0）

来源：https://github.com/katulu-io/uniwear-dataset
  data/phm2010_bundle_high_resolution.csv   PHM2010 c1/c4/c6（不锈钢 HRC52，钨钢刀），tool_wear 单位 mm
  data/nuaa_orthogonal_bundle_high_resolution.csv  NUAA 正交试验 W1-W9（钛合金 TC4，DMU 80P），tool_wear mm
  data/uniwear.csv                          两者的 ~2 Hz 对齐交集
"""
import os, urllib.request

RAW = "https://raw.githubusercontent.com/katulu-io/uniwear-dataset/main/data/"
MIRRORS = ["", "https://ghfast.top/", "https://gh-proxy.com/", "https://ghproxy.net/"]
OUT = "data/toolwear/uniwear"
os.makedirs(OUT, exist_ok=True)
FILES = ["phm2010_bundle_high_resolution.csv",
         "nuaa_orthogonal_bundle_high_resolution.csv",
         "uniwear.csv",
         "README.md"]

for fn in FILES:
    dst = os.path.join(OUT, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print("skip (exists): %s (%.2f MB)" % (fn, os.path.getsize(dst) / 1e6))
        continue
    ok = False
    for m in MIRRORS:
        url = m + RAW + fn
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if len(data) < 500:
                print("  too small from %s (%d bytes)" % (url[:60], len(data)))
                continue
            open(dst, "wb").write(data)
            print("OK  %-46s %8.2f MB   via %s" % (fn, len(data) / 1e6, m or "direct"))
            ok = True
            break
        except Exception as e:
            print("  FAIL %-58s %s" % (url[:58], str(e)[:70]))
    if not ok:
        print("ALL MIRRORS FAILED for", fn)
