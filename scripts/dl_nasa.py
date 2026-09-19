import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # drop if huggingface.co is reachable
from huggingface_hub import hf_hub_download

# NASA Ames milling dataset, parquet mirror (~46 MB).
R = "jonasmaltebecker/nasa_milling"
p = hf_hub_download(R, "data.parquet", repo_type="dataset",
                    local_dir="data/toolwear/nasa_milling")
print("DONE", p, "%.1f MB" % (os.path.getsize(p) / 1e6))
