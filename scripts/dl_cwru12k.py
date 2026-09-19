import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # drop if huggingface.co is reachable
from huggingface_hub import snapshot_download

# CWRU 12 kHz drive-end parquet mirror with rpm (load) labels, ~119 MB.
# The cwru_*.py scripts expect the mirror's own data/ subdirectory here.
snapshot_download("babylon9/cwru_12k_de", repo_type="dataset",
                  local_dir="data/bearing/cwru12k")
print("DONE -> data/bearing/cwru12k")
