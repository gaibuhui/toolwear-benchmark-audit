import os, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

R = "alpha-by-beta/Multivariate_time_series_data_of_milling_processes_with_varying_tool_wear_and_machine_tools"
t0 = time.time()
p = hf_hub_download(R, "dataset.parquet", repo_type="dataset",
                    local_dir="data/toolwear/alphabeta")
print("DONE", p, "%.2f GB" % (os.path.getsize(p) / 1e9), "%.1f min" % ((time.time() - t0) / 60))
