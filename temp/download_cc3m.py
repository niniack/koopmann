"""CC3M training shards (webdataset tars) from pixparse/cc3m-wds, into the datasets cache."""
import os
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import snapshot_download
path = snapshot_download(
    "pixparse/cc3m-wds", repo_type="dataset", allow_patterns=["cc3m-train-*.tar", "_info.json", "README.md"],
    local_dir=os.path.join(os.getenv("DATASETS_CACHE"), "cc3m-wds"), max_workers=8,
)
print("done", path, flush=True)
