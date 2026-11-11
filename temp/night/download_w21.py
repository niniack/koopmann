"""Downloads the first 200 of 2048 pre-shuffled ImageNet Winter21 shards (~114 GB, ~1.28M images)."""
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
from huggingface_hub import snapshot_download
files = [f"imagenet_w21-train-{i:04d}.tar" for i in range(200)] + ["_info.yaml", "README.md"]
path = snapshot_download("timm/imagenet-w21-wds", repo_type="dataset", allow_patterns=files, local_dir="/scratch/nsa325/datasets/imagenet-w21-wds", max_workers=8)
print("done", path)
