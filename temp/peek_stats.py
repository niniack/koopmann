import os, sys
from pathlib import Path
import torch
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
ckpt = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
stats = torch.load(ckpt / "im_dino_w21_wd_stats.pt")
print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in stats.items()})
print("mean", stats["mean"].flatten()[:3], "std", stats["std"].flatten()[:3])
d = torch.load("dino_explore.pt", weights_only=False)
print({k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in d.items()})
