import os, sys, collections, torch, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
SP = sys.argv[1]
store = torch.load(f"{SP}/removal_after.pt")
rec = pd.read_pickle(f"{SP}/removal.pkl")
y = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_ViT-B-32_val_blocks_5_11.pt")["labels"]
names = open_clip.IMAGENET_CLASSNAMES
tot = rec.groupby("mode")[["on", "fixed", "broken"]].sum()
tot["net"] = tot["fixed"] - tot["broken"]
print("snow mode 903 removed, all 50k val:", tot.loc[903].to_dict())
print("distribution of net over modes:", tot["net"].describe().round(1).to_dict())
print("modes with net > 0 over all val:", (tot["net"] > 0).sum(), "of", len(tot))
print(tot.sort_values("net", ascending=False).head(5))
# which classes lose when 903 is removed
before_path = f"{SP}/before_val.pt"
