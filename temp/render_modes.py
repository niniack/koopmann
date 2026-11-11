import os, sys, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
from datasets import load_dataset
SP = sys.argv[1]; modes = [int(m) for m in sys.argv[3].split(",")]
d = torch.load(SP + "/modes_val.pt"); coef = d["coef"]
data = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
fig, axes = plt.subplots(len(modes), 8, figsize=(12, 1.6 * len(modes)))
for row, m in zip(axes, modes):
    for ax, i in zip(row, coef[:, m].topk(8).indices.tolist()):
        im = data[i]["image"].convert("RGB"); im.thumbnail((128, 128)); ax.imshow(np.asarray(im)); ax.axis("off")
    row[0].set_title(f"mode {m}", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(SP + f"/{sys.argv[2]}.png", dpi=60)
