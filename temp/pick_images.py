"""Lists, for the modes with a known meaning, the validation images where each is largest at block 6 (hidden images left out)."""
import torch
import open_clip
d = torch.load("dino_explore.pt", weights_only=False)
hidden = torch.load("hidden_mask.pt"); hidden[22037] = True
z6, labels = d["z"][:, 0], None
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
val_y = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_patch_std_dinov3-vitb16_val_blocks_5_11.pt")["labels"]
names = open_clip.IMAGENET_CLASSNAMES
energy = z6.abs().pow(2)
for mode, meaning in {667: "winter woods, snow", 644: "snow, white water", 247: "fruit piles", 37: "open water", 71: "birds against sky", 535: "knitted", 211: "fire or glow in the dark", 670: "menus, print"}.items():
    order = z6[:, mode].abs().argsort(descending=True)
    top = order[~hidden[order]][:6].tolist()
    print(mode, meaning, [(i, names[val_y[i]], round((energy[i, mode] / energy[i].sum()).item(), 3)) for i in top])
