"""Finds the validation index of one image shown in a mode's gallery row, to hide it by hand."""
import os, sys
from pathlib import Path
import torch
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
ckpt, root = Path(os.getenv("WEIGHTS_CACHE")) / "tokae", Path(os.getenv("DATASETS_CACHE"))
stats = torch.load(ckpt / f"{NAME}_stats.pt")
mean, std = stats["mean"], stats["std"]
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
val_x, val_y = (val["states"].float() - mean) / std, val["labels"]
head = train.DinoHead(ENCODER, mean, std, torch.device("cpu"), zero_shot=True)
HIDDEN_CLASSES = torch.tensor([*range(69, 80), 110, 111, *range(113, 117), *range(118, 127), *range(300, 329), 410, 599, 815])
hidden = torch.isin(val_y, HIDDEN_CLASSES) | torch.isin(head.get_log_probs(val_x[:, 1]).topk(5).indices, HIDDEN_CLASSES).any(dim=1)
torch.save(hidden, "hidden_mask.pt")
z6 = torch.load("dino_explore.pt", weights_only=False)["z"][:, 0]
order = z6[:, 670].abs().argsort(descending=True)
shown = order[~hidden[order]][:8].tolist()
print("mode 670 row:", shown, [int(val_y[i]) for i in shown])
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
image = images[shown[1]]["image"].convert("RGB")
image.thumbnail((300, 300))
image.save("menu_check.png")
print("hidden images:", int(hidden.sum()))
