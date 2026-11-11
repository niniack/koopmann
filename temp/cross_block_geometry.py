"""
How far apart are block 6 and block 12 states, as the KAE sees them (globally standardized)?
Tells us whether distances between a block-6 state and a block-12 state are mostly one constant offset.
"""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_num_threads(8)
torch.manual_seed(0)

root = Path(os.getenv("DATASETS_CACHE"))
stats = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae" / "im_iso_2x4k_stats.pt")
val = torch.load(root / "imagenet" / "clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")["states"]
x = (val[torch.randperm(len(val))[:4096]].float() - stats["mean"]) / stats["std"]
print("states", tuple(val.shape), val.dtype, "| standardization:", stats["standardization"])

means = x.mean(dim=0)
centered = x - means
offset = (means[1] - means[0]).pow(2).sum()
spread = centered.pow(2).sum(dim=-1).mean(dim=0)
print(f"squared offset between block means: {offset:.0f}")
print(f"mean squared distance of a state from its block mean: block 6 {spread[0]:.0f}, block 12 {spread[1]:.0f}")
print(f"same image, block 6 to block 12, squared: raw {(x[:, 1] - x[:, 0]).pow(2).sum(-1).mean():.0f}, centered {(centered[:, 1] - centered[:, 0]).pow(2).sum(-1).mean():.0f}")
print(f"cosine between an image's centered block-6 and block-12 states: {torch.cosine_similarity(centered[:, 0], centered[:, 1], dim=-1).mean():.3f}")

for label, data in [("raw", x), ("centered", centered)]:
    cross = torch.cdist(data[:, 0], data[:, 1])
    same, other = cross.diagonal(), cross[~torch.eye(len(cross), dtype=torch.bool)]
    print(f"{label}: block-6-to-block-12 distance, mean {cross.mean():.1f}, std/mean {cross.std() / cross.mean():.3f}; same image {same.mean():.1f}, different images {other.mean():.1f}")
for block, label in [(0, "block 6"), (1, "block 12")]:
    within = torch.cdist(x[:, block], x[:, block])
    within = within[~torch.eye(len(within), dtype=torch.bool)]
    print(f"within {label}: distance mean {within.mean():.1f}, std/mean {within.std() / within.mean():.3f}")

# Per part of the state: CLS, patch mean, patch std
for part, name in enumerate(["CLS", "patch mean", "patch std"]):
    sl = slice(part * 768, (part + 1) * 768)
    print(f"{name}: squared offset {(means[1, sl] - means[0, sl]).pow(2).sum():.0f}, spread block 6 {centered[:, 0, sl].pow(2).sum(-1).mean():.0f}, block 12 {centered[:, 1, sl].pow(2).sum(-1).mean():.0f}, cosine {torch.cosine_similarity(centered[:, 0, sl], centered[:, 1, sl], dim=-1).mean():.3f}")
