"""Follow-up on the always-on modes: how much larger the image-specific part gets, and what the growth modes' direction means."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
torch.set_num_threads(12)
SP = Path(__file__).parent
NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
HIDDEN = {*range(69, 80), 110, 111, 113, 114, 126, *range(300, 327), 599, 815}
names = open_clip.IMAGENET_CLASSNAMES
ckpt, root = Path(os.getenv("WEIGHTS_CACHE")) / "tokae", Path(os.getenv("DATASETS_CACHE"))
stats = torch.load(ckpt / f"{NAME}_stats.pt")
mean, std = stats["mean"], stats["std"]
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
val_x, val_y = (val["states"].float() - mean) / std, val["labels"]
saved = torch.load(SP / "dino_explore.pt", weights_only=False)
z, lam = saved["z"], saved["eigenvalues"]
z6, z12 = z[:, 0], z[:, 1]
x6, x12 = val_x[:, 0], val_x[:, 1]
c6, c12 = x6 - x6.mean(dim=0), x12 - x12.mean(dim=0)
print(f"spread of states around their block's average: block 12 is {(c12.pow(2).sum() / c6.pow(2).sum()).sqrt():.2f} times block 6")
for part, s in [("CLS", slice(0, 768)), ("patch mean", slice(768, 1536)), ("patch std", slice(1536, 2304))]:
    print(f"   {part}: {(c12[:, s].pow(2).sum() / c6[:, s].pow(2).sum()).sqrt():.2f} times")
print(f"length of the average state: block 6 {x6.mean(dim=0).norm():.2f}, block 12 {x12.mean(dim=0).norm():.2f}; spread (root mean square distance) block 6 {c6.pow(2).sum(dim=1).mean().sqrt():.2f}, block 12 {c12.pow(2).sum(dim=1).mean().sqrt():.2f}")
kept = z6 != 0
share = kept.double().mean(dim=0)
feature = (share > 0.005) & (share < 0.99)
# latent energy per image
e6, e12 = z6.abs().pow(2), (lam * z6).abs().pow(2)
print(f"latent energy at block 12 over block 6 (K's prediction), square root: {(e12.sum() / e6.sum()).sqrt():.2f}")

_, s6, v6 = torch.pca_lowrank(c6, q=20, center=False, niter=4)
_, s12, v12 = torch.pca_lowrank(c12, q=20, center=False, niter=4)
p6, p12 = c6 @ v6, c12 @ v12


def r2(target, design):
    design = torch.cat([design, torch.ones(len(design), 1)], dim=1)
    fit = design @ torch.linalg.lstsq(design, target).solution
    return (1 - (fit - target).pow(2).sum() / (target - target.mean(dim=0)).pow(2).sum()).item()


for m in [331, 77]:
    value = torch.view_as_real(z6[:, m].contiguous())
    print(f"\nmode {m}: both numbers from the top 1 / 5 / 20 block-6 directions: R² {r2(value, p6[:, :1]):.3f} / {r2(value, p6[:, :5]):.3f} / {r2(value, p6[:, :20]):.3f}; from the top 5 / 20 block-12 directions: {r2(value, p12[:, :5]):.3f} / {r2(value, p12[:, :20]):.3f}")
    true12 = torch.view_as_real(z12[:, m].contiguous())
    print(f"mode {m}: its true block-12 value from the top 5 / 20 block-12 directions: R² {r2(true12, p12[:, :5]):.3f} / {r2(true12, p12[:, :20]):.3f}")
    phase = torch.angle(z6[:, m]) * 180 / math.pi
    size = z6[:, m].abs()
    print(f"mode {m}: mean value {z6[:, m].mean().abs():.3f} against typical size {size.mean():.3f}")
    for low in range(-180, 180, 45):
        inside = (phase >= low) & (phase < low + 45)
        weight = torch.zeros(1000).index_add_(0, val_y[inside], size[inside])
        total = torch.zeros(1000).index_add_(0, val_y, size)
        lift = weight / total
        order = [c for c in lift.argsort(descending=True).tolist() if c not in HIDDEN][:8]
        print(f"   direction {low:+4d}° to {low + 45:+4d}° ({inside.double().mean():.2f} of images): " + ", ".join(f"{names[c]} {lift[c]:.2f}" for c in order))
    by_class = torch.zeros(1000, dtype=torch.cfloat).index_add_(0, val_y, z6[:, m]) / 50
    within = (z6[:, m] - by_class[val_y]).abs().pow(2).mean()
    print(f"mode {m}: share of its spread that is between classes (both numbers): {1 - within / (z6[:, m] - z6[:, m].mean()).abs().pow(2).mean():.3f}")

# the same class share for feature modes, for reference
shares = []
for m in torch.nonzero(feature).flatten().tolist():
    by_class = torch.zeros(1000, dtype=torch.cfloat).index_add_(0, val_y, z6[:, m]) / 50
    within = (z6[:, m] - by_class[val_y]).abs().pow(2).mean()
    shares.append((1 - within / (z6[:, m] - z6[:, m].mean()).abs().pow(2).mean()).item())
print(f"\nfeature modes: median share of spread between classes {np.median(shares):.3f}")
