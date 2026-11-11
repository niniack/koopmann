"""What the 16 modes that every image keeps are doing in the DINOv3 KAE (im_dino_w21_wd). CPU only."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file
from scipy.stats import spearmanr

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
torch.set_num_threads(20)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
SP = Path(__file__).parent
NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
GROWTH, INERT = [331, 77], [68, 653, 877, 465]
OFFSET = [861, 338, 144, 560, 117, 988, 871, 746, 652, 533]
ALWAYS = GROWTH + OFFSET + INERT
HIDDEN_CLASSES = {*range(69, 80), 126, *range(300, 327), 599, 815}
names = open_clip.IMAGENET_CLASSNAMES

ckpt, root = Path(os.getenv("WEIGHTS_CACHE")) / "tokae", Path(os.getenv("DATASETS_CACHE"))
stats = torch.load(ckpt / f"{NAME}_stats.pt")
mean, std = stats["mean"], stats["std"]
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
raw, val_y = val["states"].float(), val["labels"]
val_x = (raw - mean) / std
kae = train.JordanKoopmanAutoencoder(256, shared_modes=True, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ckpt / f"{NAME}.safetensors"))
kae.eval()
head = train.DinoHead(ENCODER, mean, std, torch.device("cpu"), zero_shot=True)
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])


def to_latent(z):
    return torch.cat([z[..., :1].real, torch.view_as_real(z[..., 1:]).flatten(-2)], dim=-1)


def decode(z):
    return torch.cat([kae.decode(to_latent(batch)) for batch in z.split(5_000)])


z = torch.load(SP / "dino_explore.pt", weights_only=False)["z"]
z6, z12 = z[:, 0], z[:, 1]
kept = z6 != 0
share = kept.double().mean(dim=0)
used = share > 0
print("modes kept per image", int(kept.sum(dim=1).min()), int(kept.sum(dim=1).max()), "| modes ever kept", int(used.sum()))
assert all(share[m] > 0.99 for m in ALWAYS)

# ---------------------------------------------------------------- A. growth and turn against the other modes
size, angle = lam.abs(), torch.angle(lam).abs() * 180 / math.pi
feature = used & (share < 0.99)
print("\n=== A. growth and turn, ranked among the", int(used.sum()), "modes ever kept (rank 1 = largest) ===")
print(f"feature modes: median |λ| {size[feature].median():.2f}, median |angle| {angle[feature].median():.1f}°")
rows = []
for group, modes in [("growth", GROWTH), ("offset", OFFSET), ("inert", INERT)]:
    for m in modes:
        on = kept[:, m]
        phase = torch.angle(z6[on, m])
        spread = math.degrees(math.sqrt(max(-2 * math.log(torch.exp(1j * phase).mean().abs().item()), 0)))
        mags = z6[on, m].abs()
        rows.append(
            {
                "group": group,
                "mode": m,
                "|λ|": round(size[m].item(), 2),
                "|λ| rank": int((size[used] > size[m]).sum()) + 1,
                "|angle|°": round(angle[m].item(), 1),
                "angle rank": int((angle[used] > angle[m]).sum()) + 1,
                "size spread (std/mean)": round((mags.std() / mags.mean()).item(), 3),
                "phase spread°": round(spread, 1),
            }
        )
print(pd.DataFrame(rows).to_string(index=False))
spreads, cvs = [], []
for m in torch.nonzero(feature & (share > 0.005)).flatten().tolist():
    on = kept[:, m]
    phase = torch.angle(z6[on, m])
    spreads.append(math.degrees(math.sqrt(max(-2 * math.log(torch.exp(1j * phase).mean().abs().item()), 0))))
    cvs.append((z6[on, m].abs().std() / z6[on, m].abs().mean()).item())
print(f"feature modes kept ≥ 0.5%: median phase spread {np.median(spreads):.1f}°, median size spread {np.median(cvs):.3f}")

# ---------------------------------------------------------------- B. the average state at each block
x6, x12 = val_x[:, 0], val_x[:, 1]
m6, m12 = x6.mean(dim=0), x12.mean(dim=0)
print("\n=== B. the average state of each block, after the shared standardization ===")
print(f"cosine between the block-6 average and the block-12 average: {torch.cosine_similarity(m6, m12, dim=0):.3f}")
for label, x, m in [("block 6", x6, m6), ("block 12", x12, m12)]:
    total = x.pow(2).sum(dim=1).mean()
    parts = [f"{part} {m[s].pow(2).sum() / x[:, s].pow(2).sum(dim=1).mean():.2f}" for part, s in [("CLS", slice(0, 768)), ("patch mean", slice(768, 1536)), ("patch std", slice(1536, 2304))]]
    print(f"{label}: the average holds {m.pow(2).sum() / total:.2f} of the state's energy ({', '.join(parts)})")

# ---------------------------------------------------------------- C. decode with only some modes
print("\n=== C. decoding with only some of the modes (the rest set to zero) ===")
print("FVU is against the spread of states around their block's average, so 1.0 = no better than the average state")


def only(modes):
    mask = torch.zeros(z6.shape[1], dtype=torch.bool)
    mask[modes] = True
    return mask


feature_modes = torch.nonzero(feature).flatten().tolist()
sets = {
    "no modes (decoder at zero)": [],
    "10 offset": OFFSET,
    "2 growth": GROWTH,
    "4 inert": INERT,
    "offset + growth": OFFSET + GROWTH,
    "all 16 always-on": ALWAYS,
    "feature modes only": feature_modes,
    "feature + growth": feature_modes + GROWTH,
    "feature + offset": feature_modes + OFFSET,
    "all modes": list(range(z6.shape[1])),
}
rows = []
for label, modes in sets.items():
    mask = only(modes)
    row = {"modes used": label}
    for block, x, m, latent in [("6", x6, m6, z6 * mask), ("12", x12, m12, lam * z6 * mask)]:
        out = decode(latent)
        row[f"block {block} FVU"] = round(((out - x).pow(2).sum() / (x - m).pow(2).sum()).item(), 3)
        row[f"block {block} cos(avg out, avg state)"] = round(torch.cosine_similarity(out.mean(dim=0), m, dim=0).item(), 3)
        row[f"block {block} |avg out|/|avg state|"] = round((out.mean(dim=0).norm() / m.norm()).item(), 2)
        if block == "12":
            row["accuracy"] = round((head.classify(out) == val_y).double().mean().item(), 4)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- D. are they constants?
print("\n=== D. give every image the same value (the average over images) for a group of modes ===")
base = lam * z6
true_answers = head.classify(x12)


def score(latent):
    found = head.classify(decode(latent))
    return f"accuracy {(found == val_y).double().mean():.4f}, agrees with DINOv3 {(found == true_answers).double().mean():.4f}"


print("K as trained:", score(base))
for label, modes in [("10 offset", OFFSET), ("2 growth", GROWTH), ("mode 331", [331]), ("mode 77", [77]), ("4 inert", INERT)]:
    constant, removed, straight = base.clone(), base.clone(), base.clone()
    constant[:, modes] = (lam[modes] * z6[:, modes].mean(dim=0)).expand(len(z6), -1)
    removed[:, modes] = 0
    straight[:, modes] = lam[modes].abs() * z6[:, modes]
    print(f"{label:10s} same value for every image: {score(constant)}")
    print(f"{label:10s} removed:                     {score(removed)}")
    print(f"{label:10s} angle set to 0 (same size):  {score(straight)}")
# the growth modes keep their size but lose their image-to-image phase, and the reverse
for m in GROWTH:
    phase_only, size_only = base.clone(), base.clone()
    phase_only[:, m] = lam[m] * z6[:, m].abs().mean() * torch.exp(1j * torch.angle(z6[:, m]))
    size_only[:, m] = lam[m] * z6[:, m].abs() * torch.exp(1j * torch.angle(z6[:, m].mean()))
    print(f"mode {m}: average size, own phase: {score(phase_only)}")
    print(f"mode {m}: own size, average phase: {score(size_only)}")

# ---------------------------------------------------------------- E. what the growth modes track
print("\n=== E. what the growth modes track (Spearman correlation over the 50,000 images) ===")
c6, c12 = x6 - m6, x12 - m12
_, s6, v6 = torch.pca_lowrank(c6, q=20, center=False, niter=4)
_, s12, v12 = torch.pca_lowrank(c12, q=20, center=False, niter=4)
p6, p12 = c6 @ v6, c12 @ v12
print("share of block-6 spread in its top 5 directions:", [round(v, 3) for v in (s6[:5].pow(2) / c6.pow(2).sum()).tolist()])
print("share of block-12 spread in its top 5 directions:", [round(v, 3) for v in (s12[:5].pow(2) / c12.pow(2).sum()).tolist()])
log_probs = head.get_log_probs(x12)
kae_right = head.classify(decode(base)) == val_y
things = {
    "CLS norm, block 6": raw[:, 0, :768].norm(dim=1),
    "CLS norm, block 12": raw[:, 1, :768].norm(dim=1),
    "patch-mean norm, block 6": raw[:, 0, 768:1536].norm(dim=1),
    "patch-mean norm, block 12": raw[:, 1, 768:1536].norm(dim=1),
    "patch-std norm, block 6": raw[:, 0, 1536:].norm(dim=1),
    "patch-std norm, block 12": raw[:, 1, 1536:].norm(dim=1),
    "distance from the average state, block 6": c6.norm(dim=1),
    "distance from the average state, block 12": c12.norm(dim=1),
    "DINOv3's confidence": log_probs.max(dim=1).values.exp(),
    "DINOv3 is right": (true_answers == val_y).float(),
    "KAE is right": kae_right.float(),
    **{f"block-6 direction {i + 1}": p6[:, i] for i in range(5)},
    **{f"block-12 direction {i + 1}": p12[:, i] for i in range(5)},
}
for m in GROWTH:
    # the mode's value along its own main axis, as its phase barely moves
    axis = torch.exp(-1j * torch.angle(z6[:, m].mean()))
    along, across, mags = (z6[:, m] * axis).real, (z6[:, m] * axis).imag, z6[:, m].abs()
    found = sorted(((abs(spearmanr(mags.numpy(), v.numpy()).statistic), spearmanr(mags.numpy(), v.numpy()).statistic, k) for k, v in things.items()), reverse=True)
    print(f"\nmode {m} size against:")
    for _, rho, k in found[:8]:
        print(f"   {rho:+.3f}  {k}")
    found = sorted(((abs(spearmanr(across.numpy(), v.numpy()).statistic), spearmanr(across.numpy(), v.numpy()).statistic, k) for k, v in things.items()), reverse=True)
    print(f"mode {m} sideways part (std {across.std():.3f} against {along.std():.3f} along) against:")
    for _, rho, k in found[:4]:
        print(f"   {rho:+.3f}  {k}")
    for label, p in [("block 6", p6), ("block 12", p12)]:
        design = torch.cat([p, torch.ones(len(p), 1)], dim=1)
        fit = design @ torch.linalg.lstsq(design, mags[:, None]).solution
        print(f"mode {m} size from the top 20 {label} directions: R² {1 - ((fit[:, 0] - mags).pow(2).sum() / (mags - mags.mean()).pow(2).sum()).item():.3f}")
    by_class = torch.zeros(1000).index_add_(0, val_y, mags) / 50
    order = [c for c in by_class.argsort(descending=True).tolist() if c not in HIDDEN_CLASSES]
    print(f"mode {m} largest in: {', '.join(names[c] for c in order[:10])}")
    print(f"mode {m} smallest in: {', '.join(names[c] for c in order[-10:])}")
    print(f"mode {m} share of its size spread that is between classes: {(by_class.var() / mags.var()).item():.3f}")
print(f"\nsizes of 331 and 77 against each other: {spearmanr(z6[:, 331].abs().numpy(), z6[:, 77].abs().numpy()).statistic:+.3f}")
for m in OFFSET[:4]:
    mags = z6[:, m].abs()
    found = sorted(((abs(spearmanr(mags.numpy(), v.numpy()).statistic), spearmanr(mags.numpy(), v.numpy()).statistic, k) for k, v in things.items()), reverse=True)
    print(f"offset mode {m} size against: " + "; ".join(f"{rho:+.2f} {k}" for _, rho, k in found[:3]))

# ---------------------------------------------------------------- F. which part of the output each mode moves
print("\n=== F. which part of the block-12 prediction moves when the mode is removed (5,000 images) ===")
subset = torch.arange(0, 50_000, 10)
full = decode(base[subset])
rows = []
for group, modes in [("growth", GROWTH), ("offset", OFFSET), ("inert", INERT)]:
    for m in modes:
        edited = base[subset].clone()
        edited[:, m] = 0
        change = (decode(edited) - full).pow(2)
        total = change.sum()
        rows.append({"group": group, "mode": m, "CLS": round((change[:, :768].sum() / total).item(), 3), "patch mean": round((change[:, 768:1536].sum() / total).item(), 3), "patch std": round((change[:, 1536:].sum() / total).item(), 3), "change / output energy": round((total / full.pow(2).sum()).item(), 4)})
print(pd.DataFrame(rows).to_string(index=False))
generator = torch.Generator().manual_seed(0)
picks = torch.nonzero(feature & (share > 0.03) & (share < 0.3)).flatten()
picks = picks[torch.randperm(len(picks), generator=generator)[:40]].tolist()
parts = []
for m in picks:
    on = torch.nonzero(kept[:, m]).flatten()[:1000]
    edited = base[on].clone()
    edited[:, m] = 0
    full_on = decode(base[on])
    change = (decode(edited) - full_on).pow(2)
    parts.append([(change[:, :768].sum() / change.sum()).item(), (change[:, 768:1536].sum() / change.sum()).item(), (change[:, 1536:].sum() / change.sum()).item(), (change.sum() / full_on.pow(2).sum()).item()])
print("median over 40 feature modes (on their own images): CLS, patch mean, patch std, change / output energy:", np.round(np.median(np.array(parts), axis=0), 4).tolist())
