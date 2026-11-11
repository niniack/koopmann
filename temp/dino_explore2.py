"""
Second look at im_dino_w21_wd, from what dino_explore.py saved: why the KAE fails on snowy images, edits of several snow modes at once,
the always-on modes, and per-mode coherence and removal effects. Saves the per-mode table to dino_explore2.pt.
Run with HF_HUB_OFFLINE=1 PYTHONPATH=<scratchpad>/tf_target.
"""
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
torch.manual_seed(0)
torch.set_num_threads(32)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

SP = Path(__file__).parent
NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
FOLDERS = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
root, ckpt = Path(os.getenv("DATASETS_CACHE")), Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
names = open_clip.IMAGENET_CLASSNAMES

stats = torch.load(ckpt / f"{NAME}_stats.pt")
mean, std = stats["mean"], stats["std"]
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
val_x, val_y = (val["states"].float() - mean) / std, val["labels"]
kae = train.JordanKoopmanAutoencoder(
    256, shared_modes=True, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu"
)
kae.load_state_dict(load_file(ckpt / f"{NAME}.safetensors"))
kae.eval()
head = train.DinoHead(ENCODER, mean, std, torch.device("cpu"), zero_shot=True)

saved = torch.load(SP / "dino_explore.pt", weights_only=False)
z, eigenvalues, answers, true_answers, modes = saved["z"], saved["eigenvalues"], saved["answers"], saved["true_answers"], saved["modes"]
z6, z12 = z[:, 0], z[:, 1]
kept = z6 != 0
usage = kept.double().mean(dim=0)
angle = torch.rad2deg(torch.angle(eigenvalues))


def to_complex(phi):
    return torch.cat([phi[..., :1].to(torch.cfloat), torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())], dim=-1)


def to_latent(z):
    return torch.cat([z[..., :1].real, torch.view_as_real(z[..., 1:]).flatten(-2)], dim=-1)


def decode(z):
    return torch.cat([kae.decode(to_latent(batch)) for batch in z.split(5_000)])


def edit(z_start, changes):
    """The prediction with some eigenvalues multiplied by complex factors. `changes` maps mode to factor."""

    lam = eigenvalues.clone()
    for mode, factor in changes.items():
        lam[mode] = lam[mode] * factor
    return decode(lam * z_start)


def fvu(pred, target, reference):
    """Error over the spread of the reference set around its mean, so groups of images share one denominator."""

    return ((pred - target).pow(2).sum() / len(pred)).item() / ((reference - reference.mean(dim=0)).pow(2).sum() / len(reference)).item()


############# SNOW: WHY THE KAE FAILS #############

snowy_x, snowy_y = (saved["snowy_raw"] - mean) / std, saved["snowy_y"]
normal = torch.isin(val_y, torch.tensor(list(FOLDERS.values())))
x, y = torch.cat([val_x[normal], snowy_x]), torch.cat([val_y[normal], snowy_y])
is_snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(snowy_x), dtype=torch.bool)])
zs = to_complex(train.encode_pairs(kae, x))
base = decode(eigenvalues * zs[:, 0])
base_answers = head.classify(base)
kae_right = base_answers == y

print("== Snow: what the KAE answers on the snowy images it gets wrong")
print(Counter(names[c] for c in base_answers[is_snowy & ~kae_right].tolist()).most_common(12))
print("Real DINOv3's wrong answers on snowy images:", Counter(names[c] for c in head.classify(x[:, 1])[is_snowy & (head.classify(x[:, 1]) != y)].tolist()).most_common(6))

# Each stage of the KAE, on snowy vs normal images, as error over the spread of all validation states
recon6, recon12 = kae.decode(kae.encode(x[:, 0])), kae.decode(kae.encode(x[:, 1]))
for label, mask in [("snowy", is_snowy), ("normal", ~is_snowy)]:
    print(
        f"{label}: reconstruction of block 6 {fvu(recon6[mask], x[mask, 0], val_x[:, 0]):.3f}, of block 12 {fvu(recon12[mask], x[mask, 1], val_x[:, 1]):.3f}, "
        f"prediction of block 12 {fvu(base[mask], x[mask, 1], val_x[:, 1]):.3f}, of its CLS part {fvu(base[mask, :768], x[mask, 1, :768], val_x[:, 1, :768]):.3f}; "
        f"probe on the reconstruction of block 12 {(head.classify(recon12[mask]) == y[mask]).double().mean():.3f}"
    )
# How unusual the snowy states are: distance to the validation mean, in units of the typical validation image's
for block, column in [("block 6", 0), ("block 12", 1)]:
    center = val_x[:, column].mean(dim=0)
    typical = (val_x[:, column] - center).norm(dim=-1).median()
    print(f"{block}: distance to the validation mean, over the typical image's: snowy {((x[is_snowy, column] - center).norm(dim=-1).median() / typical):.2f}, normal {((x[~is_snowy, column] - center).norm(dim=-1).median() / typical):.2f}")

############# SNOW: SEVERAL MODES AT ONCE #############

aucs = saved["aucs"]
ranked = [m for m in aucs.argsort(descending=True).tolist() if usage[m] < 0.9][:8]
print("\n== Snow modes (not always-on), by AUC:", [(m, round(aucs[m].item(), 3)) for m in ranked])
FACTORS = {"size x0": 0.0, "turn +90°": 1j, "turn -90°": -1j, "turn 180°": -1.0}
rows = []
for count in [1, 2, 3, 4, 5, 6, 8]:
    chosen = ranked[:count]
    on = kept[:, chosen].any(dim=1)
    val_before = answers[on] == val_y[on]
    for label, factor in FACTORS.items():
        right = head.classify(edit(zs[:, 0], {m: factor for m in chosen})) == y
        val_right = head.classify(edit(z6[on], {m: factor for m in chosen})) == val_y[on]
        rows.append(
            {
                "modes": count,
                "edit": label,
                "snowy acc": right[is_snowy].double().mean().item(),
                "normal acc": right[~is_snowy].double().mean().item(),
                "snowy fixed": int((~kae_right & right & is_snowy).sum()),
                "snowy broken": int((kae_right & ~right & is_snowy).sum()),
                "val images touched": int(on.sum()),
                "val fixed": int((~val_before & val_right).sum()),
                "val broken": int((val_before & ~val_right).sum()),
            }
        )
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Best single turn per snow mode, on a finer grid, picked on half of the snowy images and scored on the other half
half = torch.arange(len(y)) % 2 == 0
print("\nPer snow mode: the turn that helps most, picked on even images and scored on odd ones")
for mode in ranked[:5]:
    grid = list(range(-180, 180, 30))
    accuracy = {}
    for degrees in grid:
        right = head.classify(edit(zs[:, 0], {mode: complex(math.cos(math.radians(degrees)), math.sin(math.radians(degrees)))})) == y
        accuracy[degrees] = (right[is_snowy & half].double().mean().item(), right[is_snowy & ~half].double().mean().item())
    best = max(grid, key=lambda d: accuracy[d][0])
    print(f"mode {mode} (angle {angle[mode]:+.0f}°): best turn {best:+d}° -> held-out snowy accuracy {accuracy[best][1]:.3f} (as trained {accuracy[0][1]:.3f}); all turns, held-out:", {d: round(a[1], 2) for d, a in accuracy.items()})

############# ALWAYS-ON MODES #############

print("\n== Modes on for at least 99% of images: is each a constant (the same value for every image)?")
always = (usage >= 0.99).nonzero()[:, 0].tolist()
subset = torch.randperm(len(val_y))[:5_000]
subset_right = answers[subset] == val_y[subset]
rows = []
for mode in always:
    a, b = z6[:, mode], z12[:, mode]
    removed = head.classify(edit(z6[subset], {mode: 0.0}))
    straightened = head.classify(edit(z6[subset], {mode: eigenvalues[mode].abs() / eigenvalues[mode]}))
    rows.append(
        {
            "mode": mode,
            "|λ|": eigenvalues[mode].abs().item(),
            "angle": angle[mode].item(),
            "block 6: |mean| / spread": (a.mean().abs() / (a - a.mean()).abs().pow(2).mean().sqrt()).item(),
            "block 12: |mean| / spread": (b.mean().abs() / (b - b.mean()).abs().pow(2).mean().sqrt()).item(),
            "share 6": modes.loc[mode, "share 6"],
            "share 12": modes.loc[mode, "share 12"],
            "acc, removed": (removed == val_y[subset]).double().mean().item(),
            "acc, angle set to 0": (straightened == val_y[subset]).double().mean().item(),
        }
    )
print(f"accuracy as trained on these 5,000 images: {subset_right.double().mean():.3f}")
print(pd.DataFrame(rows).set_index("mode").round(3).to_string())

############# EVERY MODE: COHERENCE AND REMOVAL #############

true_embeddings = head.embed(val_x[:, 1])


def get_coherence(indices):
    similarity = true_embeddings[indices] @ true_embeddings[indices].T
    n = len(indices)
    return ((similarity.sum() - n) / (n * (n - 1))).item()


random_sets = np.array([get_coherence(torch.randperm(len(val_y))[:16]) for _ in range(1_000)])
class_sets = np.array([get_coherence((val_y == c).nonzero()[:, 0][:16]) for c in range(1_000)])
threshold = np.quantile(random_sets, 0.99)

start = time.time()
rows = []
for mode in (usage >= 0.005).nonzero()[:, 0].tolist():
    top = z6[:, mode].abs().topk(16).indices
    on = kept[:, mode].nonzero()[:, 0]
    on = on[torch.randperm(len(on))[:2_000]]
    before, truth, real = answers[on], val_y[on], true_answers[on]
    after = head.classify(edit(z6[on], {mode: 0.0}))
    scale = usage[mode].item() * len(val_y) / len(on)
    rows.append(
        {
            "mode": mode,
            "coherence": get_coherence(top),
            "classes in top 16": len(set(val_y[top].tolist())),
            "top classes": ", ".join(f"{names[c][:16]} {k}" for c, k in Counter(val_y[top].tolist()).most_common(3)),
            # Scaled from the sample of at most 2,000 kept images to every validation image that keeps the mode
            "fixed": ((before != truth) & (after == truth)).sum().item() * scale,
            "broken": ((before == truth) & (after != truth)).sum().item() * scale,
            "answers changed": (before != after).double().mean().item(),
            "agreement change": ((after == real).double().mean() - (before == real).double().mean()).item(),
        }
    )
table = modes.join(pd.DataFrame(rows).set_index("mode"), how="inner")
table["net"] = table["fixed"] - table["broken"]
print(f"\n== Every mode kept on at least 0.5% of images ({len(table)}), removal measured on up to 2,000 kept images each, {time.time() - start:.0f}s")
print(f"Random sets of 16: 99th percentile {threshold:.3f}; one class: median {np.median(class_sets):.3f}")
print(f"{(table['coherence'] > threshold).mean():.0%} of modes beat random; top 16 from 1-3 classes: {(table['classes in top 16'] <= 3).sum()}, 4-8: {((table['classes in top 16'] > 3) & (table['classes in top 16'] <= 8)).sum()}, 9-16: {(table['classes in top 16'] > 8).sum()}")
columns = ["|λ|", "angle", "kept", "coherence", "classes in top 16", "fixed", "broken", "net", "answers changed", "top classes"]
print("\nRemoval helps most (net fixed, scaled to all of validation)")
print(table.sort_values("net", ascending=False).head(15)[columns].round(3).to_string())
print("\nRemoval hurts most")
print(table.sort_values("net").head(15)[columns].round(3).to_string())
print("\nMost coherent among modes whose top 16 span at least 9 classes")
print(table[table["classes in top 16"] >= 9].sort_values("coherence", ascending=False).head(25)[columns].round(3).to_string())
print("\nBy |angle|: medians")
print(table.groupby(pd.cut(table["angle"].abs(), [0, 5, 15, 45, 180], include_lowest=True), observed=True)[["kept", "coherence", "classes in top 16", "answers changed", "net", "R²"]].agg(["count", "median"]).round(3).to_string())
print("Total over modes: fixed", round(table["fixed"].sum()), "broken", round(table["broken"].sum()))

torch.save({"table": table, "threshold": threshold, "class_median": float(np.median(class_sets)), "snow_ranked": ranked}, SP / "dino_explore2.pt")
print("saved", SP / "dino_explore2.pt")
