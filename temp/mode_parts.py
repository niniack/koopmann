"""
Where does each mode's push land: in the CLS token (what the probe reads) or in the patch statistics (patch mean and patch std)?

A mode's "push" is the change of the KAE's predicted block-12 state when the mode is taken out of K's prediction,
on images that keep the mode. The state is [CLS, patch mean, patch std], 768 numbers each, in standardized units,
so the push splits exactly into three shares that sum to 1.

The per-channel mean and std of a feature map are the "style statistics" of AdaIN, so a mode whose push lands in the
patch statistics is a candidate style mode, and a mode whose push lands in the CLS is a candidate object mode.
The same split is done for the kept part of K's prediction, Re(λ)·z, and the built part, i·Im(λ)·z.

DINOv3 run im_dino_w21_wd, ImageNet validation. Usage: mode_parts.py
"""

import csv
import os

import torch

import inversion
from inversion import to_complex, to_latent, train

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
MIN_IMAGES, MOST_IMAGES = 200, 600
PARTS = {"cls": slice(0, 768), "patch mean": slice(768, 1536), "patch std": slice(1536, 2304)}
kae, stats = inversion.load_kae()
states, labels = train.get_states(inversion.ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
ROWS = [slice(first, first + 5000) for first in range(0, len(states), 5000)]
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[rows])[:, 0]) for rows in ROWS])
predicted = lam * z6
kept = z6 != 0


def get_shares(x):
    """Share of the summed squares of x in each part of the state. x: [images, 2304]"""

    total = x.pow(2).sum()
    return {part: (x[:, columns].pow(2).sum() / total).item() for part, columns in PARTS.items()}


# What to compare against: how the image-to-image variation of the true block-12 state splits over the parts
truth = states[:, 1] - states[:, 1].mean(dim=0)
print("true block-12 state, variation across images:", {k: round(v, 3) for k, v in get_shares(truth).items()})
truth = states[:, 0] - states[:, 0].mean(dim=0)
print("true block-6 state, variation across images: ", {k: round(v, 3) for k, v in get_shares(truth).items()})
sample = torch.randperm(len(states))[:5000]
output = kae.decode(to_latent(predicted[sample]))
print("KAE's predicted block 12, variation:         ", {k: round(v, 3) for k, v in get_shares(output - output.mean(dim=0)).items()})

modes = (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0].tolist()
always = (kept.float().mean(dim=0) > 0.99).nonzero()[:, 0].tolist()
print(f"{len(modes)} modes kept on at least {MIN_IMAGES} images, {len(always)} of them on every image", flush=True)
factors = torch.load("alter_factors.pt", weights_only=False)["factors"]
table = []
for m in modes:
    on = kept[:, m].nonzero()[:, 0]
    rows = on[torch.randperm(len(on))[:MOST_IMAGES]]
    base_latent = predicted[rows]
    base = kae.decode(to_latent(base_latent))
    row = {
        "mode": m,
        "usage": round(len(on) / len(states), 4),
        "always on": m in always,
        "size of λ": round(lam[m].abs().item(), 3),
        "angle of λ": round(torch.rad2deg(torch.angle(lam[m])).item(), 2),
        "gray factor size": round(factors["gray"][m].abs().item(), 3),
        "sepia factor size": round(factors["sepia"][m].abs().item(), 3),
    }
    z = z6[rows, m]
    # value left in the mode once the named part is taken out
    for name, left in {"mode": torch.zeros_like(z), "kept": 1j * lam[m].imag * z, "built": lam[m].real * z}.items():
        edited = base_latent.clone()
        edited[:, m] = left
        push = base - kae.decode(to_latent(edited))
        for part, share in get_shares(push).items():
            row[f"{name}: {part}"] = round(share, 4)
        row[f"{name}: push size"] = round((push.pow(2).sum() / base.pow(2).sum()).sqrt().item(), 4)
    table.append(row)

with open("mode_parts.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
torch.save(table, "mode_parts.pt")

column = lambda key, rows: torch.tensor([float(row[key]) for row in rows])
quartiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
features = [row for row in table if not row["always on"]]
print(f"\n== {len(features)} feature modes: share of the push in each part, 10 / 25 / 50 / 75 / 90% of modes")
for name in ["mode", "kept", "built"]:
    for part in PARTS:
        print(f"  {name:5s} {part:10s}", " / ".join(f"{v:.3f}" for v in column(f"{name}: {part}", features).quantile(quartiles).tolist()))
    print(f"  {name:5s} push size ", " / ".join(f"{v:.3f}" for v in column(f"{name}: push size", features).quantile(quartiles).tolist()))


def correlate(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


cls_share = column("mode: cls", features)
angle, size = column("angle of λ", features).abs(), column("size of λ", features)
print("\n== feature modes: does the eigenvalue say where the push lands?")
print(f"  correlation of the CLS share with |angle of λ|: {correlate(cls_share, angle):.3f}; with the size of λ: {correlate(cls_share, size):.3f}; with usage: {correlate(cls_share, column('usage', features)):.3f}")
order = cls_share.argsort()
for name, rows in zip(["lowest third", "middle third", "highest third"], order.chunk(3)):
    print(f"  CLS share {name:13s} ({cls_share[rows].min():.2f} to {cls_share[rows].max():.2f}): median |angle| {angle[rows].median():.1f}°, median size {size[rows].median():.2f}, median usage {column('usage', features)[rows].median():.3f}")
print("\n== feature modes: is the built part more about the CLS than the kept part?")
gap = column("built: cls", features) - column("kept: cls", features)
turning = angle > 5
print(f"  modes that turn by over 5°: {turning.sum().item()}; built CLS share minus kept CLS share, 25 / 50 / 75%:", " / ".join(f"{v:.3f}" for v in gap[turning].quantile(quartiles[1:4]).tolist()), f"; built higher in {(gap[turning] > 0).float().mean():.0%}")
print("\n== do the modes that gray and sepia copies change most push the patch statistics?")
for key in ["gray", "sepia"]:
    change = (column(f"{key} factor size", features) - 1).abs()
    print(f"  {key}: correlation of |factor size - 1| with the CLS share {correlate(change, cls_share):.3f}; with the patch-mean share {correlate(change, column('mode: patch mean', features)):.3f}; with the patch-std share {correlate(change, column('mode: patch std', features)):.3f}")
print("\n== modes on every image")
for row in table:
    if row["always on"]:
        print(f"  mode {row['mode']:4d} λ {row['size of λ']:6.2f} at {row['angle of λ']:7.1f}°: CLS {row['mode: cls']:.2f}, patch mean {row['mode: patch mean']:.2f}, patch std {row['mode: patch std']:.2f}; push size {row['mode: push size']:.3f}; gray {row['gray factor size']:.2f}, sepia {row['sepia factor size']:.2f}")
print("\n== the 8 feature modes with the lowest and the highest CLS share")
for row in sorted(features, key=lambda row: row["mode: cls"])[:8] + sorted(features, key=lambda row: row["mode: cls"])[-8:]:
    print(f"  mode {row['mode']:4d} usage {row['usage']:.3f} λ {row['size of λ']:5.2f} at {row['angle of λ']:6.1f}°: CLS {row['mode: cls']:.2f}, patch mean {row['mode: patch mean']:.2f}, patch std {row['mode: patch std']:.2f}; gray {row['gray factor size']:.2f}, sepia {row['sepia factor size']:.2f}")
print("done")
