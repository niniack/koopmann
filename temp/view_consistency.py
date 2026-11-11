"""
Do a mode's size and phase stay the same across the three training views (random crops and flips) of one image?

If sizes were stable and phases were not, a 2x2 block would already work like a "complex cell": size says what is in the picture,
phase says how this view shows it. If both are stable, the phase is part of the content.

For each quantity the score is the share of its image-to-image variation that two views of the same image agree on
(1 = the views always agree, 0 = two views of one image are as different as two different images).
"""

import os
from pathlib import Path

import torch

import inversion
from inversion import to_complex, train

torch.set_num_threads(6)
torch.set_grad_enabled(False)
COUNT, VIEWS, MIN_IMAGES = 6000, 3, 100

kae, stats = inversion.load_kae()
path = Path(os.getenv("DATASETS_CACHE")) / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_train_3_views_blocks_5_11.pt"
cache = torch.load(path, mmap=True)
num_images = len(cache["states"]) // VIEWS
chosen = torch.randperm(num_images, generator=torch.Generator().manual_seed(0))[:COUNT].sort().values
# shape: [images, views, blocks, 2304]. Views are stored one after another
states = torch.stack([cache["states"][view * num_images + chosen].float() for view in range(VIEWS)], dim=1)
assert (cache["labels"][chosen] == cache["labels"][num_images + chosen]).all()
states = (states - stats["mean"]) / stats["std"]
# shape: [images, views, blocks, modes], complex. Each view picks its own modes at block 6
z = to_complex(train.encode_pairs(kae, states.flatten(0, 1))).unflatten(0, (COUNT, VIEWS))
kept = z[:, :, 0] != 0

same = (kept[:, 0] & kept[:, 1]).sum(dim=1) / (kept[:, 0] | kept[:, 1]).sum(dim=1)
other = (kept[:, 0] & kept[:, 1].roll(1, dims=0)).sum(dim=1) / (kept[:, 0] | kept[:, 1].roll(1, dims=0)).sum(dim=1)
print(f"modes shared by two views of one image (Jaccard): {same.mean():.3f}; by two different images: {other.mean():.3f}")


def get_agreement(a, b):
    """Share of the variation across images that two views agree on, for real or complex values. shape of a, b: [images]"""

    centre = torch.cat([a, b]).mean()
    spread = torch.cat([a - centre, b - centre]).abs().pow(2).mean()
    return (1 - (a - b).abs().pow(2).mean() / (2 * spread)).item()


rows = []
for m in range(1, z.shape[-1]):
    both = kept[:, 0, m] & kept[:, 1, m]
    if both.sum() < MIN_IMAGES:
        continue
    a, b = z[both, 0, :, m], z[both, 1, :, m]
    unit = lambda value: value / value.abs().clamp_min(1e-8)
    turn_a, turn_b = unit(a[:, 1] * a[:, 0].conj()), unit(b[:, 1] * b[:, 0].conj())
    rows.append(
        [
            m,
            both.float().mean().item(),
            # Among images where one view keeps the mode, how often the other view does too
            (both.sum() / (kept[:, 0, m] | kept[:, 1, m]).sum()).item(),
            get_agreement(a[:, 0].abs().log(), b[:, 0].abs().log()),
            get_agreement(unit(a[:, 0]), unit(b[:, 0])),
            get_agreement(a[:, 1].abs().log(), b[:, 1].abs().log()),
            get_agreement(unit(a[:, 1]), unit(b[:, 1])),
            get_agreement(turn_a, turn_b),
            # How much the phase varies across images at all: 0 = the same phase everywhere
            1 - unit(torch.cat([a[:, 0], b[:, 0]])).mean().abs().item(),
        ]
    )
table = torch.tensor(rows)
columns = ["mode", "usage", "kept by both", "size at 6", "phase at 6", "size at 12", "phase at 12", "turn 6 to 12", "phase spread at 6"]
always = torch.isin(table[:, 0].long(), torch.tensor([861, 338, 144, 560, 117, 988, 871, 746, 652, 533, 331, 77, 68, 653, 877, 465]))
print(f"{len(table)} modes kept by both views on at least {MIN_IMAGES} of {COUNT} images; {int((~always).sum())} feature modes")
quartiles = torch.tensor([0.25, 0.5, 0.75])
print("\nfeature modes: agreement between two views of one image, 25% / median / 75%")
for column in range(2, len(columns)):
    values = table[~always, column].quantile(quartiles).tolist()
    print(f"  {columns[column]:18s} {values[0]:.3f} / {values[1]:.3f} / {values[2]:.3f}")
wide = ~always & (table[:, 8] > 0.2)
print(f"\nthe same for the {int(wide.sum())} feature modes whose block-6 phase varies a lot across images (spread over 0.2)")
for column in range(3, 8):
    values = table[wide, column].quantile(quartiles).tolist()
    print(f"  {columns[column]:18s} {values[0]:.3f} / {values[1]:.3f} / {values[2]:.3f}")
print(f"\nmodes where the size agrees more than the phase at block 6: {(table[wide, 3] > table[wide, 4]).float().mean():.2f} of those")
print("\nalways-on modes")
for row in table[always].tolist():
    print(f"  mode {int(row[0]):4d}: " + ", ".join(f"{columns[c]} {row[c]:.2f}" for c in range(3, 9)))
torch.save({"table": table, "columns": columns}, "view_consistency.pt")
