"""
Pictures for the phase test: for one mode, the images that keep it, laid out by the direction of the mode in its plane.

Each mode of the KAE is a pair of numbers, so an image that keeps the mode has a size (how strongly) and a direction
(a phase, 0° to 360°). The phase test in ms_score.py says that images with the same direction look alike and images with
opposite directions do not. This script shows it: six columns, one per 60° slice of the circle, each holding the images
with the largest size in that slice. Under each column: how many images fall in the slice and their most common classes.

Modes shown: the ones with the largest gap in the phase test (same-direction pairs less opposite pairs), and, for honesty,
a few modes at the median gap. Images of insects, arachnids and their relatives are never shown (same rule as the notebook:
the label or one of DINOv3's top 5 answers is such a class); a mode whose strongest images are mostly hidden is skipped.

DINOv3 run im_dino_w21_wd, ImageNet validation. Usage: phase_wheel.py <best modes> <median modes> [comma list of modes]
"""

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open_clip
import torch
from datasets import load_dataset
from torchvision import transforms

import inversion
from inversion import to_complex, train

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
INK, MUTED = "#52514e", "#898781"
SLICES, ROWS, MIN_IN_SLICE = 6, 4, 12
HIDDEN_CLASSES = torch.tensor([*range(69, 80), 110, 111, *range(113, 117), *range(118, 127), *range(300, 329), 410, 599, 815])
best_count, median_count = int(sys.argv[1]), int(sys.argv[2])
out = Path("wheel")
out.mkdir(exist_ok=True)
names = open_clip.IMAGENET_CLASSNAMES

kae, stats = inversion.load_kae()
states, labels = train.get_states(inversion.ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
hidden = torch.isin(labels, HIDDEN_CLASSES) | torch.isin(head.get_log_probs(states[:, 1]).topk(5).indices, HIDDEN_CLASSES).any(dim=1)
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])[:, 0]) for first in range(0, len(states), 5000)])
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
saved = torch.load("ms_score_im_dino_w21_wd.pt", weights_only=False)
gap = saved["phase"]["block 6"][:, 2] - saved["phase"]["block 6"][:, 6]
span = saved["ceiling"] - saved["floor"]
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
to_thumb = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(160), train.to_rgb])


def get_slices(mode):
    """Rows of the images that keep the mode and may be shown, per 60° slice, largest size first."""

    rows = ((z6[:, mode] != 0) & ~hidden).nonzero()[:, 0]
    size, phase = z6[rows, mode].abs(), torch.rad2deg(torch.angle(z6[rows, mode])) % 360
    # NOTE: Slices are centred on 0°, 60°, ... so the first slice straddles the positive first axis
    slice_of = ((phase + 180 / SLICES) % 360 / (360 / SLICES)).long()
    return [rows[slice_of == s][size[slice_of == s].argsort(descending=True)] for s in range(SLICES)]


def is_showable(mode):
    keeps = (z6[:, mode] != 0).nonzero()[:, 0]
    strongest = keeps[z6[keeps, mode].abs().argsort(descending=True)[:200]]
    filled = sum(len(rows) >= MIN_IN_SLICE for rows in get_slices(mode))
    return hidden[strongest].float().mean() < 0.2 and filled >= 4


def show(mode, position, label):
    slices = get_slices(mode)
    fig, axes = plt.subplots(ROWS, SLICES, figsize=(1.9 * SLICES, 1.9 * ROWS + 1.5), squeeze=False)
    report = []
    for s, rows in enumerate(slices):
        classes = Counter(labels[rows[:30]].tolist()).most_common(2)
        described = ", ".join(f"{names[c][:16]} ({n})" for c, n in classes)
        report.append(f"    {s * 360 // SLICES:3d}°: {len(rows):4d} images; of the strongest 30: {described}")
        for r in range(ROWS):
            ax = axes[r, s]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if r == 0:
                ax.set_title(f"direction {s * 360 // SLICES}°\n{len(rows)} images", fontsize=9, color=INK)
            if r < len(rows):
                ax.imshow(to_thumb(images[int(rows[r])]["image"]))
            if r == ROWS - 1:
                ax.set_xlabel("\n".join(f"{names[c][:18]} ({n} of 30)" for c, n in classes), fontsize=7.5, color=MUTED)
    eigen = lam[mode]
    fig.suptitle(
        f"Mode {mode}: images that keep it, by the mode's direction in its plane at block 6 (strongest first)\n"
        f"{label}: same-direction pairs score {gap[position] / span:.0%} of the floor-to-class range above opposite pairs; "
        f"λ has size {eigen.abs():.2f} and angle {torch.rad2deg(torch.angle(eigen)):.1f}°",
        fontsize=10, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"wheel_{label.split()[0]}_{mode}.png", dpi=100, facecolor="white")
    plt.close(fig)
    print(f"  mode {mode} ({label}; gap {gap[position]:+.3f}, {gap[position] / span:.0%} of the range; size of λ {eigen.abs():.2f}, angle {torch.rad2deg(torch.angle(eigen)):.1f}°)")
    print("\n".join(report), flush=True)


order = gap.nan_to_num(-1).argsort(descending=True).tolist()
valid = [p for p in order if gap[p] == gap[p]]
print(f"{len(valid)} modes with a phase test; gap 10 / 50 / 90%:", [round(v, 3) for v in gap[valid].quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()])
# Optional third argument: a comma list of modes to show whatever their gap (hidden images stay hidden)
if len(sys.argv) > 3:
    positions = {int(mode): position for position, mode in enumerate(saved["modes"].tolist())}
    for mode in [int(m) for m in sys.argv[3].split(",")]:
        show(mode, positions[mode], "chosen mode")
    print("done")
    sys.exit()
shown, skipped = 0, []
for position in valid:
    mode = int(saved["modes"][position])
    if shown == best_count:
        break
    if not is_showable(mode):
        skipped.append(mode)
        continue
    show(mode, position, "largest gap")
    shown += 1
print("skipped among the best (mostly hidden images, or fewer than 4 filled slices):", skipped)
middle = valid[len(valid) // 2 :]
shown = 0
for position in middle:
    mode = int(saved["modes"][position])
    if shown == median_count:
        break
    if not is_showable(mode):
        continue
    show(mode, position, "median gap")
    shown += 1
print("done")
