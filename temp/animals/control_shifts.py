"""
What does the bilinear control do to each mode's eigenvalue? Per image the step is z12 = (λ + δ)·z6, so the image's own
eigenvalue is λ + δ. For every mode, over the images that keep it:
  size change = log(|λ + δ| / |λ|),   turn = angle of (λ + δ) / λ
and how far each varies across images, and how much of that variation the species label explains.

Usage: control_shifts.py <bilinear KAE run name>
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from control_train import train  # noqa: E402
from probe_kaes import load_kae  # noqa: E402

torch.set_num_threads(16)
torch.set_grad_enabled(False)
MIN_IMAGES = 200


def explained_by_label(values, kept, labels):
    """Per mode: share of the variation of `values` over the images that keep the mode that the label explains."""

    one_hot = torch.nn.functional.one_hot(labels).float()
    count = one_hot.T @ kept.float()
    class_mean = (one_hot.T @ (values * kept)) / count.clamp_min(1)
    mean = (values * kept).sum(dim=0) / kept.sum(dim=0)
    between = (count * (class_mean - mean) ** 2).sum(dim=0)
    total = (kept * (values - mean) ** 2).sum(dim=0)
    return between / total.clamp_min(1e-12)


data = torch.load(Path(__file__).parent / "hoofed_clip_patches.pt")
states, patches, labels = data["states"].float(), data["patches"], data["labels"]
kae, stats = load_kae(sys.argv[1])
latent = train.encode_pairs(kae, (states - stats["mean"]) / stats["std"])[:, 0]
kae.set_control(patches)
z6 = torch.complex(latent[:, 1::2], latent[:, 2::2])
delta = torch.complex(kae.signal[:, 1::2], kae.signal[:, 2::2])
lam = torch.complex(kae.block_real, kae.block_imag)
kept = z6 != 0
live = kept.sum(dim=0) >= MIN_IMAGES
ratio = ((lam + delta) / lam)[:, live]
kept, lam = kept[:, live], lam[live]
size, turn = ratio.abs().log(), torch.rad2deg(ratio.angle())

show = lambda values: " / ".join(f"{v:.3f}" for v in values.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist())
count = kept.sum(dim=0)
spread = lambda values: (((values - (values * kept).sum(dim=0) / count) ** 2 * kept).sum(dim=0) / count).sqrt()
print(f"== {sys.argv[1]}: {int(live.sum())} modes kept on at least {MIN_IMAGES} of {len(states)} hoofed images")
print(f"|λ|, 10 / 50 / 90% of modes: {show(lam.abs())}; |angle of λ| in degrees: {show(torch.rad2deg(lam.angle()).abs())}")
print(f"|δ| / |λ|, median over a mode's images: {show(torch.stack([(ratio[kept[:, m], m] - 1).abs().median() for m in range(len(lam))]))}")
print(f"mean size change per mode: {show((size * kept).sum(dim=0) / count)}; spread across images: {show(spread(size))}")
print(f"mean turn per mode (degrees): {show((turn * kept).sum(dim=0) / count)}; spread across images: {show(spread(turn))}")
print(f"share of the variation across images the species explains: size change {show(explained_by_label(size, kept, labels))}; turn {show(explained_by_label(turn, kept, labels))}")
print(f"  for comparison, the mode's own size |z6|: {show(explained_by_label(z6[:, live].abs(), kept, labels))}")
torch.save({"modes": live.nonzero()[:, 0] + 1, "size": size, "turn": turn, "kept": kept, "eigenvalues": lam, "labels": labels}, Path(__file__).parent / f"control_shifts_{sys.argv[1]}.pt")
