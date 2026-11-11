"""
Which modes are held constant from block 6 to block 12? Two checks per mode, over the images that keep it:
  1. the KAE's own step: the image's eigenvalue λ + δ (λ alone without control). Constant = size near 1, angle near 0.
  2. real CLIP: the encoder's code of the real block-12 state against its code of the real block-6 state.
     slope = the complex number s that best fits z12 = s·z6; fit = share of z12's energy that s·z6 explains.

Usage: constant_modes.py <KAE run name>
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
to_complex = lambda latent: torch.complex(latent[..., 1::2], latent[..., 2::2])
show = lambda values: " / ".join(f"{v:.3f}" for v in values.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist())

data = torch.load(Path(__file__).parent / "hoofed_clip_patches.pt")
states, patches = data["states"].float(), data["patches"]
kae, stats = load_kae(sys.argv[1])
codes = to_complex(train.encode_pairs(kae, (states - stats["mean"]) / stats["std"]))  # [images, 2, modes]
z6, z12 = codes[:, 0], codes[:, 1]
kae.set_control(patches)
lam = torch.complex(kae.block_real, kae.block_imag)
step = lam + to_complex(kae.signal) if kae.control == "bilinear" else lam.expand_as(z6)
kept = z6 != 0
live = (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0]
z6, z12, step, kept, lam = z6[:, live], z12[:, live], step[:, live], kept[:, live], lam[live]

count = kept.sum(dim=0)
mean_step = (step * kept).sum(dim=0) / count
near_one = ((step - 1).abs() < 0.2) & kept  # The image's eigenvalue is within 0.2 of 1
share_near_one = near_one.sum(dim=0) / count
slope = (z6.conj() * z12).sum(dim=0) / (z6.abs() ** 2).sum(dim=0)
fit = 1 - ((z12 - slope * z6).abs() ** 2 * kept).sum(dim=0) / ((z12.abs() ** 2 * kept).sum(dim=0)).clamp_min(1e-12)

print(f"== {sys.argv[1]} (control: {kae.control}): {len(live)} modes kept on at least {MIN_IMAGES} of {len(states)} hoofed images")
print(f"shared λ: size {show(lam.abs())}, |angle| {show(torch.rad2deg(lam.angle()).abs())}°")
print(f"mean of the image's eigenvalue: size {show(mean_step.abs())}, |angle| {show(torch.rad2deg(mean_step.angle()).abs())}°")
print(f"share of a mode's images whose eigenvalue is within 0.2 of 1: {show(share_near_one)}")
print(f"real CLIP: size of slope {show(slope.abs())}, |angle of slope| {show(torch.rad2deg(slope.angle()).abs())}°, fit {show(fit)}")
for label, constant in [
    ("constant by the KAE's step (at least 90% of images within 0.2 of 1)", share_near_one >= 0.9),
    ("constant in real CLIP (slope within 0.2 of 1 and fit at least 0.5)", ((slope - 1).abs() < 0.2) & (fit >= 0.5)),
    ("both", (share_near_one >= 0.9) & ((slope - 1).abs() < 0.2) & (fit >= 0.5)),
]:
    print(f"{label}: {int(constant.sum())} modes; usage {show(count[constant] / len(states)) if constant.any() else '-'}")
rank = lambda a: a.argsort().argsort().float()
print(f"rank correlation over modes: |mean eigenvalue - 1| with |real slope - 1|: {torch.corrcoef(torch.stack([rank((mean_step - 1).abs()), rank((slope - 1).abs())]))[0, 1]:+.3f}")
torch.save({"modes": live + 1, "mean_step": mean_step, "share_near_one": share_near_one, "slope": slope, "fit": fit, "usage": count / len(states)}, Path(__file__).parent / f"constant_modes_{sys.argv[1]}.pt")
