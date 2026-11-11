"""
The modes real CLIP holds constant from block 6 to block 12, on ImageNet validation (no patches needed: only the
encoder's codes of the two real states are compared). slope and fit as in constant_modes.py. For each constant mode:
usage, the share of its variation the class label explains, and which part of the block-6 state it follows
(share of its squared correlation with the CLS token, the patch mean, the patch std).

Usage: constant_val.py <KAE run name>
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
from control_train import train  # noqa: E402
from demo_utils import class_tie, load_val  # noqa: E402
from probe_kaes import load_kae  # noqa: E402

torch.set_num_threads(16)
MIN_IMAGES = 1_000
states, labels = load_val()
kae, stats = load_kae(sys.argv[1])
states = (states - stats["mean"]) / stats["std"]
latent = torch.cat([train.encode_pairs(kae, part) for part in states.split(5_000)])
codes = torch.complex(latent[..., 1::2], latent[..., 2::2])
z6, z12 = codes[:, 0], codes[:, 1]
kept = z6 != 0
lam = torch.complex(kae.block_real, kae.block_imag)
slope = (z6.conj() * z12).sum(dim=0) / (z6.abs() ** 2).sum(dim=0).clamp_min(1e-12)
fit = 1 - ((z12 - slope * z6).abs() ** 2 * kept).sum(dim=0) / (z12.abs() ** 2 * kept).sum(dim=0).clamp_min(1e-12)
constant = ((slope - 1).abs() < 0.2) & (fit >= 0.5) & (kept.sum(dim=0) >= MIN_IMAGES)
modes = constant.nonzero()[:, 0]
print(f"== {sys.argv[1]}: {int((kept.sum(dim=0) >= MIN_IMAGES).sum())} modes in use, {len(modes)} held constant by real CLIP (slope within 0.2 of 1, fit at least 0.5)")

tie = class_tie(z6[:, modes].abs(), labels)
start = states[:, 0] - states[:, 0].mean(dim=0)
start = start / start.norm(dim=0)
print("mode | usage | λ size, angle | real slope size, angle | fit | class explains | follows CLS / patch mean / patch std")
for i, mode in enumerate(modes.tolist()):
    value = z6[:, mode].abs()
    corr = ((value - value.mean()) / (value - value.mean()).norm()) @ start
    parts = torch.stack([part.pow(2).sum() for part in corr.split(768)])
    parts = (parts / parts.sum()).tolist()
    print(
        f"{mode + 1:5d} | {kept[:, mode].float().mean():.2f} | {lam[mode].abs():.2f}, {torch.rad2deg(lam[mode].angle()):+.1f}° | "
        f"{slope[mode].abs():.2f}, {torch.rad2deg(slope[mode].angle()):+.1f}° | {fit[mode]:.2f} | {tie[i]:.2f} | {parts[0]:.2f} / {parts[1]:.2f} / {parts[2]:.2f}"
    )
others = (kept.sum(dim=0) >= MIN_IMAGES) & ~constant
print(f"for comparison, the other {int(others.sum())} modes in use: class explains (median) {class_tie(z6[:, others].abs(), labels).median():.2f}, usage (median) {kept[:, others].float().mean(dim=0).median():.2f}")
torch.save({"modes": modes, "slope": slope, "fit": fit, "z6": z6[:, modes]}, Path(__file__).parent / f"constant_val_{sys.argv[1]}.pt")
