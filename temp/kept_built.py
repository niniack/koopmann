"""
Splits what K does to each mode into a "kept" part and a "built" part, and scores each through CLIP's zero-shot head.

K multiplies a mode's block-6 value z by its eigenvalue λ = |λ|(cos θ + i sin θ). So the predicted block-12 value is
  kept part:  |λ| cos θ · z    (the block-6 value, only made larger or smaller)
  built part: |λ| sin θ · i z  (a quarter turn away from the block-6 value: something block 6 did not have in this mode)
With the `real_start` rule a block-6 value has no second number, so the kept part is the mode's first number at block 12 and the built part its second.

Usage: kept_built.py <run name> <real_start: 0 or 1> [<hidden width>]
"""

import math
import os
import sys

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

name, real_start = sys.argv[1], sys.argv[2] == "1"
torch.set_num_threads(8)
torch.set_grad_enabled(False)
ENCODER = "ViT-B-32"
save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae"
stats = torch.load(f"{save_dir}/{name}_stats.pt")
states, labels = train.get_states(ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
kae = train.JordanKoopmanAutoencoder(
    256,
    shared_modes=True,
    real_start=real_start,
    in_features=states.shape[-1],
    latent_features=2049,
    hidden_config=stats["hidden_config"],
    bias=True,
    batchnorm=False,
    nonlinearity="leaky_relu",
)
kae.load_state_dict(load_file(f"{save_dir}/{name}.safetensors"))
kae.eval()
# NOTE: Building the zero-shot classifier takes half an hour on the CPU, so the one saved by check_surrogate.py is used.
# It gives CLIP 63.0% on the first 10,000 validation images, as the freshly built one does
head = train.ClipHead(ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=False)
head.classes = torch.load("zero_shot_head.pt").float().cpu()


def to_complex(latent):
    return torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)


def to_latent(z):
    return torch.cat([z[..., :1].real, torch.stack([z[..., 1:].real, z[..., 1:].imag], dim=-1).flatten(-2)], dim=-1)


def score(z_end):
    """Accuracy, agreement with CLIP and log probabilities for these block-12 mode values."""

    log_probs = torch.cat([head.get_log_probs(kae.decode(to_latent(z_end[rows]))) for rows in ROWS])
    answers = log_probs.argmax(dim=1)
    return (answers == labels).float().mean().item(), (answers == clip_answers).float().mean().item(), log_probs


ROWS = [slice(first, first + 5000) for first in range(0, len(states), 5000)]
# shape: [images, 2, modes], complex. Mode 0 is the real mode
z = torch.cat([to_complex(train.encode_pairs(kae, states[rows])) for rows in ROWS])
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
clip_answers = torch.cat([head.classify(states[rows, 1]) for rows in ROWS])
print(f"CLIP itself {(clip_answers == labels).float().mean():.4f}")

z6, z12 = z[:, 0], z[:, 1]
kept = z6 != 0
usage = kept.float().mean(dim=0)
used = usage > 0.001
size, angle = lam.abs(), torch.rad2deg(torch.angle(lam))
print(f"{int(used.sum())} modes kept on over 0.1% of images; always on (over 99%): {int((usage > 0.99).sum())}")
quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
print("|λ| at 10/25/50/75/90%:", [round(v, 2) for v in size[used].quantile(quantiles).tolist()])
print("|angle| at 10/25/50/75/90%:", [round(v, 1) for v in angle[used].abs().quantile(quantiles).tolist()])
for low, high in [(0, 5), (5, 20), (20, 45), (45, 90), (90, 135), (135, 181)]:
    inside = used & (angle.abs() >= low) & (angle.abs() < high)
    energy = (z6[:, inside].abs().pow(2).sum() / z6[:, used].abs().pow(2).sum()).item()
    print(f"  |angle| {low:3d}-{high:3d}°: {int(inside.sum()):4d} modes, {energy:.3f} of the block-6 latent energy, median |λ| {size[inside].median().item() if inside.any() else float('nan'):.2f}")
if real_start:
    print(f"block-6 values with a second number (should be none): {int((z6[:, 1:].imag != 0).sum())}")

print("\n== whole-latent edits: accuracy | agreement with CLIP")
full = score(lam * z6)
edits = {
    "K as trained": lam * z6,
    "kept part only (|λ| cos θ · z)": lam.real * z6,
    "built part only (|λ| sin θ · iz)": 1j * lam.imag * z6,
    "no turn, same size (|λ| · z)": size * z6,
    "turn only, size 1 (e^{iθ} · z)": (lam / size.clamp_min(1e-8)) * z6,
    "no K at all (z)": z6,
    "true block-12 values of the kept modes": z12,
    "true kept part, K's built part": torch.complex((z12 * z6.conj()).real / z6.abs().clamp_min(1e-8), torch.zeros(()))
    * z6
    / z6.abs().clamp_min(1e-8)
    + 1j * lam.imag * z6,
}
for label, value in edits.items():
    accuracy, agreement, _ = score(value)
    print(f"  {label:42s} {accuracy:.4f} | {agreement:.4f}")

print("\n== does the true block 12 turn by the eigenvalue's angle? (per mode, over the images that keep it)")
turn = torch.angle(z12 * z6.conj())  # the true turn from block 6 to block 12, per image and mode
rows = []
for mode in used.nonzero()[:, 0].tolist():
    if mode == 0:
        continue
    on = kept[:, mode]
    weights = (z12[on, mode] * z6[on, mode].conj()).abs()
    mean_turn = torch.angle((weights * torch.exp(1j * turn[on, mode])).sum())
    together = ((weights * torch.exp(1j * turn[on, mode])).sum().abs() / weights.sum()).item()
    gain = (z12[on, mode].abs().pow(2).sum() / z6[on, mode].abs().pow(2).sum()).sqrt().item()
    # Share of the true block-12 value that is a quarter turn away from the block-6 value
    built_share = ((z12[on, mode] * z6[on, mode].conj()).imag.pow(2).sum() / (z12[on, mode] * z6[on, mode].conj()).abs().pow(2).sum()).item()
    rows.append((mode, usage[mode].item(), size[mode].item(), angle[mode].item(), math.degrees(mean_turn.item()), together, gain, built_share))
table = torch.tensor(rows)
gap = (table[:, 3] - table[:, 4] + 180) % 360 - 180
print(f"  |eigenvalue angle - true mean turn| at 25/50/75%: {[round(v, 1) for v in gap.abs().quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()]}°")
print(f"  how alike the true turn is across images (1 = the same for every image) at 25/50/75%: {[round(v, 2) for v in table[:, 5].quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()]}")
print(f"  correlation of sin²(eigenvalue angle) with the true built share: {torch.corrcoef(torch.stack([torch.sin(torch.deg2rad(table[:, 3])).pow(2), table[:, 7]]))[0, 1]:.3f}")
print(f"  correlation of |λ| with the true gain: {torch.corrcoef(torch.stack([table[:, 2], table[:, 6]]))[0, 1]:.3f}")

print("\n== per image: share of the predicted block-12 latent energy that is built, against how the image fares")
energy_kept = (lam.real * z6).abs().pow(2).sum(dim=1)
energy_built = (lam.imag * z6).abs().pow(2).sum(dim=1)
built_share = energy_built / (energy_kept + energy_built)
right = full[2].argmax(dim=1) == labels
agrees = full[2].argmax(dim=1) == clip_answers
print(f"  built share at 10/50/90%: {[round(v, 3) for v in built_share.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()]}")
order = built_share.argsort()
for part, label in enumerate(["lowest fifth", "second", "third", "fourth", "highest fifth"]):
    chunk = order[part * len(order) // 5 : (part + 1) * len(order) // 5]
    print(f"  {label:14s} built share {built_share[chunk].mean():.3f}: KAE right {right[chunk].float().mean():.3f}, agrees with CLIP {agrees[chunk].float().mean():.3f}, CLIP right {(clip_answers[chunk] == labels[chunk]).float().mean():.3f}")
torch.save({"table": table, "columns": ["mode", "usage", "|λ|", "angle", "true mean turn", "turn alike", "true gain", "true built share"], "built_share": built_share, "right": right, "agrees": agrees}, f"kept_built_{name}.pt")
