"""
Does a mode's eigenvalue angle say how much blocks 7-12 REWRITE the mode's pattern in the model's own state?

A check on the real model's states only (no K, no decoder; the KAE is used only to pick each mode's top images).
The real-start rule's overlap loss asks that a mode's first number at block 12 be "how much of its block-6 pattern is
still there", so under the rule a mode with a small angle should stand for a pattern the later blocks pass along, and a
mode with a large angle for a pattern they replace. Blocks 6 and 12 share one residual stream, so the two patterns can be
compared directly.

For each mode:
  pattern at block 6  = mean real block-6 state of the 50 images that fire the mode hardest, less the mean over all images
  pattern at block 12 = the same with the real block-12 states of the same images
  persistence         = cosine of the two patterns (1 = passed along unchanged, 0 = replaced by something unrelated)
Reported: rank correlation of persistence with |angle of λ| (and with size of λ and usage held fixed), also for the
CLS part and the patch parts of the state alone, and how repeatable persistence is (odd against even top images).

Usage: angle_persist.py <run name> [<encoder, default ViT-B-32>]
"""

import os
import sys

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

name = sys.argv[1]
ENCODER = sys.argv[2] if len(sys.argv) > 2 else "ViT-B-32"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
MIN_IMAGES, TOP_IMAGES = 200, 50
save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae"
stats = torch.load(f"{save_dir}/{name}_stats.pt")
real_start = bool(stats.get("real_start", False))
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
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])[:, 0]) for first in range(0, len(states), 5000)])
kept = z6 != 0
modes = torch.tensor([m for m in (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0].tolist() if m > 0])
usage = kept.float().mean(dim=0)[modes]
size, angle = lam.abs()[modes], torch.rad2deg(torch.angle(lam))[modes].abs()
# shape: [modes, 2 * TOP_IMAGES]; the images that fire each mode hardest at block 6, strongest first
tops = torch.stack([kept[:, m].nonzero()[:, 0][z6[kept[:, m], m].abs().topk(2 * TOP_IMAGES).indices] for m in modes.tolist()])
centred = states - states.mean(dim=0)
width = states.shape[-1] // 3
PARTS = {"whole state": slice(0, 3 * width), "CLS": slice(0, width), "patch mean": slice(width, 2 * width), "patch std": slice(2 * width, 3 * width)}
cosine = torch.nn.functional.cosine_similarity
rank = lambda a: a.argsort().argsort().float()


def get_patterns(columns):
    """Mean centred state of the chosen top images. shape: [modes, 2 (block 6, block 12), state]"""

    return torch.stack([centred[top].mean(dim=0) for top in tops[:, columns]])


def correlate(a, b, held=()):
    a, b = rank(a), rank(b)
    if held:
        design = torch.stack([torch.ones(len(a)), *[rank(h) for h in held]], dim=1)
        a = a - design @ torch.linalg.lstsq(design, a[:, None]).solution[:, 0]
        b = b - design @ torch.linalg.lstsq(design, b[:, None]).solution[:, 0]
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


patterns = get_patterns(slice(0, TOP_IMAGES))
odd, even = get_patterns(slice(0, 2 * TOP_IMAGES, 2)), get_patterns(slice(1, 2 * TOP_IMAGES, 2))
quantiles = torch.tensor([0.1, 0.5, 0.9])
show = lambda values: " / ".join(f"{v:.3f}" for v in values.quantile(quantiles).tolist())
print(f"\n== {name} ({ENCODER}; real start {real_start}): {len(modes)} modes kept on at least {MIN_IMAGES} validation images")
print(f"|angle of λ|, 10 / 50 / 90%: {show(angle)}°; size of λ: {show(size)}; usage: {show(usage)}")
print("part of the state | persistence 10 / 50 / 90% | repeatable (odd against even top images) | rank correlation with |angle| (size, usage fixed) | with size (angle, usage fixed) | with usage (angle, size fixed)")
saved = {"modes": modes, "size": size, "angle": angle, "usage": usage}
for label, part in PARTS.items():
    persistence = cosine(patterns[:, 0, part], patterns[:, 1, part], dim=-1)
    repeat = correlate(cosine(odd[:, 0, part], odd[:, 1, part], dim=-1), cosine(even[:, 0, part], even[:, 1, part], dim=-1))
    saved[label] = persistence
    print(
        f"  {label:11s} | {show(persistence)} | {repeat:+.3f} | {correlate(angle, persistence):+.3f} ({correlate(angle, persistence, (size, usage)):+.3f}) | "
        f"{correlate(size, persistence):+.3f} ({correlate(size, persistence, (angle, usage)):+.3f}) | {correlate(usage, persistence):+.3f} ({correlate(usage, persistence, (angle, size)):+.3f})"
    )
persistence = saved["whole state"]
feature = usage < 0.99
print(f"whole state, over the {int(feature.sum())} modes not on for every image: |angle| {correlate(angle[feature], persistence[feature]):+.3f} ({correlate(angle[feature], persistence[feature], (size[feature], usage[feature])):+.3f})")
for label, half in [("first", torch.arange(len(modes)) % 2 == 0), ("second", torch.arange(len(modes)) % 2 == 1)]:
    print(f"  {label} half of the modes (every other one): |angle| {correlate(angle[half], persistence[half]):+.3f} ({correlate(angle[half], persistence[half], (size[half], usage[half])):+.3f})")
kept_share = torch.cos(torch.deg2rad(angle))
print(f"cos(angle) against persistence, plain correlation: {torch.corrcoef(torch.stack([kept_share, persistence]))[0, 1].item():+.3f}")
print("by |angle of λ|: modes | persistence (median), whole state | CLS | patch mean | patch std | size of λ (median) | usage (median)")
for low, high in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75), (75, 180)]:
    inside = (angle >= low) & (angle < high)
    if inside.sum() >= 3:
        row = " | ".join(f"{saved[label][inside].median():.3f}" for label in PARTS)
        print(f"  {low:3d}-{high:3d}°: {int(inside.sum()):4d} | {row} | {size[inside].median():.2f} | {usage[inside].median():.3f}")
print("by usage (thirds), inside each: rank correlation of |angle| with persistence (whole state)")
edges = usage.quantile(torch.tensor([0, 1 / 3, 2 / 3, 1]))
for i in range(3):
    inside = (usage >= edges[i]) & (usage <= edges[i + 1])
    print(f"  usage {edges[i]:.3f} to {edges[i + 1]:.3f}: {int(inside.sum())} modes | {correlate(angle[inside], persistence[inside]):+.3f}")
torch.save(saved, f"angle_persist_{name}.pt")
print("done")
