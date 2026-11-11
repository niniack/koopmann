"""
Does a mode's eigenvalue angle say how much blocks 7-12 REWRITE the mode's pattern in the model's own state?
Second version of angle_persist.py: a mode's pattern is its PARTIAL effect on the real state, with the other modes held
fixed, in place of the mean state of its top images (128 modes are on per image, so top images share many modes).

Patterns, by ridge regression over the 50,000 validation images (real states of the model as targets; no K, no decoder):
  U[c] = effect of block-6 coordinate c on the real block-6 state        (what the coordinate stands for at block 6)
  W[c] = effect of block-6 coordinate c on the real block-12 state       (what it leads to at block 12)
  A[c], B[c] = effect on the real block-12 state of the first and second coordinate of the block-12 code of the same mode
A mode has two coordinates (one at block 6 under the real-start rule).

Checks:
  1. persistence = cosine of U and W (mean over the mode's coordinates). If the mode's plane is a fixed pair of patterns
     that K turns by the angle of λ, persistence = cos(angle). Rank correlation with |angle|, size and usage held fixed.
  2. the rule's structure: is the first coordinate at block 12 the block-6 pattern (cosine of U and A high), and the second
     coordinate something else (cosine of U and B near 0)?
  3. repeatability: persistence from the odd images against the even images.

Usage: angle_persist_reg.py <run name> [<encoder, default ViT-B-32>]
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
MIN_IMAGES, RIDGE = 200, 1e-3
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
# shape: [images, 2 (block 6, block 12), 2049]; the KAE's code of the real state at each block
codes = torch.cat([train.encode_pairs(kae, states[first : first + 5000]) for first in range(0, len(states), 5000)])
first, second = codes[..., 1::2], codes[..., 2::2]
kept = (first[:, 0] != 0) | (second[:, 0] != 0)
live = (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0]
modes = live + 1
usage = kept.float().mean(dim=0)[live]
size, angle = lam.abs()[modes], torch.rad2deg(torch.angle(lam))[modes].abs()
has_second = bool((second[:, 0, live] != 0).any())
centred = (states - states.mean(dim=0)).double()
cosine = torch.nn.functional.cosine_similarity
rank = lambda a: a.argsort().argsort().float()


def get_patterns(block, rows):
    """Partial effect of every coordinate of the code at `block` on the real states of both blocks.

    Returns (patterns [coordinates, 2 (block 6, block 12), state], share of the state's variance explained [2]).
    Coordinates: the real mode, then the first coordinate of every live mode, then (if in use) the second.
    """

    parts = [codes[rows, block, :1], first[rows, block][:, live], *([second[rows, block][:, live]] if block == 1 or has_second else [])]
    x = torch.cat(parts, dim=1).double()
    x = x - x.mean(dim=0)
    gram = x.T @ x
    gram = gram + RIDGE * gram.diagonal().mean() * torch.eye(len(gram), dtype=torch.double)
    y = centred[rows]
    w = torch.linalg.solve(gram, x.T @ y.flatten(1)).view(x.shape[1], 2, -1)
    explained = 1 - (y - (x @ w.flatten(1)).view_as(y)).pow(2).sum(dim=(0, 2)) / y.pow(2).sum(dim=(0, 2))

    return w.float(), explained


def get_checks(rows):
    """Per mode: persistence, cosine of U and A, cosine of U and B (first coordinate; and second where there is one)."""

    count = len(live)
    start, explained_start = get_patterns(0, rows)
    end, explained_end = get_patterns(1, rows)
    u, w = start[1 : 1 + count, 0], start[1 : 1 + count, 1]
    a, b = end[1 : 1 + count, 1], end[1 + count :, 1]
    checks = {"persistence": cosine(u, w, dim=-1), "U with A": cosine(u, a, dim=-1), "U with B": cosine(u, b, dim=-1)}
    if has_second:
        v, w2 = start[1 + count :, 0], start[1 + count :, 1]
        checks["persistence"] = (checks["persistence"] + cosine(v, w2, dim=-1)) / 2
        checks["V with B"], checks["V with A"] = cosine(v, b, dim=-1), cosine(v, a, dim=-1)
    return checks, explained_start, explained_end


def correlate(a, b, held=()):
    a, b = rank(a), rank(b)
    if held:
        design = torch.stack([torch.ones(len(a)), *[rank(h) for h in held]], dim=1)
        a = a - design @ torch.linalg.lstsq(design, a[:, None]).solution[:, 0]
        b = b - design @ torch.linalg.lstsq(design, b[:, None]).solution[:, 0]
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


everything = torch.arange(len(states))
checks, explained_start, explained_end = get_checks(everything)
odd, _, _ = get_checks(everything[1::2])
even, _, _ = get_checks(everything[::2])
quantiles = torch.tensor([0.1, 0.5, 0.9])
show = lambda values: " / ".join(f"{v:.3f}" for v in values.quantile(quantiles).tolist())
print(f"\n== {name} ({ENCODER}; real start {real_start}): {len(live)} modes kept on at least {MIN_IMAGES} validation images")
print(f"|angle of λ|, 10 / 50 / 90%: {show(angle)}°; size of λ: {show(size)}; usage: {show(usage)}")
print(f"share of the real state a linear map from the code explains: block-6 code to block 6 {explained_start[0]:.3f}, to block 12 {explained_start[1]:.3f}; block-12 code to block 12 {explained_end[1]:.3f}")
print("check | 10 / 50 / 90% of modes | repeatable (odd against even images)")
for label, values in checks.items():
    print(f"  {label:11s} | {show(values)} | {correlate(odd[label], even[label]):+.3f}")
persistence = checks["persistence"]
print("rank correlation of persistence with (in brackets: the other two held fixed):")
print(f"  |angle of λ|  {correlate(angle, persistence):+.3f} ({correlate(angle, persistence, (size, usage)):+.3f})")
print(f"  size of λ     {correlate(size, persistence):+.3f} ({correlate(size, persistence, (angle, usage)):+.3f})")
print(f"  usage         {correlate(usage, persistence):+.3f} ({correlate(usage, persistence, (angle, size)):+.3f})")
print(f"  plain correlation of cos(angle) with persistence: {torch.corrcoef(torch.stack([torch.cos(torch.deg2rad(angle)), persistence]))[0, 1].item():+.3f}")
feature = usage < 0.99
print(f"  over the {int(feature.sum())} modes not on for every image: |angle| {correlate(angle[feature], persistence[feature]):+.3f} ({correlate(angle[feature], persistence[feature], (size[feature], usage[feature])):+.3f})")
for label, half in [("first", torch.arange(len(live)) % 2 == 0), ("second", torch.arange(len(live)) % 2 == 1)]:
    print(f"  {label} half of the modes (every other one): |angle| {correlate(angle[half], persistence[half]):+.3f} ({correlate(angle[half], persistence[half], (size[half], usage[half])):+.3f})")
print("by |angle of λ|: modes | cos(angle) (median) | persistence (median) | U with A | U with B | size of λ | usage")
for low, high in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75), (75, 180)]:
    inside = (angle >= low) & (angle < high)
    if inside.sum() >= 3:
        print(
            f"  {low:3d}-{high:3d}°: {int(inside.sum()):4d} | {torch.cos(torch.deg2rad(angle[inside])).median():.3f} | {persistence[inside].median():.3f} | "
            f"{checks['U with A'][inside].median():.3f} | {checks['U with B'][inside].median():.3f} | {size[inside].median():.2f} | {usage[inside].median():.3f}"
        )
torch.save({"modes": modes, "size": size, "angle": angle, "usage": usage, **checks}, f"angle_persist_reg_{name}.pt")
print("done")
