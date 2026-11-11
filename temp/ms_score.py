"""
How single-meaning is each mode, and does the direction of a mode within its plane (its phase) pick out a variant?

The yardstick is the monosemanticity score of Pach et al. (NeurIPS 2025, arXiv 2504.02821), written from the paper's
description, not from its code. For one feature with activations a_n over images, scaled to 0..1 over the images:

    MS = sum over pairs of images (n, m) of a_n · a_m · s_nm, divided by the sum of a_n · a_m,

where s_nm is the cosine similarity of the two images in a separate image encoder (the "judge"). A feature that fires on
images that look alike to the judge scores high. Differences from the paper: the judge here is the other model of our two
(DINOv3 judges a CLIP KAE, as the paper's DINOv2 judges CLIP; CLIP's image embedding judges the DINOv3 KAE), the images
are ImageNet's 50,000 validation images, and a mode's activation is the size of its 2-number block at block 6.

Scale for reading a score: the "floor" is the score of a feature that fires equally on every image (the average similarity
of two images), and the "class ceiling" is the score of a feature that fires on exactly the images of one class.

The phase test. For one mode and the images that keep it, pairs of images are split by the difference of their phases:
same direction (within 60°), right angle, opposite (within 60° of 180°). If the direction in the plane picks a variant of
the mode's meaning, same-direction pairs look more alike to the judge than opposite pairs.

Usage: ms_score.py dino | ms_score.py <CLIP run name> <real_start: 0 or 1>
"""

import os
import sys

import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
MIN_IMAGES, MOST_IMAGES = 200, 2000
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from dotenv import load_dotenv  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)
save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae"
if sys.argv[1] == "dino":
    name, encoder, judge, real_start = "im_dino_w21_wd", "dinov3-vitb16", "ViT-B-32", False
else:
    name, encoder, judge, real_start = sys.argv[1], "ViT-B-32", "dinov3-vitb16", sys.argv[2] == "1"
stats = torch.load(f"{save_dir}/{name}_stats.pt")
raw, labels = train.get_states(encoder, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
raw = raw.float()
kae = train.JordanKoopmanAutoencoder(256, shared_modes=True, real_start=real_start, in_features=raw.shape[-1], latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(f"{save_dir}/{name}.safetensors"))
kae.eval()

############# THE JUDGE #############

judge_raw, judge_labels = train.get_states(judge, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
assert torch.equal(labels, judge_labels), "the two caches do not list the images in the same order"
zeros, ones = torch.zeros(judge_raw.shape[-1]), torch.ones(judge_raw.shape[-1])
Head = train.ClipHead if judge == "ViT-B-32" else train.DinoHead
# shape: [images, width], unit length
embeddings = Head(judge, zeros, ones, "cpu", zero_shot=False).embed(judge_raw[:, 1].float()).double()
del judge_raw


def get_score(activations):
    """The score of each column of [images, features], over all pairs of different images."""

    low, high = activations.min(dim=0).values, activations.max(dim=0).values
    scaled = ((activations - low) / (high - low).clamp(min=1e-12)).double()
    squares = scaled.pow(2).sum(dim=0)
    # NOTE: Sum over pairs of a_n·a_m·(e_n·e_m) is the squared length of the sum of a_n·e_n, less the pairs of an image with itself
    return (((scaled.T @ embeddings).pow(2).sum(dim=1) - squares) / (scaled.sum(dim=0).pow(2) - squares).clamp(min=1e-12)).float()


############# SCORES #############

states = (raw - stats["mean"]) / stats["std"]
z = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])) for first in range(0, len(states), 5000)])
z6, z12 = z[:, 0], z[:, 1]
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
kept = z6 != 0
usage = kept.float().mean(dim=0)
modes = ((kept.sum(dim=0) >= MIN_IMAGES) & (usage < 0.99)).nonzero()[:, 0]
angle, size, use = torch.rad2deg(torch.angle(lam[modes])).abs(), lam[modes].abs(), usage[modes]
# The score of a feature that fires equally on every image
floor = ((embeddings.sum(dim=0).pow(2).sum() - len(labels)) / (len(labels) ** 2 - len(labels))).item()
scores = {
    "modes, size at block 6": get_score(z6[:, modes].abs()),
    "modes, size at block 12": get_score(z12[:, modes].abs()),
    "one class": get_score(torch.nn.functional.one_hot(labels, 1000).float()),
    "CLS units of the model, block 6": get_score(raw[:, 0, :768]),
    "CLS units of the model, block 12": get_score(raw[:, 1, :768]),
    "random directions of the block-6 state": get_score(states[:, 0] @ torch.randn(states.shape[-1], 512)),
    "random directions, top 12% kept": None,
}
# NOTE: A mode is kept on about 12% of images. Random directions cut the same way show what sparsity alone buys.
random = states[:, 0] @ torch.randn(states.shape[-1], 512)
cut = random.quantile(1 - use.median().item(), dim=0)
scores["random directions, top 12% kept"] = get_score((random - cut).clamp(min=0))


def keep_top(activations, share):
    """Only the strongest share of each column's images, measured from the cut."""

    return (activations - activations.quantile(1 - share, dim=0)).clamp(min=0)


# NOTE: 128 of 1,024 modes are kept per image, so a mode fires on about 1 image in 8, far more than one meaning covers.
# The strongest 1% of a mode's images (500 images) shows what its top end holds, against other features cut the same way.
scores["modes, strongest 1% of images"] = get_score(keep_top(z6[:, modes].abs(), 0.01))
scores["modes at block 12, strongest 1%"] = get_score(keep_top(z12[:, modes].abs(), 0.01))
scores["random directions, strongest 1%"] = get_score(keep_top(random, 0.01))
scores["CLS units, block 12, strongest 1%"] = get_score(keep_top(raw[:, 1, :768], 0.01))
ceiling = scores["one class"].median().item()
print(f"== {name}, judged by {judge}: {len(modes)} feature modes; floor {floor:.3f}, class ceiling (median class) {ceiling:.3f}")
quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
print("score, 10 / 25 / 50 / 75 / 90%; then the median as a share of the way from the floor to the class ceiling")
for key, values in scores.items():
    print(f"  {key:40s}", " / ".join(f"{v:.3f}" for v in values.quantile(quantiles).tolist()), f"   {(values.median().item() - floor) / (ceiling - floor):6.1%}")
mode_score = scores["modes, size at block 6"]
for key in ["CLS units of the model, block 6", "CLS units of the model, block 12", "random directions, top 12% kept"]:
    print(f"  modes above the 95% point of '{key}': {(mode_score > scores[key].quantile(0.95)).float().mean():.0%}")
print(f"  modes above half-way to the class ceiling: {(mode_score > (floor + ceiling) / 2).float().mean():.0%}; above the median class: {(mode_score > ceiling).float().mean():.0%}")

rank = lambda a: a.argsort().argsort().float()
correlate = lambda a, b: torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()


def partial(a, b, controls):
    """Rank correlation of a and b once the controls are fitted out of both."""

    design = torch.stack([torch.ones(len(a))] + [rank(c) for c in controls], dim=1)
    left = lambda v: rank(v) - design @ torch.linalg.lstsq(design, rank(v)[:, None]).solution[:, 0]
    return correlate(left(a), left(b))


print("\nrank correlation of a mode's score with: size of λ (then |angle| and usage fixed) | |angle| (size and usage fixed) | usage")
for key in ["modes, size at block 6", "modes, size at block 12"]:
    values = scores[key]
    print(f"  {key:26s} {correlate(rank(size), rank(values)):6.3f} ({partial(size, values, [angle, use]):6.3f}) | {correlate(rank(angle), rank(values)):6.3f} ({partial(angle, values, [size, use]):6.3f}) | {correlate(rank(use), rank(values)):6.3f}")
print("by size of λ: modes | score at block 6 | at block 12")
for part, rows in enumerate(size.argsort().chunk(5)):
    print(f"  fifth {part + 1} (size {size[rows].min():.2f} to {size[rows].max():.2f}): {len(rows):4d} | {scores['modes, size at block 6'][rows].median():.3f} | {scores['modes, size at block 12'][rows].median():.3f}")

############# THE PHASE TEST #############

KINDS = ["all pairs", "same direction", "right angle", "opposite"]
table = {"block 6": [], "block 12": [], "turn 6 to 12": []}
for count, m in enumerate(modes.tolist()):
    on = (kept[:, m] & (z12[:, m] != 0)).nonzero()[:, 0]
    rows = on[torch.randperm(len(on))[:MOST_IMAGES]]
    similar = (embeddings[rows] @ embeddings[rows].T).float()
    off = ~torch.eye(len(rows), dtype=torch.bool)
    for key, values, weights in [("block 6", z6[rows, m], z6[rows, m].abs()), ("block 12", z12[rows, m], z12[rows, m].abs()), ("turn 6 to 12", z12[rows, m] / z6[rows, m], z6[rows, m].abs())]:
        unit = values / values.abs()
        # cosine of the difference of two images' phases
        agree = (unit[:, None] * unit[None, :].conj()).real
        weight = (weights[:, None] * weights[None, :]) * off
        row = []
        for mask in [torch.ones_like(off), agree > 0.5, agree.abs() <= 0.5, agree < -0.5]:
            total = (weight * mask).sum()
            row += [((weight * mask * similar).sum() / total).item() if total > 0 else float("nan"), (mask & off).float().sum().item() / off.sum().item()]
        table[key].append(row)
    if count % 100 == 0:
        print(f"  {count} modes done", flush=True)
print("\nthe phase test: similarity of pairs of images that keep a mode, weighted by both sizes, by the difference of their phases")
print("median over modes (share of pairs in brackets): all pairs | same direction | right angle | opposite | same above opposite in | gap as a share of floor-to-ceiling")
results = {}
for key, rows in table.items():
    rows = torch.tensor(rows)
    results[key] = rows
    gap = rows[:, 2] - rows[:, 6]
    valid = ~gap.isnan()
    print(f"  phase at {key:13s}" + " | ".join(f"{rows[:, 2 * k].nanmedian():.3f} ({rows[:, 2 * k + 1].nanmedian():.2f})" for k in range(4)) + f" | {(gap[valid] > 0).float().mean():.0%} | {gap[valid].median() / (ceiling - floor):.1%}")
    print(f"      gap (same less opposite), 10 / 50 / 90%: " + " / ".join(f"{v:+.3f}" for v in gap[valid].quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()) + f"; rank correlation of the gap with size of λ {correlate(rank(size[valid]), rank(gap[valid])):.3f} ({partial(size[valid], gap[valid], [angle[valid], use[valid]]):.3f}), with |angle| {correlate(rank(angle[valid]), rank(gap[valid])):.3f} ({partial(angle[valid], gap[valid], [size[valid], use[valid]]):.3f})")
torch.save({"modes": modes, "angle": angle, "size": size, "usage": use, "floor": floor, "ceiling": ceiling, "scores": scores, "kinds": KINDS, "phase": results}, f"ms_score_{name}.pt")
print("done")
