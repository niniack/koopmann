"""
Does a mode's eigenvalue angle say WHEN the mode's concept arises in the real model?

A check that uses the real model only (no K, no decoder), so it can confirm or refute a meaning of the angle from outside.

For each mode, its concept is taken from the real model's final embedding: the mean embedding of the 50 validation
images that fire the mode hardest at block 6, less the mean over all images (the same definition as name_modes.py).
An image's "concept score" is its real final embedding along that direction. Then:
  readable at block 6 = how well a linear map from the real block-6 state (CLS, patch mean, patch std) predicts the
                        concept score, as R² on held-out images (ridge regression; strength picked on a third split)
A concept that block 6 already holds in plain sight scores high; one that blocks 7-12 still have to build scores low.

The question: do modes with a large eigenvalue angle stand for concepts that are less readable at block 6 (built
late), also with the size of λ and the mode's usage held fixed?

Usage: angle_readable.py <run name> <real_start: 0 or 1> [<encoder, default ViT-B-32>]
"""

import os
import sys

import open_clip
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

name, real_start = sys.argv[1], sys.argv[2] == "1"
ENCODER = sys.argv[3] if len(sys.argv) > 3 else "ViT-B-32"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
FIT, PICK, MIN_IMAGES, TOP_IMAGES = 25000, 5000, 200, 50
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
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
normalize = torch.nn.functional.normalize
if ENCODER in train.DINO_MODELS:
    head = train.DinoHead(ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
    embed = lambda rows: normalize(head.features(rows), dim=-1)
else:
    head = train.ClipHead(ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=False)
    embed = head.embed
to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)
ROWS = [slice(first, first + 5000) for first in range(0, len(states), 5000)]
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[rows])[:, 0]) for rows in ROWS])
# shape: [images, width]; the real model's final embedding
real = torch.cat([embed(states[rows, 1]) for rows in ROWS])
kept = z6 != 0
modes = torch.tensor([m for m in (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0].tolist() if m > 0])
usage = kept.float().mean(dim=0)[modes]
tops = [kept[:, m].nonzero()[:, 0][z6[kept[:, m], m].abs().topk(TOP_IMAGES).indices] for m in modes.tolist()]
# shape: [modes, width]; what the images that fire a mode hardest have in common, in the real model's final embedding
concepts = torch.stack([real[top].mean(dim=0) for top in tops]) - real.mean(dim=0)
# shape: [images, modes]; each image's score on each mode's concept
scores = (real - real.mean(dim=0)) @ normalize(concepts, dim=-1).T
live = kept.sum(dim=0)[1:]
print(f"modes kept on no validation image: {int((live == 0).sum())} of 1024; on 1 to {MIN_IMAGES - 1}: {int(((live > 0) & (live < MIN_IMAGES)).sum())}; on {MIN_IMAGES} or more: {int((live >= MIN_IMAGES).sum())}; on 99% or more: {int((live >= 0.99 * len(states)).sum())}")

order = torch.randperm(len(states))
fit, pick, test = order[:FIT], order[FIT : FIT + PICK], order[FIT + PICK :]


def get_r2(inputs):
    """Held-out R² of a ridge map from `inputs` to every concept score. shape: [modes]"""

    centre, target_centre = inputs[fit].mean(dim=0), scores[fit].mean(dim=0)
    x, y = (inputs[fit] - centre).double(), (scores[fit] - target_centre).double()
    gram, cross = x.T @ x, x.T @ y
    r2 = lambda rows, w: 1 - (scores[rows] - target_centre - ((inputs[rows] - centre).double() @ w).float()).pow(2).mean(dim=0) / scores[rows].var(dim=0)
    best = None
    for strength in [1, 10, 100, 1000, 10000]:
        w = torch.linalg.solve(gram + strength * torch.eye(len(gram), dtype=torch.double), cross)
        picked = r2(pick, w).mean().item()
        if best is None or picked > best[0]:
            best = (picked, strength, w)
    print(f"  ridge strength {best[1]} (mean R² on the picking split {best[0]:.3f})")

    return r2(test, best[2])


print("linear map from the real block-6 state to the concept scores")
at_6 = get_r2(states[:, 0])
print("linear map from the real block-12 state (a ceiling: the score is nearly a linear read of it)")
at_12 = get_r2(states[:, 1])
size, angle = lam.abs()[modes], torch.rad2deg(torch.angle(lam))[modes].abs()
rank = lambda a: a.argsort().argsort().float()


def correlate(a, b, held=(), pick=None):
    """Rank correlation of a and b over the modes in `pick`, with the ranks of `held` regressed out of both."""

    pick = torch.ones(len(a), dtype=torch.bool) if pick is None else pick
    a, b = rank(a[pick]), rank(b[pick])
    if held:
        design = torch.stack([torch.ones(len(a)), *[rank(h[pick]) for h in held]], dim=1)
        a = a - design @ torch.linalg.lstsq(design, a[:, None]).solution[:, 0]
        b = b - design @ torch.linalg.lstsq(design, b[:, None]).solution[:, 0]
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


quantiles = torch.tensor([0.1, 0.5, 0.9])
show = lambda values: " / ".join(f"{v:.3f}" for v in values.quantile(quantiles).tolist())
print(f"\n== {name} ({ENCODER}): {len(modes)} modes")
print(f"readable at block 6 (R²), 10 / 50 / 90% of modes: {show(at_6)}; at block 12: {show(at_12)}")
print(f"|angle of λ|, 10 / 50 / 90%: {show(angle)}°; size of λ: {show(size)}")
print("rank correlation with 'readable at block 6' (then with the other two of angle, size, usage held fixed):")
print(f"  |angle of λ|  {correlate(angle, at_6):+.3f} ({correlate(angle, at_6, (size, usage)):+.3f})")
print(f"  size of λ     {correlate(size, at_6):+.3f} ({correlate(size, at_6, (angle, usage)):+.3f})")
print(f"  usage         {correlate(usage, at_6):+.3f} ({correlate(usage, at_6, (angle, size)):+.3f})")
print(f"  kept gain |λ|·cos(angle)   {correlate(size * torch.cos(torch.deg2rad(angle)), at_6):+.3f}")
print(f"  built gain |λ|·sin(angle)  {correlate(size * torch.sin(torch.deg2rad(angle)), at_6):+.3f}")
feature = usage < 0.99
print(f"the same over the {int(feature.sum())} modes that are not on for every image:")
print(f"  |angle of λ|  {correlate(angle, at_6, pick=feature):+.3f} ({correlate(angle, at_6, (size, usage), feature):+.3f});  size of λ  {correlate(size, at_6, pick=feature):+.3f} ({correlate(size, at_6, (angle, usage), feature):+.3f})")
for label, half in [("first", torch.arange(len(modes)) % 2 == 0), ("second", torch.arange(len(modes)) % 2 == 1)]:
    print(f"  {label} half of the modes (every other one): |angle of λ|  {correlate(angle, at_6, pick=half):+.3f} ({correlate(angle, at_6, (size, usage), half):+.3f})")
print("by |angle of λ|: modes | readable at block 6 (median) | at block 12 | size of λ (median) | usage (median)")
for low, high in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75), (75, 180)]:
    inside = (angle >= low) & (angle < high)
    if inside.sum() >= 3:
        print(f"  {low:3d}-{high:3d}°: {int(inside.sum()):4d} | {at_6[inside].median():.3f} | {at_12[inside].median():.3f} | {size[inside].median():.2f} | {usage[inside].median():.3f}")
print("by usage (thirds), inside each: rank correlation of |angle of λ| with 'readable at block 6' | of size of λ")
edges = usage.quantile(torch.tensor([0, 1 / 3, 2 / 3, 1]))
for i in range(3):
    inside = (usage >= edges[i]) & (usage <= edges[i + 1])
    print(f"  usage {edges[i]:.3f} to {edges[i + 1]:.3f}: {int(inside.sum())} modes | {correlate(angle, at_6, pick=inside):+.3f} | {correlate(size, at_6, pick=inside):+.3f}")
if ENCODER not in train.DINO_MODELS:
    class_names = open_clip.IMAGENET_CLASSNAMES
    print("examples, most and least readable at block 6: mode | angle | size | usage | R² at 6 | most common classes of the mode's top images")
    by_r2 = at_6.argsort(descending=True).tolist()
    for i in by_r2[:6] + by_r2[-6:]:
        common = torch.bincount(labels[tops[i]], minlength=1000).topk(3)
        described = "; ".join(f"{class_names[c]} ({n})" for n, c in zip(common.values.tolist(), common.indices.tolist()))
        print(f"  {int(modes[i]):5d} | {torch.rad2deg(torch.angle(lam))[modes[i]]:6.1f}° | {size[i]:.2f} | {usage[i]:.3f} | {at_6[i]:.3f} | {described}")
torch.save({"modes": modes, "at 6": at_6, "at 12": at_12, "size": size, "angle": angle, "usage": usage}, f"angle_readable_{name}.pt")
print("done")
