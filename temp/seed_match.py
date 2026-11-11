"""
Are a mode's eigenvalue (size and angle) and its phase properties of the FEATURE, or accidents of one training run?

Two KAEs trained with the same settings and different seeds are compared on the same validation images.
  1. Match the modes of run A to the modes of run B by how alike they fire: the correlation, over images, of the
     mode's size at block 6 (0 where the mode is not kept). One-to-one matching (Hungarian), and the stricter
     "mutual best" pairs (each is the other's best partner).
  2. Over matched pairs: does the size of λ agree? Does the |angle| of λ agree? (rank correlation; also with each
     run's size of λ and usage held fixed; chance = the same over shuffled pairs)
  3. Phase: among the images that keep both modes of a pair, is the phase in run A the phase in run B plus a fixed
     offset (or its mirror image plus an offset)? Measured by the length of the mean of exp(i(a - b)) (or a + b for
     the mirror image): 1 = a fixed offset, about 1/sqrt(images) = unrelated.
     If the plane's orientation is a property of the feature, a pair related by an offset should have eigenvalue
     angles of the same sign, and a pair related by a mirror image angles of opposite sign.

What would count against a meaning for the angle: matched modes whose angles are unrelated across seeds.

Usage: seed_match.py <run A> <run B> [<encoder, default ViT-B-32>]
"""

import os
import sys

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

names = sys.argv[1:3]
ENCODER = sys.argv[3] if len(sys.argv) > 3 else "ViT-B-32"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
MIN_IMAGES, MIN_SHARED = 200, 100
save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae"
raw, labels = train.get_states(ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)
rank = lambda a: a.argsort().argsort().float()


def load(name):
    stats = torch.load(f"{save_dir}/{name}_stats.pt")
    states = (raw.float() - stats["mean"]) / stats["std"]
    kae = train.JordanKoopmanAutoencoder(
        256,
        shared_modes=True,
        real_start=bool(stats.get("real_start", False)),
        in_features=states.shape[-1],
        latent_features=2049,
        hidden_config=stats["hidden_config"],
        bias=True,
        batchnorm=False,
        nonlinearity="leaky_relu",
    )
    kae.load_state_dict(load_file(f"{save_dir}/{name}.safetensors"))
    kae.eval()
    z6 = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])[:, 0]) for first in range(0, len(states), 5000)])
    lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
    modes = torch.tensor([m for m in ((z6 != 0).sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0].tolist() if m > 0])
    print(f"{name}: {len(modes)} modes kept on at least {MIN_IMAGES} validation images", flush=True)

    return {"z": z6[:, modes], "size": lam.abs()[modes], "angle": torch.rad2deg(torch.angle(lam))[modes], "usage": (z6[:, modes] != 0).float().mean(dim=0), "modes": modes}


def correlate(a, b, held=()):
    a, b = rank(a), rank(b)
    if held:
        design = torch.stack([torch.ones(len(a)), *[rank(h) for h in held]], dim=1)
        a = a - design @ torch.linalg.lstsq(design, a[:, None]).solution[:, 0]
        b = b - design @ torch.linalg.lstsq(design, b[:, None]).solution[:, 0]
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def get_phase_tie(i, j):
    """How fixed the phase relation of mode i (run A) and mode j (run B) is, over the images that keep both.

    Returns (offset relation, mirror relation, shared images); each relation is in 0..1.
    """

    both = (a["z"][:, i] != 0) & (b["z"][:, j] != 0)
    if both.sum() < MIN_SHARED:
        return float("nan"), float("nan"), int(both.sum())
    unit_a, unit_b = a["z"][both, i] / a["z"][both, i].abs(), b["z"][both, j] / b["z"][both, j].abs()
    return (unit_a * unit_b.conj()).mean().abs().item(), (unit_a * unit_b).mean().abs().item(), int(both.sum())


a, b = load(names[0]), load(names[1])
standardize = lambda sizes: (sizes - sizes.mean(dim=0)) / sizes.std(dim=0)
# shape: [modes of A, modes of B]; correlation over images of the two modes' sizes at block 6
alike = standardize(a["z"].abs()).T @ standardize(b["z"].abs()) / len(raw)
rows, columns = linear_sum_assignment(-alike.numpy())
rows, columns = torch.tensor(rows), torch.tensor(columns)
match = alike[rows, columns]
mutual = (alike.argmax(dim=1)[rows] == columns) & (alike.argmax(dim=0)[columns] == rows)
quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
show = lambda values: " / ".join(f"{v:.3f}" for v in values.quantile(quantiles).tolist())
print(f"\n== {names[0]} against {names[1]}: {len(rows)} one-to-one pairs")
print(f"how alike matched modes fire (correlation of sizes over images), 10/25/50/75/90%: {show(match)}")
print(f"  for comparison, a mode against a random mode of the other run: {show(alike.flatten()[torch.randperm(alike.numel())[:20000]])}")
print(f"  pairs above 0.3: {(match > 0.3).float().mean():.1%}; above 0.5: {(match > 0.5).float().mean():.1%}; above 0.7: {(match > 0.7).float().mean():.1%}; mutual best pairs: {int(mutual.sum())} ({mutual.float().mean():.1%})")

size_a, size_b = a["size"][rows], b["size"][columns]
angle_a, angle_b = a["angle"][rows], b["angle"][columns]
usage_a, usage_b = a["usage"][rows], b["usage"][columns]
print("\nagreement of matched modes across the two runs (rank correlation; in brackets with the other properties of both runs held fixed)")
print("pairs | count | usage | size of λ (usage fixed) | |angle of λ| (size and usage fixed) | |angle| over shuffled pairs")
for label, pick in [("all one-to-one pairs", torch.ones(len(rows), dtype=torch.bool)), ("mutual best", mutual), ("mutual best, alike > 0.5", mutual & (match > 0.5)), ("mutual best, alike > 0.7", mutual & (match > 0.7)), ("not mutual best", ~mutual)]:
    if pick.sum() < 10:
        continue
    sa, sb, aa, ab, ua, ub = size_a[pick], size_b[pick], angle_a[pick].abs(), angle_b[pick].abs(), usage_a[pick], usage_b[pick]
    shuffled = torch.tensor([correlate(aa, ab[torch.randperm(len(ab))]) for _ in range(200)])
    print(
        f"  {label}: {int(pick.sum())} | {correlate(ua, ub):+.3f} | {correlate(sa, sb):+.3f} ({correlate(sa, sb, (ua, ub)):+.3f}) | "
        f"{correlate(aa, ab):+.3f} ({correlate(aa, ab, (sa, sb, ua, ub)):+.3f}) | {shuffled.mean():+.3f} ± {shuffled.std():.3f}"
    )
good = mutual & (match > 0.5)
if good.sum() >= 10:
    gap = (angle_a[good].abs() - angle_b[good].abs()).abs()
    print(f"mutual best pairs with alike > 0.5: |difference of the two |angles||, 25/50/75%: " + " / ".join(f"{v:.1f}°" for v in gap.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()))
    spread = (angle_a[good].abs() - angle_a[good].abs()[torch.randperm(int(good.sum()))]).abs()
    print(f"  the same between random modes of run A: " + " / ".join(f"{v:.1f}°" for v in spread.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()))
    ratio = (size_a[good] / size_b[good]).log().abs().exp()
    print(f"  ratio of the two sizes of λ (larger over smaller), 25/50/75%: " + " / ".join(f"{v:.2f}" for v in ratio.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()))

print("\nphase: is the phase in run A the phase in run B plus a fixed offset (or its mirror image plus an offset)?")
ties = torch.tensor([get_phase_tie(int(i), int(j)) for i, j in zip(rows, columns)])
valid = ties[:, 0] == ties[:, 0]
best = ties[:, :2].max(dim=1).values
random_pairs = torch.tensor([get_phase_tie(int(i), int(j)) for i, j in zip(rows[torch.randperm(len(rows))], columns)])
random_valid = random_pairs[:, 0] == random_pairs[:, 0]
print(f"  matched pairs with at least {MIN_SHARED} shared images: {int(valid.sum())}; shared images (median) {ties[valid, 2].median():.0f}")
print(f"  strength of the better relation, 10/25/50/75/90%: {show(best[valid])}")
print(f"  the same for random pairs ({int(random_valid.sum())} with enough shared images): {show(random_pairs[random_valid, :2].max(dim=1).values)}")
for label, pick in [("mutual best", valid & mutual), ("mutual best, alike > 0.5", valid & mutual & (match > 0.5)), ("mutual best, alike > 0.7", valid & mutual & (match > 0.7))]:
    if pick.sum() >= 10:
        print(f"  {label} ({int(pick.sum())} pairs): {show(best[pick])}; above 0.5: {(best[pick] > 0.5).float().mean():.1%}; above 0.8: {(best[pick] > 0.8).float().mean():.1%}")
print(f"  rank correlation of the relation's strength with how alike the pair fires: {correlate(best[valid], match[valid]):+.3f}")
clear = valid & (best > 0.5) & (angle_a.abs() > 3) & (angle_b.abs() > 3)
is_offset = ties[:, 0] > ties[:, 1]
same_sign = angle_a.sign() == angle_b.sign()
if clear.sum() >= 10:
    print(f"orientation: pairs with a clear relation (strength > 0.5) and both |angles| above 3°: {int(clear.sum())}; of these, offset relation {int((clear & is_offset).sum())}, mirror relation {int((clear & ~is_offset).sum())}")
    print(f"  the eigenvalue angles have the same sign when the relation is an offset, opposite signs when it is a mirror image: {(same_sign == is_offset)[clear].float().mean():.1%} of pairs (chance 50%)")
torch.save({"rows": a["modes"][rows], "columns": b["modes"][columns], "alike": match, "mutual": mutual, "ties": ties, "size": torch.stack([size_a, size_b]), "angle": torch.stack([angle_a, angle_b]), "usage": torch.stack([usage_a, usage_b])}, f"seed_match_{names[0]}_{names[1]}.pt")
print("examples (mutual best, most alike): mode A | mode B | alike | λ A: size, angle | λ B: size, angle | phase relation: offset, mirror")
for index in (match * mutual).argsort(descending=True)[:12].tolist():
    print(f"  {int(a['modes'][rows[index]]):5d} | {int(b['modes'][columns[index]]):5d} | {match[index]:.2f} | {size_a[index]:.2f}, {angle_a[index]:+6.1f}° | {size_b[index]:.2f}, {angle_b[index]:+6.1f}° | {ties[index, 0]:.2f}, {ties[index, 1]:.2f}")
print("done")
