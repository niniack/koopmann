"""
The dial test: does turning a mode swap one family of images for another, while rescaling only gives more or less of the same?

Background (phase_wheel.py, ms_score.py): the images that keep a mode differ by the mode's direction in its plane, and images
with the same direction look alike. Cut the circle into six 60° slices; call the images of a slice a "family"
(for mode 750: long-haired dog faces near 0° and 60°, spotted salamanders near 120° and 180°, hay fields near 300°).

The test. Take the images of one family (the source), turn the mode's block-6 value by the angle that leads to another
family (the destination), let K and the decoder predict block 12 and read DINOv3's final embedding of the prediction.
The change of the embedding is compared with where each family sits: the mean embedding of each family's true block-12
states, centred over the families of the mode. If the direction is a dial, the change points at the destination family
more than at any other family that is not the source, and the source family loses. Chance for that is 1 in
(families - 1). The same is read for two rescalings (mode removed, mode doubled), which have no destination.
A turn of a mode's value for every image is the same edit as adding that angle to the mode's eigenvalue, and a rescaling
is the same as changing the eigenvalue's size, so this is also a test of what the two parts of the eigenvalue do as knobs.

Also read through the class probe: the log-probability of the destination family's 3 most common classes, before and after.

DINOv3 run im_dino_w21_wd, ImageNet validation. Usage: dial_test.py
"""

import os
from collections import Counter

import torch

import inversion
from inversion import to_complex, to_latent, train

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
SLICES, MIN_IN_SLICE, MOST_IN_SLICE, MIN_FAMILIES = 6, 30, 120, 3
kae, stats = inversion.load_kae()
states, labels = train.get_states(inversion.ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
get_log_probs = lambda features: (features @ head.classes + head.bias).log_softmax(dim=-1)
normalize = torch.nn.functional.normalize
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])[:, 0]) for first in range(0, len(states), 5000)])
# shape: [images, width], unit length; DINOv3's final embedding of the true block-12 state
true_embed = torch.cat([normalize(head.features(states[first : first + 5000, 1]), dim=-1) for first in range(0, len(states), 5000)])
kept = z6 != 0
usage = kept.float().mean(dim=0)
modes = ((kept.sum(dim=0) >= 200) & (usage < 0.99)).nonzero()[:, 0].tolist()
print(f"{len(modes)} feature modes", flush=True)

# One row per (mode, source family, destination family)
turns = {key: [] for key in ["mode", "step", "families", "wins", "destination", "others", "source", "class destination", "class others", "class source", "answer before", "answer after"]}
# One row per (mode, source family)
scales = {key: [] for key in ["mode", "removed source", "removed others", "doubled source", "doubled others", "removed class source", "doubled class source"]}
for count, m in enumerate(modes):
    rows = kept[:, m].nonzero()[:, 0]
    size, phase = z6[rows, m].abs(), torch.rad2deg(torch.angle(z6[rows, m])) % 360
    slice_of = ((phase + 180 / SLICES) % 360 / (360 / SLICES)).long()
    families = {}
    for s in range(SLICES):
        members = rows[slice_of == s][size[slice_of == s].argsort(descending=True)[:MOST_IN_SLICE]]
        if len(members) >= MIN_IN_SLICE:
            families[s] = members
    if len(families) < MIN_FAMILIES:
        continue
    keys = list(families)
    means = torch.stack([true_embed[families[s]].mean(dim=0) for s in keys])
    # shape: [families, width]; where each family sits, relative to the mode's other families
    centred = normalize(means - means.mean(dim=0), dim=-1)
    top_classes = {s: torch.tensor([c for c, _ in Counter(labels[families[s][:30]].tolist()).most_common(3)]) for s in keys}

    def read(latent):
        features = head.features(kae.decode(to_latent(latent)))
        return normalize(features, dim=-1), get_log_probs(features)

    for position, s in enumerate(keys):
        latent = lam * z6[families[s]]
        base_embed, base_log_probs = read(latent)
        mass = lambda log_probs, classes: log_probs[:, classes].logsumexp(dim=1).mean().item()
        others = [i for i in range(len(keys)) if i != position]

        def edit(factor):
            edited = latent.clone()
            edited[:, m] = factor * latent[:, m]
            embed, log_probs = read(edited)
            # shape: [families]; cosine of the mean change of the embedding with each family's place
            return normalize((embed - base_embed).mean(dim=0), dim=0) @ centred.T, log_probs

        for index in others:
            d = keys[index]
            step = (d - s) % SLICES
            gains, log_probs = edit(inversion.turn(step * 360 / SLICES))
            rest = [i for i in others if i != index]
            turns["mode"].append(m)
            turns["step"].append(step)
            turns["families"].append(len(keys))
            turns["wins"].append(float(gains[index] > gains[rest].max()) if rest else float("nan"))
            turns["destination"].append(gains[index].item())
            turns["others"].append(gains[rest].mean().item() if rest else float("nan"))
            turns["source"].append(gains[position].item())
            turns["class destination"].append(mass(log_probs, top_classes[d]) - mass(base_log_probs, top_classes[d]))
            turns["class others"].append(sum(mass(log_probs, top_classes[keys[i]]) - mass(base_log_probs, top_classes[keys[i]]) for i in rest) / max(len(rest), 1))
            turns["class source"].append(mass(log_probs, top_classes[s]) - mass(base_log_probs, top_classes[s]))
            turns["answer before"].append(torch.isin(base_log_probs.argmax(dim=1), top_classes[d]).float().mean().item())
            turns["answer after"].append(torch.isin(log_probs.argmax(dim=1), top_classes[d]).float().mean().item())
        removed, removed_log_probs = edit(0)
        doubled, doubled_log_probs = edit(2)
        scales["mode"].append(m)
        scales["removed source"].append(removed[position].item())
        scales["removed others"].append(removed[others].mean().item())
        scales["doubled source"].append(doubled[position].item())
        scales["doubled others"].append(doubled[others].mean().item())
        scales["removed class source"].append(mass(removed_log_probs, top_classes[s]) - mass(base_log_probs, top_classes[s]))
        scales["doubled class source"].append(mass(doubled_log_probs, top_classes[s]) - mass(base_log_probs, top_classes[s]))
    if count % 50 == 0:
        print(f"  {count} modes done", flush=True)

turns = {key: torch.tensor(value) for key, value in turns.items()}
scales = {key: torch.tensor(value) for key, value in scales.items()}
torch.save({"turns": turns, "scales": scales, "size": lam.abs(), "angle": torch.rad2deg(torch.angle(lam))}, "dial_test.pt")
tested = turns["mode"].unique()
valid = turns["wins"] == turns["wins"]
chance = (1 / (turns["families"][valid] - 1).float()).mean()
print(f"\n== {len(tested)} modes with at least {MIN_FAMILIES} families; {len(turns['mode'])} turns (mode, source family, destination family)")
print(f"the destination family gains most among the families that are not the source: {turns['wins'][valid].mean():.1%} of turns (chance {chance:.1%})")
per_mode = torch.stack([turns["wins"][valid & (turns["mode"] == m)].mean() for m in tested])
per_chance = torch.stack([(1 / (turns["families"][valid & (turns["mode"] == m)] - 1).float()).mean() for m in tested])
print(f"per mode: share of its turns won, 10 / 50 / 90%: " + " / ".join(f"{v:.2f}" for v in per_mode.nanquantile(torch.tensor([0.1, 0.5, 0.9])).tolist()) + f"; above its own chance level in {(per_mode > per_chance).float().mean():.0%} of modes")
print("cosine of the change of the embedding with a family's place (mean over turns): destination | other families | source")
print(f"  all turns: {turns['destination'].mean():+.3f} | {turns['others'].nanmean():+.3f} | {turns['source'].mean():+.3f}")
for step in range(1, SLICES):
    pick = turns["step"] == step
    print(f"  turn by {step * 360 // SLICES:3d}°: {turns['destination'][pick].mean():+.3f} | {turns['others'][pick].nanmean():+.3f} | {turns['source'][pick].mean():+.3f}; wins {turns['wins'][pick & valid].mean():.1%}; turns {int(pick.sum())}")
print("rescalings, the same cosine: source family | other families")
print(f"  mode removed: {scales['removed source'].mean():+.3f} | {scales['removed others'].mean():+.3f}")
print(f"  mode doubled: {scales['doubled source'].mean():+.3f} | {scales['doubled others'].mean():+.3f}")
print("through the class probe: change of the log-probability of a family's 3 most common classes (mean over turns)")
print(f"  turn: destination {turns['class destination'].mean():+.3f} | other families {turns['class others'].mean():+.3f} | source {turns['class source'].mean():+.3f}")
print(f"  removed: source {scales['removed class source'].mean():+.3f}; doubled: source {scales['doubled class source'].mean():+.3f}")
print(f"  the KAE's answer is one of the destination's 3 classes: before {turns['answer before'].mean():.3f}, after the turn {turns['answer after'].mean():.3f}")
size, angle = lam.abs()[tested], torch.rad2deg(torch.angle(lam))[tested].abs()
rank = lambda a: a.argsort().argsort().float()
correlate = lambda a, b: torch.corrcoef(torch.stack([rank(a), rank(b)]))[0, 1].item()
print(f"rank correlation of a mode's share of turns won with size of λ {correlate(per_mode.nan_to_num(0), size):.3f}, with |angle| {correlate(per_mode.nan_to_num(0), angle):.3f}")
edges = size.quantile(torch.linspace(0, 1, 6))
for i in range(5):
    pick = (size >= edges[i]) & (size <= edges[i + 1])
    print(f"  size of λ {edges[i]:.2f} to {edges[i + 1]:.2f}: {int(pick.sum())} modes, turns won {per_mode[pick].nanmean():.1%} (chance {per_chance[pick].mean():.1%})")
print("done")
