"""
The dial test with gain: can turning a mode, and turning it up, change the KAE's answer to the family the turn points at?

dial_test.py found that a turn of one mode moves the predicted embedding away from the source family and toward the
destination family, but the answer hardly changes: one mode of the roughly 128 an image keeps is turned, the rest still
describe the image. Here the turned mode is also made stronger (gain 1, 2, 4, 8), and the same gains without a turn are
the control, so the turn and the gain can be told apart.

Families as in dial_test.py: the six 60° slices of a mode's block-6 phase, each slice's strongest images. A family's
"classes" are the 3 most common labels among its strongest 30 images. Read through DINOv3's class probe on the KAE's
prediction of block 12:
  - share of the source images whose answer is one of the destination family's classes (before, after each edit)
  - share whose answer is still one of the source family's classes
  - ceiling: share of the destination family's own images whose answer is one of its classes (unedited prediction)

DINOv3 run im_dino_w21_wd, ImageNet validation. Usage: dial_gain.py <take every n-th mode>
"""

import os
import sys
from collections import Counter

import torch

import inversion
from inversion import to_complex, to_latent, train

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
SLICES, MIN_IN_SLICE, MOST_IN_SLICE, MIN_FAMILIES, GAINS = 6, 30, 60, 3, [1, 2, 4, 8]
every = int(sys.argv[1])
kae, stats = inversion.load_kae()
states, labels = train.get_states(inversion.ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])[:, 0]) for first in range(0, len(states), 5000)])
kept = z6 != 0
usage = kept.float().mean(dim=0)
modes = ((kept.sum(dim=0) >= 200) & (usage < 0.99)).nonzero()[:, 0].tolist()[::every]
print(f"{len(modes)} feature modes tested", flush=True)


def get_answers(latent):
    """The class DINOv3's probe reads from the KAE's prediction, and the entropy of its answer."""

    features = head.features(kae.decode(to_latent(latent)))
    log_probs = (features @ head.classes + head.bias).log_softmax(dim=-1)
    return log_probs.argmax(dim=1), -(log_probs.exp() * log_probs).sum(dim=1).mean().item()


# One row per (mode, source family, destination family, gain)
rows = {key: [] for key in ["mode", "step", "gain", "destination before", "destination turned", "destination control", "source before", "source turned", "source control", "ceiling", "entropy before", "entropy turned", "entropy control"]}
for count, m in enumerate(modes):
    members = kept[:, m].nonzero()[:, 0]
    size, phase = z6[members, m].abs(), torch.rad2deg(torch.angle(z6[members, m])) % 360
    slice_of = ((phase + 180 / SLICES) % 360 / (360 / SLICES)).long()
    families = {}
    for s in range(SLICES):
        family = members[slice_of == s][size[slice_of == s].argsort(descending=True)[:MOST_IN_SLICE]]
        if len(family) >= MIN_IN_SLICE:
            families[s] = family
    if len(families) < MIN_FAMILIES:
        continue
    classes = {s: torch.tensor([c for c, _ in Counter(labels[families[s][:30]].tolist()).most_common(3)]) for s in families}
    latents = {s: lam * z6[families[s]] for s in families}
    base = {s: get_answers(latents[s]) for s in families}
    share = lambda answers, wanted: torch.isin(answers, wanted).float().mean().item()
    for s in families:
        controls = {}
        for gain in GAINS[1:]:
            edited = latents[s].clone()
            edited[:, m] = gain * latents[s][:, m]
            controls[gain] = get_answers(edited)
        controls[1] = base[s]
        for d in families:
            if d == s:
                continue
            step = (d - s) % SLICES
            for gain in GAINS:
                edited = latents[s].clone()
                edited[:, m] = gain * inversion.turn(step * 360 / SLICES) * latents[s][:, m]
                answers, entropy = get_answers(edited)
                rows["mode"].append(m)
                rows["step"].append(step)
                rows["gain"].append(gain)
                rows["destination before"].append(share(base[s][0], classes[d]))
                rows["destination turned"].append(share(answers, classes[d]))
                rows["destination control"].append(share(controls[gain][0], classes[d]))
                rows["source before"].append(share(base[s][0], classes[s]))
                rows["source turned"].append(share(answers, classes[s]))
                rows["source control"].append(share(controls[gain][0], classes[s]))
                rows["ceiling"].append(share(base[d][0], classes[d]))
                rows["entropy before"].append(base[s][1])
                rows["entropy turned"].append(entropy)
                rows["entropy control"].append(controls[gain][1])
    if count % 20 == 0:
        print(f"  {count} modes done", flush=True)

rows = {key: torch.tensor(value, dtype=torch.float) for key, value in rows.items()}
torch.save({"rows": rows, "size": lam.abs(), "angle": torch.rad2deg(torch.angle(lam))}, "dial_gain.pt")
print(f"\n== {len(rows['mode'].unique())} modes; {int((rows['gain'] == 1).sum())} (mode, source family, destination family) cases")
print("share of the source images whose answer is one of the DESTINATION family's classes; and one of the SOURCE family's classes")
print(f"  before any edit: destination {rows['destination before'].mean():.3f} | source {rows['source before'].mean():.3f} | ceiling (the destination family's own images) {rows['ceiling'].mean():.3f}")
for gain in GAINS:
    pick = rows["gain"] == gain
    print(
        f"  gain x{gain}: turned: destination {rows['destination turned'][pick].mean():.3f}, source {rows['source turned'][pick].mean():.3f}, entropy {rows['entropy turned'][pick].mean():.2f}"
        f" | same gain, no turn: destination {rows['destination control'][pick].mean():.3f}, source {rows['source control'][pick].mean():.3f}, entropy {rows['entropy control'][pick].mean():.2f}"
    )
print(f"  entropy before: {rows['entropy before'].mean():.2f}")
print("by turn step, share of answers in the destination's classes after the turn (gain x1 | x2 | x4 | x8), and the ceiling")
for step in range(1, SLICES):
    values = [rows["destination turned"][(rows["step"] == step) & (rows["gain"] == gain)].mean().item() for gain in GAINS]
    print(f"  turn by {step * 360 // SLICES:3d}°: " + " | ".join(f"{v:.3f}" for v in values) + f"; ceiling {rows['ceiling'][rows['step'] == step].mean():.3f}")
# Cases where the destination family is concentrated (its own images mostly answer one of its classes)
for floor in [0.3, 0.6]:
    for gain in [4, 8]:
        pick = (rows["gain"] == gain) & (rows["ceiling"] >= floor)
        print(f"destination families with ceiling >= {floor}: {int(pick.sum())} cases at gain x{gain}: destination before {rows['destination before'][pick].mean():.3f}, turned {rows['destination turned'][pick].mean():.3f}, no turn {rows['destination control'][pick].mean():.3f}, ceiling {rows['ceiling'][pick].mean():.3f}")
print("done")
