"""
Do a turn and a rescaling of a mode change different things, when both move the latent by the same distance?

The control is borrowed from "Disentangling Direction and Magnitude in Transformer Representations" (arXiv 2602.11169):
compare the two edits at the same Euclidean displacement. For a mode with predicted value v = λ·z6:
  a turn by θ moves v by |v|·2·sin(θ/2), and a rescaling by (1 ± δ) moves it by |v|·δ, so δ = 2·sin(θ/2) matches them.
  θ = 30° pairs with δ = 0.52, and θ = 60° pairs with δ = 1 (scale down = the mode removed, scale up = the mode doubled).
Each edit is applied to one mode at a time, on images that keep the mode, and read through the decoder and DINOv3's probe.

What is recorded per mode and edit:
  state move      : size of the change of the decoded state, over the size of the state (is the decoder as sensitive to both?)
  answers changed : share of images whose answer changes
  KL              : how far the class probabilities move
  confidence      : change of the probability of the answer the KAE gave before the edit
  entropy         : change of the entropy of the class probabilities
  new answer      : among changed answers, the share that was outside the top 5 before the edit
  right           : change of accuracy against the labels
and, between edits, the cosine of the changes of the class scores (is a turn's effect one that a rescaling could make?).

DINOv3 run im_dino_w21_wd, ImageNet validation. Usage: matched_edits.py
"""

import math
import os

import torch

import inversion
from inversion import to_complex, to_latent, train

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
MIN_IMAGES, MOST_IMAGES = 200, 600
kae, stats = inversion.load_kae()
states, labels = train.get_states(inversion.ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
get_scores = lambda x: head.features(x) @ head.classes + head.bias
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])[:, 0]) for first in range(0, len(states), 5000)])
kept = z6 != 0
usage = kept.float().mean(dim=0)
modes = ((kept.sum(dim=0) >= MIN_IMAGES) & (usage < 0.99)).nonzero()[:, 0].tolist()

EDITS = {}
for degrees in [30, 60]:
    delta = 2 * math.sin(math.radians(degrees) / 2)
    EDITS[f"turn +{degrees}°"] = inversion.turn(degrees)
    EDITS[f"turn -{degrees}°"] = inversion.turn(-degrees)
    EDITS[f"scale x{1 + delta:.2f}"] = complex(1 + delta)
    EDITS[f"scale x{1 - delta:.2f}"] = complex(1 - delta)
NAMES = list(EDITS)
OUTCOMES = ["state move", "answers changed", "KL", "confidence", "entropy", "new answer", "right"]
PAIRS = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3), (4, 5), (6, 7), (4, 6), (4, 7), (5, 6), (5, 7)]
print(f"{len(modes)} feature modes; edits: {NAMES}", flush=True)
centre = lambda scores: scores - scores.mean(dim=-1, keepdim=True)
entropy = lambda log_probs: -(log_probs.exp() * log_probs).sum(dim=-1)
table, cosines = [], []
for count, m in enumerate(modes):
    on = kept[:, m].nonzero()[:, 0]
    rows = on[torch.randperm(len(on))[:MOST_IMAGES]]
    latent = lam * z6[rows]
    base_state = kae.decode(to_latent(latent))
    base_scores = get_scores(base_state)
    base = base_scores.log_softmax(dim=-1)
    answer = base.argmax(dim=-1)
    top5 = base.topk(5).indices
    per_edit, moves = [], []
    for factor in EDITS.values():
        edited = latent.clone()
        edited[:, m] = factor * latent[:, m]
        state = kae.decode(to_latent(edited))
        scores = get_scores(state)
        log_probs = scores.log_softmax(dim=-1)
        new = log_probs.argmax(dim=-1)
        changed = new != answer
        outside = (new[:, None] != top5).all(dim=1)
        per_edit.append(
            torch.stack(
                [
                    (state - base_state).norm(dim=-1).mean() / base_state.norm(dim=-1).mean(),
                    changed.float().mean(),
                    (base.exp() * (base - log_probs)).sum(dim=-1).mean(),
                    (log_probs.exp().gather(1, answer[:, None]) - base.exp().gather(1, answer[:, None])).mean(),
                    (entropy(log_probs) - entropy(base)).mean(),
                    outside[changed].float().mean() if changed.any() else torch.tensor(float("nan")),
                    (new == labels[rows]).float().mean() - (answer == labels[rows]).float().mean(),
                ]
            )
        )
        moves.append(centre(scores) - centre(base_scores))
    table.append(torch.stack(per_edit))
    cosines.append(torch.stack([torch.nn.functional.cosine_similarity(moves[a], moves[b], dim=-1).mean() for a, b in PAIRS]))
    if count % 100 == 0:
        print(f"  {count} modes done", flush=True)
# shape: [modes, edits, outcomes] and [modes, pairs]
table, cosines = torch.stack(table), torch.stack(cosines)
index = torch.tensor(modes)
torch.save({"modes": index, "edits": NAMES, "outcomes": OUTCOMES, "table": table, "pairs": PAIRS, "cosines": cosines, "size": lam[index].abs(), "angle": torch.rad2deg(torch.angle(lam[index]))}, "matched_edits.pt")

print("\n== median over modes (each mode edited alone, on the images that keep it)")
print(f"{'edit':14s}" + "".join(f"{name:>17s}" for name in OUTCOMES))
for e, name in enumerate(NAMES):
    print(f"{name:14s}" + "".join(f"{table[:, e, o].nanmedian().item():17.4f}" for o in range(len(OUTCOMES))))
print("\n== the same, summed over modes and images where it is a count: mean over modes")
print(f"{'edit':14s}" + "".join(f"{name:>17s}" for name in OUTCOMES))
for e, name in enumerate(NAMES):
    print(f"{name:14s}" + "".join(f"{table[:, e, o].nanmean().item():17.4f}" for o in range(len(OUTCOMES))))
print("\n== cosine between the changes of the class scores made by two edits, 10 / 50 / 90% of modes")
for p, (a, b) in enumerate(PAIRS):
    print(f"  {NAMES[a]:12s} against {NAMES[b]:12s}:", " / ".join(f"{v:6.3f}" for v in cosines[:, p].quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()))
print("\n== per mode: which edit changes more answers, the turn (mean of + and -) or the rescaling (mean of up and down)?")
for first, degrees in [(0, 30), (4, 60)]:
    turn, scale = table[:, first : first + 2, 1].mean(dim=1), table[:, first + 2 : first + 4, 1].mean(dim=1)
    kl_turn, kl_scale = table[:, first : first + 2, 2].mean(dim=1), table[:, first + 2 : first + 4, 2].mean(dim=1)
    print(f"  {degrees}°: the turn changes more answers in {(turn > scale).float().mean():.0%} of modes; median ratio turn / rescaling {(turn / scale.clamp(min=1e-6)).median():.2f}; KL ratio {(kl_turn / kl_scale.clamp(min=1e-9)).median():.2f}")
    print(f"       confidence: turn {table[:, first : first + 2, 3].mean(dim=1).median():+.4f}, scale up {table[:, first + 2, 3].median():+.4f}, scale down {table[:, first + 3, 3].median():+.4f}; entropy: turn {table[:, first : first + 2, 4].mean(dim=1).median():+.4f}, scale up {table[:, first + 2, 4].median():+.4f}, scale down {table[:, first + 3, 4].median():+.4f}")
print("done")
