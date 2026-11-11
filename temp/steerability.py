"""
How many modes of a CLIP KAE are "steerable", in the sense of Joseph et al. (arXiv 2504.08729)?

Their test, on SAE features of the same model (CLIP ViT-B/32): replace one feature by a large value on every image of a
set, read CLIP's softmax over a vocabulary of text concepts before (P) and after (P~), and score
    S = sum over the vocabulary of the mean over images of (P~ - P)^2.
A feature is called steerable if S > 0.10. They find about 11% of 12,000 sampled SAE features at layer 11.

Ours, and what differs from theirs:
  - the unit is a mode of the KAE at block 6 (a pair of numbers: a size and a direction), not an SAE feature at layer 11;
    the edit passes through K and the decoder, and CLIP's last layers are read on the KAE's prediction of block 12;
  - a mode has a direction, so each mode is pushed in six directions (0°, 60°, ...; a real-start run has two: + and -),
    and a mode's score is that of its best direction;
  - strength: a multiple (2, 4, 8) of the mode's own large value (the 99th percentile of its size among the images that
    keep it), not their absolute value of 150;
  - vocabulary: CLIP's zero-shot prompts of ImageNet's 1,000 classes, and a list of 19,404 words (theirs: 5,000 concepts);
  - images: 200 random validation images (theirs: not named in the paper);
  - the outer |.| of their formula is read as the sum over the vocabulary.
Also reported, for the "code book" reading of a mode's plane: do different directions of one mode steer to different classes?

Usage: steerability.py <run name> <real_start: 0 or 1> [<take every n-th mode>]
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
every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 6)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
ENCODER, MIN_IMAGES, IMAGES, GAINS, THRESHOLD = "ViT-B-32", 200, 200, [2, 4, 8], 0.10
DIRECTIONS = [0, 180] if real_start else [0, 60, 120, 180, 240, 300]
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
head = train.ClipHead(ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=False)
class_names = open_clip.IMAGENET_CLASSNAMES
words = torch.load("clip_vocab.pt", weights_only=False)
# shape: [512, concepts]; the two vocabularies as CLIP text embeddings of unit length
vocabularies = {"classes": torch.load("zero_shot_head.pt").float().cpu(), "words": words["embeddings"].float().T.contiguous()}
concept_names = {"classes": class_names, "words": words["words"]}


def to_complex(latent):
    return torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)


def get_probs(latent, vocabulary):
    return (head.scale * head.embed(kae.decode(latent)) @ vocabularies[vocabulary]).softmax(dim=-1)


ROWS = [slice(first, first + 5000) for first in range(0, len(states), 5000)]
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[rows])[:, 0]) for rows in ROWS])
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
kept = z6 != 0
modes = [m for m in (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0].tolist() if m > 0][::every]
print(f"{len(modes)} modes kept on at least {MIN_IMAGES} validation images (every {every})", flush=True)
picked = torch.randperm(len(states))[:IMAGES]
predicted = lam * z6[picked]
# shape: [images, 2049]; K's prediction in the KAE's real coordinates
latent = torch.cat([predicted[:, :1].real, torch.stack([predicted[:, 1:].real, predicted[:, 1:].imag], dim=-1).flatten(-2)], dim=-1)
before = {key: get_probs(latent, key) for key in vocabularies}
turns = torch.polar(torch.ones(len(DIRECTIONS)), torch.deg2rad(torch.tensor(DIRECTIONS, dtype=torch.float)))

table = {key: [] for key in ["mode", "direction", "gain", "score classes", "score words", "top class", "top word", "top class gain"]}
for number, m in enumerate(modes):
    large = z6[kept[:, m], m].abs().quantile(0.99)
    # shape: [directions * gains]; the block-6 value the mode is replaced by, then K's step
    values = lam[m] * large * (turns[:, None] * torch.tensor(GAINS)[None]).flatten()
    edited = latent.repeat(len(values), 1)
    edited[:, 2 * m - 1] = values.real.repeat_interleave(IMAGES)
    edited[:, 2 * m] = values.imag.repeat_interleave(IMAGES)
    state = kae.decode(edited)
    embedding = head.embed(state)
    scores, tops, top_gains = {}, {}, {}
    for key in vocabularies:
        # shape: [directions * gains, images, concepts]
        change = (head.scale * embedding @ vocabularies[key]).softmax(dim=-1).view(len(values), IMAGES, -1) - before[key]
        scores[key] = change.pow(2).mean(dim=1).sum(dim=-1)
        top_gains[key], tops[key] = change.mean(dim=1).max(dim=-1)
    for position in range(len(values)):
        table["mode"].append(m)
        table["direction"].append(DIRECTIONS[position // len(GAINS)])
        table["gain"].append(GAINS[position % len(GAINS)])
        table["score classes"].append(scores["classes"][position].item())
        table["score words"].append(scores["words"][position].item())
        table["top class"].append(tops["classes"][position].item())
        table["top word"].append(tops["words"][position].item())
        table["top class gain"].append(top_gains["classes"][position].item())
    if number % 50 == 0:
        best = scores["classes"].view(len(DIRECTIONS), len(GAINS))[:, -1].argmax().item()
        index = best * len(GAINS) + len(GAINS) - 1
        print(f"  {number}: mode {m}, best direction {DIRECTIONS[best]}° at x{GAINS[-1]}: score {scores['classes'][index]:.3f}, to {class_names[tops['classes'][index]]} / {words['words'][tops['words'][index]]}", flush=True)

table = {key: torch.tensor(value) for key, value in table.items()}
torch.save({"table": table, "modes": modes, "size": lam.abs(), "angle": torch.rad2deg(torch.angle(lam))}, f"steerability_{name}.pt")
shape = (len(modes), len(DIRECTIONS), len(GAINS))
print(f"\n== {name}: {len(modes)} modes, {len(DIRECTIONS)} directions, {IMAGES} images; steerable means score > {THRESHOLD}")
for key in ["classes", "words"]:
    score = table[f"score {key}"].view(shape)
    print(f"vocabulary: {key} ({vocabularies[key].shape[1]} concepts)")
    for g, gain in enumerate(GAINS):
        best = score[:, :, g].max(dim=1).values
        quartiles = " / ".join(f"{v:.3f}" for v in best.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist())
        print(
            f"  strength x{gain}: steerable in its best direction {(best > THRESHOLD).float().mean():.1%} of modes; score > 0.5: {(best > 0.5).float().mean():.1%}; "
            f"best-direction score 25/50/75%: {quartiles}; steerable (mode, direction) pairs {(score[:, :, g] > THRESHOLD).float().mean():.1%}"
        )
score, top = table["score classes"].view(shape), table["top class"].view(shape)
print("code book: among modes steerable (classes) in two or more directions, do the directions steer to different classes?")
for g, gain in enumerate(GAINS):
    good = score[:, :, g] > THRESHOLD
    several = good.sum(dim=1) >= 2
    distinct = torch.tensor([len(set(top[i, good[i], g].tolist())) for i in range(len(modes)) if several[i]]).float()
    if len(distinct) == 0:
        print(f"  strength x{gain}: no mode with two steerable directions")
        continue
    count = good.sum(dim=1)[several].float()
    print(
        f"  strength x{gain}: {int(several.sum())} modes; steerable directions per mode {count.mean():.1f}; distinct top classes per mode {distinct.mean():.1f}; "
        f"all directions to one class {(distinct == 1).float().mean():.1%}; every direction to its own class {(distinct == count).float().mean():.1%}"
    )
rank = lambda a: a.argsort().argsort().float()
correlate = lambda a, b: torch.corrcoef(torch.stack([rank(a), rank(b)]))[0, 1].item()
best = score[:, :, -1].max(dim=1).values
size, angle = lam.abs()[modes], torch.rad2deg(torch.angle(lam))[modes].abs()
print(f"rank correlation of a mode's best score at x{GAINS[-1]} with size of λ {correlate(best, size):.3f}, with |angle| {correlate(best, angle):.3f}")
print("examples: the 12 most steerable modes at the largest strength, and where each direction steers")
for i in best.argsort(descending=True)[:12].tolist():
    described = "; ".join(f"{d}° {class_names[top[i, j, -1]][:18]} ({score[i, j, -1]:.2f})" for j, d in enumerate(DIRECTIONS))
    print(f"  mode {modes[i]} (λ size {size[i]:.2f}, angle {torch.rad2deg(torch.angle(lam))[modes[i]]:.1f}°): {described}")
print("done")
