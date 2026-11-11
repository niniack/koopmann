"""
Names every mode of a CLIP KAE with words, and names separately the part of it that K keeps and the part that K builds.

A mode's "push" is how much CLIP's final image embedding moves when the mode is taken out of K's prediction,
averaged over images that keep the mode. The push is a direction in CLIP's shared image-text space, so the words
whose text embeddings lie closest to it name the mode (the idea of Discover-then-Name and SpLiCE, applied to a push
instead of a dictionary vector, because our decoder is not linear).

K multiplies a mode's block-6 value z by λ. The prediction λz splits into
  the kept part   Re(λ)·z   (the block-6 value, rescaled)
  the built part  i·Im(λ)·z (a quarter turn away: what blocks 7-12 add that block 6 did not have in this mode)
and each gets its own push and its own words.

Usage: name_modes.py <run name> <real_start: 0 or 1> [<vocabulary file>]
"""

import csv
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
vocabulary = sys.argv[3] if len(sys.argv) > 3 else "clip_vocab.pt"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 6)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
ENCODER, MIN_IMAGES, MOST_IMAGES, TOP_IMAGES = "ViT-B-32", 200, 600, 50
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
classes = torch.load("zero_shot_head.pt").float().cpu()
class_names = open_clip.IMAGENET_CLASSNAMES
words = torch.load(vocabulary)
# NOTE: Text embeddings share a large common part; taking it out leaves what tells words apart (as SpLiCE does)
texts = torch.nn.functional.normalize(words["embeddings"] - words["embeddings"].mean(dim=0), dim=-1)
words = words["words"]


def to_complex(latent):
    return torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)


def embed(latent):
    return head.embed(kae.decode(latent))


ROWS = [slice(first, first + 5000) for first in range(0, len(states), 5000)]
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[rows])[:, 0]) for rows in ROWS])
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
predicted = lam * z6
latent = torch.cat([predicted[:, :1].real, torch.stack([predicted[:, 1:].real, predicted[:, 1:].imag], dim=-1).flatten(-2)], dim=-1)
base = torch.cat([embed(latent[rows]) for rows in ROWS])
real = torch.cat([head.embed(states[rows, 1]) for rows in ROWS])
real_centre = real.mean(dim=0)
kept = z6 != 0
modes = [m for m in (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0].tolist() if m > 0]
print(f"{len(modes)} modes kept on at least {MIN_IMAGES} validation images", flush=True)


def get_push(rows, m, value):
    """Mean move of the embedding when mode m's predicted value is replaced by `value`. shape: [512]"""

    edited = latent[rows].clone()
    edited[:, 2 * m - 1], edited[:, 2 * m] = value.real, value.imag

    return (base[rows] - embed(edited)).mean(dim=0)


def get_words(direction, count=5):
    return [words[i] for i in (texts @ torch.nn.functional.normalize(direction, dim=0)).topk(count).indices.tolist()]


table, kept_pushes = [], {"mode": [], "kept": [], "built": [], "top images": []}
for number, m in enumerate(modes):
    on = kept[:, m].nonzero()[:, 0]
    rows = on[torch.randperm(len(on))[:MOST_IMAGES]]
    z = z6[rows, m]
    pushes = {
        "mode": get_push(rows, m, torch.zeros_like(z)),
        "kept": get_push(rows, m, 1j * lam[m].imag * z),
        "built": get_push(rows, m, lam[m].real * z),
    }
    # What the images that fire the mode hardest have in common, in CLIP's own embedding: a check that needs no KAE output
    top = on[z6[on, m].abs().topk(min(TOP_IMAGES, len(on))).indices]
    shared = real[top].mean(dim=0) - real_centre
    top_classes = torch.bincount(labels[top], minlength=1000).topk(3)
    cosine = torch.nn.functional.cosine_similarity
    table.append(
        {
            "mode": m,
            "usage": round(len(on) / len(states), 4),
            "size of λ": round(lam[m].abs().item(), 2),
            "angle of λ": round(torch.rad2deg(torch.angle(lam[m])).item(), 1),
            "words": " ".join(get_words(pushes["mode"])),
            "words kept": " ".join(get_words(pushes["kept"])),
            "words built": " ".join(get_words(pushes["built"])),
            "words of top images": " ".join(get_words(shared)),
            "classes of top images": "; ".join(f"{class_names[i]} ({n})" for n, i in zip(top_classes.values.tolist(), top_classes.indices.tolist())),
            "classes pushed": "; ".join(class_names[i] for i in (pushes["mode"] @ classes).topk(3).indices.tolist()),
            "push matches top images": round(cosine(pushes["mode"], shared, dim=0).item(), 3),
            "kept matches top images": round(cosine(pushes["kept"], shared, dim=0).item(), 3),
            "built matches top images": round(cosine(pushes["built"], shared, dim=0).item(), 3),
            "built against kept": round(cosine(pushes["built"], pushes["kept"], dim=0).item(), 3),
            "push size": round(pushes["mode"].norm().item(), 4),
            "kept size": round(pushes["kept"].norm().item(), 4),
            "built size": round(pushes["built"].norm().item(), 4),
            "top words match": len(set(get_words(pushes["mode"], 20)) & set(get_words(shared, 20))),
        }
    )
    if number % 50 == 0:
        print(number, table[-1], flush=True)
    for key in pushes:
        kept_pushes[key].append(pushes[key])
    kept_pushes["top images"].append(shared)

with open(f"names_{name}.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
column = lambda key: torch.tensor([row[key] for row in table])
quartiles = torch.tensor([0.25, 0.5, 0.75])
print("\n25% / median / 75% over modes")
for key in ["push matches top images", "kept matches top images", "built matches top images", "built against kept", "push size", "kept size", "built size", "top words match"]:
    print(f"  {key:26s}", " / ".join(f"{v:.3f}" for v in column(key).float().quantile(quartiles).tolist()))
kept_pushes = {key: torch.stack(value) for key, value in kept_pushes.items()}
torch.save({"table": table, "modes": modes, **kept_pushes}, f"names_{name}.pt")
# Chance level for the match: a mode's push against the top images of the other modes
unit = lambda x: torch.nn.functional.normalize(x, dim=-1)
print("\na mode's push against the top images of its own mode and of the other modes (cosine), and how often its own mode is the best match")
for key in ["mode", "kept", "built"]:
    match = unit(kept_pushes[key]) @ unit(kept_pushes["top images"]).T
    own, others = match.diag(), match[~torch.eye(len(match), dtype=torch.bool)]
    rank = (match > own[:, None]).sum(dim=1)
    print(f"  {key:5s} own median {own.median():.3f}, others median {others.median():.3f}, others 95% {others.quantile(0.95):.3f}; own mode is the best match for {(rank == 0).float().mean():.1%}, in the best 5 for {(rank < 5).float().mean():.1%} of {len(match)} modes")
print("done")
