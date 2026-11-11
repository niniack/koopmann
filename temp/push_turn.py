"""
Does the angle of a mode's eigenvalue say how much the mode changes what it writes between block 6 and block 12?

The decoder is shared by both blocks, so each mode owns a plane of decoder inputs, and K turns the mode's value in that plane.
If the angle has a meaning in state space, it is this: a mode with a small angle writes the same pattern into the state at
block 12 as at block 6 (only larger), and a mode with a large angle writes a different pattern.

A mode's "write" at a block is the change of the decoded state when the mode is taken out of the latent, on images that keep it.
  write at 6       : from the block-6 latent z6
  write at 12 (K)  : from K's prediction λ·z6
  write at 12 (true): from the encoding of the true block-12 state, with the modes picked at block 6
"Same pattern" is the cosine between two writes, image by image, over the whole state and over the CLS part alone.
The same is done after the head: the change of the class scores ("which classes does the mode vote for"), where the
block-6 write is read by the block-12 head, as a logit lens would.
With a linear decoder and a mode whose two decoder directions are orthogonal and equally long, the cosine is cos(angle).

Usage: push_turn.py dino | push_turn.py <CLIP run name> <real_start: 0 or 1>
"""

import os
import sys

import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
torch.manual_seed(0)
MIN_IMAGES, MOST_IMAGES = 200, 600
to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)


def to_latent(z):
    blocks = torch.stack([z[..., 1:].real, z[..., 1:].imag], dim=-1).flatten(-2)
    return torch.cat([z[..., :1].real, blocks], dim=-1)


if sys.argv[1] == "dino":
    import inversion
    from inversion import train

    name = "im_dino_w21_wd"
    kae, stats = inversion.load_kae()
    states, labels = train.get_states(inversion.ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
    head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
    get_scores = lambda x: head.features(x) @ head.classes + head.bias
else:
    sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
    from dotenv import load_dotenv
    from safetensors.torch import load_file

    load_dotenv("/home/nsa325/work/koopmann/.env")
    import train

    name, real_start = sys.argv[1], sys.argv[2] == "1"
    save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae"
    stats = torch.load(f"{save_dir}/{name}_stats.pt")
    states, labels = train.get_states("ViT-B-32", "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
    kae = train.JordanKoopmanAutoencoder(256, shared_modes=True, real_start=real_start, in_features=states.shape[-1], latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(f"{save_dir}/{name}.safetensors"))
    kae.eval()
    # NOTE: Building the zero-shot classifier takes over 30 minutes on a busy CPU, so it is read from a cache
    head = train.ClipHead("ViT-B-32", stats["mean"], stats["std"], "cpu", zero_shot=False)
    head.classes = torch.load("zero_shot_head.pt").float().cpu()
    get_scores = lambda x: head.scale * head.embed(x) @ head.classes

states = (states.float() - stats["mean"]) / stats["std"]
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
z = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])) for first in range(0, len(states), 5000)])
z6, z12 = z[:, 0], z[:, 1]
kept = z6 != 0
usage = kept.float().mean(dim=0)
modes = ((kept.sum(dim=0) >= MIN_IMAGES) & (usage < 0.99)).nonzero()[:, 0].tolist()
print(f"== {name}: {len(modes)} feature modes", flush=True)
cosine = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
centre = lambda scores: scores - scores.mean(dim=-1, keepdim=True)
KEYS = ["state: 6 against 12 (K)", "state: 6 against 12 (true)", "state: 12 (K) against 12 (true)", "CLS: 6 against 12 (K)", "CLS: 6 against 12 (true)", "votes: 6 against 12 (K)", "votes: 6 against 12 (true)", "votes: 12 (K) against 12 (true)", "mean write: 6 against 12 (K)", "write size 12 (K) over 6", "same top class: 6 and 12 (K)"]
table = []
for count, m in enumerate(modes):
    on = kept[:, m].nonzero()[:, 0]
    rows = on[torch.randperm(len(on))[:MOST_IMAGES]]
    writes, votes = {}, {}
    for key, latent in {"6": z6[rows], "12 (K)": lam * z6[rows], "12 (true)": z12[rows]}.items():
        without = latent.clone()
        without[:, m] = 0
        with_mode, without_mode = kae.decode(to_latent(latent)), kae.decode(to_latent(without))
        writes[key] = with_mode - without_mode
        votes[key] = centre(get_scores(with_mode)) - centre(get_scores(without_mode))
    row = [
        cosine(writes["6"], writes["12 (K)"]).mean(),
        cosine(writes["6"], writes["12 (true)"]).mean(),
        cosine(writes["12 (K)"], writes["12 (true)"]).mean(),
        cosine(writes["6"][:, :768], writes["12 (K)"][:, :768]).mean(),
        cosine(writes["6"][:, :768], writes["12 (true)"][:, :768]).mean(),
        cosine(votes["6"], votes["12 (K)"]).mean(),
        cosine(votes["6"], votes["12 (true)"]).mean(),
        cosine(votes["12 (K)"], votes["12 (true)"]).mean(),
        cosine(writes["6"].mean(dim=0), writes["12 (K)"].mean(dim=0)),
        writes["12 (K)"].norm(dim=-1).mean() / writes["6"].norm(dim=-1).mean(),
        (votes["6"].mean(dim=0).argmax() == votes["12 (K)"].mean(dim=0).argmax()).float(),
    ]
    table.append(torch.stack(row))
    if count % 100 == 0:
        print(f"  {count} modes done", flush=True)
table = torch.stack(table)
index = torch.tensor(modes)
angle, size, use = torch.rad2deg(torch.angle(lam[index])).abs(), lam[index].abs(), usage[index]
torch.save({"modes": index, "angle": angle, "size": size, "usage": use, "columns": KEYS, "table": table}, f"push_turn_{name}.pt")

rank = lambda a: a.argsort().argsort().float()
correlate = lambda a, b: torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()


def partial(a, b, controls):
    """Rank correlation of a and b once the controls are fitted out of both."""

    design = torch.stack([torch.ones(len(a))] + [rank(c) for c in controls], dim=1)
    left = lambda v: rank(v) - design @ torch.linalg.lstsq(design, rank(v)[:, None]).solution[:, 0]
    return correlate(left(a), left(b))


quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
print("\nover modes, 10 / 25 / 50 / 75 / 90%; then the rank correlation with |angle| alone | with |angle|, size and usage fixed | with size, |angle| and usage fixed")
for column, key in enumerate(KEYS):
    values = table[:, column]
    print(f"  {key:34s}", " / ".join(f"{v:6.3f}" for v in values.quantile(quantiles).tolist()), f"   {correlate(rank(angle), rank(values)):6.3f} | {partial(angle, values, [size, use]):6.3f} | {partial(size, values, [angle, use]):6.3f}")
print("\nif the decoder were linear and each mode's plane square, the cosine would be cos(angle):")
print(f"  correlation of cos(angle) with the state cosine (K): {correlate(torch.cos(torch.deg2rad(angle)), table[:, 0]):.3f}; with the votes cosine (K): {correlate(torch.cos(torch.deg2rad(angle)), table[:, 5]):.3f}")
print("\nby |angle of λ|: modes | cos(angle) | state cosine 6 against 12 (K) | (true) | CLS (K) | votes (K) | votes (true) | same top class")
for low, high in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 180)]:
    rows = (angle >= low) & (angle < high)
    if rows.sum() == 0:
        continue
    median = lambda column: table[rows, column].median().item()
    print(f"  {low:3d}-{high:3d}°: {rows.sum().item():4d} | {torch.cos(torch.deg2rad(angle[rows])).median():.3f} | {median(0):.3f} | {median(1):.3f} | {median(3):.3f} | {median(5):.3f} | {median(6):.3f} | {table[rows, 10].mean():.2f}")
print("\nby size of λ: modes | median |angle| | state cosine 6 against 12 (K) | (true) | votes (K) | votes (true)")
for part, rows in enumerate(size.argsort().chunk(5)):
    print(f"  fifth {part + 1} (size {size[rows].min():.2f} to {size[rows].max():.2f}): {len(rows):4d} | {angle[rows].median():5.1f}° | {table[rows, 0].median():.3f} | {table[rows, 1].median():.3f} | {table[rows, 5].median():.3f} | {table[rows, 6].median():.3f}")
print("done")
