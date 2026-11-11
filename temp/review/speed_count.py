"""
Parameters and multiply-adds of real CLIP's image tower and of each KAE, counted by hand from the layer sizes.

A multiply-add is one weight times one number, added to a sum: a linear layer from `a` to `b` numbers costs a x b for
each token it is applied to. Biases, norms, activations and TopK are left out (well under 1% of the total).
The parameter counts are checked against the loaded models.

Also a what-if, BY ARITHMETIC ONLY: the CLS-only KAE with a narrower hidden layer or a narrower control encoder.
NOTE: No model of those sizes was trained. Their accuracy is not measured.
"""

from speed_models import KAES, Kae, count, load_clip, update

WIDTH, TOKENS, PATCHES, LATENT = 768, 50, 49, 2049
DIM, QUERIES = 512, 16  # The control encoder's width and its number of learned queries
SETTINGS = {"im_iso_2x4k": {"state": 3 * WIDTH, "control": 0}, "cls_add_real": {}, "cls_mlp": {"mlp": 4096}}
WHAT_IF = [(4096, 2048), (2048, 2048), (1024, 2048), (4096, 1024), (2048, 1024), (1024, 1024)]  # (hidden, control hidden)


def linear(a, b):
    """Parameters of a linear layer with a bias."""

    return a * b + b


def clip_parts():
    """{part: (parameters, multiply-adds per image)} for CLIP ViT-B/32's image tower."""

    # A block: attention's query, key, value and output layers, two norms, an MLP that is 4 times wider
    block = 4 * linear(WIDTH, WIDTH) + 4 * WIDTH + linear(WIDTH, 4 * WIDTH) + linear(4 * WIDTH, WIDTH)
    # Its arithmetic: the four attention layers on every token, the scores and the weighted sums, then the MLP
    block_work = TOKENS * 4 * WIDTH**2 + 2 * TOKENS**2 * WIDTH + TOKENS * 8 * WIDTH**2
    patch = 3 * 32 * 32 * WIDTH

    return {
        "patch embedding": (patch + WIDTH + TOKENS * WIDTH + 2 * WIDTH, PATCHES * patch),
        "blocks 1-6": (6 * block, 6 * block_work),
        "blocks 7-12": (6 * block, 6 * block_work),
        "head": (2 * WIDTH + WIDTH * 512, WIDTH * 512 + 512 * 1000),
    }


def kae_parts(state=WIDTH, hidden=4096, control=2048, mlp=0):
    """
    {part: (parameters, multiply-adds per image)} for a KAE.
    `control` is the hidden width of the control encoder's MLP (0: no control encoder).
    `mlp` is the hidden width of the MLP that takes the eigenvalues' place in the baseline (0: eigenvalues).
    """

    layers = state * hidden + hidden * hidden + hidden * LATENT
    parts = {"encoder": (layers + 2 * hidden + LATENT, layers), "decoder": (layers + 2 * hidden + state, layers)}
    if mlp:
        parts["MLP in place of eigenvalues"] = (linear(LATENT, mlp) + linear(mlp, LATENT), 2 * LATENT * mlp)
    else:
        parts["eigenvalues"] = (LATENT, 2 * LATENT - 1)
    if control:
        # Norm, embedding, positions, queries, then the query, key and value layers. All but the query layer run on 49 patch tokens
        patch_layers = 2 * WIDTH + linear(WIDTH, DIM) + PATCHES * DIM + QUERIES * DIM + 3 * linear(DIM, DIM)
        patch_work = PATCHES * WIDTH * DIM + 2 * PATCHES * DIM**2 + QUERIES * DIM**2 + 2 * QUERIES * PATCHES * DIM
        parts["control encoder: layers on the patch tokens"] = (patch_layers, patch_work)
        parts["control encoder: its MLP"] = (linear(QUERIES * DIM, control) + linear(control, LATENT), QUERIES * DIM * control + control * LATENT)

    return parts


def total(parts, column):
    return sum(values[column] for values in parts.values())


def check(clip, kaes):
    """The hand counts of parameters against the loaded models."""

    visual = load_clip().visual
    assert count(visual) == total(clip, 0), "CLIP's image tower"
    assert count(visual.transformer.resblocks[6:]) == clip["blocks 7-12"][0], "blocks 7-12"
    for name, parts in kaes.items():
        assert count(Kae(name).model) == total(parts, 0), name


def what_if(clip):
    """The CLS-only KAE at other sizes: its own cost, and the whole model's (patch embedding, blocks 1-6, KAE, head)."""

    rest = [clip[part] for part in ("patch embedding", "blocks 1-6", "head")]
    rows = []
    for hidden, control in WHAT_IF:
        parts = kae_parts(hidden=hidden, control=control)
        parameters, work = total(parts, 0), total(parts, 1)
        rows.append({
            "hidden": hidden, "control_hidden": control, "trained": (hidden, control) == WHAT_IF[0],
            "kae_parameters": parameters, "kae_megabytes": 4 * parameters / 1e6, "kae_multiply_adds": work,
            "whole_parameters": parameters + sum(p for p, _ in rest), "whole_multiply_adds": work + sum(w for _, w in rest),
        })  # fmt: skip

    return rows


def main():
    clip, kaes = clip_parts(), {name: kae_parts(**SETTINGS[name]) for name in KAES}
    check(clip, kaes)
    for name, parts in {"real CLIP": clip, **kaes}.items():
        print(f"{name}: {total(parts, 0):,} parameters ({4 * total(parts, 0) / 1e6:.1f} MB), {total(parts, 1) / 1e6:,.1f} million multiply-adds per image")
        for part, (parameters, work) in parts.items():
            print(f"    {part:<45} {parameters:>12,} parameters {work / 1e6:>9,.2f} million multiply-adds")
    rows = what_if(clip)
    for row in rows:
        print(row)
    update("count", {"clip": clip, "kaes": kaes, "what_if": rows})


if __name__ == "__main__":
    main()
