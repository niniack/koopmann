import json

path = "/home/nsa325/work/koopmann/second_analysis/20_steering.ipynb"
nb = json.load(open(path))


def get(i):
    return "".join(nb["cells"][i]["source"])


def put(i, text):
    nb["cells"][i]["source"] = text.splitlines(keepends=True)


def swap(i, old, new):
    text = get(i)
    assert text.count(old) == 1, old
    put(i, text.replace(old, new))


swap(10, '''kae = load_model("b6_novc", 0, False)
operator = kae.koopman_weights.T.double().contiguous()
bundles["dense K ICA"] = get_rotated_features(
    kae.encode(train_x[:, 0]).double() @ operator,
    kae.encode(val_x[:, 0]).double() @ operator,
    lambda x: kae.decode(x.float()),
    "ica",
)
''', '''

def get_output_ica(name, top_k, jordan):
    """ICA on the operator's outputs. K still does the prediction, and ICA reads features from it."""

    kae = load_model(name, top_k, jordan)

    def forward(x):
        return kae.koopman_forward(kae.encode(x), 1).double()

    return get_rotated_features(forward(train_x[:, 0]), forward(val_x[:, 0]), lambda x: kae.decode(x.float()), "ica")


for name, config in MODELS.items():
    bundles[f"{name} ICA"] = get_output_ica(name, **config)
''')

put(11, """##### Steering

Most images barely change under a sparse edit, so averaging over all images hides side effects. Everything here is measured on the images an edit actually changes.

- `gap closed`: how far black-and-white images move toward colour images on CLIP's black-and-white score, in predicted block 12. 1 means the gap is gone.
- `size ratio`: how much bigger the edit is on black-and-white images than on colour images.
- `other, b&w`: how much 19 other concepts move on black-and-white images, relative to their normal variation within a class.
- `other, colour`: the same, on the 10% of colour images the edit changes most.
- `flips`: the percentage of predictions that change, on those same images.
""")

text = get(12)
start = text.index("def measure(base, edited):")
end = text.index("true_scores =")
put(12, text[:start] + '''def measure(base, edited):
    """How much of the black-and-white gap closes, and what else moves on the images that change."""

    base_embeddings, edited_embeddings = to_embeddings(base), to_embeddings(edited)
    base_scores, edited_scores = base_embeddings @ grey_direction, edited_embeddings @ grey_direction
    gap = base_scores[val_grey].mean() - base_scores[~val_grey].mean()
    closed = (base_scores[val_grey].mean() - edited_scores[val_grey].mean()) / gap

    base_panel = base_embeddings @ panel_directions
    change = (edited_embeddings @ panel_directions - base_panel).pow(2)
    variance = remove_class_means(base_panel, val_y).pow(2).mean(dim=0)

    # The colour images this edit changes most
    size = (edited - base).norm(dim=1) / base.norm(dim=1)
    colour = (~val_grey).nonzero()[:, 0]
    touched = colour[size[colour].argsort(descending=True)[: len(colour) // 10]]

    def predict(states):
        return (F.pad(states.double(), (0, 1), value=1) @ classifier).argmax(dim=1)

    flips = predict(base) != predict(edited)

    def accuracy(mask):
        return (predict(edited)[mask] == val_y[mask]).double().mean().item()

    return {
        "gap closed": closed.item(),
        "size ratio": (size[val_grey].mean() / size[~val_grey].mean().clamp_min(1e-12)).item(),
        "other, b&w": (change[val_grey].mean(dim=0) / variance).median().item(),
        "other, colour": (change[touched].mean(dim=0) / variance).median().item(),
        "flips, b&w": 100 * flips[val_grey].double().mean().item(),
        "flips, colour": 100 * flips[touched].double().mean().item(),
        "accuracy, b&w": accuracy(val_grey),
        "accuracy, colour": accuracy(~val_grey),
    }


''' + text[end:])

swap(12, '''for name in ["b6_jordan", "b6_jordan_s1", "b6_jordan128", "b6_jordan128_s1", "b6_novc", "b6_novc_s1", "b6_k128", "linear ICA", "dense K ICA"]:
''', '''for name in [*MODELS, *[f"{name} ICA" for name in MODELS], "linear ICA"]:
''')

control = {
    "cell_type": "code",
    "execution_count": None,
    "id": "control-other-features",
    "metadata": {},
    "outputs": [],
    "source": '''# Control: does the chosen feature stand out from the model's other 19?
rows = {}
for name in [*MODELS, *[f"{name} ICA" for name in MODELS], "linear ICA"]:
    _, _, edit, base, labels = bundles[name]
    scores = pd.DataFrame([measure(base, edit(i)) for i in range(NUM_FEATURES)])
    others = scores.drop(index=best[name])
    rows[name] = {
        "gap closed": scores["gap closed"][best[name]],
        "next best feature": others["gap closed"].max(),
        "median feature": others["gap closed"].median(),
        "other, b&w": scores["other, b&w"][best[name]],
        "median other, b&w": others["other, b&w"].median(),
        "other, colour": scores["other, colour"][best[name]],
        "median other, colour": others["other, colour"].median(),
    }

pd.DataFrame(rows).T.round(3)
'''.splitlines(keepends=True),
}
nb["cells"].insert(13, control)
json.dump(nb, open(path, "w"), indent=1)
