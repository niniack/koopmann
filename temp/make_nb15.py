import json

source = json.load(open("second_analysis/19_feature_modes.ipynb"))
code = ["".join(c["source"]) for c in source["cells"] if c["cell_type"] == "code"]
setup, load, words, modes, methods = code[:5]


def cell(kind, text):
    new = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        new.update({"outputs": [], "execution_count": None})
    return new


intro = """### Can dampening a mode steer images?

Looking inside each class, one of Jordan's modes separates black-and-white photos from colour photos. Black-and-white is easy to measure from pixels, so it is a clean stand-in for a feature like snow.

1. **Detection:** does any single feature pick out black-and-white images, on training images we didn't look at?
2. **Steering:** we dampen that feature in the operator. Does block 12 stop looking black-and-white, while the class stays the same?

A **supervised direction**, the difference between black-and-white and colour means, is the reference. It needs labels. Modes don't. Nothing is trained here.
"""

setup = setup[: setup.index("MODELS = ")] + '''MODELS = {
    "b6_novc": {"top_k": 0, "jordan": False},
    "b6_novc_s1": {"top_k": 0, "jordan": False},
    "b6_k128": {"top_k": 128, "jordan": False},
    "b6_jordan128": {"top_k": 128, "jordan": True},
    "b6_jordan128_s1": {"top_k": 128, "jordan": True},
    "b6_jordan": {"top_k": 0, "jordan": True},
    "b6_jordan_s1": {"top_k": 0, "jordan": True},
}
STRENGTHS = [1.0, 0.75, 0.5, 0.25, 0.0, -0.5]
GREY_THRESHOLD = 0.02
PANEL = [
    "the sky", "grass", "water", "snow", "night", "an indoor room", "the outdoors", "a crowd", "a face", "hands",
    "text", "a close-up", "a landscape", "a white background", "a building", "a vehicle", "an animal", "a machine",
    "metal",
]
'''

# Modes can now be dampened rather than only removed
methods = methods.split("def evaluate(train_features")[0].rstrip() + "\n"
methods = methods.replace('''    def edit(i):
        edited = val_modes.clone()
        edited[:, order[i]] = mean[order[i]]
        return decode(edited * eigenvalues)
''', '''    def edit(i, strength=0.0):
        edited = val_modes.clone()
        edited[:, order[i]] = mean[order[i]] + strength * (val_modes[:, order[i]] - mean[order[i]])
        return decode(edited * eigenvalues)
''')
methods = methods.replace('''    def edit(i):
        return finish(val_out - (val_features[:, i : i + 1] @ mixing[i : i + 1]) @ directions.T)
''', '''    def edit(i, strength=0.0):
        change = (val_features[:, i : i + 1] @ mixing[i : i + 1]) @ directions.T
        return finish(val_out - (1 - strength) * change)
''')
assert "strength=0.0" in methods

grey = '''resize = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])


def get_colour_amount(split):
    """How far pixels are from grey. Black-and-white images are close to zero."""

    images = Imagenette(root, split=split, size="full", transform=resize)
    amounts = []
    for batch, _ in DataLoader(images, batch_size=256, num_workers=8):
        red, green, blue = batch.unbind(dim=1)
        amounts.append(((red - green).abs() + (green - blue).abs()).mean(dim=[1, 2]))
    return torch.cat(amounts)


train_colour, val_colour = get_colour_amount("train"), get_colour_amount("val")
train_grey, val_grey = train_colour < GREY_THRESHOLD, val_colour < GREY_THRESHOLD

print(f"Black-and-white: {train_grey.sum()} of {len(train_grey)} training, {val_grey.sum()} of {len(val_grey)} validation")
pd.DataFrame(
    {"train": [train_grey[train_y == c].sum().item() for c in range(10)], "val": [val_grey[val_y == c].sum().item() for c in range(10)]},
    index=CLASSES,
).T
'''

detect = '''def get_auc(features, is_positive):
    """For each feature, the chance a positive image ranks above a negative one, in either direction."""

    ranks = features.argsort(dim=0).argsort(dim=0).double() + 1
    positives, negatives = is_positive.sum(), (~is_positive).sum()
    auc = (ranks[is_positive].sum(dim=0) - positives * (positives + 1) / 2) / (positives * negatives)
    return torch.maximum(auc, 1 - auc)


linear = fit_linear(train_x[:, 0], train_x[:, 1])
linear_train = F.pad(train_x[:, 0], (0, 1), value=1).double() @ linear
linear_val = F.pad(val_x[:, 0], (0, 1), value=1).double() @ linear

bundles = {name: get_mode_features(name, **config) for name, config in MODELS.items()}
bundles["linear principal"] = get_rotated_features(linear_train, linear_val, lambda x: x, "principal")
bundles["linear ICA"] = get_rotated_features(linear_train, linear_val, lambda x: x, "ica")
for trial in range(NUM_RANDOM):
    bundles[f"linear random {trial}"] = get_rotated_features(linear_train, linear_val, lambda x: x, "random")

kae = load_model("b6_novc", 0, False)
operator = kae.koopman_weights.T.double().contiguous()
bundles["dense K ICA"] = get_rotated_features(
    kae.encode(train_x[:, 0]).double() @ operator,
    kae.encode(val_x[:, 0]).double() @ operator,
    lambda x: kae.decode(x.float()),
    "ica",
)

start = train_x[:, 0].double()
start = start - start.mean(dim=0)
pca_features = start @ get_principal_directions(start)[:, :NUM_FEATURES]

rows = {}
best = {}
for name, bundle in bundles.items():
    auc = get_auc(bundle[0], train_grey)
    best[name] = auc.argmax().item()
    validation = get_auc(bundle[1][:, best[name] : best[name] + 1], val_grey)[0]
    rows[name] = {"feature": bundle[4][best[name]], "train AUC": auc.max().item(), "validation AUC": validation.item()}
rows["PCA of block 6"] = {"feature": "best of 20", "train AUC": get_auc(pca_features, train_grey).max().item(), "validation AUC": float("nan")}

pd.DataFrame(rows).T
'''

steer = '''clip, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    force_quick_gelu=True,
    cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
)
clip = clip.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def encode_text(prompts):
    return F.normalize(clip.encode_text(tokenizer(prompts)), dim=-1)


grey_direction = (encode_text(["a black and white photo"]) - encode_text(["a color photo"]))[0]
panel_directions = (encode_text([f"a photo of {concept}" for concept in PANEL]) - encode_text(["a photo"])).T


def to_embeddings(states):
    raw = states.float() * stats["std"] + stats["mean"]
    return F.normalize(clip.visual.ln_post(raw) @ clip.visual.proj, dim=-1)


def remove_class_means(x, labels):
    x = x.clone()
    for c in range(10):
        x[labels == c] -= x[labels == c].mean(dim=0)
    return x


def measure(base, edited):
    """How much of the black-and-white gap closes, and what else moves."""

    base_scores, edited_scores = to_embeddings(base) @ grey_direction, to_embeddings(edited) @ grey_direction
    gap = base_scores[val_grey].mean() - base_scores[~val_grey].mean()
    closed = (base_scores[val_grey].mean() - edited_scores[val_grey].mean()) / gap

    base_panel, edited_panel = to_embeddings(base) @ panel_directions, to_embeddings(edited) @ panel_directions
    change = remove_class_means(edited_panel - base_panel, val_y).pow(2).mean(dim=0)
    other = (change / remove_class_means(base_panel, val_y).pow(2).mean(dim=0)).median()

    def accuracy(states, mask):
        predictions = (F.pad(states.double(), (0, 1), value=1) @ classifier).argmax(dim=1)
        return (predictions[mask] == val_y[mask]).double().mean().item()

    return {
        "gap closed": closed.item(),
        "other concepts moved": other.item(),
        "accuracy, b&w": accuracy(edited, val_grey),
        "accuracy, colour": accuracy(edited, ~val_grey),
    }


true_scores = to_embeddings(val_x[:, 1]) @ grey_direction
print(f"Gap in true block 12: {true_scores[val_grey].mean() - true_scores[~val_grey].mean():.3f}")

results = {}
for name in ["b6_jordan", "b6_jordan_s1", "b6_jordan128", "b6_jordan128_s1", "b6_novc", "b6_novc_s1", "b6_k128", "linear ICA", "dense K ICA"]:
    _, _, edit, base, labels = bundles[name]
    results[(name, labels[best[name]], "no edit")] = measure(base, base)
    results[(name, labels[best[name]], "removed")] = measure(base, edit(best[name]))

# The supervised reference, with labels for black-and-white
difference = linear_train[train_grey].mean(dim=0) - linear_train[~train_grey].mean(dim=0)
direction = F.normalize(difference, dim=0)
center = linear_train.mean(dim=0)
supervised = linear_val - ((linear_val - center) @ direction)[:, None] * direction
results[("linear", "supervised", "no edit")] = measure(linear_val, linear_val)
results[("linear", "supervised", "removed")] = measure(linear_val, supervised)

pd.DataFrame(results).T.round(3)
'''

dampen = '''rows = {}
for name in ["b6_jordan", "b6_jordan_s1", "b6_jordan128", "b6_jordan128_s1"]:
    _, _, edit, base, labels = bundles[name]
    for strength in STRENGTHS:
        rows[(name, strength)] = measure(base, edit(best[name], strength))

pd.DataFrame(rows).T.round(3)
'''

nb = {
    "cells": [
        cell("markdown", intro),
        cell("code", setup),
        cell("markdown", "##### Load data\n"),
        cell("code", load),
        cell("markdown", "##### Models\n"),
        cell("code", modes),
        cell("code", methods),
        cell("markdown", "##### Which images are black-and-white?\n\nMeasured from pixels alone.\n"),
        cell("code", grey),
        cell("markdown", "##### Detection\n\nThe best single feature of each method, chosen on training images. An AUC of 1 is a perfect detector, and 0.5 is chance.\n"),
        cell("code", detect),
        cell("markdown", "##### Steering\n\n`gap closed` is how far black-and-white images move toward colour images on CLIP's black-and-white score, in predicted block 12. 1 means the gap is gone.\n"),
        cell("code", steer),
        cell("markdown", "##### Dampening\n\nA strength of 1 changes nothing, 0 removes the mode, and a negative strength pushes past the mean.\n"),
        cell("code", dampen),
    ],
    "metadata": source["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open("second_analysis/20_steering.ipynb", "w"), indent=1)
