import json
import sys

models = sys.argv[1]
source = json.load(open("second_analysis/18_sparse_modes.ipynb"))
code = ["".join(c["source"]) for c in source["cells"] if c["cell_type"] == "code"]
setup, load, concepts, modes = code[0], code[1], code[2], code[3]


def cell(kind, text):
    new = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        new.update({"outputs": [], "execution_count": None})
    return new


intro = """### Do modes pick out features?

Earlier notebooks scored modes by class. A mode that tracks a feature shared by several classes scores badly on those tests, so here class is removed from everything. Nothing is trained.

1. **Naming:** each feature is correlated with every word in a vocabulary taken from CLIP's tokenizer, among images of the same class. Words are chosen on training images and scored on validation images.
2. **Editing:** we remove a feature, then see how much its word changes in the predicted block 12 compared with a typical word, again within each class.
3. **Looking:** the top and bottom images of a mode inside each class.

**Controls:** random rotations of the same top 20 directions. If modes aren't more nameable than random mixtures, naming means little.
"""

# Setup
start = setup.index("MODELS = ")
setup = setup[:start] + f"MODELS = {models}\n"
setup = setup.replace("from dotenv import load_dotenv\n", "from dotenv import load_dotenv\nfrom sklearn.decomposition import FastICA\nfrom sklearn.feature_extraction.text import ENGLISH_STOP_WORDS\n")
setup = setup.replace("from safetensors.torch import load_file\n", "from safetensors.torch import load_file\nfrom torch.utils.data import DataLoader\n")
setup = setup.replace("NUM_REMOVED = [1, 2, 5, 20]\nNUM_SHOWN = 7\nNUM_IMAGES = 6\nNUM_TOP = 50\nNUM_CANDIDATES = 30\n", "NUM_FEATURES = 20\nNUM_WORDS = 4000\nNUM_NAMED = 5\nNUM_PER_CLASS = 3\nNUM_RANDOM = 3\n")
setup = setup[: setup.index("NEUTRAL = [")] + setup[setup.index("NUM_FEATURES = 20") :]
load = load.replace('val_x, val_y, _ = load_split("val")', 'val_x, val_y, val_final = load_split("val")')

# Keep the model and mode helpers, without the class-based analysis
modes = modes[: modes.index("def get_scores(modes, c):")].rstrip() + "\n"

words = '''clip, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    force_quick_gelu=True,
    cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
)
clip = clip.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Whole words from CLIP's tokenizer, most common first
tokens = sorted(tokenizer.encoder.items(), key=lambda item: item[1])
words = [token[:-4] for token, _ in tokens if token.endswith("</w>")]
words = [word for word in words if word.isascii() and word.isalpha() and len(word) >= 4]
dictionary = {line.strip().lower() for line in open("/usr/share/dict/words")}
words = [word for word in words if word in dictionary and word not in ENGLISH_STOP_WORDS][:NUM_WORDS]
print(len(words), words[:12], words[-6:])

plain = F.normalize(clip.encode_text(tokenizer(["a photo"])), dim=-1)[0]
batches = [words[i : i + 256] for i in range(0, len(words), 256)]
word_directions = torch.cat(
    [F.normalize(clip.encode_text(tokenizer([f"a photo of {word}" for word in batch])), dim=-1) for batch in batches]
)
word_directions = (word_directions - plain).T


def get_word_scores(final_states):
    """How much each word applies to each image, from block 12 states in their raw scale."""

    return F.normalize(clip.visual.ln_post(final_states) @ clip.visual.proj, dim=-1) @ word_directions


def remove_class_means(x, labels):
    x = x.clone()
    for c in range(10):
        x[labels == c] -= x[labels == c].mean(dim=0)
    return x


train_words = F.normalize(remove_class_means(get_word_scores(train_final), train_y), dim=0).double()
val_words = F.normalize(remove_class_means(get_word_scores(val_final), val_y), dim=0).double()
'''

methods = '''def get_principal_directions(x):
    return torch.linalg.svd(x - x.mean(dim=0), full_matrices=False)[2].T


def get_mode_features(name, top_k, jordan):
    """A model's top modes, each as one feature, and how to remove each one."""

    kae = load_model(name, top_k, jordan)
    eigenvalues, encode, decode = (get_jordan_modes if jordan else get_dense_modes)(kae)
    train_modes, val_modes = encode(train_x[:, 0]), encode(val_x[:, 0])

    mean = train_modes.mean(dim=0)
    energy = (train_modes - mean).abs().pow(2).mean(dim=0) * eigenvalues.abs().pow(2)
    order = energy.argsort(descending=True) if jordan else eigenvalues.abs().argsort(descending=True)
    order = order[:NUM_FEATURES]

    train_features, val_features = [], []
    for i in order:
        train_plane = torch.stack([train_modes[:, i].real, train_modes[:, i].imag], dim=1)
        val_plane = torch.stack([val_modes[:, i].real, val_modes[:, i].imag], dim=1)
        center = train_plane.mean(dim=0)
        direction = torch.linalg.svd(train_plane - center, full_matrices=False)[2][0]
        train_features.append((train_plane - center) @ direction)
        val_features.append((val_plane - center) @ direction)

    def edit(i):
        edited = val_modes.clone()
        edited[:, order[i]] = mean[order[i]]
        return decode(edited * eigenvalues)

    labels = [f"mode {i}" for i in order.tolist()]
    return torch.stack(train_features, dim=1), torch.stack(val_features, dim=1), edit, decode(val_modes * eigenvalues), labels


def get_rotated_features(train_out, val_out, finish, kind):
    """Top output directions, either as they are, rotated by ICA, or rotated at random."""

    directions = get_principal_directions(train_out)[:, :NUM_FEATURES]
    mean = train_out.mean(dim=0)
    train_coordinates, val_coordinates = (train_out - mean) @ directions, (val_out - mean) @ directions

    if kind == "ica":
        ica = FastICA(whiten="unit-variance", max_iter=2000, random_state=0).fit(train_coordinates.numpy())
        unmixing = torch.tensor(ica.components_, dtype=torch.double).T
    elif kind == "random":
        rotation = torch.linalg.qr(torch.randn(NUM_FEATURES, NUM_FEATURES, dtype=torch.double))[0]
        unmixing = torch.diag(1 / train_coordinates.std(dim=0)) @ rotation
    else:
        unmixing = torch.eye(NUM_FEATURES, dtype=torch.double)

    mixing = torch.linalg.inv(unmixing)
    train_features, val_features = train_coordinates @ unmixing, val_coordinates @ unmixing

    def edit(i):
        return finish(val_out - (val_features[:, i : i + 1] @ mixing[i : i + 1]) @ directions.T)

    labels = [f"{kind} {i}" for i in range(NUM_FEATURES)]
    return train_features, val_features, edit, finish(val_out), labels


def evaluate(train_features, val_features, edit, base, labels):
    """Names each feature on training images, then scores names and edits on validation images."""

    train_r = F.normalize(remove_class_means(train_features, train_y), dim=0).T @ train_words
    val_r = F.normalize(remove_class_means(val_features, val_y), dim=0).T @ val_words
    named = train_r.abs().topk(NUM_NAMED, dim=1).indices

    # Held-out correlation counts only if it keeps the sign found on training images
    held_out = val_r.gather(1, named) * train_r.gather(1, named).sign()

    def to_scores(states):
        return get_word_scores(states.float() * stats["std"] + stats["mean"])

    base_scores = to_scores(base)
    within = remove_class_means(base_scores, val_y).pow(2).mean(dim=0)
    base_accuracy = get_class_accs(base).mean()

    rows = []
    for i, label in enumerate(labels):
        edited = edit(i)
        change = remove_class_means(to_scores(edited) - base_scores, val_y).pow(2).mean(dim=0) / within
        rows.append(
            {
                "feature": label,
                "words": ", ".join(words[j] for j in named[i].tolist()),
                "held-out r": held_out[i, 0].item(),
                "held-out r, top 5": held_out[i].mean().item(),
                "word change": change[named[i, 0]].item(),
                "typical change": change.median().item(),
                "accuracy drop": (base_accuracy - get_class_accs(edited).mean()).item(),
            }
        )

    return pd.DataFrame(rows)


linear = fit_linear(train_x[:, 0], train_x[:, 1])
linear_train = F.pad(train_x[:, 0], (0, 1), value=1).double() @ linear
linear_val = F.pad(val_x[:, 0], (0, 1), value=1).double() @ linear

tables = {}
features = {}


def add(name, bundle):
    features[name] = bundle
    tables[name] = evaluate(*bundle)


for name, config in MODELS.items():
    add(name, get_mode_features(name, **config))

    kae = load_model(name, **config)
    if not config["jordan"] and not config["top_k"]:
        operator = kae.koopman_weights.T.double().contiguous()
        train_out, val_out = kae.encode(train_x[:, 0]).double() @ operator, kae.encode(val_x[:, 0]).double() @ operator

        def finish(x):
            return kae.decode(x.float())

        add(f"{name} ICA", get_rotated_features(train_out, val_out, finish, "ica"))
        for trial in range(NUM_RANDOM):
            add(f"{name} random {trial}", get_rotated_features(train_out, val_out, finish, "random"))

add("linear principal", get_rotated_features(linear_train, linear_val, lambda x: x, "principal"))
add("linear ICA", get_rotated_features(linear_train, linear_val, lambda x: x, "ica"))
for trial in range(NUM_RANDOM):
    add(f"linear random {trial}", get_rotated_features(linear_train, linear_val, lambda x: x, "random"))
'''

summary = '''def summarize(table):
    return {
        "held-out r": table["held-out r"].mean(),
        "r above 0.3": (table["held-out r"] > 0.3).sum(),
        "distinct top words": table["words"].str.split(", ").str[0].nunique(),
        "word change": table["word change"].median(),
        "typical change": table["typical change"].median(),
        "accuracy drop": table["accuracy drop"].median(),
    }


pd.DataFrame({name: summarize(table) for name, table in tables.items()}).T.round(3)
'''

named = '''for name in MODELS:
    print(name)
    print(tables[name].round(3).to_string())
    print()
'''

images = '''def show_within_classes(name, index):
    """A mode's top and bottom images inside each class."""

    feature = features[name][1][:, index]
    row = tables[name].iloc[index]
    columns = 2 * NUM_PER_CLASS + 1
    fig, axes = plt.subplots(5, 2 * columns, figsize=(2 * columns * 0.95, 5 * 1.05))

    for c, class_name in enumerate(CLASSES):
        members = (val_y == c).nonzero()[:, 0]
        ranked = members[feature[members].argsort(descending=True)].tolist()
        picks = ranked[:NUM_PER_CLASS] + [None] + ranked[-NUM_PER_CLASS:][::-1]
        offset = columns * (c // 5)

        for column, index_in_data in enumerate(picks):
            ax = axes[c % 5, offset + column]
            ax.set_xticks([])
            ax.set_yticks([])
            if index_in_data is None:
                ax.axis("off")
                continue
            ax.imshow(dataset[index_in_data][0])

        axes[c % 5, offset].set_ylabel(class_name, fontsize=7)

    fig.suptitle(f"{name}, {row['feature']} (r {row['held-out r']:.2f}): {row['words']}. Left of each block: high. Right: low.", fontsize=9)
    fig.tight_layout()
    plt.show()


for name in MODELS:
    best = tables[name]["held-out r"].argsort()[::-1][:4].tolist()
    for index in best:
        show_within_classes(name, index)
'''

pixels = '''# NOTE: These were suggested by looking at validation images, so we test on training images
resize = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])
train_images = Imagenette(root, split="train", size="full", transform=resize)
assert [label for _, label in train_images._samples] == train_y.tolist()


def get_pixel_features(images):
    \"\"\"Image statistics that use pixels alone. Images are [batch, 3, height, width], in 0 to 1.\"\"\"

    red, green, blue = images.unbind(dim=1)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    brightest, darkest = images.amax(dim=1), images.amin(dim=1)
    border = torch.cat(
        [luminance[:, :4].flatten(1), luminance[:, -4:].flatten(1), luminance[:, :, :4].flatten(1), luminance[:, :, -4:].flatten(1)],
        dim=1,
    )
    vertical = (luminance[:, 1:] - luminance[:, :-1]).abs().mean(dim=[1, 2])
    horizontal = (luminance[:, :, 1:] - luminance[:, :, :-1]).abs().mean(dim=[1, 2])

    return torch.stack(
        [
            luminance.mean(dim=[1, 2]),
            border.mean(dim=1),
            luminance.std(dim=[1, 2]),
            ((brightest - darkest) / (brightest + 1e-6)).mean(dim=[1, 2]),
            ((green > red) & (green > blue)).float().mean(dim=[1, 2]),
            ((blue > red) & (blue > green)).float().mean(dim=[1, 2]),
            vertical + horizontal,
        ],
        dim=1,
    )


PIXEL_NAMES = ["brightness", "border brightness", "contrast", "colourfulness", "green", "blue", "fine detail"]
pixel_values = torch.cat(
    [get_pixel_features(images) for images, _ in DataLoader(train_images, batch_size=256, num_workers=8)]
).double()
pixel_centered = F.normalize(remove_class_means(pixel_values, train_y), dim=0)


def get_pixel_correlations(train_features):
    \"\"\"Absolute correlation of each feature with each pixel statistic, within classes.\"\"\"

    return (F.normalize(remove_class_means(train_features, train_y), dim=0).T @ pixel_centered).abs()


start = train_x[:, 0].double()
start = start - start.mean(dim=0)
pca_features = start @ get_principal_directions(start)[:, :NUM_FEATURES]

columns = {name: get_pixel_correlations(bundle[0]).amax(dim=0) for name, bundle in features.items() if "random" not in name}
columns["PCA of block 6"] = get_pixel_correlations(pca_features).amax(dim=0)
columns["random mixes of linear"] = torch.stack(
    [get_pixel_correlations(features[f"linear random {trial}"][0]).amax(dim=0) for trial in range(NUM_RANDOM)]
).mean(dim=0)

pd.DataFrame(columns, index=PIXEL_NAMES).T.round(2)
'''

which = '''rows = {}
for name in MODELS:
    correlations = get_pixel_correlations(features[name][0])
    labels = features[name][4]
    rows[name] = {
        pixel: f"{labels[correlations[:, i].argmax()]} ({correlations[:, i].max():.2f})"
        for i, pixel in enumerate(PIXEL_NAMES)
    }

pd.DataFrame(rows).T
'''

colour = '''for name, config in MODELS.items():
    if config["jordan"]:
        show_within_classes(name, get_pixel_correlations(features[name][0])[:, PIXEL_NAMES.index("colourfulness")].argmax().item())
'''

nb = {
    "cells": [
        cell("markdown", intro),
        cell("code", setup),
        cell("markdown", "##### Load data\n"),
        cell("code", load),
        cell("markdown", "##### Vocabulary\n"),
        cell("code", words),
        cell("markdown", "##### Models\n"),
        cell("code", modes),
        cell("markdown", "##### Name and edit every feature\n"),
        cell("code", methods),
        cell("markdown", "##### Summary\n\n`held-out r` is the validation correlation of each feature's top word, averaged over 20 features. `word change` and `typical change` are medians of how much the feature's own word, and a typical word, move within classes when the feature is removed.\n"),
        cell("code", summary),
        cell("markdown", "##### Every feature\n"),
        cell("code", named),
        cell("markdown", "##### Inside each class\n\nThe four most nameable modes of each model.\n"),
        cell("code", images),
        cell("markdown", "##### Testing what the images show\\n\\nThe best absolute correlation with each pixel statistic, among each method's 20 features, within classes, on training images. No CLIP words are involved.\\n"),
        cell("code", pixels),
        cell("markdown", "##### Which mode tracks each statistic\\n"),
        cell("code", which),
        cell("markdown", "##### Each Jordan model's colourfulness mode, inside each class\\n"),
        cell("code", colour),
    ],
    "metadata": source["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open("second_analysis/19_feature_modes.ipynb", "w"), indent=1)
