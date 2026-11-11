import json
import sys

models = sys.argv[1]
target = sys.argv[2]
nb = json.load(open("/home/nsa325/work/koopmann/second_analysis/21_imagenet_features.ipynb"))
for cell in nb["cells"]:
    cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs")
    if cell["cell_type"] == "markdown":
        cell.pop("outputs", None)
    cell["execution_count"] = None if cell["cell_type"] == "code" else cell.get("execution_count")
    if cell["cell_type"] == "markdown":
        cell.pop("execution_count", None)


def get(i):
    return "".join(nb["cells"][i]["source"])


def put(i, text):
    nb["cells"][i]["source"] = text.splitlines(keepends=True)


def swap(i, old, new):
    text = get(i)
    assert text.count(old) >= 1, old
    put(i, text.replace(old, new))


put(0, """### KAEs trained longer and wider on ImageNet

The same tests as `21_imagenet_features.ipynb`, on models trained for 100 epochs. We also look at the top images of each chosen snow feature, on validation images outside the snow set. A high score on 129 images is not enough: a snow feature should fire on snow everywhere.
""")

text = get(1)
start = text.index("MODELS = {")
end = text.index("PANEL = [")
put(1, text[:start] + "MODELS = " + models + "\n" + text[end:])

swap(5, 'def load_model(name, top_k, jordan):', 'def load_model(name, top_k, jordan, width=1024):')
swap(5, '''"latent_features": 1025 if jordan else 1024,
        "hidden_config": [1024],''', '''"latent_features": width + 1 if jordan else width,
        "hidden_config": [width],''')
swap(5, '''def get_mode_bundle(name, top_k, jordan):''', '''def get_mode_bundle(name, top_k, jordan, width=1024):''')
swap(5, '''    kae = load_model(name, top_k, jordan)
    eigenvalues, encode, decode''', '''    kae = load_model(name, top_k, jordan, width)
    eigenvalues, encode, decode''')
swap(5, '''def get_kae_ica(name, top_k, jordan):
    kae = load_model(name, top_k, jordan)''', '''def get_kae_ica(name, top_k, jordan, width=1024):
    kae = load_model(name, top_k, jordan, width)''')

swap(17, '''rows = {}
for name, (features, edit, predict, labels) in bundles.items():
    scores = features(x[:, 0])''', '''rows, chosen = {}, {}
for name, (features, edit, predict, labels) in bundles.items():
    scores = features(x[:, 0])''')
swap(17, '''        halves.append({"AUC": auc, **result})''', '''        halves.append({"AUC": auc, **result})
        chosen[name] = best''')

nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["##### What each snow feature fires on\n", "\n", "The 16 validation images highest on each chosen feature, and how much of the top 200 sits in CLIP's snowiest 5% of validation images.\n"]})
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": '''import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

data = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
class_names = list(open_clip.IMAGENET_CLASSNAMES)
snowiness = to_embeddings(val_x[:, 1]) @ snow_direction
is_snowy = snowiness > snowiness.quantile(0.95)

rows = {}
for name, (features, edit, predict, labels) in bundles.items():
    best = chosen[name]
    scores = features(x[:, 0])[:, best]
    sign = (scores[is_snow].mean() - scores[~is_snow].mean()).sign()
    top = (sign * features(val_x[:, 0])[:, best]).argsort(descending=True)
    rows[name] = {"feature": labels[best], "snowy share of top 200": is_snowy[top[:200]].double().mean().item()}

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
    for ax, i in zip(axes.flat, top[:16].tolist()):
        image = data[i]["image"].convert("RGB")
        image.thumbnail((256, 256))
        ax.imshow(np.asarray(image))
        ax.set_title(class_names[val_y[i]][:22], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{name}, {labels[best]}")
    fig.tight_layout()
    plt.show()

pd.DataFrame(rows).T
'''.splitlines(keepends=True),
})
json.dump(nb, open(target, "w"), indent=1)
print("written", target)
