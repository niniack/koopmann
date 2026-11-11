import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""### CLIP on snowy images

Reproduces the "Pretrained" row of Table 2a in Class Vectors (Kim & Do, NeurIPS 2025). Nothing is trained.

Images come from `MadryLab/EditingClassifiers` (`data/snow/test`), 129 Flickr photos. Two are exact duplicates, with `(1)` in the filename."""),
code("""import os
from pathlib import Path

import matplotlib.pyplot as plt
import open_clip
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image

from koopmann.utils import get_device

load_dotenv("../.env")
torch.set_grad_enabled(False)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

device = get_device()

%config InlineBackend.figure_format = 'jpeg'"""),
md("##### Load data"),
code("""data_root = Path(os.getenv("DATASETS_CACHE")) / "snowy_imagenet"
class_names = list(open_clip.IMAGENET_CLASSNAMES)

# Folder name to ImageNet index
folder_to_index = {
    "armytank": 847,
    "carwheel": 479,
    "firetruck": 555,
    "motorscooter": 670,  # named `vespa` in CLIP's class list
    "racingcar": 751,
    "schoolbus": 779,
    "trafficlight": 920,
}
snow_indices = torch.tensor(list(folder_to_index.values()))

paths = sorted(data_root.glob("*/*.jpg"))
folders = [path.parent.name for path in paths]
labels = torch.tensor([folder_to_index[folder] for folder in folders])

print(f"{len(paths)} images")
print({folder: class_names[index] for folder, index in folder_to_index.items()})"""),
md("##### Run encoders"),
code("""ENCODERS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
TEMPLATES = {
    "1 template": [lambda c: f"a photo of a {c}."],
    "80 templates": open_clip.OPENAI_IMAGENET_TEMPLATES,
}


def get_features(model, preprocess, batch_size=32):
    features = []
    for i in range(0, len(paths), batch_size):
        images = [preprocess(Image.open(path).convert("RGB")) for path in paths[i : i + batch_size]]
        features.append(model.encode_image(torch.stack(images).to(device)).float())

    features = torch.cat(features)
    return features / features.norm(dim=-1, keepdim=True)


def get_head(model, tokenizer, templates):
    weights = []
    for name in class_names:
        tokens = tokenizer([template(name) for template in templates]).to(device)
        embeddings = model.encode_text(tokens).float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean_embedding = embeddings.mean(dim=0)
        weights.append(mean_embedding / mean_embedding.norm())

    return torch.stack(weights)


# shape: [encoder][templates] -> [images, classes]
logits = {}
for encoder in ENCODERS:
    model, _, preprocess = open_clip.create_model_and_transforms(
        encoder,
        pretrained="openai",
        force_quick_gelu=True,  # OpenAI's weights were trained with QuickGELU
        cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(encoder)

    features = get_features(model, preprocess)
    logits[encoder] = {
        name: (features @ get_head(model, tokenizer, templates).T).cpu()
        for name, templates in TEMPLATES.items()
    }

    del model
    torch.cuda.empty_cache()"""),
md("""##### Compare to the paper

The paper reports mean and std across classes. 1000-way with 80 templates on all 129 images matches its "Pretrained" row, using the population std."""),
code("""PAPER = {"ViT-B-16": "55.2 ± 24.6", "ViT-B-32": "53.4 ± 29.9", "ViT-L-14": "60.2 ± 22.5"}


def get_class_accs(predictions):
    correct = (predictions == labels).float()
    return torch.stack([correct[labels == index].mean() for index in snow_indices]) * 100


rows = []
for encoder in ENCODERS:
    row = {"encoder": encoder, "paper": PAPER[encoder]}
    for name in TEMPLATES:
        scores = logits[encoder][name]
        accs = get_class_accs(scores.argmax(dim=1))
        row[f"1000-way, {name}"] = f"{accs.mean():.1f} ± {accs.std(unbiased=False):.1f}"

        accs = get_class_accs(snow_indices[scores[:, snow_indices].argmax(dim=1)])
        row[f"7-way, {name}"] = f"{accs.mean():.1f} ± {accs.std(unbiased=False):.1f}"
    rows.append(row)

pd.DataFrame(rows).set_index("encoder")"""),
md("""##### Per-class accuracy

From here on, everything uses the setting that matches the paper: 1000-way with 80 templates."""),
code("""scores = {encoder: logits[encoder]["80 templates"] for encoder in ENCODERS}
predictions = {encoder: scores[encoder].argmax(dim=1) for encoder in ENCODERS}
top5 = {encoder: scores[encoder].topk(5, dim=1).indices for encoder in ENCODERS}

table = {"images": [folders.count(folder) for folder in folder_to_index]}
for encoder in ENCODERS:
    table[encoder] = [round(acc, 1) for acc in get_class_accs(predictions[encoder]).tolist()]
    hits = (top5[encoder] == labels[:, None]).any(dim=1).float()
    table[f"{encoder} top-5"] = [
        round(hits[labels == index].mean().item() * 100, 1) for index in snow_indices
    ]

pd.DataFrame(table, index=list(folder_to_index))"""),
md("##### What does it predict instead?"),
code("""rows = []
for folder, index in folder_to_index.items():
    row = {"class": folder}
    for encoder in ENCODERS:
        wrong = predictions[encoder][(labels == index) & (predictions[encoder] != labels)]
        names, counts = wrong.unique(return_counts=True)
        order = counts.argsort(descending=True)[:4]
        row[encoder] = ", ".join(f"{class_names[names[i]]} ({counts[i]})" for i in order)
    rows.append(row)

pd.DataFrame(rows).set_index("class")"""),
code("""SNOW_CLASSES = ["snowplow", "snowmobile"]
snow_ids = torch.tensor([class_names.index(name) for name in SNOW_CLASSES])

rows = []
for encoder in ENCODERS:
    is_wrong = predictions[encoder] != labels
    is_snow = torch.isin(predictions[encoder], snow_ids) & is_wrong
    is_other_snow_class = torch.isin(predictions[encoder], snow_indices) & is_wrong
    rows.append(
        {
            "encoder": encoder,
            "wrong": int(is_wrong.sum()),
            "wrong, snowplow or snowmobile": int(is_snow.sum()),
            "wrong, another of the 7 classes": int(is_other_snow_class.sum()),
        }
    )

pd.DataFrame(rows).set_index("encoder")"""),
md("""##### All images

Each line is one encoder: ✓ if correct, ✗ if wrong, followed by its prediction."""),
code("""SHORT = {"ViT-B-32": "B/32", "ViT-B-16": "B/16", "ViT-L-14": "L/14"}
COLUMNS = 6


def plot_class(folder):
    indices = [i for i, name in enumerate(folders) if name == folder]
    rows = -(-len(indices) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(2.6 * COLUMNS, 3.1 * rows), squeeze=False)

    for ax in axes.flatten():
        ax.axis("off")

    for ax, i in zip(axes.flatten(), indices):
        ax.imshow(Image.open(paths[i]).convert("RGB").resize((224, 224)))

        lines = []
        for encoder in ENCODERS:
            mark = "✓" if predictions[encoder][i] == labels[i] else "✗"
            lines.append(f"{SHORT[encoder]} {mark} {class_names[predictions[encoder][i]][:20]}")

        ax.set_title("\\n".join(lines), fontsize=8.5, loc="left")

    fig.suptitle(f"{folder} ({class_names[folder_to_index[folder]]})", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    plt.show()


for folder in folder_to_index:
    plot_class(folder)"""),
md("##### Full table"),
code("""table = {"class": folders, "file": [path.name for path in paths]}
for encoder in ENCODERS:
    table[encoder] = [class_names[index] for index in predictions[encoder]]
    table[f"{SHORT[encoder]} ok"] = (predictions[encoder] == labels).tolist()

pd.DataFrame(table)"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "first_analysis/06_snowy_clip.ipynb")
