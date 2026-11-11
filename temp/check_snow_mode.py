import json, os
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("21_imagenet_features.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]

# Imports, then only what we need: no classifier or linear fits
exec(cells[1], globals())
source = cells[3]
exec(source.split("all_x, all_y = load_split")[0], globals())
val_x, val_y = load_split("val")
exec(cells[5].split("def get_mode_bundle")[0], globals())
measures = cells[11]
exec(measures.split("def remove_class_means")[0], globals())

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

MODE = 414
names = list(open_clip.IMAGENET_CLASSNAMES)
kae = load_model("im_jordan128", 128, True)
_, encode, _ = get_jordan_modes(kae)

value = encode(val_x[:, 0])[:, MODE].abs()
on = value > 0
print(f"Mode {MODE} is on for {100 * on.double().mean():.1f}% of validation images")

snow_score = to_embeddings(val_x[:, 1]) @ snow_direction
print(f"Correlation with CLIP's snow score on true block 12: {torch.corrcoef(torch.stack([value, snow_score.double()]))[0, 1]:.3f}")

# Inside each class, so it can't just be picking snowy classes
sums = torch.zeros(1000, dtype=torch.double).index_add_(0, val_y, value)
score_sums = torch.zeros(1000, dtype=torch.double).index_add_(0, val_y, snow_score.double())
within = torch.corrcoef(torch.stack([value - (sums / 50)[val_y], snow_score.double() - (score_sums / 50)[val_y]]))[0, 1]
print(f"Same correlation inside classes: {within:.3f}")

top = value.argsort(descending=True)
print("\nClasses among the top 200 images:")
for label, count in pd.Series([names[i] for i in val_y[top[:200]].tolist()]).value_counts().head(20).items():
    print(f"  {count:3d} {label}")

print("\nClasses with the highest mean value:")
means = sums / 50
for i in means.argsort(descending=True)[:15].tolist():
    print(f"  {means[i]:.2f} {names[i]}")

print("\nClasses in the snow set, mean value on normal validation images:")
for folder, index in SNOW_FOLDERS.items():
    print(f"  {means[index]:.2f} {names[index]}")

data = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
fig, axes = plt.subplots(5, 8, figsize=(16, 10.5))
for ax, i in zip(axes.flat, top[:40].tolist()):
    image = data[i]["image"].convert("RGB")
    image.thumbnail((256, 256))
    ax.imshow(np.asarray(image))
    ax.set_title(names[val_y[i]][:22], fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig("/home/nsa325/work/koopmann/temp/mode414_top.png", dpi=70)

# Random images where the mode is on, excluding the very top, as a check on weak activity
active = on.nonzero()[:, 0]
middle = active[value[active].argsort(descending=True)][len(active) // 2 : len(active) // 2 + 16]
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for ax, i in zip(axes.flat, middle.tolist()):
    image = data[i]["image"].convert("RGB")
    image.thumbnail((256, 256))
    ax.imshow(np.asarray(image))
    ax.set_title(names[val_y[i]][:22], fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig("/home/nsa325/work/koopmann/temp/mode414_middle.png", dpi=70)
