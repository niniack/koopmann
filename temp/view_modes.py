import json, sys, warnings
warnings.filterwarnings("ignore")
out = sys.argv[1]
nb = json.load(open("19_feature_modes.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
setup, load, words, modes, methods = code[:5]
ns = {}
exec(setup, ns); exec(load, ns); exec(modes, ns)
exec('''
def remove_class_means(x, labels):
    x = x.clone()
    for c in range(10):
        x[labels == c] -= x[labels == c].mean(dim=0)
    return x
''', ns)
exec(methods.split("linear = fit_linear(train_x")[0], ns)
torch, plt, F = ns["torch"], ns["plt"], ns["F"]
dataset, val_y = ns["dataset"], ns["val_y"]
CLASSES = ns["CLASSES"]
a = ns["get_mode_features"]("b6_jordan128", 128, True)
b = ns["get_mode_features"]("b6_jordan128_s1", 128, True)

def within(x):
    return F.normalize(ns["remove_class_means"](x, val_y), dim=0)

agreement = (within(a[1]).T @ within(b[1])).abs()
print("mode | best partner in seed 1 | within-class agreement | class share")
for i, label in enumerate(a[4]):
    j = agreement[i].argmax().item()
    x = a[1][:, i] - a[1][:, i].mean()
    share = (sum((val_y == c).sum() * x[val_y == c].mean().pow(2) for c in range(10)) / x.pow(2).sum()).item()
    print(f"{i:>2} {label:<9} {b[4][j]:<9} {agreement[i, j]:.2f}  {share:.2f}")

def draw(bundle, i, path):
    """Within each class, the top 2 and bottom 2 images. One row of images per class pair."""
    feature = bundle[1][:, i]
    fig, axes = plt.subplots(5, 10, figsize=(11, 5.6))
    for c in range(10):
        members = (val_y == c).nonzero()[:, 0]
        ranked = members[feature[members].argsort(descending=True)].tolist()
        picks = ranked[:2] + [None] + ranked[-2:][::-1]
        offset = 5 * (c // 5)
        for column, index in enumerate(picks):
            ax = axes[c % 5, offset + column]
            ax.set_xticks([]); ax.set_yticks([])
            if index is None:
                ax.axis("off"); continue
            ax.imshow(dataset[index][0])
        axes[c % 5, offset].set_ylabel(CLASSES[c], fontsize=7)
    fig.suptitle(f"{bundle[4][i]}. Left pair: high. Right pair: low.", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=75); plt.close(fig)

for i in range(12):
    draw(a, i, f"{out}/s0_{i:02d}.png")
