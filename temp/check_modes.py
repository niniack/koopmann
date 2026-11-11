import json, sys, warnings
warnings.filterwarnings("ignore")
out = sys.argv[1]
nb = json.load(open("19_feature_modes.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
setup, load, words, modes, methods, summary, named, images, pixels, which = code
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
exec(pixels.split("start = train_x[:, 0]")[0], ns)
torch, plt, F, pd = ns["torch"], ns["plt"], ns["F"], ns["pd"]
pd.set_option("display.width", 200)
CLASSES, dataset, val_y, train_y = ns["CLASSES"], ns["dataset"], ns["val_y"], ns["train_y"]
names = ns["PIXEL_NAMES"]
bundles = {name: ns["get_mode_features"](name, **config) for name, config in ns["MODELS"].items()}
pixel_r = ns["get_pixel_correlations"]

def row(name, label):
    train_f, val_f, _, _, labels = bundles[name]
    i = labels.index(label)
    return i, pixel_r(train_f[:, i:i + 1])[0]

picks = [("b6_novc", "mode 6"), ("b6_novc", "mode 7"), ("b6_jordan128", "mode 501"), ("b6_jordan128_s1", "mode 476"), ("b6_jordan128", "mode 422"), ("b6_jordan128_s1", "mode 41")]
print(pd.DataFrame({f"{n} {l}": row(n, l)[1].tolist() for n, l in picks}, index=names).T.round(2).to_string())

def val_within(name, label):
    i = bundles[name][4].index(label)
    return F.normalize(ns["remove_class_means"](bundles[name][1][:, i:i + 1], val_y), dim=0)[:, 0]

print("within-class agreement across seeds on validation images:")
for a, b in [(("b6_jordan128", "mode 501"), ("b6_jordan128_s1", "mode 476")), (("b6_jordan128", "mode 422"), ("b6_jordan128_s1", "mode 41"))]:
    print(" ", a, b, round((val_within(*a) @ val_within(*b)).abs().item(), 3))

def show(name, label, path):
    i = bundles[name][4].index(label)
    feature = bundles[name][1][:, i]
    fig, axes = plt.subplots(5, 14, figsize=(13.5, 5.2))
    for c, class_name in enumerate(CLASSES):
        members = (val_y == c).nonzero()[:, 0]
        ranked = members[feature[members].argsort(descending=True)].tolist()
        picks = ranked[:3] + [None] + ranked[-3:][::-1]
        offset = 7 * (c // 5)
        for column, index in enumerate(picks):
            ax = axes[c % 5, offset + column]
            ax.set_xticks([]); ax.set_yticks([])
            if index is None:
                ax.axis("off"); continue
            ax.imshow(dataset[index][0])
        axes[c % 5, offset].set_ylabel(class_name, fontsize=7)
    fig.suptitle(f"{name}, {label}. Left of each block: high. Right: low.", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)

show("b6_jordan128", "mode 422", f"{out}/colour_s0.png")
show("b6_jordan128_s1", "mode 41", f"{out}/colour_s1.png")
