import json, os, sys, warnings
warnings.filterwarnings("ignore")
import torch
nb = json.load(open("18_sparse_modes.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = {}
exec(code[0], ns)
exec(code[1], ns)
exec(code[3].split("results = {name")[0], ns)
pd = ns["pd"]; CLASSES = ns["CLASSES"]; val_x = ns["val_x"]; val_y = ns["val_y"]; train_x = ns["train_x"]; train_y = ns["train_y"]
pd.set_option("display.width", 200)
kae = ns["load_model"]("b6_jordan128", 128, True)
eigenvalues, encode, decode = ns["get_jordan_modes"](kae)
get_class_accs, get_scores = ns["get_class_accs"], ns["get_scores"]
train_modes, val_modes = encode(train_x[:, 0]), encode(val_x[:, 0])
mean = train_modes.mean(dim=0)
base = get_class_accs(decode(val_modes * eigenvalues))

# Purity: share of a mode's energy that comes from its top class, on validation images
energy = val_modes.abs().pow(2)
per_class = torch.stack([energy[val_y == c].sum(dim=0) for c in range(10)])
total = per_class.sum(dim=0)
used = (val_modes.abs() > 0).double().mean(dim=0) > 0.01
purity = per_class.amax(dim=0) / total.clamp(min=1e-12)
owner = per_class.argmax(dim=0)
print("modes used on >1% of images:", used.sum().item())
for t in [0.5, 0.7, 0.9]:
    print(f"purity above {t}:", (used & (purity > t)).sum().item())
print("pure modes (>0.7) per class:", {CLASSES[c]: (used & (purity > 0.7) & (owner == c)).sum().item() for c in range(10)})
print("median purity of used modes:", round(purity[used].median().item(), 3), "| chance 0.1")

# Removing more modes, with both drops, per class
def run(m, fill):
    rows = {}
    for c in range(10):
        removed = get_scores(train_modes, c).argsort(descending=True)[:m]
        modes = val_modes.clone()
        modes[:, removed] = fill[removed]
        drops = base - get_class_accs(decode(modes * eigenvalues))
        rows[CLASSES[c]] = {"target": drops[c].item(), "other": ((drops.sum() - drops[c]) / 9).item(), "worst other": max(drops[j].item() for j in range(10) if j != c)}
    return pd.DataFrame(rows).T

zero = torch.zeros_like(mean)
for m in [5, 20, 50, 100, 200]:
    table = run(m, zero)
    print(f"m={m}: target {table['target'].mean():.3f}  other {table['other'].mean():.3f}  specificity {table['target'].mean() - table['other'].mean():.3f}  worst other {table['worst other'].mean():.3f}")
print(run(20, zero).round(3).to_string())
