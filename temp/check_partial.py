import json, os, sys, torch, pandas as pd
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("jordan_topk_modes.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for i in [0, 1, 2]:
    exec(cells[i], globals())
exec(next(c for c in cells if "snowy_imagenet" in c).split("true_correct =")[0], globals())
torch.set_num_threads(48)
pd.set_option("display.width", 250)
on = activity[:, SNOW_MODE] > 0
groups = {"snowy": (x[is_snowy, 0], y[is_snowy]), "normal": (x[~is_snowy, 0], y[~is_snowy]), "ImageNet val, mode on": (val_x[on, 0], val_y[on])}
base = {k: classify(predict(kae, s)) == t for k, (s, t) in groups.items()}
rows = []
for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    edited = dampen(SNOW_MODE, alpha)
    row = {"alpha": alpha}
    for k, (s, t) in groups.items():
        now = classify(predict(edited, s)) == t
        row[f"{k}: fixed"] = int((~base[k] & now).sum()); row[f"{k}: broken"] = int((base[k] & ~now).sum())
    rows.append(row)
print(f"images: snowy {int(is_snowy.sum())} ({int(base['snowy'].sum())} right), normal {int((~is_snowy).sum())} ({int(base['normal'].sum())} right), ImageNet val with snow mode on {int(on.sum())} ({int(base['ImageNet val, mode on'].sum())} right)")
print(pd.DataFrame(rows).set_index("alpha").to_string())
