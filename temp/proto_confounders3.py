import collections, sys
import numpy as np, open_clip, pandas as pd, torch
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
SP = sys.argv[1]
names = open_clip.IMAGENET_CLASSNAMES
saved = torch.load(f"{SP}/train_on.pt"); on, clip_answer, ty = saved["on"], saved["clip_answer"], saved["y"]
wrong = clip_answer != ty
pairs = pd.read_pickle(f"{SP}/confound_pairs.pkl")
modes = pd.read_pickle(f"{SP}/mode_coherence.pkl")
usage = on.double().mean(0)
sizes = torch.bincount(ty, minlength=1000)
pairs["usage"] = usage[pairs["mode"].values].numpy()
pairs["share of class"] = pairs["on"] / sizes[pairs["class"].values].numpy()
pairs["drop"] = pairs["CLIP accuracy, mode off"] - pairs["CLIP accuracy, mode on"]
print("usage of modes in the first ranking:", {m: round(usage[m].item(), 3) for m in [448, 524, 907, 28, 811, 903]})
print("how often a mode is on for most of a class (share > 0.9):", (pairs["share of class"] > 0.9).mean().round(3))
# snow-like: a minority context within the class, a mode that isn't on everywhere, and cross-class
sel = pairs[(pairs["usage"] <= 0.2) & (pairs["share of class"] <= 0.4) & (pairs["on"] >= 50) & (pairs["classes in mode's top 50"] >= 20)].copy()
p_on, p_off = 1 - sel["CLIP accuracy, mode on"], 1 - sel["CLIP accuracy, mode off"]
n_on, n_off = sel["on"], sel["on"] / sel["share of class"] - sel["on"]
p = (p_on * n_on + p_off * n_off) / (n_on + n_off)
sel["z"] = (p_on - p_off) / np.sqrt(p * (1 - p) * (1 / n_on + 1 / n_off))
print("pairs left:", len(sel), "| with z > 5:", (sel["z"] > 5).sum())
top = sel.sort_values("z", ascending=False).head(25).reset_index(drop=True)
def spurious(c, m):
    idx = ((ty == c) & on[:, m] & wrong).nonzero()[:, 0]
    return "; ".join(f"{names[k]} {v}" for k, v in collections.Counter(clip_answer[idx].tolist()).most_common(2))
top["CLIP's wrong answers, mode on"] = [spurious(int(c), int(m)) for c, m in zip(top["class"], top["mode"])]
top["class name"] = top["class"].map(lambda c: names[int(c)])
print(top[["class name", "mode", "usage", "on", "share of class", "CLIP accuracy, mode on", "CLIP accuracy, mode off", "z", "classes in mode's top 50", "CLIP's wrong answers, mode on"]].round(3).to_string())
print("modes in top 25:", collections.Counter(top["mode"]).most_common())
snow = sel[sel["mode"] == 903].sort_values("z", ascending=False).head(5)
snow = snow.assign(name=snow["class"].map(lambda c: names[int(c)]))
print("snow mode rows:\n", snow[["name", "on", "share of class", "CLIP accuracy, mode on", "CLIP accuracy, mode off", "z"]].round(3).to_string())
top.to_pickle(f"{SP}/confound_top.pkl")
