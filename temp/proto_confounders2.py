import os, sys, collections
from pathlib import Path
import numpy as np, open_clip, pandas as pd, torch
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
SP = sys.argv[1]
NAME, NUM_PAIRS = "im_jordan256_2k_both", 20
names = open_clip.IMAGENET_CLASSNAMES
saved = torch.load(f"{SP}/train_on.pt"); on, clip_answer, ty = saved["on"], saved["clip_answer"], saved["y"]
wrong = clip_answer != ty
import numpy as np
pairs = pd.read_pickle(f"{SP}/confound_pairs.pkl")
usage = on.double().mean(0); sizes = torch.bincount(ty, minlength=1000)
pairs["usage"] = usage[pairs["mode"].values].numpy()
pairs["share of class"] = pairs["on"] / sizes[pairs["class"].values].numpy()
sel = pairs[(pairs["usage"] <= 0.2) & (pairs["share of class"] <= 0.4) & (pairs["on"] >= 50) & (pairs["classes in mode's top 50"] >= 20)].copy()
p_on, p_off = 1 - sel["CLIP accuracy, mode on"], 1 - sel["CLIP accuracy, mode off"]
n_on = sel["on"]; n_off = n_on / sel["share of class"] - n_on
pool = (p_on * n_on + p_off * n_off) / (n_on + n_off)
sel["z"] = (p_on - p_off) / np.sqrt(pool * (1 - pool) * (1 / n_on + 1 / n_off))
# NOTE: One pair per class, its strongest mode
top = sel.sort_values("z", ascending=False).drop_duplicates("class").head(NUM_PAIRS).reset_index(drop=True)
def spurious(c, m):
    idx = ((ty == c) & on[:, m] & wrong).nonzero()[:, 0]
    return "; ".join(f"{names[k]} {v}" for k, v in collections.Counter(clip_answer[idx].tolist()).most_common(2))
top["CLIP's wrong answers, mode on"] = [spurious(int(c), int(m)) for c, m in zip(top["class"], top["mode"])]
top["class name"] = top["class"].map(lambda c: names[int(c)])
print(top[["class name", "mode", "usage", "on", "CLIP accuracy, mode on", "CLIP accuracy, mode off", "z", "classes in mode's top 50", "CLIP's wrong answers, mode on"]].round(3).to_string())
top.to_pickle(f"{SP}/confound_top_classes.pkl")

# test on validation with the surrogate
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
stats = torch.load(save_dir / f"{NAME}_stats.pt")
kae = JordanKoopmanAutoencoder(top_k=256, in_features=768, latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir / f"{NAME}.safetensors")); kae.eval()
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(save_dir / "zero_shot_head_ViT-B-32.pt")
def classify(s):
    o = clip.visual.ln_post(s * stats["std"] + stats["mean"]) @ clip.visual.proj
    return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(dim=-1)
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_ViT-B-32_val_blocks_5_11.pt")
x, y = (val["states"] - stats["mean"]) / stats["std"], val["labels"]
phi = torch.cat([kae.encode(b) for b in x[:, 0].split(5000)])
out = kae.koopman_forward(phi, 1)
val_on = torch.cat([phi[:, :1].abs(), phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=-1) > 0
before = torch.cat([classify(kae.decode(b)) for b in out.split(5000)])
clip_val = torch.cat([classify(b) for b in x[:, 1].split(5000)])
rows, cache = [], {}
for c, m in zip(top["class"].astype(int), top["mode"].astype(int)):
    idx = val_on[:, m].nonzero()[:, 0]
    if m not in cache:
        cache[m] = {}
        for alpha in [0.0, 0.5]:
            edited = out[idx].clone()
            edited[:, 2 * m - 1 : 2 * m + 1] *= alpha
            cache[m][alpha] = classify(kae.decode(edited))
    t, b = y[idx], before[idx]
    target = t == c
    row = {"class": names[c], "mode": m, "val images, class with mode on": int(target.sum()),
           "CLIP accuracy there": (clip_val[idx][target] == c).double().mean().item() if target.any() else float("nan"),
           "KAE accuracy there": (b[target] == c).double().mean().item() if target.any() else float("nan")}
    for alpha, after in cache[m].items():
        fixed, broken = (b != t) & (after == t), (b == t) & (after != t)
        row[f"alpha {alpha}: class fixed"] = int((fixed & target).sum())
        row[f"alpha {alpha}: class broken"] = int((broken & target).sum())
        row[f"alpha {alpha}: elsewhere fixed"] = int((fixed & ~target).sum())
        row[f"alpha {alpha}: elsewhere broken"] = int((broken & ~target).sum())
    rows.append(row)
result = pd.DataFrame(rows)
print(result.round(3).to_string())
print("pooled over pairs:", result.filter(like="alpha").sum().to_dict())
result.to_pickle(f"{SP}/confound_val_classes.pkl")
