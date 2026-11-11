import os, sys, collections
from pathlib import Path
import numpy as np, open_clip, pandas as pd, torch
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 40)
SP = sys.argv[1]
NAME, MIN_ON = "im_jordan256_2k_both", 30
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
stats = torch.load(save_dir / f"{NAME}_stats.pt")
kae = JordanKoopmanAutoencoder(top_k=256, in_features=768, latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir / f"{NAME}.safetensors")); kae.eval()
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(save_dir / "zero_shot_head_ViT-B-32.pt"); names = open_clip.IMAGENET_CLASSNAMES
def classify(s):
    o = clip.visual.ln_post(s * stats["std"] + stats["mean"]) @ clip.visual.proj
    return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(dim=-1)
def get_on(start):
    phi = kae.encode(start)
    return torch.cat([phi[:, :1].abs(), phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=-1) > 0

root = Path(os.getenv("DATASETS_CACHE")) / "imagenet"
train = torch.load(root / "clip_cls_ViT-B-32_train_blocks_5_11.pt")
ty = train["labels"]
on, clip_answer = [], []
for b in train["states"].split(20_000):
    b = (b - stats["mean"]) / stats["std"]
    on.append(get_on(b[:, 0])); clip_answer.append(classify(b[:, 1]))
on, clip_answer = torch.cat(on), torch.cat(clip_answer)
del train
wrong = clip_answer != ty
print("CLIP zero-shot on ImageNet train:", 1 - wrong.double().mean().item(), flush=True)
torch.save({"on": on, "clip_answer": clip_answer, "y": ty}, f"{SP}/train_on.pt")

# per class and mode: CLIP's error rate with the mode on vs off
rows = []
for c in range(1000):
    idx = (ty == c).nonzero()[:, 0]
    o, w = on[idx], wrong[idx]
    n_on = o.sum(0).double(); w_on = (o & w[:, None]).sum(0).double()
    n_off = len(idx) - n_on; w_off = w.sum().double() - w_on
    ok = (n_on >= MIN_ON) & (n_off >= MIN_ON)
    excess = w_on - n_on * w_off / n_off.clamp(min=1)
    for m in ok.nonzero()[:, 0].tolist():
        rows.append((c, m, int(n_on[m]), (1 - w_on[m] / n_on[m]).item(), (1 - w_off[m] / n_off[m]).item(), excess[m].item()))
pairs = pd.DataFrame(rows, columns=["class", "mode", "on", "CLIP accuracy, mode on", "CLIP accuracy, mode off", "excess errors"])
modes = pd.read_pickle(f"{SP}/mode_coherence.pkl")
pairs["classes in mode's top 50"] = pairs["mode"].map(modes["classes in top 50"])
pairs.to_pickle(f"{SP}/confound_pairs.pkl")
print("pairs tested:", len(pairs))
snow = pairs[(pairs["mode"] == 903)].sort_values("excess errors", ascending=False).head(8).copy()
snow["class"] = snow["class"].map(lambda c: names[c])
print("snow mode 903, worst classes:\n", snow.round(3).to_string())
top = pairs[pairs["classes in mode's top 50"] >= 20].sort_values("excess errors", ascending=False).head(25).copy()
def spurious(r):
    idx = ((ty == r["class"]) & on[:, r["mode"]] & wrong).nonzero()[:, 0]
    return "; ".join(f"{names[k]} {v}" for k, v in collections.Counter(clip_answer[idx].tolist()).most_common(2))
top["CLIP's wrong answers"] = top.apply(spurious, axis=1)
top["class"] = top["class"].map(lambda c: names[c])
print("cross-class modes, most excess CLIP errors:\n", top.round(3).to_string())
