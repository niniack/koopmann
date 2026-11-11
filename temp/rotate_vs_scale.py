"""Editing one mode: scaling its eigenvalue (|λ| times alpha) vs rotating it (arg λ plus theta). What changes in CLIP's reading of the prediction?"""
import os, sys, math, torch, numpy as np, pandas as pd, open_clip
from collections import Counter
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.manual_seed(0); torch.set_num_threads(16)
pd.set_option("display.width", 220)
name, ck, snow_mode = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
root = Path(os.getenv("DATASETS_CACHE"))
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
val = torch.load(root / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
vx, vy = (val["states"].float() - mean) / std, val["labels"]
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip"); clip.eval()
tok = open_clip.get_tokenizer("ViT-B-32")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
names = open_clip.IMAGENET_CLASSNAMES
def embed(s):
    o = clip.visual.ln_post(s[..., :768] * std[1, :768] + mean[1, :768]) @ clip.visual.proj
    return o / o.norm(dim=-1, keepdim=True)
def cls_norm(s):
    return (s[..., :768] * std[1, :768] + mean[1, :768]).norm(dim=-1)
t = clip.encode_text(tok(["a photo in the snow", "a photo"])); t = t / t.norm(dim=-1, keepdim=True)
snow_dir = (t[0] - t[1]) / (t[0] - t[1]).norm()
lam0 = torch.complex(kae.block_real.clone(), kae.block_imag.clone())
def predict(start, mode=None, edit=None):
    if mode is not None:
        lam = lam0[mode - 1]
        new = lam * edit
        kae.block_real[mode - 1] = new.real; kae.block_imag[mode - 1] = new.imag
    out = torch.cat([kae.decode(kae.koopman_forward(kae.encode(b), 1)) for b in start.split(5000)])
    if mode is not None:
        kae.block_real[mode - 1] = lam0[mode - 1].real; kae.block_imag[mode - 1] = lam0[mode - 1].imag
    return out
EDITS = {"scale x0 (remove)": 0.0, "scale x0.5": 0.5, "scale x2": 2.0,
         "rotate +45°": complex(math.cos(math.pi / 4), math.sin(math.pi / 4)), "rotate +90°": 1j, "rotate -90°": -1j, "rotate 180° (= x-1)": -1.0}
EDITS = {k: complex(v) for k, v in EDITS.items()}

# Snow test: 350 normal val images of 7 vehicle classes + 129 snowy photos
FOLD = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
paths = sorted((root / "snowy_imagenet").glob("*/*.jpg"))
st = []
def keep(_, __, o):
    p = o[:, 1:]; st.append(torch.cat([o[:, 0], p.mean(1), p.std(1)], -1))
hs = [clip.visual.transformer.resblocks[b].register_forward_hook(keep) for b in stats["blocks"]]
clip.encode_image(torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]))
for h in hs: h.remove()
sx = (torch.stack(st, 1) - mean) / std
sy = torch.tensor([FOLD[p.parent.name] for p in paths])
normal = torch.isin(vy, torch.tensor(list(FOLD.values())))
x, y = torch.cat([vx[normal], sx]), torch.cat([vy[normal], sy])
snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(sx), dtype=torch.bool)])
base = predict(x[:, 0]); be = embed(base); ba = (be @ head).argmax(-1)
rows = {}
for label, e in {"as trained": complex(1.0), **EDITS}.items():
    p = predict(x[:, 0], snow_mode, e); pe = embed(p); a = (pe @ head).argmax(-1)
    new = Counter(names[c] for c in a[snowy & (a != ba)].tolist()).most_common(3)
    rows[label] = {"snowy acc": (a == y)[snowy].double().mean().item(), "normal acc": (a == y)[~snowy].double().mean().item(),
                   "snow score, snowy": (pe[snowy] @ snow_dir).mean().item(), "snowy answers changed": (a != ba)[snowy].double().mean().item(),
                   "normal answers changed": (a != ba)[~snowy].double().mean().item(), "snowy embedding move (1-cos)": (1 - (pe * be).sum(-1))[snowy].median().item(),
                   "snowy CLS norm ratio": (cls_norm(p) / cls_norm(base))[snowy].median().item(), "snowy: new answers": ", ".join(f"{n} {k}" for n, k in new)}
print(f"== {name}: snow mode {snow_mode}, λ = {lam0[snow_mode - 1].real:.2f}{lam0[snow_mode - 1].imag:+.2f}i")
print(pd.DataFrame(rows).T.round(3).to_string())

# Many modes: on the val images that keep each mode
phi = torch.cat([kae.encode(b) for b in vx[:, 0].split(5000)])
act = phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)
usage = (act > 0).double().mean(0)
cands = ((usage > 0.03) & (usage < 0.3)).nonzero()[:, 0]
modes = (cands[torch.randperm(len(cands))[:40]] + 1).tolist()
per = {k: [] for k in EDITS}
for m in modes:
    on = (act[:, m - 1] > 0).nonzero()[:, 0]
    on = on[torch.randperm(len(on))[:2000]]
    s0 = vx[on, 0]
    p0 = predict(s0); e0 = embed(p0); a0 = (e0 @ head).argmax(-1); right0 = a0 == vy[on]
    up = embed(predict(s0, m, complex(2.0))) - e0
    for k, e in EDITS.items():
        p = predict(s0, m, e); pe = embed(p); a = (pe @ head).argmax(-1)
        d = pe - e0
        per[k].append({"answers changed": (a != a0).double().mean().item(), "accuracy change": ((a == vy[on]).double().mean() - right0.double().mean()).item(),
                       "embedding move (1-cos)": (1 - (pe * e0).sum(-1)).median().item(),
                       "direction vs scale x2 (cos)": torch.nn.functional.cosine_similarity(d, up, dim=-1).median().item(),
                       "CLS norm ratio": (cls_norm(p) / cls_norm(p0)).median().item()})
print(f"\n40 random modes kept on 3-30% of val; per edit, the median over modes of each measure (on images that keep the mode)")
print(pd.DataFrame({k: pd.DataFrame(v).median() for k, v in per.items()}).T.round(3).to_string())
