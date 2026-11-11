import copy, os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, open_clip, pandas as pd, torch
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
SP = sys.argv[1]
NAME, SNOW_MODE = "im_jordan256_2k_both", 903
SNOW_FOLDERS = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
stats = torch.load(save_dir / f"{NAME}_stats.pt")
print("blocks", stats["blocks"])
kae = JordanKoopmanAutoencoder(top_k=256, in_features=768, latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir / f"{NAME}.safetensors")); kae.eval()
eigenvalues = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
print("|eig| quantiles", torch.quantile(eigenvalues.abs(), torch.tensor([0, .01, .5, .99, 1])).tolist(), "n > 2:", (eigenvalues.abs() > 2).sum().item())
predict = lambda m, s: m.decode(m.koopman_forward(m.encode(s), 1))
def get_activity(start):
    phi = kae.encode(start)
    return torch.cat([phi[:, :1].abs(), phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=-1)
def dampen(mode, alpha):
    edited = copy.deepcopy(kae)
    if mode == 0: edited.real_eigenvalue.mul_(alpha)
    else:
        edited.block_real[mode - 1].mul_(alpha); edited.block_imag[mode - 1].mul_(alpha)
    return edited
root = Path(os.getenv("DATASETS_CACHE"))
val = torch.load(root / "imagenet" / "clip_cls_ViT-B-32_val_blocks_5_11.pt")
val_x = (val["states"] - stats["mean"]) / stats["std"]; val_y = val["labels"]
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
clip.eval(); tokenizer = open_clip.get_tokenizer("ViT-B-32")
head = torch.load(save_dir / "zero_shot_head_ViT-B-32.pt")
def embed(states):
    out = clip.visual.ln_post(states * stats["std"] + stats["mean"]) @ clip.visual.proj
    return out / out.norm(dim=-1, keepdim=True)
classify = lambda s: (embed(s) @ head).argmax(dim=-1)
def embed_text(p):
    out = clip.encode_text(tokenizer([p]))[0]; return out / out.norm()
snow_direction = embed_text("a photo in the snow") - embed_text("a photo"); snow_direction /= snow_direction.norm()
paths = sorted((root / "snowy_imagenet").glob("*/*.jpg"))
states = []
hooks = [clip.visual.transformer.resblocks[b].register_forward_hook(lambda _, __, o: states.append(o[:, 0])) for b in stats["blocks"]]
clip.encode_image(torch.stack([preprocess(Image.open(p)) for p in paths]))
for h in hooks: h.remove()
snowy_x = (torch.stack(states, dim=1) - stats["mean"]) / stats["std"]
snowy_y = torch.tensor([SNOW_FOLDERS[p.parent.name] for p in paths])
normal = torch.isin(val_y, torch.tensor(list(SNOW_FOLDERS.values())))
x = torch.cat([val_x[normal], snowy_x]); y = torch.cat([val_y[normal], snowy_y])
is_snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(snowy_x), dtype=torch.bool)])
act = get_activity(x[:, 0])
aucs = torch.tensor([roc_auc_score(is_snowy.numpy(), act[:, m].numpy()) for m in range(act.shape[1])])
top = aucs.argsort(descending=True)[:5]
print("top AUC modes", [(m, round(aucs[m].item(), 3)) for m in top.tolist()])
true = classify(x[:, 1])
print("real CLIP snowy", (true == y)[is_snowy].double().mean().item(), "normal", (true == y)[~is_snowy].double().mean().item())
rows = []
for alpha in [0, .25, .5, .75, 1, 1.5, 2]:
    p = predict(dampen(SNOW_MODE, alpha), x[:, 0]); c = classify(p) == y; s = embed(p) @ snow_direction
    rows.append({"alpha": alpha, "snowy acc": c[is_snowy].double().mean().item(), "normal acc": c[~is_snowy].double().mean().item(), "snowy snow": s[is_snowy].mean().item(), "normal snow": s[~is_snowy].mean().item()})
print(pd.DataFrame(rows).round(3).to_string())
b, a = embed(predict(kae, x[:, 0])), embed(predict(dampen(SNOW_MODE, 0.0), x[:, 0]))
t = embed(x[:, 1])
for label, fit in [("pca_pred", b), ("pca_all", torch.cat([b, a]))]:
    pca = PCA(2).fit(fit.numpy()); pb, pa = pca.transform(b.numpy()), pca.transform(a.numpy())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*pb[~is_snowy].T, s=8, color="#898781", alpha=.5, label="normal")
    ax.scatter(*pb[is_snowy].T, s=12, color="#2a78d6", label="snowy, K as trained")
    ax.scatter(*pa[is_snowy].T, s=12, color="#eb6834", label="snowy, snow eigenvalue 0")
    for i in is_snowy.nonzero()[:, 0].tolist():
        ax.annotate("", pa[i], pb[i], arrowprops=dict(arrowstyle="-", color="#898781", lw=.5))
    ax.legend(); fig.savefig(f"{SP}/{label}.png", dpi=70)
    print(label, "explained", pca.explained_variance_ratio_)
