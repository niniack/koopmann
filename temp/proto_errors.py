import os, sys, collections
from pathlib import Path
import numpy as np, open_clip, pandas as pd, torch
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.manual_seed(0)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
SP = sys.argv[1]
NAME = "im_jordan256_2k_both"
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
stats = torch.load(save_dir / f"{NAME}_stats.pt")
kae = JordanKoopmanAutoencoder(top_k=256, in_features=768, latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir / f"{NAME}.safetensors")); kae.eval()
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_ViT-B-32_val_blocks_5_11.pt")
x, y = (val["states"] - stats["mean"]) / stats["std"], val["labels"]
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
clip.eval()
head = torch.load(save_dir / "zero_shot_head_ViT-B-32.pt")
names = open_clip.IMAGENET_CLASSNAMES
def embed(s):
    o = clip.visual.ln_post(s * stats["std"] + stats["mean"]) @ clip.visual.proj
    return o / o.norm(dim=-1, keepdim=True)
classify = lambda s: (embed(s) @ head).argmax(dim=-1)

phi = torch.cat([kae.encode(b) for b in x[:, 0].split(5000)])
out = kae.koopman_forward(phi, 1)  # K phi, shape [images, 2049]
activity = torch.cat([phi[:, :1].abs(), phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=-1)
before = torch.cat([classify(kae.decode(b)) for b in out.split(5000)])
clip_answer = torch.cat([classify(b) for b in x[:, 1].split(5000)])
true_emb = torch.cat([embed(b) for b in x[:, 1].split(5000)])
print("KAE acc", (before == y).double().mean().item(), "CLIP acc", (clip_answer == y).double().mean().item())

# ---- coherence of each mode's top 16 images, in CLIP's embedding space
def coherence(idx):
    e = true_emb[idx]; s = e @ e.T; n = len(idx)
    return ((s.sum() - n) / (n * (n - 1))).item()
random_sets = [coherence(torch.randperm(len(y))[:16]) for _ in range(1000)]
class_sets = []
for c in torch.randperm(1000)[:200].tolist():
    idx = (y == c).nonzero()[:, 0]; class_sets.append(coherence(idx[torch.randperm(len(idx))[:16]]))
threshold = np.quantile(random_sets, 0.99)
print(f"random 16-sets: median {np.median(random_sets):.3f}, 99th pct {threshold:.3f}; same-class 16-sets: median {np.median(class_sets):.3f}, 5th pct {np.quantile(class_sets, .05):.3f}")
usage = (activity > 0).double().mean(0)
rows = []
for m in range(activity.shape[1]):
    if usage[m] < 0.01: continue
    top16 = activity[:, m].topk(16).indices; top50 = activity[:, m].topk(50).indices
    rows.append({"mode": m, "usage": usage[m].item(), "coherence": coherence(top16), "classes in top 16": len(set(y[top16].tolist())),
                 "classes in top 50": len(set(y[top50].tolist())), "top class share (50)": torch.bincount(y[top50]).max().item() / 50})
modes = pd.DataFrame(rows).set_index("mode")
modes["coherent"] = modes["coherence"] > threshold
print(f"modes on >= 1% of images: {len(modes)}; coherent: {modes['coherent'].mean():.3f}")
for lo, hi in [(1, 3), (4, 8), (9, 16)]:
    sel = modes[(modes["classes in top 16"] >= lo) & (modes["classes in top 16"] <= hi)]
    print(f"  {lo}-{hi} classes in top 16: {len(sel)} modes, coherent {sel['coherent'].mean():.3f}, median coherence {sel['coherence'].median():.3f}")
print(modes.loc[[903, 878, 647, 818, 281, 271, 310, 94, 618, 73, 662, 234, 767, 319, 813, 912, 358]].round(3).to_string())
modes.to_pickle(f"{SP}/mode_coherence.pkl")

# ---- remove each mode (eigenvalue 0) and see which errors it fixes, on two halves of validation
half = torch.randperm(len(y)) % 2
records = []
for m in range(activity.shape[1]):
    on = (activity[:, m] > 0).nonzero()[:, 0]
    if len(on) < 50: continue
    edited = out[on].clone()
    if m == 0: edited[:, 0] = 0
    else: edited[:, 2 * m - 1 : 2 * m + 1] = 0
    after = classify(kae.decode(edited))
    b, t = before[on], y[on]
    fixed, broken = (b != t) & (after == t), (b == t) & (after != t)
    for h in [0, 1]:
        mask = half[on] == h
        records.append({"mode": m, "half": h, "on": int(mask.sum()), "fixed": int((fixed & mask).sum()), "broken": int((broken & mask).sum()),
                        "fixed, CLIP right": int((fixed & mask & (clip_answer[on] == t)).sum())})
    if m % 200 == 0: print("mode", m, flush=True)
    torch.save({"m": m}, f"{SP}/progress.pt")
    if m == 0: store = {}
    store[m] = {"on": on, "after": after}
torch.save(store, f"{SP}/removal_after.pt")
rec = pd.DataFrame(records)
rec["net"] = rec["fixed"] - rec["broken"]
a, b = rec[rec.half == 0].set_index("mode"), rec[rec.half == 1].set_index("mode")
best = a["net"].sort_values(ascending=False).index[:15]
table = pd.DataFrame({"usage": usage[best].numpy(), "net, pick half": a.loc[best, "net"], "fixed, held-out": b.loc[best, "fixed"], "broken, held-out": b.loc[best, "broken"],
                      "net, held-out": b.loc[best, "net"], "fixed where CLIP right": b.loc[best, "fixed, CLIP right"]}, index=best)
print(table.to_string())
print("held-out net summed over all modes with net > 0 on pick half:", b.loc[a.index[a["net"] > 0], "net"].sum(), "| modes:", (a["net"] > 0).sum())
print("correlation of net between halves:", np.corrcoef(a["net"], b.loc[a.index, "net"])[0, 1])
rec.to_pickle(f"{SP}/removal.pkl")
# what each top mode fixes: most common (true, before -> after) on held-out half
for m in best[:10]:
    on, after = store[m]["on"], store[m]["after"]
    t, bb = y[on], before[on]
    mask = (half[on] == 1) & (bb != t) & (after == t)
    pairs = collections.Counter((names[bb[i]], names[t[i]]) for i in mask.nonzero()[:, 0].tolist()).most_common(4)
    print(m, "fixes:", "; ".join(f"{w} -> {r} x{n}" for (w, r), n in pairs))
