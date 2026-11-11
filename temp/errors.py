"""What kind of mistakes does the KAE make that CLIP doesn't? Near misses, fine-grained classes, embedding closeness."""
import os, sys, torch, numpy as np, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(24)
name, ck = sys.argv[1], Path(sys.argv[2])
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
x, y = (val["states"].float() - mean) / std, val["labels"]
def emb(s):
    o = clip.visual.ln_post(s[..., :768] * std[1, :768] + mean[1, :768]) @ clip.visual.proj
    return o / o.norm(dim=-1, keepdim=True)
pred = torch.cat([kae.decode(kae.koopman_forward(kae.encode(b), 1)) for b in x[:, 0].split(5000)])
ek, ec = emb(pred), emb(x[:, 1])
lk, lc = ek @ head, ec @ head
ak, ac = lk.argmax(-1), lc.argmax(-1)
cos = (ek * ec).sum(-1)
top5k = lk.topk(5, -1).indices
print(f"{name}: zero-shot KAE {(ak == y).double().mean():.3f}, CLIP {(ac == y).double().mean():.3f}; agreement top-1 {(ak == ac).double().mean():.3f}; "
      f"CLIP's answer in KAE's top 5: {(top5k == ac[:, None]).any(-1).double().mean():.3f}; true label in KAE top 5 {(top5k == y[:, None]).any(-1).double().mean():.3f} (CLIP {(lc.topk(5, -1).indices == y[:, None]).any(-1).double().mean():.3f})")
print(f"cosine between KAE and CLIP final embeddings: median {cos.median():.3f}, 10th percentile {cos.quantile(0.1):.3f}; when they agree {cos[ak == ac].median():.3f}, disagree {cos[ak != ac].median():.3f}")
# CLIP's confidence on the images where the KAE disagrees
pc = lc.mul(100).softmax(-1).max(-1).values
print(f"CLIP's own top probability: median {pc.median():.3f}; where KAE agrees {pc[ak == ac].median():.3f}, disagrees {pc[ak != ac].median():.3f}")
# Kinds of classes: ImageNet index ranges for dogs (151-268), birds (7-24, 80-100, 127-146), and the rest
groups = {"dogs (118 classes)": range(151, 269), "birds": [*range(7, 25), *range(80, 101), *range(127, 147)], "other": None}
rows = {}
used = set()
for g, r in groups.items():
    cls = torch.tensor(list(r)) if r is not None else torch.tensor([c for c in range(1000) if c not in used])
    used |= set(cls.tolist())
    m = torch.isin(y, cls)
    rows[g] = {"images": int(m.sum()), "CLIP": (ac == y)[m].double().mean().item(), "KAE": (ak == y)[m].double().mean().item(),
               "agreement": (ak == ac)[m].double().mean().item(), "KAE wrong but CLIP right": ((ak != y) & (ac == y))[m].double().mean().item()}
print(pd.DataFrame(rows).T.round(3).to_string())
# Of the KAE's disagreements, how often is its answer a "sibling" (CLIP's 2nd-5th choice)?
dis = ak != ac
sib = (lc.topk(5, -1).indices[:, 1:] == ak[:, None]).any(-1)
print(f"When the KAE disagrees with CLIP, its answer is among CLIP's choices 2-5 in {sib[dis].double().mean():.3f} of cases")
