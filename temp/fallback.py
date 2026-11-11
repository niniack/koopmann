"""Speed with a fallback: trust the KAE when it looks confident, otherwise finish with CLIP's blocks 7-12. How well can we tell?"""
import os, sys, torch, numpy as np, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.manual_seed(0); torch.set_num_threads(16)
name, ck = sys.argv[1], Path(sys.argv[2])
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
x, y = (val["states"].float() - mean) / std, val["labels"]
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
def logits(s):
    o = clip.visual.ln_post(s[..., :768] * std[1, :768] + mean[1, :768]) @ clip.visual.proj
    return 100 * (o / o.norm(dim=-1, keepdim=True)) @ head
z6 = torch.cat([kae.encode(b) for b in x[:, 0].split(5000)])
pred = torch.cat([kae.decode(kae.koopman_forward(b, 1)) for b in z6.split(5000)])
recon6 = torch.cat([kae.decode(b) for b in z6.split(5000)])
lk, lc = logits(pred), logits(x[:, 1])
pk = lk.softmax(-1); top = pk.topk(2, dim=-1).values
kae_ans, clip_ans = lk.argmax(-1), lc.argmax(-1)
agree = (kae_ans == clip_ans).numpy()
feats = {"KAE top probability": top[:, 0], "KAE margin (top1 - top2)": top[:, 0] - top[:, 1],
         "minus KAE entropy": (pk * pk.clamp_min(1e-12).log()).sum(-1),
         "minus block-6 reconstruction error": -((recon6 - x[:, 0]).pow(2).sum(-1) / x[:, 0].pow(2).sum(-1))}
print(f"{name}: KAE zero-shot {(kae_ans == y).double().mean():.3f}, CLIP {(clip_ans == y).double().mean():.3f}, agreement {agree.mean():.3f}")
print("How well each signal predicts 'KAE agrees with CLIP' (AUC):", {k: round(roc_auc_score(agree, v.numpy()), 3) for k, v in feats.items()})
F = torch.stack(list(feats.values()), 1).numpy()
half = np.arange(len(y)) % 2 == 0
gate = LogisticRegression(max_iter=1000).fit(F[half], agree[half])
score = gate.predict_proba(F)[:, 1]
print(f"Small gate on all four signals, trained on half of val: AUC {roc_auc_score(agree[~half], score[~half]):.3f} on the other half")
# Keep the most confident fraction on the KAE path, send the rest to CLIP's blocks 7-12
s, kc, cc = score[~half], (kae_ans == y).numpy()[~half], (clip_ans == y).numpy()[~half]
ag = agree[~half]
order = np.argsort(-s)
rows = []
for q in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
    k = order[: int(q * len(order))]
    on_kae = np.zeros(len(order), bool); on_kae[k] = True
    acc = np.where(on_kae, kc, cc).mean()
    rows.append({"share answered by KAE": q, "KAE agrees with CLIP on those": ag[on_kae].mean(), "overall zero-shot": acc,
                 "overall agrees with CLIP": np.where(on_kae, ag, True).mean(), "compute vs full CLIP": 0.5 + 0.016 + 0.5 * (1 - q)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
