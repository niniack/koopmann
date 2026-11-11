"""Did KL 0.1 close the gap where CLIP is unsure? Agreement with CLIP on val, binned by CLIP's top probability, control vs KL 0.1."""
import os, sys, torch, numpy as np, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(8)
ck = Path(sys.argv[1])
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
y = val["labels"]
out = {}
for name in sys.argv[2:]:
    stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
    kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
    x = (val["states"].float() - mean) / std
    def logits(s):
        o = clip.visual.ln_post(s[..., :768] * std[1, :768] + mean[1, :768]) @ clip.visual.proj
        return 100 * (o / o.norm(dim=-1, keepdim=True)) @ head
    pred = torch.cat([kae.decode(kae.koopman_forward(kae.encode(b), 1)) for b in x[:, 0].split(5000)])
    lk, lc = logits(pred), logits(x[:, 1])
    out[name] = (lk, lc)
lc = out[sys.argv[2]][1]
pc = lc.softmax(-1)
clip_top = pc.max(-1).values
clip_ans = lc.argmax(-1)
bins = pd.cut(clip_top.numpy(), [0, 0.3, 0.5, 0.7, 0.9, 1.0])
rows = {}
for name, (lk, _) in out.items():
    kae_ans = lk.argmax(-1)
    agree = (kae_ans == clip_ans).numpy()
    kl = (pc * (pc.clamp_min(1e-12).log() - lk.log_softmax(-1))).sum(-1).numpy()
    top5 = (lk.topk(5, -1).indices == clip_ans[:, None]).any(-1).numpy()
    frame = pd.DataFrame({"bin": bins, "agree": agree, "KL": kl, "CLIP answer in KAE top 5": top5, "KAE right": (kae_ans == y).numpy(), "CLIP right": (clip_ans == y).numpy()})
    g = frame.groupby("bin", observed=True)
    rows[name] = g[["agree", "KL", "CLIP answer in KAE top 5", "KAE right"]].mean()
    print(f"{name}: agree {agree.mean():.4f}, KAE zero-shot {(kae_ans == y).double().mean():.4f}, mean KL {kl.mean():.4f}")
counts = frame.groupby("bin", observed=True).size().rename("images")
clip_right = frame.groupby("bin", observed=True)["CLIP right"].mean().rename("CLIP right")
table = pd.concat([counts, clip_right] + [rows[n].add_prefix(n.replace("im_iso_2x4k", "ctrl") + ": ") for n in rows], axis=1)
pd.set_option("display.width", 250)
print(table.round(3).T.to_string())
