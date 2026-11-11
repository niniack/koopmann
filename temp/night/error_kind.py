"""When CLIP is right and the KAE is wrong, how close is the KAE's answer to the right class? Text-embedding cosine between classes, vs CLIP's own errors and random pairs."""
import os, sys, torch, numpy as np, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(8); torch.manual_seed(0)
ck, name = Path(sys.argv[1]), sys.argv[2]
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
y = val["labels"]
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
x = (val["states"].float() - mean) / std
def logits(s):
    o = clip.visual.ln_post(s[..., :768] * std[1, :768] + mean[1, :768]) @ clip.visual.proj
    return 100 * (o / o.norm(dim=-1, keepdim=True)) @ head
pred = torch.cat([kae.decode(kae.koopman_forward(kae.encode(b), 1)) for b in x[:, 0].split(5000)])
lk, lc = logits(pred), logits(x[:, 1])
kae_ans, clip_ans = lk.argmax(-1), lc.argmax(-1)
text = head / head.norm(dim=0, keepdim=True)
cos = (text.T @ text)
dog = lambda c: (c >= 151) & (c <= 268)
clip_top = lc.softmax(-1).max(-1).values
lost = (clip_ans == y) & (kae_ans != y)
gained = (clip_ans != y) & (kae_ans == y)
both_wrong_clip = clip_ans != y
rand = torch.randint(0, 1000, (len(y),))
rows = {
    "CLIP right, KAE wrong: KAE answer vs true": (lost, kae_ans),
    "CLIP wrong: CLIP answer vs true": (both_wrong_clip, clip_ans),
    "random class vs true": (torch.ones_like(lost), rand),
}
out = []
for label, (mask, ans) in rows.items():
    c = cos[y[mask], ans[mask]]
    out.append({"": label, "images": int(mask.sum()), "text cosine (median)": c.median().item(),
                "answer is CLIP's runner-up": ((lc.topk(2, -1).indices[:, 1] == ans)[mask].float().mean().item()),
                "true class in KAE top 5": (lk.topk(5, -1).indices == y[:, None]).any(-1)[mask].float().mean().item(),
                "both dogs, of true dogs": (dog(ans) & dog(y))[mask].sum().item() / max(1, dog(y)[mask].sum().item())})
pd.set_option("display.width", 250)
print(pd.DataFrame(out).set_index("").round(3).to_string())
print(f"gained (CLIP wrong, KAE right): {int(gained.sum())}, lost: {int(lost.sum())}")
for lo, hi in [(0, .5), (.5, .9), (.9, 1.01)]:
    m = lost & (clip_top > lo) & (clip_top <= hi)
    print(f"lost with CLIP top prob in ({lo}, {hi}]: {int(m.sum())}, KAE answer = CLIP runner-up {((lc.topk(2, -1).indices[:, 1] == kae_ans)[m].float().mean()):.3f}, text cosine median {cos[y[m], kae_ans[m]].median():.3f}")
