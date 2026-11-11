import os, sys, collections
from pathlib import Path
import open_clip, torch
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
NAME, MODE = "im_jordan256_2k_both", 903
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
stats = torch.load(save_dir / f"{NAME}_stats.pt")
kae = JordanKoopmanAutoencoder(top_k=256, in_features=768, latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir / f"{NAME}.safetensors")); kae.eval()
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_ViT-B-32_val_blocks_5_11.pt")
x, y = (val["states"] - stats["mean"]) / stats["std"], val["labels"]
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(save_dir / "zero_shot_head_ViT-B-32.pt"); names = open_clip.IMAGENET_CLASSNAMES
def classify(s):
    o = clip.visual.ln_post(s * stats["std"] + stats["mean"]) @ clip.visual.proj
    return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(dim=-1)
phi = torch.cat([kae.encode(b) for b in x[:, 0].split(5000)])
on = (phi[:, 2 * MODE - 1 : 2 * MODE + 1].norm(dim=-1) > 0).nonzero()[:, 0]
out = kae.koopman_forward(phi[on], 1)
before = classify(kae.decode(out)); out[:, 2 * MODE - 1 : 2 * MODE + 1] = 0; after = classify(kae.decode(out))
t = y[on]
for label, mask in [("broken", (before == t) & (after != t)), ("fixed", (before != t) & (after == t))]:
    c = collections.Counter(names[i] for i in t[mask].tolist())
    print(label, int(mask.sum()), "by true class:", "; ".join(f"{k} {v}" for k, v in c.most_common(10)))
    if label == "broken":
        moves = collections.Counter((names[t[i]], names[after[i]]) for i in mask.nonzero()[:, 0].tolist())
        print("  broken moves:", "; ".join(f"{a} -> {b} x{n}" for (a, b), n in moves.most_common(8)))
