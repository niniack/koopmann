"""Where does the KAE lose accuracy against CLIP? Autoencoding block 12 alone, vs predicting it from block 6, on val and on training images."""
import os, sys, torch, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs, compute_fvu
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.manual_seed(0); torch.set_num_threads(24)
name, ck = sys.argv[1], Path(sys.argv[2])
root = Path(os.getenv("DATASETS_CACHE")) / "imagenet"
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
def answers(s):
    o = clip.visual.ln_post(s[..., :768] * std[1, :768] + mean[1, :768]) @ clip.visual.proj
    return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(-1)
val = torch.load(root / "clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
tr = torch.load(root / "clip_cls_patch_std_ViT-B-32_train_3_views_blocks_5_11.pt", mmap=True)
idx = torch.randperm(len(tr["labels"]))[:30000].sort().values
sets = {"val": ((val["states"].float() - mean) / std, val["labels"]),
        "train (seen crops)": ((tr["states"][idx].float() - mean) / std, tr["labels"][idx])}
# A dense linear map between latents, fit on other training images, as an upper bound for any linear K
fit_idx = torch.randperm(len(tr["labels"]))[:60000]
xf = (tr["states"][fit_idx].float() - mean) / std
pf = torch.cat([encode_pairs(kae, b) for b in xf.split(5000)])
A = torch.nn.functional.pad(pf[:, 0], (0, 1), value=1).double()
dense = torch.linalg.solve(A.T @ A + 10 * torch.eye(A.shape[1], dtype=A.dtype), A.T @ pf[:, 1].double()).float()
rows = {}
for split, (x, y) in sets.items():
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
    free12 = torch.cat([kae.encode(b) for b in x[:, 1].split(5000)])
    clip_ans = answers(x[:, 1])
    def row(states):
        a = answers(states)
        return {"zero-shot": (a == y).double().mean().item(), "agrees with CLIP": (a == clip_ans).double().mean().item(),
                "CLS FVU": compute_fvu(states[:, None, :768], x[:, 1:, :768]).item()}
    rows[(split, "CLIP")] = row(x[:, 1])
    rows[(split, "autoencode block 12, own TopK")] = row(kae.decode(free12))
    rows[(split, "autoencode block 12, block 6's modes")] = row(kae.decode(phi[:, 1]))
    rows[(split, "predict: decode(K · z6)")] = row(kae.decode(kae.koopman_forward(phi[:, 0], 1)))
    rows[(split, "predict with a dense linear map in latent")] = row(kae.decode(torch.nn.functional.pad(phi[:, 0], (0, 1), value=1) @ dense))
print(name)
print(pd.DataFrame(rows).T.round(3).to_string())
