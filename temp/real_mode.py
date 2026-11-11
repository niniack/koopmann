"""What the single real-eigenvalue mode (latent coordinate 0) does, per checkpoint."""
import os, sys, copy, torch, numpy as np, open_clip, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
from datasets import load_dataset
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
root = Path(os.getenv("DATASETS_CACHE"))
val = torch.load(root / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
y = val["labels"]
names = open_clip.IMAGENET_CLASSNAMES
for name, ck in [("im_cc3m_2x4k_e20", "ckpt_live"), ("im_shared_noiso_2x4k", "ckpt"), ("im_std_shared", "ckpt")]:
    ck = Path(sys.argv[1]) / ck
    stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
    kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats.get("hidden_config", [2048]), bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
    x = (val["states"].float() - mean) / std
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
    z0 = phi[:, :, 0]
    on = z0[:, 0] != 0
    blocks = phi[:, 0, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)
    kept_block_size = blocks[blocks > 0].median()
    def classify(s):
        c = s[..., :768] * std[1, :768] + mean[1, :768]
        o = clip.visual.ln_post(c) @ clip.visual.proj
        return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(-1)
    def predict(m, s):
        return torch.cat([m.decode(m.koopman_forward(m.encode(b), 1)) for b in s.split(5000)])
    base = classify(predict(kae, x[on, 0])) == y[on]
    edited = copy.deepcopy(kae); edited.real_eigenvalue.zero_()
    after = classify(predict(edited, x[on, 0])) == y[on]
    # Is it a global quantity? Correlate z0 at block 6 with simple image statistics
    cls_norm6 = val["states"][:, 0, :768].float().norm(dim=-1)
    patch_std6 = val["states"][:, 0, 1536:].float().mean(dim=-1)
    r = lambda a, b: np.corrcoef(a.numpy(), b.numpy())[0, 1]
    print(f"== {name}: real eigenvalue {kae.real_eigenvalue.item():.3f}; kept for {on.double().mean():.1%} of val; "
          f"block-12 energy share {(z0[:, 1].pow(2).sum() / phi[:, 1].pow(2).sum()):.4f}; "
          f"median |z0| when kept {z0[on, 0].abs().median():.3f} vs median kept 2D mode size {kept_block_size:.3f}; "
          f"sign + {(z0[on, 0] > 0).double().mean():.2f}")
    print(f"   zero-shot where kept {base.double().mean():.3f} -> {after.double().mean():.3f} with it removed ({int((~base & after).sum())} fixed, {int((base & ~after).sum())} broken)")
    if on.sum() > 10:
        print(f"   corr of z0 (block 6, where kept) with: block-6 CLS norm {r(z0[on, 0], cls_norm6[on]):.2f}, mean patch std {r(z0[on, 0], patch_std6[on]):.2f}")
        top = z0[:, 0].abs().topk(min(16, int(on.sum()))).indices
        print(f"   top-16 classes: {[names[c][:18] for c in y[top].tolist()]}")
