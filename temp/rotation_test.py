"""Do the complex (rotating) modes earn their second dimension? Post-hoc rotation removal, and how 2D each mode's use is."""
import os, sys, copy, torch, numpy as np, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs, compute_fvu
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
root = Path(os.getenv("DATASETS_CACHE"))
val = torch.load(root / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
y = val["labels"]
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
for name, ck in [("im_cc3m_iso0.1_2x4k_e20", "ckpt_live"), ("im_cc3m_2x4k_e20", "ckpt_live"), ("im_std_shared", "ckpt")]:
    ck = Path(sys.argv[1]) / ck
    stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
    kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats.get("hidden_config", [2048]), bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
    x = (val["states"].float() - mean) / std
    def classify(s):
        c = s[..., :768] * std[1, :768] + mean[1, :768]
        o = clip.visual.ln_post(c) @ clip.visual.proj
        return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(-1)
    def score(m):
        p = torch.cat([m.decode(m.koopman_forward(m.encode(b), 1)) for b in x[:, 0].split(5000)])
        return (classify(p) == y).double().mean().item(), compute_fvu(p[:, None], x[:, 1:]).item()
    lam = torch.complex(kae.block_real, kae.block_imag)
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
    z6 = phi[:, 0, 1:].unflatten(-1, (-1, 2))
    w = z6.norm(dim=-1).pow(2).sum(0)  # each mode's block-6 energy
    angle = lam.angle().abs().rad2deg()
    print(f"== {name}")
    print(f"   rotation angle |arg λ|: median {angle.median():.1f}°, energy-weighted median {angle[w.argsort()][(w[w.argsort()].cumsum(0) / w.sum() >= 0.5).nonzero()[0]].item():.1f}°, "
          f"share of block-6 energy in modes rotating > 15°: {(w[angle > 15].sum() / w.sum()):.2f}")
    # How 2D is each mode's use? Fraction of its coefficient variance (over images that keep it) along its main direction
    flat = []
    for m in range(z6.shape[1]):
        on = z6[:, m].norm(dim=-1) > 0
        if on.sum() < 50: continue
        c = z6[on, m]
        ev = torch.linalg.eigvalsh((c.T @ c) / len(c))  # second moment, so a fixed direction counts as 1D
        flat.append((ev[-1] / ev.sum()).item())
    flat = np.array(flat)
    print(f"   {len(flat)} modes kept on 50+ images: share of each mode's 2D spread along one direction: median {np.median(flat):.3f}; >0.95 for {np.mean(flat > 0.95):.0%}, >0.99 for {np.mean(flat > 0.99):.0%}")
    zs, fvu = score(kae)
    no_rot = copy.deepcopy(kae); no_rot.block_real.copy_(lam.abs()); no_rot.block_imag.zero_()
    zs2, fvu2 = score(no_rot)
    real_only = copy.deepcopy(kae); real_only.block_imag.zero_()
    zs3, fvu3 = score(real_only)
    print(f"   zero-shot / state FVU: as trained {zs:.4f} / {fvu:.4f}; rotation removed, |λ| kept {zs2:.4f} / {fvu2:.4f}; imaginary part dropped {zs3:.4f} / {fvu3:.4f}")
