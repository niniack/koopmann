import os, sys, torch, numpy as np, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.manual_seed(0); torch.set_num_threads(48)
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_ViT-B-32_val_blocks_5_11.pt")
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
y = val["labels"]
for name, top_k in [("im_jordan256_2k_both", 256), ("im_jordan_2k_v3_both", 0), ("im_jordan_2k_v3", 0)]:
    stats = torch.load(save_dir / f"{name}_stats.pt")
    kae = JordanKoopmanAutoencoder(top_k=top_k, in_features=768, latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(save_dir / f"{name}.safetensors")); kae.eval()
    x = (val["states"] - stats["mean"]) / stats["std"]
    o = clip.visual.ln_post(x[:, 1] * stats["std"] + stats["mean"]) @ clip.visual.proj
    emb = o / o.norm(dim=-1, keepdim=True)
    def coherence(idx):
        s = emb[idx] @ emb[idx].T; n = len(idx); return ((s.sum() - n) / (n * (n - 1))).item()
    threshold = np.quantile([coherence(torch.randperm(len(y))[:16]) for _ in range(1000)], 0.99)
    phi = torch.cat([kae.encode(b) for b in x[:, 0].split(5000)])
    act = torch.cat([phi[:, :1].abs(), phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=-1)
    usage = (act > 0).double().mean(0)
    modes = (usage >= 0.01).nonzero()[:, 0] if top_k else torch.arange(act.shape[1])
    rows = []
    for m in modes.tolist():
        top = act[:, m].topk(16).indices
        rows.append((coherence(top), len(set(y[top].tolist()))))
    r = pd.DataFrame(rows, columns=["coherence", "classes"])
    cross = r[r["classes"] >= 9]
    print(f"{name} (top_k {top_k}): {len(r)} modes, beat random {(r.coherence > threshold).mean():.2f}, median coherence {r.coherence.median():.3f}; "
          f"cross-class (9+ classes in top 16): {len(cross)} modes, beat random {(cross.coherence > threshold).mean():.2f}; single-class (1-3): {(r.classes <= 3).sum()}")
