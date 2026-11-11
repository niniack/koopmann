"""Reconstruction and prediction FVU split into the three parts of the input: CLS, patch mean, patch std."""
import os, sys, torch, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs, compute_fvu
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
ck = Path(sys.argv[1])
raw = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt", mmap=True)["states"][::2].float()
parts = {"CLS": slice(0, 768), "patch mean": slice(768, 1536), "patch std": slice(1536, 2304)}
rows = {}
for name in sys.argv[2:]:
    stats = torch.load(ck / f"{name}_stats.pt")
    kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049,
                                   hidden_config=stats.get("hidden_config", [2048]), bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
    x = (raw - stats["mean"]) / stats["std"]
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
    recons = kae.decode(phi.flatten(0, 1)).unflatten(0, phi.shape[:2])
    pred = kae.decode(kae.koopman_forward(phi[:, 0], 1))
    for p, s in parts.items():
        rec = compute_fvu(recons[..., s], x[..., s])
        rows[(name, p)] = {"recon block 6": rec[0].item(), "recon block 12": rec[1].item(),
                           "prediction block 12": compute_fvu(pred[:, None, s], x[:, 1:, s]).item(),
                           "share of block-12 variance": (x[:, 1, s].var(0).sum() / x[:, 1].var(0).sum()).item()}
print(pd.DataFrame(rows).T.round(3).to_string())
