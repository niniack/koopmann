"""Reconstruction and prediction errors of several checkpoints, all scored in the reference's standardized space."""
import os, sys, torch, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs, compute_fvu
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
ck = Path(sys.argv[1])
raw = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")["states"]
ref = torch.load(ck / f"{os.getenv('REF', 'im_std_kl0.03')}_stats.pt")
ref = {k: ref[k].expand(2, -1) for k in ["mean", "std"]}
target = (raw - ref["mean"]) / ref["std"]
rows = {}
for name in sys.argv[2:]:
    stats = torch.load(ck / f"{name}_stats.pt")
    mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
    x = (raw - mean) / std
    kae = JordanKoopmanAutoencoder(256, shared_modes=stats.get("shared_modes", False), in_features=2304, latent_features=2049,
                                   hidden_config=stats.get("hidden_config", [2048]), bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
    recons = kae.decode(phi.flatten(0, 1)).unflatten(0, phi.shape[:2])
    pred = kae.decode(kae.koopman_forward(phi[:, 0], 1))
    # back to raw activations, then into the reference's space
    recons = (recons * std + mean - ref["mean"]) / ref["std"]
    pred = (pred * std[1] + mean[1] - ref["mean"][1]) / ref["std"][1]
    rec = compute_fvu(recons, target)
    rows[name] = {"recon block 6": rec[0].item(), "recon block 12": rec[1].item(),
                  "prediction (state FVU)": compute_fvu(pred[:, None], target[:, 1:]).item(),
                  "prediction, CLS only": compute_fvu(pred[:, None, :768], target[:, 1:, :768]).item()}
print(pd.DataFrame(rows).T.round(4).to_string())
