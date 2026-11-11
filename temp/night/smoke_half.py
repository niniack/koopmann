"""Runs train.main end to end on a small synthetic half-precision cache, on the CPU, without wandb or saving into the real weights folder."""
import os, sys, tempfile, torch
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
os.environ["WANDB_MODE"] = "disabled"
import train
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
os.environ["WEIGHTS_CACHE_REAL"] = os.getenv("WEIGHTS_CACHE")
tmp = tempfile.mkdtemp()
# The zero-shot head lives in the weights folder, so link it into a temporary one
os.makedirs(f"{tmp}/tokae")
os.symlink(f"{os.getenv('WEIGHTS_CACHE')}/tokae/zero_shot_head_ViT-B-32.pt", f"{tmp}/tokae/zero_shot_head_ViT-B-32.pt")
os.environ["WEIGHTS_CACHE"] = tmp
train.get_device = lambda: torch.device("cpu")
def fake_states(encoder, dataset, split, device, blocks, views=1, tokens="cls"):
    n = 30_000 if split == "train" else 2_000
    return (torch.randn(n, 2, 4608) * 3 + 1).half(), torch.randint(0, 1000, (n,))
train.get_states = fake_states
train.main(dataset="imagenet", start_block=6, end_block=12, views=3, num_epochs=1, lr=0.0075, adam_lr=0.00075, lambda_variance=0, lambda_covariance=0,
           lambda_isometric=0.1, ae_dim=2049, ae_hidden=256, ae_layers=2, lambda_embedding=1, lambda_kl=0.03, jordan=True, top_k=256, shared_modes=True,
           tokens="cls_grid_std", name="smoke_half", batch_size=1024)
print("SMOKE OK", os.listdir(f"{tmp}/tokae"))
