"""Print config and best/final metrics for tokae runs. Usage: summary.py <run id>..."""
import sys
import wandb
from dotenv import load_dotenv

load_dotenv(".env")
api = wandb.Api()
CONFIG = ["lambda_kl", "lambda_isometric", "lambda_embedding", "ae_hidden", "ae_layers", "tokens", "num_epochs", "train_fraction", "weight_decay", "shared_modes", "mlp", "jordan", "top_k", "views", "lr", "adam_lr", "dataset"]
KEYS = ["zero_shot_accuracy", "clip_agreement", "one_step_fvu", "reconstruction_fvu", "latent_error_per_image", "latent_energy_top1pct", "kl_loss", "embedding_loss"]
for rid in sys.argv[1:]:
    run = api.run(f"nishantaswani/tokae/{rid}")
    print(f"== {run.name} ({rid}) {run.state}")
    print("   config:", {k: run.config.get(k) for k in CONFIG if k in run.config})
    s = run.summary
    print("   best:", {k: s.get(k) for k in ["best/epoch", "best/val_accuracy", "best/val_one_step_fvu"]})
    print("   baseline:", {k.split('/')[1]: round(s.get(k), 4) for k in s.keys() if k.startswith("baseline/") and s.get(k) is not None})
    h = run.history(keys=["val/zero_shot_accuracy"] + [f"train/{k}" for k in KEYS], pandas=True, samples=200)
    hv = run.history(keys=[f"val/{k}" for k in KEYS], pandas=True, samples=200)
    for split, frame in [("val", hv), ("train", h)]:
        cols = [c for c in frame.columns if c.startswith(split + "/")]
        if len(frame):
            print(f"   {split} per epoch:")
            print(frame[cols].rename(columns=lambda c: c.split("/")[1][:14]).round(4).to_string())
