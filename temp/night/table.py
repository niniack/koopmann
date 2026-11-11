"""One row per overnight run, read from wandb by the ids in the night logs. Final epoch, plus the epoch with the best val zero-shot."""
import re
import sys
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
NIGHT = Path(__file__).parent
api = wandb.Api()
names = ["im_iso_2x4k", "im_iso_2x4k_kl0.1", "im_grid_2x4k_r2", "im_iso_2x4k_f25", "im_iso_2x4k_f50", "im_iso_2x4k_s1", "im_iso_2x4k_wd", "im_w21_wd_2x4k", "im_mlpk_2x4k", "im_iso_2x8k", "im_dino_2x4k", "im_dino_w21_wd"]
keys = ["zero_shot_accuracy", "clip_agreement", "kl_loss", "one_step_fvu", "reconstruction_fvu", "latent_error_per_image"]
rows = []
for name in names:
    log = NIGHT / f"{name}.log"
    found = re.findall(r"runs/(\w+)", log.read_text()) if log.exists() else []
    if not found:
        continue
    run = api.run(f"nishantaswani/tokae/{found[0]}")
    h = run.history(keys=[f"{s}/{k}" for s in ["val", "train"] for k in keys], pandas=True, samples=500)
    if not len(h):
        continue
    best = h["val/zero_shot_accuracy"].idxmax()
    last = h.iloc[-1]
    rows.append(
        {
            "run": name,
            "id": found[0],
            "state": run.state,
            "epochs": f"{len(h)}/{run.config.get('num_epochs')}",
            "best val zs (epoch)": f"{h.loc[best, 'val/zero_shot_accuracy']:.3f} ({best + 1})",
            "val zs": last["val/zero_shot_accuracy"],
            "val agree": last["val/clip_agreement"],
            "train zs": last["train/zero_shot_accuracy"],
            "train agree": last["train/clip_agreement"],
            "val KL": last["val/kl_loss"],
            "train KL": last["train/kl_loss"],
            "val state": last["val/one_step_fvu"],
            "val recon": last["val/reconstruction_fvu"],
            "val latent/img": last.get("val/latent_error_per_image"),
        }
    )
table = pd.DataFrame(rows).set_index("run")
pd.set_option("display.width", 250)
print(table.round(4).to_string())
