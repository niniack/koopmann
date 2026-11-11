"""Per-epoch validation curves of a few runs, from wandb."""
import sys

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
api = wandb.Api(timeout=60)
KEYS = ["val/zero_shot_accuracy", "val/clip_agreement", "val/one_step_fvu", "val/reconstruction_fvu", "train/zero_shot_accuracy", "train/one_step_fvu"]
pd.set_option("display.width", 250)
for run_id in sys.argv[1:]:
    run = api.run(f"nishantaswani/tokae/{run_id}")
    history = run.history(keys=KEYS, pandas=True)
    print(f"=== {run.name} ({run_id}), state {run.state}, epochs {run.config.get('num_epochs')}, wd {run.config.get('weight_decay')}, hidden {run.config.get('ae_hidden')}")
    print(history.drop(columns=["_step"], errors="ignore").round(4).to_string())
