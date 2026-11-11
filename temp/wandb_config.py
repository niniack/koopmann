"""Print a wandb run's state and a few config keys. Usage: wandb_config.py <run id> <key> [<key> ...]"""
import sys

import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
run = wandb.Api().run(f"nishantaswani/tokae/{sys.argv[1]}")
print(run.name, run.state, {key: run.config.get(key) for key in sys.argv[2:]})
