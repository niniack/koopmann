import sys
import time

import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
names = sys.argv[2:]
target = sys.argv[1]
seen = {}
deadline = time.time() + 14400
while time.time() < deadline:
    api = wandb.Api()
    runs = {r.name: r for r in api.runs("nishantaswani/tokae", order="-created_at", per_page=20) if r.name in names and r.name not in seen}
    for name, run in runs.items():
        rows = run.summary.get("_step")
        if run.state in ("crashed", "failed", "killed"):
            seen[name] = run.state
            print(name, run.id, "ENDED", run.state, flush=True)
        elif target == "started" and rows is not None:
            seen[name] = "started"
            print(name, run.id, run.state, "epoch", rows + 1, "val fvu", round(run.summary["val/one_step_fvu"], 4), "train fvu", round(run.summary["train/one_step_fvu"], 4), "acc", round(run.summary["val/predicted_accuracy"], 4), "linear", round(run.summary.get("baseline/linear_one_step_fvu", -1), 4), "true acc", round(run.summary.get("baseline/true_end_accuracy", -1), 4), flush=True)
        elif target == "finished" and run.state == "finished":
            seen[name] = "finished"
            print(name, run.id, "finished", flush=True)
    if len(seen) == len(names):
        break
    time.sleep(20)
print("watch over", seen, flush=True)
