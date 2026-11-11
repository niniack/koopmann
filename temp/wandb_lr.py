"""Read only: which learning rates the project's runs used, and the per-epoch accuracy of the runs with 12 or more epochs."""
import collections

import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
api = wandb.Api(timeout=60)
runs = list(api.runs("nishantaswani/tokae", order="-created_at", per_page=200))
print(f"{len(runs)} runs")
counts = collections.Counter((run.config.get("lr"), run.config.get("adam_lr"), run.config.get("num_epochs")) for run in runs)
print("(lr, adam_lr, num_epochs): runs")
for key, count in sorted(counts.items(), key=lambda item: -item[1]):
    print(f"  {key}: {count}")
print("\nruns whose lr is not 0.0075 (most recent first, up to 40): id | name | state | lr | adam_lr | epochs | dataset | KAE accuracy | agreement")
shown = 0
for run in runs:
    if run.config.get("lr") != 0.0075 and shown < 40:
        s = run.summary
        number = lambda key: "-" if s.get(key) is None else f"{s.get(key):.4f}"
        print(f"  {run.id} | {run.name[:30]} | {run.state} | {run.config.get('lr')} | {run.config.get('adam_lr')} | {run.config.get('num_epochs')} | {run.config.get('dataset')} | {number('val/zero_shot_accuracy')} | {number('val/clip_agreement')}")
        shown += 1
print("\nruns with 12 or more epochs: per-epoch val/zero_shot_accuracy")
for run in runs:
    epochs = run.config.get("num_epochs") or 0
    if epochs >= 12:
        history = run.history(keys=["val/zero_shot_accuracy"], pandas=True)
        if len(history) == 0:
            continue
        values = history["val/zero_shot_accuracy"].tolist()
        print(f"  {run.id} | {run.name[:30]} | {run.state} | lr {run.config.get('lr')} | epochs {epochs} | logged {len(values)}")
        print("     " + " ".join(f"{v:.4f}" for v in values))
