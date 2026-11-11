"""
Reads every run of wandb project nishantaswani/tokae (read only) and stores what the review needs in data/wandb_raw.json:
config, tags, summary numbers and the logged curve of every validation and training number.

Usage: fetch_runs.py
"""

import json
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
OUT = Path(__file__).parent / "data" / "wandb_raw.json"


def scalars(mapping):
    """The plain numbers, strings and short lists of a wandb config or summary."""

    return {key: value for key, value in dict(mapping).items() if isinstance(value, (int, float, str, bool, list, type(None))) and not key.startswith("_wandb")}


def get_curve(run):
    """Every logged row of the run, keeping only the columns that hold numbers."""

    rows = []
    for row in run.scan_history(page_size=1_000):
        rows.append({key: value for key, value in row.items() if isinstance(value, (int, float)) and value is not None})
    return rows


if __name__ == "__main__":
    api = wandb.Api(timeout=90)
    runs = []
    for run in api.runs("nishantaswani/tokae", order="+created_at", per_page=200):
        summary = scalars(run.summary._json_dict)
        runs.append({"id": run.id, "name": run.name, "state": run.state, "created": run.created_at, "tags": list(run.tags),
                     "runtime": summary.get("_runtime"), "config": scalars(run.config), "summary": summary, "curve": get_curve(run)})  # fmt: skip
        print(f"{run.id} {run.name[:32]:32s} {run.state:9s} {run.created_at} rows {len(runs[-1]['curve'])}", flush=True)
    OUT.write_text(json.dumps(runs))
    print(f"{len(runs)} runs written to {OUT}")
