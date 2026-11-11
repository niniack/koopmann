"""
Builds the run table of the review from data/wandb_raw.json (written by fetch_runs.py) and the training logs.

Writes:
  data/runs.csv       one row per run: what was trained and its last and best numbers
  data/curves.csv     accuracy, agreement and state FVU at every logged epoch or step
  data/log_check.csv  wandb's last numbers next to the same numbers in the log files

"accuracy" is always `val/zero_shot_accuracy` (CLIP's zero-shot head on the decoded block-12 state; for DINOv3 its ImageNet
linear probe). Runs of 7 Nov did not log it; they only have `val/predicted_accuracy` (a ridge classifier fitted on real
block-12 states), kept in its own column `fitted_accuracy`.

Usage: build_runs.py
"""

import csv
import json
import re
from datetime import datetime, timedelta

from common import DATA, SCRATCH

IMAGES = {"imagenet": 1_281_167, "imagenette": 9_469}
STATE_SIZE = {"cls": 768, "cls_patch": 1_536, "cls_patch_std": 2_304, "cls_grid_std": 4_608}
ONLINE_LOGS = {"on_": "night/on.log", "cls_": "night/cls.log", "share_": "night/share.log"}
STEP_LINE = re.compile(r"step (\d+) \| (\w+): accuracy ([\d.]+), agreement ([\d.]+), state FVU ([\d.]+) \(last training batch: accuracy ([\d.]+)\)")
SUMMARY_LINE = re.compile(r"wandb:\s+(val/zero_shot_accuracy|val/clip_agreement|val/one_step_fvu|val/predicted_accuracy|train/zero_shot_accuracy) ([\d.]+)\s*$")
# Code checks that ran with wandb offline, so they are not in the project. (name, started EST, script, length, source)
SMOKE = [
    ("smoke (smoke_dense, smoke_jordan)", "11-07 00:16", "scripts/tokae/train.py", "", "smoke.log (file time; the log holds no numbers); review/history/full_11-07.txt lines 1575-1576"),
    ("smoke_debug", "11-09 23:18", "scripts/tokae/train.py on Imagenette, CPU", "", "smoke_debug.log (file time)"),
    ("smoke_real_start", "11-09 23:19", "scripts/tokae/train.py", "", "smoke_real_start.log (file time; the log holds no numbers)"),
    ("smoke_off", "11-09 23:20", "scripts/tokae/train.py", "", "smoke_off.log (file time; the log holds no numbers)"),
    ("probe_smoke_clip", "11-10 09:39", "ceiling_probe.py", "6 epochs on a small sample", "probe_smoke_clip.log (file time)"),
    ("probe_smoke_dino", "11-10 09:40", "ceiling_probe.py", "3 epochs on a small sample", "probe_smoke_dino.log (file time)"),
    ("smoke_none / smoke_add / smoke_bilinear", "11-10 12:44", "control_train.py", "4 steps", "review/history/full_11-10.txt lines 11908-11913 (time of that entry)"),
    ("smoke_cls_add / smoke_cls_add_real / smoke_cls_mlp", "11-10 17:26", "cls_train.py", "4 steps", "smoke_cls.log; smoke_wandb/wandb/offline-run-20261111_0226*"),
    ("smoke_real_w1 / smoke_add_real", "11-11 02:04", "cls_train.py", "40 steps", "wandb/offline-run-20261111_110442-9tsznewl, -105isq1h"),
]


def est(utc, seconds=0):
    """A wandb UTC time, moved by `seconds`, as EST (UTC - 5 h)."""
    moment = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ") + timedelta(seconds=seconds or 0, hours=-5)
    return moment.strftime("%m-%d %H:%M")


def get_tokens(run):
    """Which tokens make the state: from the config, or from the tags of the first runs."""
    tokens = run["config"].get("tokens")
    if tokens:
        return tokens
    if run["name"].startswith(("cls_", "share_")):
        return "cls"
    return "cls_patch_std" if "cls+patch+std" in run["tags"] else "cls_patch" if "cls+patch" in run["tags"] else "cls"


def get_family(run):
    """The group a run belongs to, numbered in time order."""
    name, tags = run["name"], run["tags"]
    if name.startswith("probe"):
        return "6 probe, no KAE"
    if name.startswith("share_"):
        return "9 CLS-only state, penalty on the control"
    if name.startswith("cls_"):
        return "8 CLS-only state, additive control"
    if name.startswith("on_"):
        return "7 fresh crops, three-part state, with and without control"
    if "dinov3-vitb16" in tags:
        return "5 DINOv3"
    if "imagenette" in tags:
        return "1 Imagenette, CLS state"
    if get_tokens(run) == "cls":
        return "2 ImageNet, CLS state"
    return "3 ImageNet, wider state, hidden width 2,048" if run["config"]["ae_hidden"] == 2_048 else "4 ImageNet, three-part state, hidden 2 x 4,096 or wider"


def get_operator(run):
    """What moves the latent from block 6 to block 12."""
    config = run["config"]
    if config.get("mlp") or config.get("arm") == "mlp":
        return "MLP"
    if run["name"].startswith("probe"):
        return "none (plain MLP probe)"
    online = run["name"].startswith(("on_", "cls_", "share_"))
    return "eigenvalues" if config.get("jordan") or online else "dense matrix"


def get_setup(run):
    """The part of the config that matters, in plain columns."""
    config, name, online = run["config"], run["name"], run["name"].startswith(("on_", "cls_", "share_"))
    arm, like_kae = config.get("arm", ""), online or name.startswith("probe")  # The probe has the KAE's embedding and KL weights (ceiling_probe.py line 42)
    dataset = config.get("dataset") or ("imagenette" if "imagenette" in run["tags"] else "imagenet")
    views = config.get("views", 3 if name.startswith("probe") else 1)  # The probe's 3,843,501 states are 3 crops per image
    states = config.get("num_train_states") or IMAGES[dataset] * views * config.get("train_fraction", 1)
    return {
        "encoder": "DINOv3 ViT-B/16" if "dinov3-vitb16" in run["tags"] else "CLIP ViT-B/32",
        "dataset": dataset + (f" + {config['extra_dataset']}" if config.get("extra_dataset") else ""),
        "crops": "fresh crops every step" if online else f"{views} stored crop{'s' if views > 1 else ''} per image",
        "train_fraction": config.get("train_fraction", 1),
        "start_block": config.get("start_block", 6),
        "state": get_tokens(run),
        "state_size": STATE_SIZE[get_tokens(run)],
        "latent_size": config.get("ae_dim", ""),
        "hidden_width": config.get("ae_hidden", config.get("width", "")),
        "hidden_layers": config.get("ae_layers", 2 if name.startswith("probe") else 1),
        "operator": get_operator(run),
        "top_k": "" if get_operator(run) != "eigenvalues" and online else config.get("top_k", ""),
        "batch_top_k": bool(config.get("batch_top_k")),
        "shared_modes": bool(config.get("shared_modes")) or (online and arm != "mlp"),
        "real_start": bool(config.get("real_start")) or "real" in arm,
        "control": config.get("control") or ("add" if name.startswith(("cls_", "share_")) else "none"),
        "control_penalty": {"real_w01": 0.1, "real_w03": 0.3, "real_w1": 1.0}.get(arm, 0),
        "weight_decay": config.get("weight_decay") or 0,
        "kl_weight": 0.03 if like_kae else config.get("lambda_kl", 0),
        "embedding_weight": 1 if like_kae else config.get("lambda_embedding", 0),
        "isometric_weight": 0.1 if online else config.get("lambda_isometric", ""),
        "overlap_weight": config.get("lambda_overlap", 0),
        "standardization": config.get("standardization") or (f"that of {config['reference']}" if config.get("reference") else ""),
        "lr": config.get("lr"),
        "seed": config.get("seed"),
        "planned": f"{config['total_steps']} steps" if online else f"{config['num_epochs']} epochs",
        "steps_per_epoch": 1 if online else round(states / config["batch_size"]),
    }


def get_curve(run, setup):
    """One row per logged epoch or step, from wandb."""
    online = setup["steps_per_epoch"] == 1
    rows = []
    for row in run["curve"]:
        if "val/one_step_fvu" not in row:
            continue
        at = row["_step"] if online else row["_step"] + 1
        rows.append({
            "run": run["name"], "id": run["id"], "unit": "step" if online else "epoch", "at": at, "steps": at * setup["steps_per_epoch"],
            "accuracy": row.get("val/zero_shot_accuracy", ""), "agreement": row.get("val/clip_agreement", ""), "fvu": row["val/one_step_fvu"],
            "train_accuracy": row.get("train/zero_shot_accuracy", ""), "train_agreement": row.get("train/clip_agreement", ""), "fitted_accuracy": row.get("val/predicted_accuracy", ""),
            "control_share": row.get("val/control_share", ""), "source": f"wandb {run['id']}",
        })  # fmt: skip
    return rows


def read_online_log(path):
    """The step lines of an online training log: {(run name, step): numbers}."""
    found = {}
    for number, line in enumerate((SCRATCH / path).read_text().splitlines(), start=1):
        match = STEP_LINE.search(line)
        if match:
            step, name, accuracy, agreement, fvu, train = match.groups()
            found[(name, int(step))] = {"accuracy": float(accuracy), "agreement": float(agreement), "fvu": float(fvu), "train_accuracy": float(train), "source": f"{path} line {number}"}
    return found


def read_summary_log(name):
    """The last numbers wandb printed at the end of a stored-crop run's log, and the log's path."""
    for path in (f"night/{name}.log", f"{name}.log"):
        if (SCRATCH / path).exists():
            return path, {key: float(value) for line in (SCRATCH / path).read_text().splitlines() if (match := SUMMARY_LINE.search(line)) for key, value in [match.groups()]}
    return None, {}


def add_log_steps(run, curve, checks):
    """Checks an online run's curve against its log, and appends the steps that only the log has."""
    path = next(path for prefix, path in ONLINE_LOGS.items() if run["name"].startswith(prefix))
    have = {row["at"]: row for row in curve}
    for (name, step), numbers in sorted(read_online_log(path).items(), key=lambda item: item[0][1]):
        if name != run["name"]:
            continue
        if step in have:
            for key in ("accuracy", "agreement", "fvu"):
                checks.append({"run": name, "at": f"step {step}", "number": key, "wandb": round(have[step][key], 5), "log": numbers[key], "log_file": numbers["source"]})
        else:
            curve.append({"run": name, "id": run["id"], "unit": "step", "at": step, "steps": step, **numbers, "fitted_accuracy": "", "control_share": ""})
            checks.append({"run": name, "at": f"step {step}", "number": "accuracy", "wandb": "missing", "log": numbers["accuracy"], "log_file": numbers["source"]})


def add_log_summary(run, curve, checks):
    """Checks a stored-crop run's last wandb row against the numbers at the end of its log."""
    path, numbers = read_summary_log(run["name"])
    names = {"val/zero_shot_accuracy": "accuracy", "val/clip_agreement": "agreement", "val/one_step_fvu": "fvu", "val/predicted_accuracy": "fitted_accuracy", "train/zero_shot_accuracy": "train_accuracy"}
    for key, value in numbers.items():
        if curve and curve[-1][names[key]] != "":
            checks.append({"run": run["name"], "at": f"epoch {curve[-1]['at']}", "number": names[key], "wandb": round(curve[-1][names[key]], 5), "log": value, "log_file": path})


def get_results(run, curve):
    """Last and best numbers of a run. The headline metric is the zero-shot accuracy where it exists."""
    if not curve:
        return {"done": "nothing logged"}
    last = curve[-1]
    metric = "accuracy" if last["accuracy"] != "" else "fitted_accuracy" if last["fitted_accuracy"] != "" else ""
    best = max(curve, key=lambda row: row[metric]) if metric else last
    summary = run["summary"]
    return {
        "done": f"{last['at']} {last['unit']}s",
        "steps_done": last["steps"],
        "accuracy_metric": {"accuracy": "val/zero_shot_accuracy", "fitted_accuracy": "val/predicted_accuracy (fitted classifier, not the zero-shot head)", "": ""}[metric],
        "final_accuracy": last["accuracy"],
        "best_accuracy": best["accuracy"],
        "best_at": f"{best['unit']} {best['at']}",
        "final_agreement": last["agreement"],
        "final_fvu": last["fvu"],
        "final_train_accuracy": last["train_accuracy"],
        "final_train_agreement": last.get("train_agreement", ""),
        "train_accuracy_is": "" if last["train_accuracy"] == "" else "last training batch of 1,024 fresh crops" if last["unit"] == "step" else "sample of the stored training crops",
        "final_fitted_accuracy": last["fitted_accuracy"],
        "final_control_share": last["control_share"],
        "wandb_best_val_accuracy": summary.get("best/val_accuracy", ""),
        "linear_map_accuracy": summary.get("baseline/linear_zero_shot_accuracy", ""),
        "real_model_accuracy": summary.get("baseline/true_zero_shot_accuracy", summary.get("val/real_model_accuracy", "")),
        "numbers_from": last["source"],
    }


def write(name, rows):
    """Writes dict rows as csv with the union of their keys, in first-seen order."""
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(DATA / name, "w", newline="") as file:
        writer = csv.DictWriter(file, keys, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows in {DATA / name}")


if __name__ == "__main__":
    table, curves, checks = [], [], []
    for run in json.loads((DATA / "wandb_raw.json").read_text()):
        setup = get_setup(run)
        curve = get_curve(run, setup)
        if setup["steps_per_epoch"] == 1:
            add_log_steps(run, curve, checks)
        else:
            add_log_summary(run, curve, checks)
        row = {"id": run["id"], "name": run["name"], "family": get_family(run), "created_est": est(run["created"]), "finished_est": est(run["created"], run["runtime"]),
               "state_on_wandb": run["state"], "smoke": False, **setup, **get_results(run, curve)}  # fmt: skip
        table.append(row)
        curves += curve
    for name, started, script, length, source in SMOKE:
        table.append({"id": "offline", "name": name, "family": "0 smoke tests (code checks, wandb offline)", "created_est": started, "smoke": True, "planned": length, "done": "code check only", "numbers_from": source, "crops": script})

    write("runs.csv", table)
    write("curves.csv", curves)
    write("log_check.csv", checks)
    differ = [check for check in checks if check["wandb"] == "missing" or abs(check["wandb"] - check["log"]) > 0.00015]
    print(f"{len(checks)} numbers checked against logs, {len(differ)} differ or are missing on wandb:")
    for check in differ:
        print("  ", check)
