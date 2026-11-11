"""
Paints every target of a spec file (built on the CPU by build_spec.py) and saves the pictures, their generators and a few numbers.

Targets without a `start` are painted from scratch. Targets with a `start` continue the generator of an earlier picture,
so an edit keeps that picture's layout and only what the edit asks for has to change.
The step counts are set from a time budget, after timing a few steps on this device.

Usage: invert_batch.py <spec.pt> <out dir> <part> <minutes from scratch> <minutes for continued> [<most steps from scratch> <most steps continued>]
"""

import copy
import os
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

import inversion
from inversion import train

spec_path, out, part = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
scratch_minutes, continued_minutes = float(sys.argv[4]), float(sys.argv[5])
most_scratch = int(sys.argv[6]) if len(sys.argv) > 6 else 20_000
most_continued = int(sys.argv[7]) if len(sys.argv) > 7 else 4_000
CHUNK = 16
out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))

kae, stats = inversion.load_kae()
reader = inversion.Reader(stats["mean"], stats["std"]).to(device)
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], device, zero_shot=True).to(device)
spec = torch.load(spec_path)
done_path = out / "results.pt"
results = torch.load(done_path, weights_only=False) if done_path.exists() else {}


def get_steps(entries, minutes, most, **kwargs):
    """Times 20 steps on the first chunk, then spreads the budget over the chunks."""

    chunk = entries[:CHUNK]
    targets = torch.stack([e["target"] for e in chunk]).to(device)
    starts = None
    if "start" in chunk[0]:
        starts = [copy.deepcopy(results[e["start"]]["generator"]) for e in chunk]
    inversion.invert(reader, targets, part, 5, [e["seed"] for e in chunk], starts=starts, log_every=1000, **kwargs)
    begin = time.time()
    inversion.invert(reader, targets, part, 20, [e["seed"] for e in chunk], starts=starts, log_every=1000, **kwargs)
    per_picture_step = (time.time() - begin) / 20 / len(chunk)
    steps = int(min(most, minutes * 60 / (per_picture_step * len(entries))))
    print(f"{per_picture_step * 1000:.1f} ms per picture and step, {len(entries)} pictures, {steps} steps each", flush=True)

    return steps


def run(entries, steps, **kwargs):
    for first in range(0, len(entries), CHUNK):
        chunk = entries[first : first + CHUNK]
        targets = torch.stack([e["target"] for e in chunk]).to(device)
        starts = None
        if "start" in chunk[0]:
            starts = [copy.deepcopy(results[e["start"]]["generator"]) for e in chunk]
        label = f"[{first}:{first + len(chunk)}]"
        pictures, generators, history = inversion.invert(
            reader, targets, part, steps, [e["seed"] for e in chunk], starts=starts, label=label, **kwargs
        )
        with torch.no_grad():
            states = reader(pictures.to(device))
            answers = head.get_log_probs(states).exp().topk(3)
            # What the probe says of the target itself, as the picture can only be as right as its target
            wanted = head.get_log_probs(targets).exp().topk(3)
            error = ((states - targets).pow(2).sum(dim=1) / targets.pow(2).sum(dim=1)).cpu()
        for i, entry in enumerate(chunk):
            results[entry["name"]] = {
                **{key: value for key, value in entry.items() if key != "target"},
                "target": entry["target"],
                "picture": pictures[i],
                "generator": generators[i],
                "state": states[i].cpu(),
                "error": error[i].item(),
                "answers": (answers.indices[i].tolist(), answers.values[i].tolist()),
                "target answers": (wanted.indices[i].tolist(), wanted.values[i].tolist()),
                "colourfulness": inversion.get_colourfulness(pictures[i : i + 1]).item(),
                "steps": steps,
                "loss history": history[:, i],
            }
            save_image(pictures[i], out / f"{entry['name']}.png")
        torch.save(results, done_path)
        print(f"saved {label}", flush=True)


scratch = [e for e in spec if "start" not in e and e["name"] not in results]
continued = [e for e in spec if "start" in e and e["name"] not in results]
if scratch:
    run(scratch, get_steps(scratch, scratch_minutes, most_scratch))
if continued:
    # NOTE: Continued pictures keep the last, quietest stage of Splice's noise schedule
    quiet = inversion.NOISE_SCHEDULE[-1][1]
    run(continued, get_steps(continued, continued_minutes, most_continued, sigma=quiet), sigma=quiet)
print("done", flush=True)
