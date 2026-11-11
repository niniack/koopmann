"""
Times each piece of speed_models.py for batches of 1, 16 and 256 images, on one device.

The pieces are timed in rounds: a round times every piece once, so a change in the machine's load hits all of them
alike. Two rounds warm up, then 9 rounds count on the CPU and 20 on the GPU. Every round's time is saved (milliseconds).
On the GPU the clock is read only after `torch.cuda.synchronize()`. Inputs are random pixels: the time does not depend
on what is in the image. PyTorch defaults throughout (float32, no compilation, no mixed precision).

NOTE: CPUs 32-47 are the second hardware threads of physical cores 0-15, where another job of the review runs. So the
CPU times are taken on shared cores. `neighbour_busy` records how busy CPUs 0-15 were during the run (0 to 1), and each
CPU run is added to the list of runs in data/speed.json, so the script can be repeated at different moments.

Usage: OMP_NUM_THREADS=16 taskset -c 32-47 python speed_time.py cpu [<batch sizes, default 1,16,256>]
       python speed_time.py cuda
"""

import sys
import time

import torch

from speed_models import KAES, Kae, back, front, head, load, load_classes, load_clip, update

device = sys.argv[1]
batches = [int(batch) for batch in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 16, 256]
rounds = 9 if device == "cpu" else 20


def neighbour_ticks():
    """(busy, total) clock ticks of CPUs 0-15 since boot, from /proc/stat."""

    rows = [[int(x) for x in line.split()[1:6]] for line in open("/proc/stat").read().splitlines()[1:17]]
    return sum(sum(row[:3]) for row in rows), sum(sum(row) for row in rows)


def now():
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def clock(runs):
    """Milliseconds of every call in `runs`, one list per call with a time per round."""

    times = {name: [] for name in runs}
    for _ in range(2 + rounds):
        for name, run in runs.items():
            start = now()
            run()
            times[name].append(1000 * (now() - start))

    return {name: values[2:] for name, values in times.items()}


def pieces(visual, classes, kaes, batch):
    """Every timed call for one batch size. `full` and `whole:<kae>` start from pixels and end at class scores."""

    pixels = torch.randn(batch, 3, 224, 224, device=device)
    tokens = front(visual, pixels)
    blocks = visual.transformer.resblocks[6:]
    runs = {
        "full": lambda: head(visual, classes, back(blocks, front(visual, pixels))),
        "front": lambda: front(visual, pixels),
        "back": lambda: back(blocks, tokens),
        "head": lambda: head(visual, classes, tokens[:, 0]),
    }
    for name, kae in kaes.items():
        runs[f"kae:{name}"] = lambda kae=kae: kae(tokens)
        runs[f"whole:{name}"] = lambda kae=kae: head(visual, classes, kae(front(visual, pixels)))

    return runs


def main():
    visual, classes = load_clip().visual.to(device), load_classes().to(device)
    kaes = {name: Kae(name).to(device) for name in KAES}
    results, (busy, total) = {}, neighbour_ticks()
    for batch in batches:
        results[batch] = clock(pieces(visual, classes, kaes, batch))
        for name, ms in results[batch].items():
            print(f"{device} | batch {batch:>3} | {name:<20} {sorted(ms)[len(ms) // 2]:9.2f} ms  ({min(ms):.2f} to {max(ms):.2f})", flush=True)

    neighbour = neighbour_ticks()
    run = {"threads": torch.get_num_threads(), "neighbour_busy": (neighbour[0] - busy) / (neighbour[1] - total), "ms": results}
    if device == "cuda":
        update("time_cuda", {**run, "hardware": torch.cuda.get_device_name(0)})
    else:
        key = "time_cpu" if run["threads"] == 16 else f"time_cpu_{run['threads']}_threads"
        update(key, load().get(key, []) + [run])


if __name__ == "__main__":
    main()
