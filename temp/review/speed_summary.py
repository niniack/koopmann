"""
Turns the raw measurements in data/speed.json into the numbers the charts and the text use, saved under `summary`.

    time      per device, batch size and piece: median, fastest and slowest round (ms), images per second at the median
    speedup   per device, batch size and KAE: blocks 7-12 against the KAE step (`half`), the full model against the KAE
              pipeline (`whole`), and `ceiling`: the full model against blocks 1-6 and the head alone, which is the best
              any stand-in for blocks 7-12 can reach. Each is the median over rounds of that round's ratio.
    memory    GPU: total (weights + working) per batch size, MB per image, and the batch size where the KAE gets cheaper.
              CPU: median resident memory before and at the peak of a forward pass.
"""

import numpy as np

from speed_models import KAES, load, update

BATCHES = ["1", "16", "256"]


def rounds(run_list, batch, piece):
    """Every round's time for one piece, over all runs of the timing script."""

    return np.concatenate([run["ms"][batch][piece] for run in run_list if batch in run["ms"]])


def time_table(run_list, batch):
    table = {}
    for piece in run_list[0]["ms"][batch]:
        ms = rounds(run_list, batch, piece)
        table[piece] = {"median": np.median(ms), "fastest": ms.min(), "slowest": ms.max(), "images_per_second": 1000 * int(batch) / np.median(ms)}

    return table


def speedups(run_list, batch):
    full, front, back, head = [rounds(run_list, batch, piece) for piece in ("full", "front", "back", "head")]
    table = {"ceiling": np.median(full / (front + head))}
    for name in KAES:
        table[name] = {"half": np.median(back / rounds(run_list, batch, f"kae:{name}")), "whole": np.median(full / rounds(run_list, batch, f"whole:{name}"))}

    return table


def gpu_memory(sweeps):
    """Total memory per batch size, the slope in MB per image (from the largest batch), and where a KAE gets cheaper."""

    table = {}
    for name, sweep in sweeps.items():
        largest = max(sweep["working"], key=int)
        table[name] = {
            "weights": sweep["weights"],
            "per_image": sweep["working"][largest] / int(largest),
            "total": {batch: sweep["weights"] + working for batch, working in sweep["working"].items()},
        }
    clip = table["clip"]
    for name in KAES:
        kae = table[name]
        saved = clip["per_image"] - kae["per_image"]
        kae["crossover"] = (kae["weights"] - clip["weights"]) / saved if saved > 0 else None
        kae["first_batch_below_clip"] = next((int(b) for b in kae["total"] if kae["total"][b] < clip["total"][b]), None)

    return table


def cpu_memory(sweeps):
    return {
        name: {batch: {field: np.median([repeat[field] for repeat in repeats]) for field in ("before", "peak")} for batch, repeats in sweep.items()}
        for name, sweep in sweeps.items()
    }


def main():
    results = load()
    devices = {"cpu": results["time_cpu"], "gpu": [results["time_cuda"]]}
    summary = {
        "time": {device: {batch: time_table(runs, batch) for batch in BATCHES} for device, runs in devices.items()},
        "speedup": {device: {batch: speedups(runs, batch) for batch in BATCHES} for device, runs in devices.items()},
        "memory_gpu": {kind: gpu_memory(sweeps) for kind, sweeps in results["memory_gpu"].items()},
    }
    if "memory_cpu" in results:
        summary["memory_cpu"] = cpu_memory(results["memory_cpu"])
    for threads in (1, 4):
        if f"time_cpu_{threads}_threads" in results:
            runs = results[f"time_cpu_{threads}_threads"]
            summary[f"cpu_{threads}_threads"] = {batch: {"time": time_table(runs, batch), "speedup": speedups(runs, batch)} for batch in runs[0]["ms"]}
    update("summary", summary)

    for device in devices:
        for batch in BATCHES:
            print(f"\n{device}, batch {batch}: ceiling {summary['speedup'][device][batch]['ceiling']:.2f}x")
            for piece, row in summary["time"][device][batch].items():
                print(f"  {piece:<20} {row['median']:9.2f} ms ({row['fastest']:.2f} to {row['slowest']:.2f})  {row['images_per_second']:>10,.0f} images/s")
            for name in KAES:
                print(f"  {name:<20} half {summary['speedup'][device][batch][name]['half']:.2f}x  whole {summary['speedup'][device][batch][name]['whole']:.2f}x")


if __name__ == "__main__":
    main()
