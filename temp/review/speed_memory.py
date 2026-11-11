"""
Memory of a forward pass, for real CLIP and for each KAE pipeline (blocks 1-6, the KAE, the head).

    gpu   One process. `weights` is what the allocator holds once the model is on the GPU; `working` is the peak during a
          forward pass above that (torch.cuda.max_memory_allocated), the batch of pixels included. Also the replaced
          half alone (blocks 7-12 against a KAE, both starting from the tokens after block 6).
    cpu   One fresh process per pipeline and batch size. `before` is the resident memory once the model is loaded and
          has run one image; `peak` is the highest resident memory during the forward pass (VmHWM in /proc/self/status,
          reset just before the pass). Each measurement is repeated 3 times.

All sizes are in MB (1 MB = 1,000,000 bytes). The CUDA context and library workspaces are outside the GPU count.

Usage: python speed_memory.py gpu
       OMP_NUM_THREADS=16 taskset -c 32-47 python speed_memory.py cpu
"""

import ctypes
import gc
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

from speed_models import KAES, PATCHES, WIDTH, half, pipeline, update

GPU_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
CPU_BATCHES = [1, 16, 64, 256, 1024]
PIXELS, TOKENS = (3, 224, 224), (PATCHES + 1, WIDTH)


def gpu_sweep(build, name, shape):
    """Weights and peak working memory (MB) at each batch size, for one model built on the GPU."""

    run = build(name, "cuda")
    run(torch.randn(1, *shape, device="cuda"))
    weights, working = torch.cuda.memory_allocated(), {}
    for batch in GPU_BATCHES:
        inputs = torch.randn(batch, *shape, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        run(inputs)
        working[batch] = (torch.cuda.max_memory_allocated() - weights) / 1e6
        del inputs
    del run
    gc.collect()
    torch.cuda.empty_cache()
    # NOTE: About 9 MB stay allocated for good after the first forward pass (a library workspace); every `weights` has them
    assert torch.cuda.memory_allocated() < 10e6, "a model was left on the GPU"

    return {"weights": weights / 1e6, "working": working}


def resident(field):
    """This process's resident memory in MB: `VmRSS` is now, `VmHWM` is the highest since the last reset."""

    return int(re.search(rf"{field}:\s+(\d+) kB", Path("/proc/self/status").read_text()).group(1)) * 1024 / 1e6


def cpu_once(name, batch):
    """Run in a fresh process: prints the resident memory before and at the peak of one forward pass."""

    run = pipeline(name, "cpu")
    run(torch.randn(1, *PIXELS))
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)  # Hands freed memory (the text tower, unused blocks) back to the system
    before = resident("VmRSS")
    Path("/proc/self/clear_refs").write_text("5")  # Resets VmHWM
    run(torch.randn(batch, *PIXELS))
    print(json.dumps({"before": before, "peak": resident("VmHWM")}))


def cpu_sweep(name, repeats=3):
    results = {}
    for batch in CPU_BATCHES:
        outputs = [subprocess.run([sys.executable, __file__, "cpu", name, str(batch)], capture_output=True, text=True) for _ in range(repeats)]
        results[batch] = [json.loads(output.stdout.splitlines()[-1]) for output in outputs]
        print(f"cpu | {name:<14} batch {batch:>4} | " + ", ".join(f"{r['before']:.0f} -> {r['peak']:.0f} MB" for r in results[batch]), flush=True)

    return results


if __name__ == "__main__":
    names = ["clip"] + KAES
    if sys.argv[1] == "gpu":
        update("memory_gpu", {"whole": {name: gpu_sweep(pipeline, name, PIXELS) for name in names}, "half": {name: gpu_sweep(half, name, TOKENS) for name in names}})
    elif len(sys.argv) > 2:
        cpu_once(sys.argv[2], int(sys.argv[3]))
    else:
        update("memory_cpu", {name: cpu_sweep(name) for name in names})
