"""
TEMP: time of real CLIP's blocks 7-12 against the models of cls_train.py that stand in for them, on random tokens.
The models are untrained: the time does not depend on the weights.

Usage: lift_speed.py --arms=none,add,attend,mixer [--cpu]
"""

import time

import fire
import torch

import control_train as base
from cls_train import get_model


def clock(run, device, repeats=20):
    """The best time of `repeats` runs, in ms."""

    best = float("inf")
    for _ in range(repeats):
        began = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - began)

    return 1_000 * best


@torch.no_grad()
def main(arms: tuple = ("none", "add", "attend", "mixer"), encoder: str = "ViT-B-32", batches: tuple = (1, 256), cpu: bool = False):
    device = torch.device("cpu" if cpu else base.get_device())
    torch.backends.cuda.matmul.allow_tf32 = True
    blocks = base.Teacher(encoder, device).model.visual.transformer.resblocks[base.BLOCKS[0] + 1 :]
    kwargs = dict(width=768, num_patches=49, top_k=256, shared_modes=True, in_features=768, latent_features=2049,
                  hidden_config=[4096, 4096], bias=True, batchnorm=False, nonlinearity="leaky_relu")  # fmt: skip
    models = {arm: get_model(arm, kwargs).to(device).eval() for arm in arms}

    for batch in batches:
        tokens = torch.randn(batch, 50, 768, device=device)

        def run_clip():
            x = tokens
            for block in blocks:
                x = block(x)

        def run_model(model):
            model.set_control(tokens[:, 1:])
            model.decode(model.koopman_forward(model.encode(tokens[:, 0]), 1))

        clip = clock(run_clip, device)
        print(f"{device.type}, {batch} images: real CLIP's blocks 7-12 {clip:.2f} ms", flush=True)
        for arm, model in models.items():
            own = clock(lambda: run_model(model), device)
            print(f"  {arm:8s} {own:.2f} ms, {clip / own:.1f} times faster, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
