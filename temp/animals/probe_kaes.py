"""
Does a KAE's output hold the species? For each KAE: its block-12 output for the hoofed-species images, then
  own probe:   a linear probe fitted on the KAE's output (how much species information the output holds)
  real probe:  the probe fitted on real CLIP's block-12 state, applied to the KAE's output (how well it stands in for CLIP)
First two rows: the same probe on real CLIP's block-6 state (what the KAE is given) and block-12 state (the bar).

Usage: probe_kaes.py <KAE run name> ...
"""

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))
from adapt_hoofed import split  # noqa: E402
from control_train import ControlKae, train  # noqa: E402
from probe_blocks import fit_probe, get_accuracy  # noqa: E402

torch.set_num_threads(16)
torch.set_grad_enabled(False)
torch.manual_seed(0)


def load_kae(name):
    """A trained KAE (with or without the control) and its run's stats."""

    folder = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
    stats = torch.load(folder / f"{name}_stats.pt")
    kae = ControlKae(
        stats.get("control", "none"), width=768, num_patches=49, top_k=256, shared_modes=True, in_features=2304, latent_features=2049,
        hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu",
    )  # fmt: skip
    kae.load_state_dict(load_file(folder / f"{name}.safetensors"))
    return kae.eval(), stats


def get_output(kae, stats, states, patches):
    """The KAE's block-12 state for every image, standardized as in its training."""

    kae.set_control(patches)
    return kae.decode(kae.koopman_forward(train.encode_pairs(kae, (states - stats["mean"]) / stats["std"])[:, 0], 1))


if __name__ == "__main__":
    data = torch.load(Path(__file__).parent / "hoofed_clip_patches.pt")
    states, patches, labels = data["states"].float(), data["patches"], data["labels"]
    fit = split(labels)
    for name in sys.argv[1:]:
        kae, stats = load_kae(name)
        output = get_output(kae, stats, states, patches)
        real = (states - stats["mean"]) / stats["std"]
        if name == sys.argv[1]:
            real_probe = fit_probe(real[fit, 1], labels[fit])
            print(f"real CLIP, block 6: {get_accuracy(fit_probe(real[fit, 0], labels[fit]), real[~fit, 0], labels[~fit]):.1f} | block 12: {get_accuracy(real_probe, real[~fit, 1], labels[~fit]):.1f}", flush=True)
        own = get_accuracy(fit_probe(output[fit], labels[fit]), output[~fit], labels[~fit])
        print(f"{name}: own probe {own:.1f} | real probe {get_accuracy(real_probe, output[~fit], labels[~fit]):.1f}", flush=True)
