"""
Can eigenvalue tuning make a DINOv3 KAE better than DINOv3 itself on species it mixes up?

Data: 49 hoofed species (antelope, deer, sheep, goats, cattle) from ImageNet-21k, 70% of each species to fit, 30% to test.
DINOv3 has no head for these species, so "the real model" is a linear probe on its final CLS output, fitted on the 70%.
The KAE is read through that same probe. Tuned: λ' = λ·exp(size + i·turn) per mode (2,050 numbers), on the same 70%.
Last row: a probe fitted on the KAE's own output, to see how much species information the KAE's output holds at all.

Usage: adapt_hoofed.py <KAE run name>
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))
from adapt_cifar import get_accuracy, kae_log_probs, kae_states, tune  # noqa: E402
from demo_utils import Kae, train  # noqa: E402

torch.set_num_threads(16)
torch.manual_seed(0)
ENCODER = "dinov3-vitb16"


class DinoKae(Kae):
    """A trained DINOv3 KAE. Its head is DINOv3's final layer norm; the classifier on top is set by `fit_probe`."""

    def __init__(self, name):
        folder = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
        stats = torch.load(folder / f"{name}_stats.pt")
        self.mean, self.std = stats["mean"], stats["std"]
        self.model = train.JordanKoopmanAutoencoder(
            256, shared_modes=True, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu"
        )
        self.model.load_state_dict(load_file(folder / f"{name}.safetensors"))
        self.model.eval()
        self.eigenvalues = torch.cat([self.model.real_eigenvalue.to(torch.cfloat), torch.complex(self.model.block_real, self.model.block_imag)])
        self.head = train.DinoHead(ENCODER, self.mean, self.std, "cpu", zero_shot=False)


def fit_probe(head, states, labels, steps=500):
    """Fits the head's linear classifier on standardized end states, full batch."""

    features = head.features(states)
    probe = torch.nn.Linear(features.shape[1], int(labels.max()) + 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.003, weight_decay=0.01)
    with torch.enable_grad():
        for _ in range(steps):
            loss = F.cross_entropy(probe(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    head.classes, head.bias = probe.weight.detach().T, probe.bias.detach()


def split(labels, share=0.7):
    """A random 70% of every class to fit, the rest to test."""

    order = torch.rand(len(labels)).argsort()
    fit = torch.zeros(len(labels), dtype=torch.bool)
    for c in labels.unique():
        rows = order[labels[order] == c]
        fit[rows[: int(share * len(rows))]] = True
    return fit


if __name__ == "__main__":
    kae = DinoKae(sys.argv[1])
    data = torch.load(Path(__file__).parent / f"hoofed_{ENCODER}.pt")
    states, labels = data["states"].float(), data["labels"]
    fit = split(labels)
    end = (states[:, 1] - kae.mean) / kae.std
    z6 = kae.encode(states)
    by_kae = lambda z6, knobs: kae_log_probs(kae, z6, knobs)
    by_real = lambda end, knobs: kae.head.get_log_probs(end)
    untuned = torch.zeros(2, 1025)

    fit_probe(kae.head, end[fit], labels[fit])
    print(f"== {sys.argv[1]} on {len(data['names'])} hoofed species: {int(fit.sum())} images to fit, {int((~fit).sum())} to test. Accuracy in %, test (fit)")
    report = lambda name, log_probs, inputs, knobs: print(
        f"  {name}: {get_accuracy(log_probs, inputs[~fit], labels[~fit], knobs):.1f} ({get_accuracy(log_probs, inputs[fit], labels[fit], knobs):.1f})", flush=True
    )
    report("real DINOv3 + probe", by_real, end, untuned)
    report("KAE through that probe, untuned", by_kae, z6, untuned)
    for name, tuned in [("size only", (1, 0)), ("turn only", (0, 1)), ("size and turn", (1, 1))]:
        report(f"KAE, tuned {name}", by_kae, z6, tune(by_kae, z6[fit], labels[fit], torch.zeros(2, 1025), tuned))

    kae_end = torch.cat([kae_states(kae, part, untuned) for part in z6.split(2_000)])
    fit_probe(kae.head, kae_end[fit], labels[fit])
    report("KAE + its own probe", by_real, kae_end, untuned)
