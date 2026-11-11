"""How much species information does each state hold? A linear probe on the whole summary (CLS, patch mean, patch std) of block 6 and of block 12."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from adapt_hoofed import split  # noqa: E402


def fit_probe(features, labels, steps=500):
    """A linear classifier on the features, full batch."""

    probe = torch.nn.Linear(features.shape[-1], int(labels.max()) + 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.01)
    with torch.enable_grad():
        for _ in range(steps):
            loss = F.cross_entropy(probe(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return probe


def get_accuracy(probe, features, labels):
    return 100 * (probe(features).argmax(dim=1) == labels).float().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    data = torch.load(Path(__file__).parent / f"hoofed_{sys.argv[1]}.pt")
    states, labels = data["states"].float(), data["labels"]
    fit = split(labels)
    states = (states - states[fit].mean(dim=0)) / states[fit].std(dim=0)
    for block, name in enumerate(["block 6", "block 12"]):
        probe = fit_probe(states[fit, block], labels[fit])
        print(f"{sys.argv[1]} {name}: test {get_accuracy(probe, states[~fit, block], labels[~fit]):.1f}, fit {get_accuracy(probe, states[fit, block], labels[fit]):.1f}")
