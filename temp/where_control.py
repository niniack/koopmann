"""
Where the additive control writes. Under the real-start rule a mode at block 6 is one real number x, so
λ·z6 = (a·x, b·x): a large first coordinate and a small second one (b is small). This script splits z12's
squared size into first and second coordinates, for λ·z6 and for the control, and removes each piece in turn.

Usage: where_control.py <run name> [<run name> ...]
"""

import sys

import torch

import angle_tests as tests
from angle_tests import base, train

first = lambda latent: latent[..., 1::2]
second = lambda latent: latent[..., 2::2]


def keep(latent, first_on, second_on):
    """The latent with the first or second coordinates of every pair set to zero."""

    out = latent.clone()
    out[..., 1::2] *= first_on
    out[..., 2::2] *= second_on
    return out


if __name__ == "__main__":
    states, patches, labels = base.get_val(None, "ViT-B-32", tests.device)
    for name in sys.argv[1:]:
        kae, stats = tests.load_kae(name)
        mean, std = stats["mean"].to(tests.device), stats["std"].to(tests.device)
        head = train.ClipHead("ViT-B-32", mean, std, tests.device, zero_shot=True)
        x = (states[..., :768] - mean) / std
        kae.set_control(patches)
        z6 = torch.cat([train.encode_pairs(kae, part)[:, 0] for part in x.split(5_000)])
        z12 = kae.koopman_forward(z6, 1)
        own = train.JordanKoopmanAutoencoder.koopman_forward(kae, z6, 1)
        control = z12 - own

        total = z12.pow(2).sum()
        print(f"{name}: share of z12's squared size")
        for label, part in {"λ·z6": own, "control": control}.items():
            print(f"  {label:8s} first coordinates {100 * first(part).pow(2).sum() / total:6.2f}% | second coordinates {100 * second(part).pow(2).sum() / total:6.2f}%")
        rows = {"as trained": z12, "second coordinates of λ·z6 removed": keep(own, 1, 0) + control, "second coordinates of control removed": own + keep(control, 1, 0),
                "first coordinates of control removed": own + keep(control, 0, 1), "all second coordinates removed": keep(z12, 1, 0)}  # fmt: skip
        for row, latent in rows.items():
            print(f"  {row:38s} accuracy {100 * (tests.classify(kae, head, latent) == labels).float().mean():5.2f}")
