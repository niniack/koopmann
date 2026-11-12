"""
Two tests of what the size and the angle do in a CLS-only KAE with additive control (cls_train.py), on ImageNet validation.

1. Eigenvalue ablation. The step is z12 = λ·z6 + control. At inference, λ is replaced by
       |λ|   (angle dropped)      λ/|λ|   (size dropped)      1   (both dropped)      0   (only the control is left)
   and, apart from that, the control is dropped. Each row gives accuracy and how often the answer stays the KAE's own.

2. Size and phase swap (after Yıldırım, arXiv 2606.17037, ported to modes). For two images A and B with different KAE
   answers, a hybrid block-12 latent takes each mode's size from A and its phase from B: h = |z12_A| · phase(z12_B).
   Whose answer does CLIP's head give? A phase only exists for a mode an image uses, so the hybrid is built on the modes
   both images use. The rows "A on shared modes" and "B on shared modes" show how much of each answer survives that cut.

Usage: angle_tests.py <run name> ...
"""

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

import control_train as base
from cls_train import get_model
from control_train import train

torch.set_grad_enabled(False)
torch.manual_seed(0)
device = base.get_device()
rate = lambda a, b: 100 * (a == b).float().mean().item()
to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)
to_latent = lambda z: torch.cat([z[..., :1].real, torch.view_as_real(z[..., 1:]).flatten(-2)], dim=-1)


def load_kae(name, weights=None):
    """A trained model of cls_train.py (any arm) and its run's stats. `weights` names another file of the run, such as a saved step."""

    folder = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
    stats = torch.load(folder / f"{name}_stats.pt")
    kwargs = dict(width=768, num_patches=49, top_k=256, shared_modes=True, in_features=768, latent_features=2049,
                  hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")  # fmt: skip
    kae = get_model(stats["arm"], kwargs)
    kae.load_state_dict(load_file(folder / f"{weights or name}.safetensors"))
    return kae.to(device).eval(), stats


def classify(kae, head, latent):
    return torch.cat([head.classify(kae.decode(part)) for part in latent.split(5_000)])


def ablate(kae, head, z6, labels, clip):
    """Test 1. Returns each row's answers. NOTE: A KAE with no control has no control rows."""

    a, b, signal = kae.block_real.data.clone(), kae.block_imag.data.clone(), kae.signal
    size = torch.complex(a, b).abs()
    rows = {"as trained": (a, b, signal), "angle dropped (|λ|)": (size, 0 * b, signal), "size dropped (λ/|λ|)": (a / size, b / size, signal),
            "both dropped (λ = 1)": (1 + 0 * a, 0 * b, signal)}  # fmt: skip
    if signal is not None:
        rows |= {"λ = 0 (control only)": (0 * a, 0 * b, signal), "control dropped": (a, b, 0 * signal)}
    answers = {}
    for row, (kae.block_real.data, kae.block_imag.data, kae.signal) in rows.items():
        answers[row] = classify(kae, head, kae.koopman_forward(z6, 1))
        print(f"  {row:24s} accuracy {rate(answers[row], labels):5.2f} | same as real CLIP {rate(answers[row], clip):5.2f} | same as the KAE as trained {rate(answers[row], answers['as trained']):5.2f}")
    kae.block_real.data, kae.block_imag.data, kae.signal = a, b, signal

    return answers


def swap(kae, head, z12):
    """Test 2."""

    answers = classify(kae, head, to_latent(z12))
    partner = torch.randperm(len(z12), device=device)
    pairs = answers != answers[partner]
    z_a, z_b, answer_a, answer_b = z12[pairs], z12[partner][pairs], answers[pairs], answers[partner][pairs]
    shared = (z_a != 0) & (z_b != 0)
    print(f"  {len(z_a)} pairs with different answers; modes used per image {(z_a != 0).sum(dim=1).float().mean():.0f}, shared by the pair {shared.sum(dim=1).float().mean():.0f}")
    phase = lambda z: z / z.abs().clamp_min(1e-12)
    rows = {"A on shared modes": z_a * shared, "B on shared modes": z_b * shared, "size of A, phase of B": z_a.abs() * phase(z_b) * shared, "size of B, phase of A": z_b.abs() * phase(z_a) * shared}
    for row, z in rows.items():
        hybrid = classify(kae, head, to_latent(z))
        print(f"  {row:24s} answer of A {100 * (hybrid == answer_a).float().mean():5.1f}% | answer of B {100 * (hybrid == answer_b).float().mean():5.1f}%")


if __name__ == "__main__":
    states, patches, labels = base.get_val(None, "ViT-B-32", device)
    for name in sys.argv[1:]:
        kae, stats = load_kae(name)
        mean, std = stats["mean"].to(device), stats["std"].to(device)
        head = train.ClipHead("ViT-B-32", mean, std, device, zero_shot=True)
        x = (states[..., :768] - mean) / std
        clip = head.classify(x[:, 1])
        kae.set_control(patches)
        z6 = torch.cat([train.encode_pairs(kae, part)[:, 0] for part in x.split(5_000)])
        eigenvalues = torch.complex(kae.block_real, kae.block_imag)
        used = ((to_complex(z6)[:, 1:] != 0).sum(dim=0) >= 50)
        angle = eigenvalues.angle().abs()[used].rad2deg()
        print(f"{name}: {used.sum()} modes used by 50+ images; |λ| median {eigenvalues.abs()[used].median():.2f}; angle of λ median {angle.median():.1f}°, 90th percentile {angle.quantile(0.9):.1f}°")
        print(" 1. eigenvalue ablation")
        ablate(kae, head, z6, labels, clip)
        z12 = kae.koopman_forward(z6, 1)
        own_part = z12 - torch.cat([kae.signal[:, :1] * (z6[:, :1] != 0), (kae.signal[:, 1:].unflatten(-1, (-1, 2)) * (z6[:, 1:].unflatten(-1, (-1, 2)) != 0).any(dim=-1, keepdim=True)).flatten(-2)], dim=-1)
        print(f"  share of z12's squared size that is λ·z6: {100 * own_part.pow(2).sum() / z12.pow(2).sum():.1f}% (the rest is the control and its overlap with λ·z6)")
        print(" 2. size and phase swap")
        swap(kae, head, to_complex(kae.koopman_forward(z6, 1)))
