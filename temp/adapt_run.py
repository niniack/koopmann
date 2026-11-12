"""
TEMP: the adaptation tests on models of cls_train.py. Can a model be moved to a new domain by tuning its eigenvalues alone?

Tuned: one size factor and one turn per mode, λ' = λ·exp(size + i·turn), 2,050 numbers shared by all images. Encoder,
decoder and head are frozen (adapt_cifar.py's functions). Every domain is read with CLIP's zero-shot head over its class names:
  cifar100          a few training images per class to fit, scored on the 10,000 test images (adapt_cifar.py's test)
  hoofed            the 49 look-alike hoofed species of ImageNet-21k: fitted on 70% of every species, scored on the rest
  imagenet_sketch   ImageNet-Sketch, split the same way
Reference with the same labels on the REAL model: a scale and a shift per dimension of its block-12 CLS token (1,536 numbers).

Usage: adapt_run.py <weights name> ... [--domains=hoofed]        such as lift_attend_step62500
"""

from types import SimpleNamespace

import fire
import torch

import adapt_cifar as adapt
import angle_tests as tests
import baseline_eval as fixed
from angle_tests import base, device, train

FOLDER = base.HERE / "domain_cache"
SHOTS = [1, 4, 16, 64]
TUNED = {"size": (1, 0), "turn": (0, 1), "both": (1, 1)}


def split(labels, share=0.7):
    """A random 70% of every class to fit, the rest to test."""

    order = torch.rand(len(labels)).argsort()
    fit = torch.zeros(len(labels), dtype=torch.bool)
    for c in labels.unique():
        rows = order[labels[order] == c]
        fit[rows[: int(share * len(rows))]] = True
    return fit


def get_domain(name, imagenet_classes):
    """A domain's data, the head's classes for it, the rows to score, and the rows to fit on by setting."""

    if name == "cifar100":
        train_set, test_set = [torch.load(FOLDER / f"cifar100_{part}.pt") for part in ["train", "test"]]
        shots = adapt.get_shots(train_set["labels"], max(SHOTS))
        data = {key: torch.cat([train_set[key][shots], test_set[key]]) for key in ["states", "patches", "labels"]}
        fits = {f"{count} images per class": adapt.get_shots(data["labels"][: len(shots)], count) for count in SHOTS}
        return data, torch.load(FOLDER / "cifar100_classes.pt"), torch.arange(len(shots), len(data["labels"])), fits

    data = torch.load(FOLDER / f"{name}.pt")
    torch.manual_seed(0)  # NOTE: The split of this test's earlier runs
    fit = split(data["labels"])
    classes = imagenet_classes if name == "imagenet_sketch" else torch.load(FOLDER / f"{name}_classes.pt")
    return data, classes, (~fit).nonzero()[:, 0], {"70% of every class": fit.nonzero()[:, 0]}


def get_kae(name, head):
    """A model of cls_train.py with the fields adapt_cifar's functions use (those of demo_utils.Kae)."""

    model, stats = tests.load_kae(name.partition("_step")[0], name)
    eigenvalues = torch.cat([model.real_eigenvalue.to(torch.cfloat), torch.complex(model.block_real, model.block_imag)])
    return SimpleNamespace(name=name, model=model.requires_grad_(False), eigenvalues=eigenvalues, head=head, mean=stats["mean"].to(device), std=stats["std"].to(device))


def main(*names, domains: str = "cifar100,hoofed,imagenet_sketch"):
    stats = tests.load_kae(names[0].partition("_step")[0], names[0])[1]
    head = train.ClipHead(fixed.ENCODER, stats["mean"], stats["std"], device, zero_shot=True)
    imagenet_classes = head.classes
    kaes = [get_kae(name, head) for name in names]
    zeros = lambda width: torch.zeros(2, width, device=device)
    by_clip = lambda end, knobs: adapt.clip_log_probs(kaes[0], end, knobs)

    for domain in domains.split(","):
        data, classes, test, fits = get_domain(domain, imagenet_classes)
        head.classes = classes.to(device)
        states, patches, labels = data["states"][..., :768].to(device), data["patches"].to(device), data["labels"].to(device)
        score = lambda log_probs, inputs, knobs: adapt.get_accuracy(log_probs, inputs[test], labels[test], knobs)
        print(f"== {domain}: {len(test)} images to score, accuracy in %")
        print(f"real CLIP: untuned {score(by_clip, states[:, 1], zeros(768)):.1f}")
        for setting, fit in fits.items():
            knobs = adapt.tune(by_clip, states[fit, 1], labels[fit], zeros(768), (1, 1))
            print(f"  scale and shift fitted on {setting}: {score(by_clip, states[:, 1], knobs):.1f}", flush=True)

        for kae in kaes:
            latent = fixed.encode_start(kae.model, (states - kae.mean) / kae.std, patches)
            z6 = tests.to_complex(latent)
            by_kae = lambda z6, knobs: adapt.kae_log_probs(kae, z6, knobs)
            untuned = score(by_kae, z6, zeros(1025))
            own = tests.rate(tests.classify(kae.model, head, kae.model.koopman_forward(latent, 1))[test], labels[test])
            assert abs(untuned - own) < 0.1, "λ·z6 is not the model's own step"
            print(f"{kae.name}: untuned {untuned:.1f}")
            for setting, fit in fits.items():
                scores = {key: score(by_kae, z6, adapt.tune(by_kae, z6[fit], labels[fit], zeros(1025), rows)) for key, rows in TUNED.items()}
                print(f"  eigenvalues fitted on {setting}: " + " | ".join(f"{key} {value:.1f}" for key, value in scores.items()), flush=True)


if __name__ == "__main__":
    fire.Fire(main)
