"""
Adaptation test: can a KAE trained on ImageNet be moved to a new domain by tuning its eigenvalues only?

Domain: CIFAR-100 (tiny 32x32 images, 100 classes), read through CLIP's zero-shot head with CIFAR's class names.
Tuned: one size factor and one turn per mode, λ' = λ·exp(size + i·turn), 2,050 numbers. Encoder, decoder and head are frozen.
Variants: size only, turn only, both. Labels: a few training images per class ("shots").
Reference with the same labels on the REAL model: a scale and a shift per dimension of its block-12 CLS token (1,536 numbers).

Usage: adapt_cifar.py <KAE run name>
"""

import sys

import open_clip
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
from control_train import HERE  # noqa: E402
from demo_utils import ENCODER, Kae, load_clip  # noqa: E402

torch.set_num_threads(16)
torch.manual_seed(0)
STEPS, LR = 300, 0.01


def get_classes(names):
    """CLIP's zero-shot classifier for these class names, built once."""

    path = HERE / "domain_cache" / "cifar100_classes.pt"
    if not path.exists():
        classes = open_clip.build_zero_shot_classifier(load_clip(), open_clip.get_tokenizer(ENCODER), names, open_clip.OPENAI_IMAGENET_TEMPLATES)
        torch.save(classes, path)
    return torch.load(path)


def get_shots(labels, shots):
    """The first `shots` training images of every class."""

    return torch.cat([(labels == c).nonzero()[:shots, 0] for c in labels.unique()])


def kae_states(kae, z6, knobs):
    """The KAE's block-12 state, with every eigenvalue multiplied by exp(size + i·turn). knobs: [2, modes]"""

    z12 = kae.eigenvalues * torch.exp(torch.complex(knobs[0], knobs[1])) * z6
    latent = torch.cat([z12[:, :1].real, torch.view_as_real(z12[:, 1:]).flatten(1)], dim=1)
    return kae.model.decode(latent)


def kae_log_probs(kae, z6, knobs):
    """The KAE's class log probabilities, from its tuned block-12 state."""

    return kae.head.get_log_probs(kae_states(kae, z6, knobs))


def clip_log_probs(kae, end_states, knobs):
    """Real CLIP's class log probabilities, with a scale and shift on its block-12 CLS token. knobs: [2, 768]"""

    cls = (end_states[:, :768] - kae.mean[:768]) / kae.std[:768]
    return kae.head.get_log_probs(cls * torch.exp(knobs[0]) + knobs[1])


def tune(log_probs, inputs, labels, knobs, tuned):
    """Fits the tuned rows of `knobs` (the other rows stay 0) to the labels, full batch."""

    knobs.requires_grad_(True)
    mask = torch.tensor(tuned, device=knobs.device).float()[:, None]
    optimizer = torch.optim.Adam([knobs], lr=LR)
    with torch.enable_grad():
        for _ in range(STEPS):
            loss = F.cross_entropy(log_probs(inputs, knobs * mask), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return (knobs * mask).detach()


def get_accuracy(log_probs, inputs, labels, knobs):
    answers = torch.cat([log_probs(part, knobs).argmax(dim=1) for part in inputs.split(2_000)])
    return 100 * (answers == labels).float().mean().item()


if __name__ == "__main__":
    kae = Kae(sys.argv[1])
    train_set, test_set = [torch.load(HERE / "domain_cache" / f"cifar100_{split}.pt") for split in ["train", "test"]]
    kae.head.classes = get_classes(train_set["classes"])
    test_z6, test_end, test_labels = kae.encode(test_set["states"]), test_set["states"][:, 1], test_set["labels"]
    by_kae = lambda z6, knobs: kae_log_probs(kae, z6, knobs)
    by_clip = lambda end, knobs: clip_log_probs(kae, end, knobs)

    print(f"== {sys.argv[1]} on CIFAR-100 test (10,000 images), accuracy in %")
    print(f"untuned: real CLIP {get_accuracy(by_clip, test_end, test_labels, torch.zeros(2, 768)):.1f} | KAE {get_accuracy(by_kae, test_z6, test_labels, torch.zeros(2, 1025)):.1f}")
    print("shots per class | KAE, size only | turn only | both | real CLIP with scale and shift")
    for shots in [1, 4, 16, 64]:
        rows = get_shots(train_set["labels"], shots)
        z6, end, labels = kae.encode(train_set["states"][rows]), train_set["states"][rows, 1], train_set["labels"][rows]
        scores = [get_accuracy(by_kae, test_z6, test_labels, tune(by_kae, z6, labels, torch.zeros(2, 1025), tuned)) for tuned in [(1, 0), (0, 1), (1, 1)]]
        scores.append(get_accuracy(by_clip, test_end, test_labels, tune(by_clip, end, labels, torch.zeros(2, 768), (1, 1))))
        print(f"  {shots:3d} | " + " | ".join(f"{score:.1f}" for score in scores), flush=True)
