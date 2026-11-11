"""
Can a control KAE be moved to the hoofed-species set by tuning its eigenvalues alone?

The species are read with CLIP's own zero-shot head, built from the species' names. The knobs: every mode's step is multiplied by exp(size + i·turn), 2 x 1,025 numbers shared by all
images. They are fitted on the fit split (70% per species) and scored on the rest.

Usage: tune_control.py <KAE run name> ...
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
from adapt_hoofed import split  # noqa: E402
from control_train import train  # noqa: E402
from extract_hoofed import SYNSETS  # noqa: E402
from probe_kaes import load_kae  # noqa: E402

torch.set_num_threads(16)
torch.set_grad_enabled(False)
torch.manual_seed(0)
HERE = Path(__file__).parent
STEPS, LR = 300, 0.01
TUNED = {"untuned": (0, 0), "size": (1, 0), "turn": (0, 1), "both": (1, 1)}


def get_modes(kae, stats, states, patches):
    """Every mode's value after K's step (control included), complex [images, 1025]."""

    kae.set_control(patches)
    latent = kae.koopman_forward(train.encode_pairs(kae, (states - stats["mean"]) / stats["std"])[:, 0], 1)
    return torch.cat([latent[:, :1].to(torch.cfloat), torch.complex(latent[:, 1::2], latent[:, 2::2])], dim=1)


def decode(kae, z12, knobs):
    """The KAE's block-12 state, with every mode multiplied by exp(size + i·turn). knobs: [2, modes]"""

    z12 = z12 * torch.exp(torch.complex(knobs[0], knobs[1]))
    return kae.decode(torch.cat([z12[:, :1].real, torch.view_as_real(z12[:, 1:]).flatten(1)], dim=1))


def get_head(stats):
    """CLIP's zero-shot head over the species' names (the first name of each WordNet synset), built once."""

    head = train.ClipHead("ViT-B-32", stats["mean"], stats["std"], "cpu", zero_shot=False)
    path = HERE / "hoofed_classes.pt"
    if not path.exists():
        import open_clip
        from demo_utils import load_clip

        names = torch.load(HERE / "w21_hoofed.pt")["synsets"]
        names = [names[synset]["name"].split(",")[0] for synset in SYNSETS]
        torch.save(open_clip.build_zero_shot_classifier(load_clip(), open_clip.get_tokenizer("ViT-B-32"), names, open_clip.OPENAI_IMAGENET_TEMPLATES), path)
    head.classes = torch.load(path)
    return head.requires_grad_(False)


def get_accuracy(head, states, labels):
    return 100 * (head.get_log_probs(states).argmax(dim=1) == labels).float().mean().item()


def tune(kae, head, z12, labels, tuned):
    """Fits the tuned rows of the knobs (the other row stays 0) to the labels, full batch."""

    knobs = torch.zeros(2, z12.shape[1], requires_grad=True)
    mask = torch.tensor(tuned).float()[:, None]
    optimizer = torch.optim.Adam([knobs], lr=LR)
    with torch.enable_grad():
        for _ in range(STEPS if sum(tuned) else 0):
            loss = F.nll_loss(head.get_log_probs(decode(kae, z12, knobs * mask)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return (knobs * mask).detach()


if __name__ == "__main__":
    data = torch.load(HERE / "hoofed_clip_patches.pt")
    states, patches, labels = data["states"].float(), data["patches"], data["labels"]
    fit = split(labels)
    for name in sys.argv[1:]:
        kae, stats = load_kae(name)
        kae.requires_grad_(False)
        if name == sys.argv[1]:
            real = (states[:, 1] - stats["mean"]) / stats["std"]
            head = get_head(stats)
            print(f"real CLIP, zero-shot: test {get_accuracy(head, real[~fit], labels[~fit]):.1f} (fit {get_accuracy(head, real[fit], labels[fit]):.1f})", flush=True)
        z12 = get_modes(kae, stats, states, patches)
        for label, tuned in TUNED.items():
            knobs = tune(kae, head, z12[fit], labels[fit], tuned)
            scores = [get_accuracy(head, decode(kae, z12[rows], knobs), labels[rows]) for rows in (~fit, fit)]
            print(f"{name}, {label}: test {scores[0]:.1f} (fit {scores[1]:.1f}) | largest |size| {knobs[0].abs().max():.2f}, |turn| {knobs[1].abs().max():.2f}", flush=True)
