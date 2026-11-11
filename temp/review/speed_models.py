"""
The pieces of the speed review. Real CLIP ViT-B/32's image tower is cut in two at block 6:

    front   pixels -> tokens after block 6 (patch embedding and blocks 1-6)
    back    tokens after block 6 -> CLS token after block 12 (blocks 7-12)
    head    CLS token after block 12 -> 1,000 class scores (CLIP's last norm, its projection, the zero-shot classes)

A KAE's step (`Kae`) stands in for `back`. Run this file to check that the pieces give the accuracies seen in training.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from common import *  # noqa: F403  (puts the scratchpad on the path)

import cls_train  # noqa: E402
import control_train as base  # noqa: E402
from control_train import train  # noqa: E402
from demo_utils import load_clip  # noqa: E402

FOLDER = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
KAES = ["im_iso_2x4k", "cls_add_real", "cls_mlp"]
WIDTH, PATCHES = 768, 49
torch.set_grad_enabled(False)


class Kae(torch.nn.Module):
    """A trained stand-in for blocks 7-12: the older KAE (2,304 state, no control), a CLS-only KAE, or the MLP baseline."""

    def __init__(self, name):
        super().__init__()
        stats = torch.load(FOLDER / f"{name}_stats.pt")
        kwargs = dict(in_features=len(stats["mean"]), latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
        if stats.get("arm") == "mlp":
            self.model = cls_train.MlpControl(hidden=4096, width=WIDTH, num_patches=PATCHES, **kwargs)
        elif "control" in stats:
            self.model = base.ControlKae("add", width=WIDTH, num_patches=PATCHES, top_k=256, shared_modes=True, real_start=stats["real_start"], **kwargs)
        else:
            self.model = train.JordanKoopmanAutoencoder(256, shared_modes=True, **kwargs)
        self.model.load_state_dict(load_file(FOLDER / f"{name}.safetensors"))
        self.register_buffer("mean", stats["mean"])
        self.register_buffer("std", stats["std"])
        self.eval()

    def forward(self, tokens):
        """Tokens after block 6 [n, 50, 768] -> the CLS token after block 12 [n, 768]."""

        cls, patches = tokens[:, 0], tokens[:, 1:]
        if hasattr(self.model, "set_control"):
            state = cls
            self.model.set_control(patches)
        else:
            # The older state: CLS token, mean of the patch tokens, their std
            state = torch.cat([cls, patches.mean(dim=1), patches.std(dim=1)], dim=1)
        end = self.model.decode(self.model.koopman_forward(self.model.encode((state - self.mean) / self.std), 1))

        return (end * self.std + self.mean)[:, :WIDTH]


def front(visual, pixels):
    tokens = visual._embeds(pixels)
    for block in visual.transformer.resblocks[:6]:
        tokens = block(tokens)

    return tokens


def back(blocks, tokens):
    for block in blocks:
        tokens = block(tokens)

    return tokens[:, 0]


def head(visual, classes, cls):
    return F.normalize(visual.ln_post(cls) @ visual.proj, dim=-1) @ classes


def load_classes():
    """CLIP's zero-shot classes for ImageNet. shape: [512, 1000]"""

    return torch.load(base.HERE / "zero_shot_head.pt").float()


def count(module):
    return sum(p.numel() for p in module.parameters())


def pipeline(name, device):
    """A whole model from pixels to class scores, holding only the weights it needs. `clip` is the real model."""

    visual, classes = load_clip().visual, load_classes().to(device)
    if name == "clip":
        blocks = visual.transformer.resblocks[6:]
        step = lambda tokens: back(blocks, tokens)  # noqa: E731
    else:
        del visual.transformer.resblocks[6:]
        step = Kae(name).to(device)
    visual.to(device)

    return lambda pixels: head(visual, classes, step(front(visual, pixels)))


def half(name, device):
    """The replaced half alone, from the tokens after block 6 to the CLS token after block 12."""

    if name != "clip":
        return Kae(name).to(device)
    blocks = load_clip().visual.transformer.resblocks[6:].to(device)

    return lambda tokens: back(blocks, tokens)


def load():
    """Everything saved so far in data/speed.json."""

    path = DATA / "speed.json"
    return json.loads(path.read_text()) if path.exists() else {}


def update(key, value):
    """Puts one block of results into data/speed.json."""

    (DATA / "speed.json").write_text(json.dumps({**load(), key: value}, indent=1))


def check(images=50_000):
    """The pieces against real CLIP on random pixels, and each KAE's accuracy on the first validation images."""

    device = "cuda"
    visual, classes = load_clip().visual.to(device), load_classes().to(device)
    pixels = torch.randn(8, 3, 224, 224, device=device)
    whole = F.normalize(visual(pixels), dim=-1) @ classes
    pieces = head(visual, classes, back(visual.transformer.resblocks[6:], front(visual, pixels)))
    print(f"front + back + head against the model's own forward: largest gap {(whole - pieces).abs().max():.1e}")

    states, patches, labels = [x[:images].to(device) for x in base.get_val(None, "ViT-B-32", "cpu")]
    tokens = torch.cat([states[:, 0, None, :WIDTH], patches.float()], dim=1)
    answers = {"clip": head(visual, classes, states[:, 1, :WIDTH]).argmax(dim=1)}  # Real CLIP, from its stored block-12 state
    for name in KAES + ["share_real_w03"]:
        kae = Kae(name).to(device)
        answers[name] = torch.cat([head(visual, classes, kae(part)).argmax(dim=1) for part in tokens.split(1_000)])
    accuracy = {name: round((answer == labels).float().mean().item() * 100, 2) for name, answer in answers.items()}
    print(f"accuracy on the first {images} validation images: {accuracy}")
    update("check", {"images": images, "accuracy": accuracy})


if __name__ == "__main__":
    check()
