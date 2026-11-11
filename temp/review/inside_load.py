"""
Loads any trained model of the review, with the validation data and the head it is judged on.

One loader for every kind of checkpoint: KAEs on stored states (im_*), KAEs with a control (on_*, cls_*, share_*) and
the two MLP baselines. The constructors are those of demo_utils.Kae, animals/probe_kaes.py, angle_tests.py and cls_train.py.
"""

import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import DATA, MODELS, SCRATCH

import cls_train
import control_train as base
from control_train import train

torch.set_grad_enabled(False)
FOLDER = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
# Models outside common.MODELS that some questions need, and the name charts use for them
EXTRA = {
    "im_iso_2x4k_s1": "KAE, no control (older state), seed 1",
    "im_realstart_2x4k_wd": "KAE, real start and weight decay (older state)",
    "im_realstart_2x4k_wd_s1": "KAE, real start and weight decay (older state), seed 1",
    "im_dino_w21_wd_e30": "DINOv3 KAE, no control",
}
NAMES = {**MODELS, **EXTRA}
KAES = [name for name in MODELS if "mlp" not in name]  # the models of the review that have modes
get_encoder = lambda name: "dinov3-vitb16" if "dino" in name else "ViT-B-32"
is_kae = lambda model: hasattr(model, "block_real")


def load_model(name):
    """A trained model and its run's stats (standardization and settings)."""

    stats = torch.load(FOLDER / f"{name}_stats.pt")
    kwargs = {"in_features": len(stats["mean"]), "latent_features": 2049, "hidden_config": stats["hidden_config"], "bias": True, "batchnorm": False, "nonlinearity": "leaky_relu"}  # fmt: skip
    if stats.get("arm") == "mlp":
        model = cls_train.MlpControl(hidden=4096, width=768, num_patches=49, **kwargs)
    elif not stats["shared_modes"]:
        model = train.MlpKoopmanAutoencoder(4096, **kwargs)
    else:
        model = base.ControlKae(stats.get("control", "none"), width=768, num_patches=49, top_k=256, shared_modes=True, real_start=stats.get("real_start", False), **kwargs)  # fmt: skip
    model.load_state_dict(load_file(FOLDER / f"{name}.safetensors"))
    return model.eval(), stats


def load_val(encoder, limit=None):
    """Raw validation states [images, 2 (block 6, block 12), 2304], block-6 patch tokens (CLIP only) and labels."""

    if encoder == "ViT-B-32":
        states, patches, labels = base.get_val(None, encoder, "cpu")
        return states[:limit], patches[:limit], labels[:limit]
    states, labels = train.get_states(encoder, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
    return states[:limit].float(), None, labels[:limit]


def load_head(encoder, stats):
    """The head that reads a standardized block-12 state: CLIP's zero-shot head, or DINOv3's ImageNet probe."""

    if encoder != "ViT-B-32":
        return train.DinoHead(encoder, stats["mean"], stats["std"], "cpu", zero_shot=True)
    # NOTE: The zero-shot classifier is slow to build on a CPU, so the saved copy is used
    head = train.ClipHead(encoder, stats["mean"], stats["std"], "cpu", zero_shot=False)
    head.classes = torch.load(SCRATCH / "zero_shot_head.pt").float()
    return head


def get_eigenvalues(kae):
    """One complex eigenvalue per mode [1025]. Mode 0 is the single real mode."""

    return torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)]).detach()


def load_results(name):
    """What inside_pass.py saved for a model."""

    return torch.load(DATA / f"inside_{name}.pt")


def load_latents(name):
    """A model's latents on validation: `z6`, `z12` (its prediction) and `end` (block 12 encoded), each [images, 2049]."""

    return torch.load(DATA / f"inside_latents_{name}.pt")


def get_modes(results):
    """Per mode with a complex eigenvalue (mode 0, the real one, is left out): size and angle of λ, and the images that use it."""

    eigenvalues, usage = results["eigenvalues"][1:], results["usage"][1:]
    return {"size": eigenvalues.abs(), "angle": eigenvalues.angle().rad2deg().abs(), "usage": usage}


def load_z6(name):
    """Each validation image's modes at block 6, complex [images, 1024] (mode 0 left out). 0 where TopK dropped the mode."""

    latent = load_latents(name)["z6"]
    return torch.complex(latent[:, 1::2], latent[:, 2::2])
