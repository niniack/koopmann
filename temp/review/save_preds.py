"""
Saves every model's answers, image by image, on ImageNet validation and on the shift sets.

For each model of the review: its 5 best classes and the gap between its best and second-best score (log probability),
read with CLIP's zero-shot head from the model's decoded block-12 state. The real model's answers are saved once per
set, with the labels. Accuracy and agreement with the real model are printed next to the number each run's log gave.

Files in data/: preds_<model>.pt (validation), shift_<set>_<model>.pt, and preds_dino.pt for real DINOv3.
Keys: top5 [images, 5], gap [images], and labels for the real model.

Usage: save_preds.py
"""

import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import DATA, MODELS, SCRATCH

import cls_train  # noqa: E402
import control_train as base  # noqa: E402
from control_train import train  # noqa: E402

torch.set_grad_enabled(False)
TOKAE = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
SHIFT = ["imagenetv2", "imagenet_sketch", "imagenet-r", "imagenet-a", "cifar100_test"]
DINO = "im_dino_w21_wd_e30"
# Validation accuracy in percent, as each run's own log gave it (sources in digest/images.md)
KNOWN = {"clip": 63.3, "im_iso_2x4k": 52.1, "im_mlpk_2x4k": 53.8, "on_none": 52.2, "on_add": 58.7, "on_bilinear": 58.8, "cls_add": 59.08, "cls_add_real": 58.94,
         "cls_mlp": 60.46, "share_real_w01": 59.14, "share_real_w03": 59.44, "share_real_w1": 59.34, "dino": 83.9, DINO: 62.9}  # fmt: skip


def load_model(name):
    """A trained model and its run's stats. NOTE: Each is built as its training script built it; the stats say which script."""

    stats = torch.load(TOKAE / f"{name}_stats.pt")
    kwargs = dict(in_features=len(stats["mean"]), latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
    if stats.get("arm") == "mlp":
        model = cls_train.MlpControl(hidden=4096, width=768, num_patches=49, **kwargs)
    elif "control" in stats:
        model = base.ControlKae(stats["control"], width=768, num_patches=49, top_k=256, shared_modes=True, real_start=stats.get("real_start", False), **kwargs)
    elif stats["shared_modes"]:
        model = train.JordanKoopmanAutoencoder(256, shared_modes=True, **kwargs)
    else:
        model = train.MlpKoopmanAutoencoder(4096, **kwargs)
    model.load_state_dict(load_file(TOKAE / f"{name}.safetensors"))
    return model.eval(), stats


def load_set(name):
    """A set's states [images, 2, 2304], block-6 patch tokens, labels, and CIFAR's own zero-shot classifier (else None)."""

    if name == "val":
        return *base.get_val(None, "ViT-B-32", "cpu"), None
    data = torch.load(SCRATCH / "domain_cache" / f"{name}.pt")
    classes = torch.load(SCRATCH / "domain_cache" / "cifar100_classes.pt").float() if "cifar" in name else None
    return data["states"], data["patches"], data["labels"], classes


def run_model(model, stats, states, patches):
    """The model's block-12 state in the real model's units, from the real block-6 state (and patch tokens, if it has a control)."""

    mean, std = stats["mean"], stats["std"]
    states = (states[..., : len(mean)] - mean) / std
    if hasattr(model, "set_control"):
        model.set_control(patches)
    # NOTE: The control is kept for the whole set in order, so the step runs on the whole set at once
    z6 = torch.cat([train.encode_pairs(model, part)[:, 0] for part in states.split(5_000)])
    z12 = model.koopman_forward(z6, 1)
    return torch.cat([model.decode(part) for part in z12.split(5_000)]) * std + mean


def get_answers(head, states):
    """From block-12 states: the 5 best classes, and the gap between the best and second-best log probability."""

    top = [head.get_log_probs(part).topk(5, dim=-1) for part in states.split(5_000)]
    values, top5 = torch.cat([t.values for t in top]), torch.cat([t.indices for t in top])
    return {"top5": top5, "gap": values[:, 0] - values[:, 1]}


def save_set(prefix, real_name, head, names, states, patches, labels):
    """Saves the real model's answers on one set of images, then each model's. Prints accuracy and agreement."""

    real = get_answers(head, states[:, 1])
    torch.save({**real, "labels": labels}, DATA / f"{prefix}_{real_name}.pt")
    print(f"{prefix} | {len(labels)} images | real model ({real_name}): accuracy {(real['top5'][:, 0] == labels).float().mean():.4f}", flush=True)

    for name in names:
        model, stats = load_model(name)
        answers = get_answers(head, run_model(model, stats, states, patches))
        torch.save(answers, DATA / f"{prefix}_{name}.pt")
        accuracy = (answers["top5"][:, 0] == labels).float().mean()
        agreement = (answers["top5"][:, 0] == real["top5"][:, 0]).float().mean()
        known = f" (log: {KNOWN[name]})" if prefix == "preds" else ""
        print(f"{prefix} | {name}: accuracy {accuracy:.4f}{known}, agreement {agreement:.4f}", flush=True)


def main():
    # NOTE: The head gets states in the real model's units, so its own mean and std are 0 and 1
    zero, one = torch.zeros(768), torch.ones(768)
    head = train.ClipHead("ViT-B-32", zero, one, "cpu", zero_shot=True)
    imagenet_classes = head.classes
    for name in ["val", *SHIFT]:
        states, patches, labels, classes = load_set(name)
        head.classes = imagenet_classes if classes is None else classes
        save_set("preds" if name == "val" else f"shift_{name}", "clip", head, list(MODELS), states, patches, labels)

    # The DINOv3 KAE: stored DINOv3 states, judged by DINOv3's ImageNet probe
    states, labels = train.get_states("dinov3-vitb16", "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
    save_set("preds", "dino", train.DinoHead("dinov3-vitb16", zero, one, "cpu", zero_shot=True), [DINO], states.float(), None, labels)


if __name__ == "__main__":
    main()
