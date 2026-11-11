"""
The input ceiling: how far can a plain network get from the block-6 summary, with none of the KAE's structure?

The KAE sees only a summary of block 6 (CLS token, mean of the patch tokens, their std; 2,304 numbers) and has to arrive
at the real model's block-12 state. If that summary has already lost what blocks 7-12 need, no KAE can close the gap to
the real model, however long it trains. Three plain networks are trained on exactly the KAE's data to find out:

  mimic probe        block-6 summary -> block-12 state, with the KAE's own prediction losses (state error + embedding loss
                     + 0.03 KL to the real head's answer) and nothing else: no latent, no TopK, no K, no reconstruction.
                     It is read through the real head, so its accuracy and agreement are the KAE's own two numbers.
                     Layers: the KAE's encoder and decoder stacked (4 hidden layers of 4096) without the latent between.
  label probe, 6     block-6 summary -> the 1,000 ImageNet classes, cross-entropy on the labels (2 hidden layers of 4096).
                     How much class information the summary holds at all, read out directly.
  label probe, 12    the same network on the block-12 summary: what the same probe finds after the six blocks.

Everything else is the KAE recipe: the reference run's standardization, Muon (weight decay 0.1) + AdamW from train.py,
a linear decay of the learning rate to zero, batch 1024, the repo's own layer class and initialisation.
A probe is a LOWER bound on what the input allows: a better-tuned probe could only score higher.

Usage: ceiling_probe.py --encoder=ViT-B-32 --reference=im_iso_2x4k_wd --name=probe_clip_b6 --num_epochs=8
       [--extra_dataset=w21] [--smoke=True: a few steps on validation states, no wandb]
"""

import os
import sys
import time

import fire
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.optim.lr_scheduler import LinearLR

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402
from koopmann.models import Autoencoder  # noqa: E402
from koopmann.utils import get_device, set_seed  # noqa: E402

BLOCKS, TOKENS, VIEWS = [5, 11], "cls_patch_std", 3
LAMBDA_EMBEDDING, LAMBDA_KL = 1.0, 0.03  # The KAE's weights on its prediction


def get_probe(in_features, out_features, hidden, device):
    """A plain MLP from the repo's own layers: the encoder half of an `Autoencoder`, with its initialisation."""

    autoencoder = Autoencoder(
        in_features=in_features, latent_features=out_features, hidden_config=hidden, bias=True, batchnorm=False, nonlinearity="leaky_relu"
    )

    return autoencoder.components.encoder.to(device)


def main(
    encoder: str = "ViT-B-32",
    reference: str = "im_iso_2x4k_wd",
    name: str = "probe_clip_b6",
    num_epochs: int = 8,
    extra_dataset: str | None = None,
    batch_size: int = 1_024,
    lr: float = 0.0075,
    adam_lr: float = 0.00075,
    weight_decay: float = 0.1,
    width: int = 4_096,
    seed: int = 0,
    smoke: bool = False,
):
    config = dict(locals())
    set_seed(seed)
    device = get_device()

    # The reference KAE run's standardization, so the probes see exactly what that KAE saw
    stats = torch.load(f"{os.getenv('WEIGHTS_CACHE')}/tokae/{reference}_stats.pt")
    mean, std = stats["mean"], stats["std"]
    assert stats["blocks"] == BLOCKS and mean.ndim == 1, "the reference run has to use blocks 6 and 12 with global standardization"

    val_states, val_labels = train.get_states(encoder, "imagenet", "val", device, BLOCKS, tokens=TOKENS, extract=False)
    val_states = (val_states.float() - mean) / std

    if smoke:
        # NOTE: A code check only: 40,000 validation states stand in for the training set
        train_states, train_labels = val_states[:40_000].half(), val_labels[:40_000]
        val_states, val_labels = val_states[40_000:], val_labels[40_000:]
        train_sample, train_sample_labels = train_states[::4].float(), train_labels[::4]
    else:
        train_states, train_labels = train.get_states(encoder, "imagenet", "train", device, BLOCKS, VIEWS, TOKENS, extract=False)
        # NOTE: In place and then to half precision, as in train.py
        train_states.sub_(mean).div_(std)
        stride = max(VIEWS, len(train_states) // 50_000)
        train_sample, train_sample_labels = train_states[::stride].clone(), train_labels[::stride]
        train_states = train_states.half()

        # Extra unlabeled images, as in train.py. Their label is -1, so only the mimic probe learns from them
        if extra_dataset:
            extra_states, extra_labels = train.get_states(encoder, extra_dataset, "train", device, BLOCKS, tokens=TOKENS, extract=False)
            extra_states = extra_states.sub_(mean).div_(std).half()
            train_states = torch.cat([train_states, extra_states])
            train_labels = torch.cat([train_labels, torch.full_like(extra_labels, -1)])
            del extra_states

    val_states, val_labels = val_states.to(device), val_labels.to(device)
    train_sample, train_sample_labels = train_sample.to(device), train_sample_labels.to(device)
    head_class = train.DinoHead if encoder in train.DINO_MODELS else train.ClipHead
    # NOTE: Building CLIP's zero-shot classifier takes 80,000 prompts, minutes on the CPU, so the code check loads a saved copy
    saved_classes = smoke and encoder not in train.DINO_MODELS
    head = head_class(encoder, mean, std, device, zero_shot=not saved_classes)
    if saved_classes:
        head.classes = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/zero_shot_head.pt").float().to(device)

    size = train_states.shape[-1]
    probes = {
        "mimic": get_probe(size, size, [width] * 4, device),
        "label_block6": get_probe(size, 1_000, [width] * 2, device),
        "label_block12": get_probe(size, 1_000, [width] * 2, device),
    }
    optimizers = [opt for probe in probes.values() for opt in train.get_optimizers(probe, lr, adam_lr, weight_decay)]
    schedulers = [LinearLR(opt, 1.0, 0.0, total_iters=num_epochs) for opt in optimizers]

    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="tokae",
        name=name,
        mode="disabled" if smoke else "online",
        tags=[encoder, "imagenet", TOKENS.replace("_", "+"), "probe", "no_kae", *([f"plus_{extra_dataset}"] if extra_dataset else [])],
        config={
            **config,
            "num_train_states": len(train_states),
            "what": "input ceiling: plain MLPs from the block-6 summary, no KAE (mimic probe to the block-12 state, label probes to the class)",
            "parameters": {key: sum(p.numel() for p in probe.parameters()) for key, probe in probes.items()},
        },
    )

    @torch.no_grad()
    def evaluate(states, labels, split):
        for probe in probes.values():
            probe.eval()
        start, end = states[:, 0], states[:, 1]
        pred = probes["mimic"](start)
        answers, real = head.classify(pred), head.classify(end)

        return {
            # NOTE: The KAE's metric names, so the mimic probe plots next to the KAE runs
            f"{split}/zero_shot_accuracy": (answers == labels).float().mean().item(),
            f"{split}/clip_agreement": (answers == real).float().mean().item(),
            f"{split}/one_step_fvu": train.compute_fvu(pred[:, None], end[:, None]).item(),
            f"{split}/embedding_loss": train.compute_embedding_loss(head, pred, end).item(),
            f"{split}/kl_loss": train.compute_kl_loss(head, pred, end).item(),
            f"{split}/label_accuracy_block6": (probes["label_block6"](start).argmax(dim=-1) == labels).float().mean().item(),
            f"{split}/label_accuracy_block12": (probes["label_block12"](end).argmax(dim=-1) == labels).float().mean().item(),
            f"{split}/real_model_accuracy": (real == labels).float().mean().item(),
        }

    for epoch in range(num_epochs):
        for probe in probes.values():
            probe.train()
        began = time.time()
        batches = torch.randperm(len(train_states)).split(batch_size)
        totals = torch.zeros(5)
        for idx in batches:
            states, labels = train_states[idx].to(device).float(), train_labels[idx].to(device)
            start, end = states[:, 0], states[:, 1]

            pred = probes["mimic"](start)
            state_loss = train.compute_fvu(pred[:, None], end[:, None]).mean()
            embedding_loss = train.compute_embedding_loss(head, pred, end)
            kl_loss = train.compute_kl_loss(head, pred, end)

            # NOTE: Images without a label (-1) are left out of the label probes' loss
            label_losses = [F.cross_entropy(probes[key](x), labels, ignore_index=-1) for key, x in [("label_block6", start), ("label_block12", end)]]

            # NOTE: The probes share no weights, so one sum trains each on its own loss
            loss = state_loss + LAMBDA_EMBEDDING * embedding_loss + LAMBDA_KL * kl_loss + sum(label_losses)
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            totals += torch.tensor([state_loss.item(), embedding_loss.item(), kl_loss.item(), *[value.item() for value in label_losses]])

        for scheduler in schedulers:
            scheduler.step()

        totals = (totals / len(batches)).tolist()
        logs = {
            "loss/state": totals[0],
            "loss/embedding": totals[1],
            "loss/kl": totals[2],
            "loss/label_block6": totals[3],
            "loss/label_block12": totals[4],
            **evaluate(train_sample, train_sample_labels, "train"),
            **evaluate(val_states, val_labels, "val"),
        }
        wandb.log(logs)
        print(
            f"epoch {epoch + 1}/{num_epochs} ({(time.time() - began) / 60:.1f} min) | mimic probe: accuracy {logs['val/zero_shot_accuracy']:.4f}, "
            f"agreement {logs['val/clip_agreement']:.4f}, state FVU {logs['val/one_step_fvu']:.4f} (train accuracy {logs['train/zero_shot_accuracy']:.4f}) | "
            f"label probe at block 6: {logs['val/label_accuracy_block6']:.4f} (train {logs['train/label_accuracy_block6']:.4f}) | "
            f"at block 12: {logs['val/label_accuracy_block12']:.4f} (train {logs['train/label_accuracy_block12']:.4f}) | "
            f"real model {logs['val/real_model_accuracy']:.4f}",
            flush=True,
        )

    if not smoke:
        folder = os.path.dirname(os.path.abspath(__file__)) + "/probes"
        os.makedirs(folder, exist_ok=True)
        torch.save({key: probe.state_dict() for key, probe in probes.items()}, f"{folder}/{name}.pt")
    wandb.finish()
    print("done")


if __name__ == "__main__":
    fire.Fire(main)
