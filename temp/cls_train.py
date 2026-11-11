"""
The one plan: a KAE whose state is the CLS token alone, with the block-6 patch tokens as an additive control.

    z12 = λ·z6 + gate·B·u        (u = control vector from the block-6 patch tokens; gate = the modes TopK kept)

Everything is control_train.py's (online crops, real CLIP as teacher, control encoder, losses, evaluation), except:
  1. The state is the CLS token (768 numbers). The patch mean and std are gone from the state.
  2. Three arms train on the same stream of batches:
       add        the KAE above
       add_real   the same with the real-start rule: at block 6 a mode is one real number (size only, no phase)
       mlp        baseline, not a KAE: no TopK and an MLP in place of the eigenvalues (train.MlpKoopmanAutoencoder),
                  with the same control added. It shows what the eigenvalue step and the sparsity cost in accuracy.
  3. Optional penalty on the control (arms `real_w*`): the loss gains `control_weight` times the control's share of z12's
     squared size. Without it the control ends up as over 90% of z12, and λ·z6 as under 10%.
No weight decay.

Usage: cls_train.py --name=cls --total_steps=30000
"""

import os
from pathlib import Path

import fire
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import control_train as base
from control_train import train
from koopmann.utils import get_device, set_seed

ARMS = {"add": {"real_start": False}, "add_real": {"real_start": True}, "mlp": None,
        "real_w01": {"real_start": True, "control_weight": 0.1}, "real_w03": {"real_start": True, "control_weight": 0.3}, "real_w1": {"real_start": True, "control_weight": 1.0}}  # fmt: skip


class ShareKae(base.ControlKae):
    """ControlKae that also keeps the control's share of z12's squared size (`control_share`) from its last step."""

    def koopman_forward(self, observable, k_steps):
        own = train.JordanKoopmanAutoencoder.koopman_forward(self, observable, 1)
        out = super().koopman_forward(observable, k_steps)
        extra = (out - own).pow(2).sum()
        self.control_share = extra / (extra + own.pow(2).sum())
        return out


class MlpControl(train.MlpKoopmanAutoencoder):
    """The baseline: an MLP in place of K, plus the control. NOTE: `set_control` before `koopman_forward`, as in ControlKae."""

    def __init__(self, width, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.control_encoder = base.ControlEncoder(width, num_patches, self.latent_features)

    def set_control(self, patches, chunk=4_096):
        self.signal = torch.cat([self.control_encoder(part.float()) for part in patches.split(chunk)])

    def koopman_forward(self, observable, k_steps):
        return super().koopman_forward(observable, 1) + self.signal.view(observable.shape)


class Arm(base.Variant):
    """One model in training. Only the construction differs from control_train's Variant."""

    def __init__(self, name, arm, config, kwargs, stats, total_steps, lr, adam_lr, device, offline):
        self.name = name
        settings = dict(ARMS[arm] or {})
        self.control_weight = settings.pop("control_weight", 0.0)
        if ARMS[arm] is None:
            self.model = MlpControl(hidden=kwargs["hidden_config"][0], **{k: v for k, v in kwargs.items() if k not in ("top_k", "shared_modes")}).to(device)
        else:
            self.model = ShareKae("add", **kwargs, **settings).to(device)
        self.optimizers = train.get_optimizers(self.model, lr, adam_lr, 0.0)
        self.schedulers = [LambdaLR(opt, lambda step: 1 - step / total_steps) for opt in self.optimizers]
        self.totals, self.best = torch.zeros(len(base.LAMBDAS)), -1.0
        self.path = Path(os.getenv("WEIGHTS_CACHE")) / "tokae" / f"{name}.safetensors"
        torch.save({**stats, "arm": arm, "control": "add", **(ARMS[arm] or {})}, self.path.with_name(f"{name}_stats.pt"))
        self.run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project="tokae",
            name=name,
            mode="offline" if offline else "online",
            reinit="create_new",  # Several runs in one process
            tags=[config["encoder"], "imagenet", "cls", "start_to_end", "online_views", "control_add", arm],
            config={**config, "arm": arm, "parameters": sum(p.numel() for p in self.model.parameters())},
        )

    def train_step(self, states, patches, head):
        """control_train's step, plus the penalty on the control's share."""

        self.model.train()
        self.model.set_control(patches)
        losses = train.compute_losses(self.model, states, 0.5, head)
        loss = sum(weight * value for weight, value in zip(base.LAMBDAS, losses))
        if self.control_weight:
            loss = loss + self.control_weight * self.model.control_share

        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers + self.schedulers:
            optimizer.step()

        self.totals += torch.tensor([value.item() for value in losses])

    def report(self, step, steps_done, batch, val, head):
        super().report(step, steps_done, batch, val, head)
        if hasattr(self.model, "control_share"):
            self.run.log({"val/control_share": self.model.control_share.item()}, step=step)
            print(f"step {step} | {self.name}: control share of z12 on validation {self.model.control_share.item():.3f}", flush=True)


def main(
    name: str = "cls",
    arms: tuple = ("add", "add_real", "mlp"),
    encoder: str = "ViT-B-32",
    reference: str = "im_iso_2x4k",
    total_steps: int = 30_000,
    report_every: int = 2_500,
    batch_size: int = 1_024,
    views: int = 2,
    workers: int = 40,
    lr: float = 0.0075,
    adam_lr: float = 0.00075,
    top_k: int = 256,
    ae_dim: int = 2_049,
    ae_hidden: int = 4_096,
    ae_layers: int = 2,
    seed: int = 0,
    smoke: bool = False,
):
    """Trains the arms on the same stream of fresh crops. `smoke` is a short code check, with wandb offline."""

    config = dict(locals())
    set_seed(seed)
    device = get_device()
    teacher = base.Teacher(encoder, device)
    width = teacher.model.visual.ln_post.normalized_shape[0]

    # The reference run's standardization of the CLS token, so the head and the numbers are those of the earlier runs
    stats = torch.load(f"{os.getenv('WEIGHTS_CACHE')}/tokae/{reference}_stats.pt")
    stats = {**stats, "mean": stats["mean"][:width], "std": stats["std"][:width], "tokens": "cls"}
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    val_states, val_patches, val_labels = base.get_val(teacher, encoder, device, limit=2_048 if smoke else None)
    val = ((val_states[..., :width] - mean) / std, val_patches, val_labels)

    head = train.ClipHead(encoder, mean, std, device, zero_shot=True)
    kwargs = {
        "width": width,
        "num_patches": val_patches.shape[1],
        "top_k": top_k,
        "shared_modes": True,
        "in_features": width,
        "latent_features": ae_dim,
        "hidden_config": [ae_hidden] * ae_layers,
        "bias": True,
        "batchnorm": False,
        "nonlinearity": "leaky_relu",
    }
    models = [Arm(f"{name}_{arm}", arm, config, kwargs, stats, total_steps, lr, adam_lr, device, smoke) for arm in arms]

    loader = DataLoader(base.Views("train", teacher, views), batch_size // views, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=True)
    step = 0
    while step < total_steps:
        for images, labels in loader:
            states, patches = teacher(images.flatten(0, 1).to(device))
            states = (states[..., :width] - mean) / std
            for model in models:
                model.train_step(states, patches, head)
            step += 1

            if step % report_every == 0 or step == total_steps:
                batch = (states, patches, labels.repeat_interleave(views).to(device))
                for model in models:
                    model.report(step, report_every, batch, val, head)
            if step == total_steps:
                break

    for model in models:
        model.run.finish()


if __name__ == "__main__":
    fire.Fire(main)
