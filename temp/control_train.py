"""
Jordan KAEs trained ONLINE, with the patch tokens as a control input.

Two changes from scripts/tokae/train.py, which is imported and otherwise used as it is (model class, losses, evaluation, head,
optimizers):

1. Online data. The real model runs on fresh random crops at every step, in place of 3 crops cached once. The cached runs
   reach ~62% accuracy on their own training crops and ~52% on validation, so they fit their fixed crops.
2. Control. A KAE's state is a summary of the tokens (CLS, patch mean, patch std). In the real model the patch tokens keep
   driving the CLS token through attention in blocks 7-12. Here they enter as a control input, as in Koopman with control:
       add       z12 = λ·z6 + gate·B·u      (u = control vector from the block-6 patch tokens; gate = the modes TopK kept)
       bilinear  z12 = (λ + δ(u))·z6        (the control shifts each mode's eigenvalue, in size and angle, per image)
       multiply  z12 = λ·exp(a(u) + i·b(u))·z6   (the control scales each mode's eigenvalue by exp(a) and turns it by b)
       none      z12 = λ·z6                 (the plain KAE)
   u comes from one cross-attention pooling layer over the patch tokens (a few learned queries), then an MLP whose last
   layer starts at zero, so every variant starts as the plain KAE.

Several KAEs train in one job on the same stream of batches, each with its own optimizers and its own wandb run.
No weight decay.

Usage: control_train.py --name=on --controls=none,add,bilinear --total_steps=30000
"""

import os
import sys
import time
from pathlib import Path

import fire
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from safetensors.torch import save_file
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

REPO = Path("/home/nsa325/work/koopmann")
sys.path.insert(0, str(REPO / "scripts" / "tokae"))
load_dotenv(REPO / ".env")
import train  # noqa: E402
from koopmann.utils import get_device, set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
BLOCKS = [5, 11]
# train.py's losses in order, and their weights in the reference runs
LOSS_NAMES = [f"loss/{name}" for name in ["reconstruction", "latent", "state", "variance", "covariance", "isometric", "embedding", "kl", "overlap"]]
LAMBDAS = (1.0, 1.0, 1.0, 0.0, 0.0, 0.1, 1.0, 0.03, 0.0)


############# DATA #############


class Views(Dataset):
    """ImageNet images as uint8 tensors: `views` random crops per image (train.py's crop and flip), or the center crop."""

    def __init__(self, split, teacher, views):
        self.data = train.ImageNet(split, None).data
        self.views = views
        crop = [transforms.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip()]
        self.transform = transforms.Compose([*(crop if views else teacher.resize), teacher.to_rgb, transforms.PILToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # shape: [views, 3, 224, 224]
        return torch.stack([self.transform(sample["image"]) for _ in range(max(self.views, 1))]), sample["label"]


class Teacher:
    """The real CLIP model. Gives each image's state after blocks 6 and 12, and its patch tokens after block 6."""

    def __init__(self, encoder, device):
        model, _, preprocess = open_clip.create_model_and_transforms(encoder, pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
        self.model = model.to(device).eval()
        self.resize, self.to_rgb = preprocess.transforms[:2], preprocess.transforms[2]
        normalize = preprocess.transforms[-1]
        self.pixel_mean = torch.tensor(normalize.mean, device=device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(normalize.std, device=device).view(1, 3, 1, 1)
        self.kept = []
        for block in BLOCKS:
            model.visual.transformer.resblocks[block].register_forward_hook(lambda _, __, output: self.kept.append(output[0] if isinstance(output, tuple) else output))

    @torch.no_grad()
    def __call__(self, images):
        """uint8 images [n, 3, 224, 224] -> (states [n, 2, 3 * width], patch tokens after block 6 [n, patches, width])."""

        self.kept.clear()
        self.model.encode_image((images.float() / 255 - self.pixel_mean) / self.pixel_std)
        # NOTE: The summary train.py caches as `cls_patch_std`: CLS token, mean of the patch tokens, their std
        states = [torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1), tokens[:, 1:].std(dim=1)], dim=-1) for tokens in self.kept]

        return torch.stack(states, dim=1), self.kept[0][:, 1:]


############# MODEL #############


class ControlEncoder(nn.Module):
    """Patch tokens -> one control signal per latent dimension: cross-attention pooling with learned queries, then an MLP."""

    def __init__(self, width, num_patches, out_features, dim=512, queries=16, heads=8, hidden=2048):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.embed = nn.Linear(width, dim)
        # NOTE: 3D, so train.get_optimizers gives these to AdamW (Muon is for weight matrices)
        self.positions = nn.Parameter(0.02 * torch.randn(1, num_patches, dim))
        self.queries = nn.Parameter(0.02 * torch.randn(1, queries, dim))
        self.to_query, self.to_key, self.to_value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(queries * dim, hidden), nn.LeakyReLU(), nn.Linear(hidden, out_features))
        # NOTE: The control starts at zero, so the model starts as the plain KAE
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, patches):
        tokens = self.embed(self.norm(patches)) + self.positions
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        query = split(self.to_query(self.queries)).expand(len(tokens), -1, -1, -1)
        # shape: [images, heads, queries, dim / heads]
        pooled = F.scaled_dot_product_attention(query, split(self.to_key(tokens)), split(self.to_value(tokens)))

        return self.mlp(pooled.transpose(1, 2).flatten(1))


class ControlKae(train.JordanKoopmanAutoencoder):
    """
    train.py's Jordan KAE, with a control input in K's step (see the top of this file).
    NOTE: `set_control` has to be called with a batch's patch tokens before that batch goes through `koopman_forward`.
    This keeps train.py's losses and evaluation usable as they are.
    """

    def __init__(self, control, width, num_patches, controlled=None, **kwargs):
        super().__init__(**kwargs)
        self.control = control
        self.controlled = controlled
        self.signal = None
        if control != "none":
            self.control_encoder = ControlEncoder(width, num_patches, self.latent_features)

    def set_control(self, patches, chunk=4_096):
        if self.control != "none":
            self.signal = torch.cat([self.control_encoder(part.float()) for part in patches.split(chunk)])
            if self.controlled:
                # Only the real coordinate and the first `controlled` pairs get control. λ alone steps the other modes.
                self.signal = self.signal * (torch.arange(self.signal.shape[1], device=self.signal.device) < 1 + 2 * self.controlled)

    def koopman_forward(self, observable, k_steps):
        assert k_steps == 1, "the control is defined for the single step from start to end"
        out = super().koopman_forward(observable, 1)
        if self.control == "none":
            return out

        signal = self.signal.view(observable.shape)
        real, blocks = observable[..., :1], observable[..., 1:].unflatten(-1, (-1, 2))
        signal_real, signal_blocks = signal[..., :1], signal[..., 1:].unflatten(-1, (-1, 2))
        if self.control == "multiply":
            # The signal is a log-scale a and a turn b of each mode's eigenvalue: exp(a + ib)·(x + iy), with x + iy = λ·z6.
            # A mode that is off at the start stays off, and with a zero signal the step is λ alone.
            x, y = out[..., 1:].unflatten(-1, (-1, 2)).unbind(dim=-1)
            c, d = signal_blocks[..., 0].exp() * signal_blocks[..., 1].cos(), signal_blocks[..., 0].exp() * signal_blocks[..., 1].sin()
            return torch.cat([signal_real.exp() * out[..., :1], torch.stack([c * x - d * y, d * x + c * y], dim=-1).flatten(-2)], dim=-1)

        if self.control == "add":
            # Only the modes TopK kept at the start receive control, so the end keeps the start's modes
            kept = (blocks != 0).any(dim=-1, keepdim=True)
            extra = torch.cat([signal_real * (real != 0), (signal_blocks * kept).flatten(-2)], dim=-1)
        else:
            # The signal is a shift δ = c + id of each mode's eigenvalue: (c + id)(x + iy)
            x, y = blocks[..., 0], blocks[..., 1]
            c, d = signal_blocks[..., 0], signal_blocks[..., 1]
            extra = torch.cat([signal_real * real, torch.stack([c * x - d * y, d * x + c * y], dim=-1).flatten(-2)], dim=-1)

        return out + extra


############# TRAINING #############


class Variant:
    """One KAE in training: its model, optimizers, learning-rate schedule, wandb run and running losses."""

    def __init__(self, name, control, config, kwargs, stats, total_steps, lr, adam_lr, device, offline):
        self.name = name
        self.model = ControlKae(control, **kwargs).to(device)
        self.optimizers = train.get_optimizers(self.model, lr, adam_lr, 0.0)
        # NOTE: The rate decays linearly to zero per step (train.py: per epoch)
        self.schedulers = [LambdaLR(opt, lambda step: 1 - step / total_steps) for opt in self.optimizers]
        self.totals, self.best = torch.zeros(len(LAMBDAS)), -1.0
        self.path = Path(os.getenv("WEIGHTS_CACHE")) / "tokae" / f"{name}.safetensors"
        torch.save({**stats, "control": control}, self.path.with_name(f"{name}_stats.pt"))
        self.run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project="tokae",
            name=name,
            mode="offline" if offline else "online",
            reinit="create_new",  # Several runs in one process
            tags=[config["encoder"], "imagenet", "cls+patch+std", "start_to_end", "online_views", "sparse", "jordan", "shared_modes", f"control_{control}"],
            config={**config, "control": control, "parameters": sum(p.numel() for p in self.model.parameters())},
        )

    def train_step(self, states, patches, head):
        self.model.train()
        self.model.set_control(patches)
        losses = train.compute_losses(self.model, states, 0.5, head)
        loss = sum(weight * value for weight, value in zip(LAMBDAS, losses))

        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers + self.schedulers:
            optimizer.step()

        self.totals += torch.tensor([value.item() for value in losses])

    @torch.no_grad()
    def report(self, step, steps_done, batch, val, head):
        """Logs the mean training losses since the last report, and train.py's evaluation on the last batch and on validation."""

        self.model.eval()
        logs = dict(zip(LOSS_NAMES, (self.totals / steps_done).tolist()))
        self.totals.zero_()
        for split, (states, patches, labels) in [("train", batch), ("val", val)]:
            self.model.set_control(patches)
            # NOTE: train.evaluate's ridge classifier is not used here, so it gets zeros and its number is dropped
            classifier = torch.zeros(states.shape[-1] + 1, 1_000, device=states.device)
            logs.update(train.evaluate(self.model, states, labels, classifier, head, split))
            del logs[f"{split}/predicted_accuracy"]
        self.run.log(logs, step=step)

        if logs["val/zero_shot_accuracy"] > self.best:
            self.best = logs["val/zero_shot_accuracy"]
            save_file(self.model.state_dict(), self.path)

        print(
            f"step {step} | {self.name}: accuracy {logs['val/zero_shot_accuracy']:.4f}, agreement {logs['val/clip_agreement']:.4f}, "
            f"state FVU {logs['val/one_step_fvu']:.4f} (last training batch: accuracy {logs['train/zero_shot_accuracy']:.4f})",
            flush=True,
        )


def get_val(teacher, encoder, device, limit=None):
    """Validation states (train.py's cache), block-6 patch tokens (extracted once, in half precision) and labels."""

    val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / f"clip_cls_patch_std_{encoder}_val_blocks_5_11.pt")
    states, labels = val["states"][:limit].float(), val["labels"][:limit]

    path = HERE / "control_cache" / f"val_patches_{encoder}_{len(states)}.pt"
    if not path.exists():
        images = Subset(Views("val", teacher, 0), range(len(states)))
        outputs = [teacher(batch[:, 0].to(device)) for batch, _ in DataLoader(images, batch_size=256, num_workers=16)]
        # NOTE: The online pipeline has to give the cached states back, or its numbers would not compare with the cached runs
        gap = (torch.cat([s.cpu() for s, _ in outputs]) - states).abs().max().item()
        assert gap < 0.05, f"online states differ from the cached ones by up to {gap}"
        path.parent.mkdir(exist_ok=True)
        torch.save(torch.cat([p.half().cpu() for _, p in outputs]), path)

    return states.to(device), torch.load(path).to(device), labels.to(device)


def main(
    name: str = "on",
    controls: tuple = ("none", "add", "bilinear"),
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
    """Trains one KAE per control, all on the same stream of fresh crops. `smoke` is a short code check, with wandb offline."""

    config = dict(locals())
    set_seed(seed)
    device = get_device()
    teacher = Teacher(encoder, device)

    # The reference run's standardization, so states, head and numbers are those of the cached runs
    stats = torch.load(f"{os.getenv('WEIGHTS_CACHE')}/tokae/{reference}_stats.pt")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    val_states, val_patches, val_labels = get_val(teacher, encoder, device, limit=2_048 if smoke else None)
    val = ((val_states - mean) / std, val_patches, val_labels)

    head = train.ClipHead(encoder, mean, std, device, zero_shot=True)
    kwargs = {
        "width": val_patches.shape[-1],
        "num_patches": val_patches.shape[1],
        "top_k": top_k,
        "shared_modes": True,
        "in_features": val_states.shape[-1],
        "latent_features": ae_dim,
        "hidden_config": [ae_hidden] * ae_layers,
        "bias": True,
        "batchnorm": False,
        "nonlinearity": "leaky_relu",
    }
    variants = [Variant(f"{name}_{control}", control, config, kwargs, stats, total_steps, lr, adam_lr, device, smoke) for control in controls]

    loader = DataLoader(Views("train", teacher, views), batch_size // views, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=True)
    step = 0
    while step < total_steps:
        for images, labels in loader:
            states, patches = teacher(images.flatten(0, 1).to(device))
            states = (states - mean) / std
            for variant in variants:
                variant.train_step(states, patches, head)
            step += 1

            if step % report_every == 0 or step == total_steps:
                batch = (states, patches, labels.repeat_interleave(views).to(device))
                for variant in variants:
                    variant.report(step, report_every, batch, val, head)
            if step == total_steps:
                break

    for variant in variants:
        variant.run.finish()


if __name__ == "__main__":
    fire.Fire(main)
