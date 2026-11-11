"""
Turns a block-12 state back into a picture, following Splice's CLS inversion (https://github.com/omerbt/Splice, inversion.py).

A small conv net (the "skip" net of Deep Image Prior, with Splice's settings) is fed fixed noise and trained, for one target only,
until frozen DINOv3 reads the picture it paints as the target state. The net is the only image prior: nothing else keeps the picture natural.
What differs from Splice: DINOv3 ViT-B/16 in place of DINO ViT-B/8, and the target is our state after block 12
(CLS, patch mean, patch std; before the final layer norm), compared in the KAE's standardized units.
"""

import math
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from transformers import AutoModel

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")

NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
SIZE = 224
NOISE_CHANNELS = 32
# Splice's generator: six scales, wide kernels at the fine scales
DOWN = UP = [16, 32, 64, 128, 128, 128]
SKIP = [4, 4, 4, 4, 4, 4]
KERNELS = [7, 7, 5, 5, 3, 3]
# Splice's schedule of the noise added to the net's input: (share of the steps, sigma)
NOISE_SCHEDULE = [(0.5, 10.0), (0.75, 2.0), (1.0, 0.5)]
PARTS = {"cls": slice(0, 768), "cls_mean": slice(0, 1536), "all": slice(0, 2304)}


def conv_block(in_channels, out_channels, kernel, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d((kernel - 1) // 2),
        nn.Conv2d(in_channels, out_channels, kernel, stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )


def crop_to(x, height, width):
    top, left = (x.shape[2] - height) // 2, (x.shape[3] - width) // 2
    return x[:, :, top : top + height, left : left + width]


class Scale(nn.Module):
    """One scale of the hourglass: a thin skip path beside a path that goes down, deeper, and back up."""

    def __init__(self, in_channels, depth):
        super().__init__()
        down, up, skip, kernel = DOWN[depth], UP[depth], SKIP[depth], KERNELS[depth]
        self.skip = conv_block(in_channels, skip, 1)
        self.down = nn.Sequential(conv_block(in_channels, down, kernel, stride=2), conv_block(down, down, kernel))
        self.deeper = Scale(down, depth + 1) if depth + 1 < len(DOWN) else None
        joined = skip + (UP[depth + 1] if self.deeper is not None else down)
        self.norm = nn.BatchNorm2d(joined)
        self.up = nn.Sequential(conv_block(joined, up, kernel), conv_block(up, up, 1))

    def forward(self, x):
        deep = self.down(x)
        if self.deeper is not None:
            deep = self.deeper(deep)
        deep = nn.functional.interpolate(deep, scale_factor=2, mode="bilinear")
        skip = self.skip(x)

        # NOTE: Odd sizes on the way down leave the two paths a pixel apart
        height, width = min(skip.shape[2], deep.shape[2]), min(skip.shape[3], deep.shape[3])
        return self.up(self.norm(torch.cat([crop_to(skip, height, width), crop_to(deep, height, width)], dim=1)))


class Generator(nn.Module):
    """Noise in, picture in [0, 1] out."""

    def __init__(self):
        super().__init__()
        self.body = Scale(NOISE_CHANNELS, 0)
        self.head = nn.Conv2d(UP[0], 3, 1)

    def forward(self, noise):
        return torch.sigmoid(self.head(self.body(noise)))


class Reader(nn.Module):
    """Frozen DINOv3: picture in [0, 1] in, standardized block-12 state out, built the way the training cache was."""

    def __init__(self, mean, std):
        super().__init__()
        self.dino = AutoModel.from_pretrained(train.DINO_MODELS[ENCODER], cache_dir=f"{os.getenv('HF_HOME')}/hub").eval()
        self.dino.requires_grad_(False)
        self.first_patch = 1 + self.dino.config.num_register_tokens
        self.normalize = transforms.Normalize(*train.IMAGENET_MEAN_STD)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def raw_states(self, pictures):
        """Unstandardized states after blocks 6 and 12. shape: [pictures, 2, 2304]"""

        states = []

        def keep(_, __, output):
            output = output[0] if isinstance(output, tuple) else output
            patches = output[:, self.first_patch :]
            states.append(torch.cat([output[:, 0], patches.mean(dim=1), patches.std(dim=1)], dim=-1))

        hooks = [self.dino.model.layer[block].register_forward_hook(keep) for block in (5, 11)]
        self.dino(pixel_values=self.normalize(pictures))
        for hook in hooks:
            hook.remove()

        return torch.stack(states, dim=1)

    def forward(self, pictures):
        return (self.raw_states(pictures)[:, 1] - self.mean) / self.std


def get_sigma(step, steps):
    return next(sigma for share, sigma in NOISE_SCHEDULE if step < share * steps)


def invert(reader, targets, part, steps, seeds, starts=None, sigma=None, lr=0.01, log_every=500, label=""):
    """
    Paints one picture per target, each with its own generator, all read by DINOv3 in one batch.

    targets: standardized block-12 states. shape: [targets, 2304]
    part: which numbers of the state the picture has to match, a key of PARTS
    seeds: one per target; the seed sets the generator's first weights and its fixed input noise, so equal seeds start equal
    starts: generators to continue from (as returned here), for edits that should keep the layout of an earlier picture
    sigma: fixed input-noise level; by default Splice's schedule
    Returns the pictures, the generators with their inputs, and the per-target loss every `log_every` steps.
    """

    device = targets.device
    if starts is None:
        starts = []
        for seed in seeds:
            torch.manual_seed(seed)
            starts.append((Generator(), torch.randn(1, NOISE_CHANNELS, SIZE, SIZE)))
    generators = [generator.to(device).train() for generator, _ in starts]
    inputs = [noise.to(device) for _, noise in starts]
    optimizer = torch.optim.Adam([p for generator in generators for p in generator.parameters()], lr=lr)
    wanted = targets[:, PARTS[part]]
    # NOTE: The jitter is seeded apart from the generators, so two runs with the same seeds repeat exactly
    jitter = torch.Generator(device=device).manual_seed(seeds[0] if seeds else 0)
    history = []

    def paint(level):
        return torch.cat(
            [g(x + level * torch.randn(x.shape, generator=jitter, device=device)) for g, x in zip(generators, inputs)]
        )

    for step in range(steps):
        level = get_sigma(step, steps) if sigma is None else sigma
        # NOTE: Summing per-target losses keeps the targets independent: Adam scales every weight by its own gradient
        losses = (reader(paint(level))[:, PARTS[part]] - wanted).pow(2).mean(dim=1)
        optimizer.zero_grad()
        losses.sum().backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            history.append(losses.detach().cpu())
            print(f"{label} step {step:5d} sigma {level:4.1f} loss {[round(v, 4) for v in losses.tolist()]}", flush=True)

    with torch.no_grad():
        pictures = paint(get_sigma(steps - 1, steps) if sigma is None else sigma).cpu()

    return pictures, [(g.cpu(), x.cpu()) for g, x in zip(generators, inputs)], torch.stack(history)


def load_kae():
    ckpt = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
    stats = torch.load(ckpt / f"{NAME}_stats.pt")
    kae = train.JordanKoopmanAutoencoder(
        256,
        shared_modes=True,
        in_features=2304,
        latent_features=2049,
        hidden_config=stats["hidden_config"],
        bias=True,
        batchnorm=False,
        nonlinearity="leaky_relu",
    )
    kae.load_state_dict(load_file(ckpt / f"{NAME}.safetensors"))

    return kae.eval(), stats


def to_complex(latent):
    """[..., 2049] reals to [..., 1025] complex: mode 0 is the real mode, mode m > 0 the block on coordinates 2m - 1 and 2m."""

    return torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)


def to_latent(z):
    blocks = torch.stack([z[..., 1:].real, z[..., 1:].imag], dim=-1).flatten(-2)
    return torch.cat([z[..., :1].real, blocks], dim=-1)


def get_colourfulness(pictures):
    """Mean distance of a pixel's colour from grey, in the chroma plane of a simple opponent space. shape: [pictures]"""

    red, green, blue = pictures.unbind(dim=1)
    return torch.stack([red - green, 0.5 * (red + green) - blue], dim=1).norm(dim=1).mean(dim=(1, 2))


def turn(degrees):
    return complex(math.cos(math.radians(degrees)), math.sin(math.radians(degrees)))
