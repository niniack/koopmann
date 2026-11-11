import os
import tarfile
from pathlib import Path

import fire
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LinearLR
from PIL import Image
from timm.data import ImageNetInfo
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.datasets import Imagenette

from koopmann.log import logger
from koopmann.models import Autoencoder, KoopmanAutoencoder
from koopmann.utils import get_device, set_seed

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# DINOv3 encoders by short name, as the name also goes into cache paths, and an ImageNet linear probe for each
# NOTE: The probe (github.com/m2b3/dinov3-in1k-probes) stands in for CLIP's zero-shot classifier
DINO_MODELS = {"dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m"}
DINO_PROBES = {"dinov3-vitb16": "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"}
IMAGENET_MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

############# DATA #############


def to_rgb(image):
    """Some ImageNet images are greyscale or CMYK."""

    return image.convert("RGB")


class ImageNet(Dataset):
    """ImageNet from the Hugging Face cache."""

    def __init__(self, split, transform):
        split = "validation" if split == "val" else split
        self.data = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=os.getenv("DATASETS_CACHE"))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return self.transform(sample["image"]), sample["label"]


class ShardImages(IterableDataset):
    """
    Images from webdataset shards, as for CC3M or a slice of ImageNet-21k. Labels are not used, so each gets -1.
    NOTE: `skip` drops images whose name starts with one of these synsets, as ImageNet-21k holds ImageNet-1k's own images.
    """

    def __init__(self, folder, pattern, transform, skip=()):
        self.shards = sorted((Path(os.getenv("DATASETS_CACHE")) / folder).glob(pattern))
        self.transform = transform
        self.skip = set(skip)

    def __iter__(self):
        # NOTE: Each loader worker reads its own shards
        worker = get_worker_info()
        shards = self.shards if worker is None else self.shards[worker.id :: worker.num_workers]

        for shard in shards:
            with tarfile.open(shard) as tar:
                for member in tar:
                    if member.name.endswith(".jpg") and member.name.split("_")[0] not in self.skip:
                        yield self.transform(Image.open(tar.extractfile(member)).convert("RGB")), -1


@torch.no_grad()
def extract_states(encoder, dataset, split, root, device, views, blocks=None, tokens="cls"):
    """
    Runs CLIP or DINOv3 over a dataset, keeping the CLS token after the chosen blocks, or every block.
    NOTE: With more than one view, each pass uses a random crop and flip.
    NOTE: `cls_patch` adds the mean patch token, and `cls_patch_std` also its spread, as later blocks read the patches too.
    NOTE: `cls_grid_std` swaps the mean patch for the mean of each quarter of the patch grid, to keep rough layout.
    It is stored in half precision, as it is twice the size.
    """

    if encoder in DINO_MODELS:
        from transformers import AutoModel  # NOTE: Only DINOv3 needs transformers

        model = AutoModel.from_pretrained(DINO_MODELS[encoder], cache_dir=f"{os.getenv('HF_HOME')}/hub").to(device).eval()
        resblocks = model.model.layer
        forward = lambda images: model(pixel_values=images)
        # NOTE: Tokens are CLS, then registers, then patches. Registers have norms ~100x the patches', so patch statistics skip them
        first_patch = 1 + model.config.num_register_tokens
        to_tensor = [to_rgb, transforms.ToTensor(), transforms.Normalize(*IMAGENET_MEAN_STD)]
        preprocess = transforms.Compose(
            [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), *to_tensor]
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            encoder,
            pretrained="openai",
            force_quick_gelu=True,  # OpenAI's weights were trained with QuickGELU
            cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
        )
        model = model.to(device).eval()
        resblocks = model.visual.transformer.resblocks
        forward = model.encode_image
        first_patch = 1
        to_tensor = preprocess.transforms[2:]  # Everything after the resize and center crop

    # NOTE: We start after block 1, as the CLS token is constant before any attention
    states = []
    def keep(_, __, output):
        output = output[0] if isinstance(output, tuple) else output
        cls, patches = output[:, 0], output[:, first_patch:]
        if tokens == "cls_grid_std":
            # shape: [images, neurons, side, side], then the mean of each quarter of the grid
            side = int(patches.shape[1] ** 0.5)
            grid = patches.unflatten(1, (side, side)).permute(0, 3, 1, 2)
            quarters = F.adaptive_avg_pool2d(grid, 2).flatten(1)
            states.append(torch.cat([cls, quarters, patches.std(dim=1)], dim=-1).half().cpu())
            return
        state = {
            "cls": [cls],
            "cls_patch": [cls, patches.mean(dim=1)],
            "cls_patch_std": [cls, patches.mean(dim=1), patches.std(dim=1)],
        }[tokens]
        states.append(torch.cat(state, dim=-1).cpu())

    hooks = [
        block.register_forward_hook(keep)
        for block in (resblocks if blocks is None else [resblocks[i] for i in blocks])
    ]

    if views > 1:
        preprocess = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                *to_tensor,
            ]
        )

    if dataset == "imagenet":
        images = ImageNet(split, preprocess)
    elif dataset == "cc3m":
        images = ShardImages("cc3m-wds", "cc3m-train-*.tar", preprocess)
    elif dataset == "w21":
        # NOTE: The first 200 of 2,048 pre-shuffled ImageNet Winter21 shards, without ImageNet-1k's classes, whose images are ImageNet-1k's own
        images = ShardImages("imagenet-w21-wds", "imagenet_w21-train-*.tar", preprocess, skip=ImageNetInfo("imagenet-1k").label_names())
    else:
        download = not (root / "imagenette2").exists()
        images = Imagenette(root, split=split, size="full", download=download, transform=preprocess)

    loader = DataLoader(images, batch_size=256, num_workers=16)

    all_states, all_labels = [], []
    for _ in range(views):
        for images, labels in loader:
            states.clear()
            forward(images.to(device))
            all_states.append(torch.stack(states, dim=1))
            all_labels.append(labels)

    for hook in hooks:
        hook.remove()

    # shape: [samples, layers, neurons]
    return torch.cat(all_states), torch.cat(all_labels)


def get_states(encoder, dataset, split, device, blocks, views=1, tokens="cls", extract=True):
    """
    Extracts the states once, then loads the chosen blocks from disk.
    NOTE: These stay on the CPU, and training moves one batch at a time to the GPU.
    NOTE: ImageNet is too large to keep every block, so its files hold the chosen blocks only.
    NOTE: Without `extract`, a missing file is an error, for datasets that need cleaning before use.
    """

    root = Path(os.getenv("DATASETS_CACHE")) / dataset
    root.mkdir(exist_ok=True)
    suffix = f"_{views}_views" if views > 1 else ""
    saved_blocks = None

    if dataset in ["imagenet", "cc3m", "w21"]:
        saved_blocks = blocks
        suffix += f"_blocks_{blocks[0]}_{blocks[1]}"

    path = root / f"clip_{tokens}_{encoder}_{split}{suffix}.pt"

    if not path.exists():
        assert extract, f"{path} is missing, and {dataset} has to be built and cleaned before training"
        states, labels = extract_states(encoder, dataset, split, root, device, views, saved_blocks, tokens)
        torch.save({"states": states, "labels": labels}, path)
        del states  # Frees memory before reloading

    data = torch.load(path)

    # shape: [samples, blocks, neurons]
    return data["states"] if saved_blocks else data["states"][:, blocks], data["labels"]


############# MODEL #############


class SparseKoopmanAutoencoder(KoopmanAutoencoder):
    """Keeps only the `top_k` largest observables. If `top_k` is 0, nothing is removed."""

    def __init__(self, top_k, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def encode(self, x):
        phi = super().encode(x)
        if not self.top_k:
            return phi

        threshold = phi.abs().topk(self.top_k, dim=-1).values[:, -1:]
        return phi * (phi.abs() >= threshold)


class JordanKoopmanAutoencoder(Autoencoder):
    """
    K in real Jordan form: one real eigenvalue, then 2x2 blocks [[a, -b], [b, a]].
    NOTE: The encoder and decoder absorb the change of basis, so each block is a mode.
    NOTE: With `real_start`, a start state has one number per mode, so only the eigenvalue's turn reaches the second.
    """

    def __init__(self, top_k, batch_top_k=False, shared_modes=False, real_start=False, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.batch_top_k = batch_top_k
        self.shared_modes = shared_modes
        self.real_start = real_start

        # NOTE: Start and end states are only told apart when encoded as pairs, which shared modes do
        if real_start and not shared_modes:
            raise ValueError("A real start needs shared modes.")

        # NOTE: With batch TopK, evaluation keeps the modes above the threshold seen in training
        if batch_top_k:
            self.register_buffer("threshold", torch.zeros(()))

        num_blocks, remainder = divmod(self.latent_features - 1, 2)
        if remainder:
            raise ValueError("One real eigenvalue plus 2x2 blocks needs an odd latent size.")

        self.real_eigenvalue = nn.Parameter(torch.ones(1))
        self.block_real = nn.Parameter(torch.ones(num_blocks))
        self.block_imag = nn.Parameter(torch.zeros(num_blocks))

    def encode(self, x, pairs=False):
        """
        Keeps the largest modes. Each mode is kept or dropped whole.
        NOTE: With batch TopK, the largest modes across the batch are kept, so a sample keeps `top_k // 2` on average.
        NOTE: With `pairs`, `x` alternates start and end states, and each end keeps the modes picked at its start.
        NOTE: With `real_start`, every state is a start state unless `pairs` says otherwise.
        """

        phi = super().encode(x)
        real, blocks = phi[:, :1], phi[:, 1:].unflatten(-1, (-1, 2))

        if self.real_start:
            # shape: [samples, 1]
            is_end = (torch.arange(len(phi), device=phi.device)[:, None] % 2 == 1) & pairs
            blocks = torch.stack([blocks[..., 0], blocks[..., 1] * is_end], dim=-1)

        if not self.top_k:
            return torch.cat([real, blocks.flatten(1)], dim=-1)

        # shape: [samples, modes]
        magnitudes = torch.cat([real.abs(), blocks.norm(dim=-1)], dim=-1)
        if pairs:
            magnitudes = magnitudes[::2].repeat_interleave(2, dim=0)

        if not self.batch_top_k:
            threshold = magnitudes.topk(self.top_k // 2, dim=-1).values[:, -1:]
        elif self.training:
            threshold = magnitudes.flatten().topk(len(magnitudes) * (self.top_k // 2)).values[-1]
            self.threshold.lerp_(threshold.detach(), 0.01)
        else:
            threshold = self.threshold

        keep = magnitudes >= threshold

        return torch.cat([real * keep[:, :1], (blocks * keep[:, 1:, None]).flatten(1)], dim=-1)

    def koopman_forward(self, observable, k_steps):
        for _ in range(k_steps):
            real, blocks = observable[..., :1], observable[..., 1:].unflatten(-1, (-1, 2))
            x, y = blocks[..., 0], blocks[..., 1]
            a, b = self.block_real, self.block_imag

            blocks = torch.stack([a * x - b * y, b * x + a * y], dim=-1)
            observable = torch.cat([self.real_eigenvalue * real, blocks.flatten(-2)], dim=-1)

        return observable


class MlpKoopmanAutoencoder(Autoencoder):
    """
    Not a KAE: an MLP takes K's place, so we can see what keeping K linear costs.
    NOTE: The MLP is residual with a zero last layer, so it starts at the identity, like the Jordan K.
    """

    def __init__(self, hidden, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_features, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, self.latent_features),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def koopman_forward(self, observable, k_steps):
        for _ in range(k_steps):
            observable = observable + self.mlp(observable)

        return observable


class ClipHead(nn.Module):
    """CLIP's layers after the last block, and its zero-shot classifier. Nothing here is trained."""

    def __init__(self, encoder, mean, std, device, zero_shot):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            encoder,
            pretrained="openai",
            force_quick_gelu=True,  # OpenAI's weights were trained with QuickGELU
            cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
        )
        model = model.to(device).eval()

        self.ln_post = model.visual.ln_post
        self.proj = model.visual.proj
        self.scale = model.logit_scale.exp().item()

        # NOTE: States may also hold patch statistics after the CLS token, and only the CLS token goes through the head
        width = self.ln_post.normalized_shape[0]
        self.mean, self.std = mean[:width].to(device), std[:width].to(device)
        self.classes = None

        # NOTE: Class names are ImageNet's, so this only applies to ImageNet labels
        if zero_shot:
            self.classes = open_clip.build_zero_shot_classifier(
                model,
                open_clip.get_tokenizer(encoder),
                open_clip.IMAGENET_CLASSNAMES,
                open_clip.OPENAI_IMAGENET_TEMPLATES,
                device=device,
            )

        self.requires_grad_(False)

    def embed(self, states):
        """CLIP's final image embedding, from a standardized end state."""

        states = states[..., : len(self.mean)]
        return F.normalize(self.ln_post(states * self.std + self.mean) @ self.proj, dim=-1)

    def get_log_probs(self, states):
        """CLIP's zero-shot log probabilities over ImageNet's classes."""

        return (self.scale * self.embed(states) @ self.classes).log_softmax(dim=-1)

    def classify(self, states):
        return (self.embed(states) @ self.classes).argmax(dim=-1)


class DinoHead(nn.Module):
    """
    DINOv3's final layer norm, then a linear probe trained on ImageNet. Nothing here is trained.
    NOTE: The probe stands in for CLIP's zero-shot classifier, with the same methods.
    """

    def __init__(self, encoder, mean, std, device, zero_shot):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoModel  # NOTE: Only DINOv3 needs transformers

        # NOTE: The cache is passed explicitly, as huggingface_hub is imported before .env is read
        hub = f"{os.getenv('HF_HOME')}/hub"
        self.norm = AutoModel.from_pretrained(DINO_MODELS[encoder], cache_dir=hub).norm.to(device).eval()

        # NOTE: States may also hold patch statistics after the CLS token, and only the CLS token goes through the head
        width = self.norm.normalized_shape[0]
        self.mean, self.std = mean[:width].to(device), std[:width].to(device)
        self.classes = None

        # NOTE: The probe's classes are ImageNet's, so this only applies to ImageNet labels
        if zero_shot:
            probe = load_file(hf_hub_download(DINO_PROBES[encoder], "model.safetensors", cache_dir=hub))
            self.classes, self.bias = probe["weight"].T.to(device), probe["bias"].to(device)

        self.requires_grad_(False)

    def features(self, states):
        """DINOv3's final CLS output, from a standardized end state."""

        return self.norm(states[..., : len(self.mean)] * self.std + self.mean)

    def embed(self, states):
        return F.normalize(self.features(states), dim=-1)

    def get_log_probs(self, states):
        """The probe's log probabilities over ImageNet's classes."""

        return (self.features(states) @ self.classes + self.bias).log_softmax(dim=-1)

    def classify(self, states):
        return (self.features(states) @ self.classes + self.bias).argmax(dim=-1)


############# OPTIMIZATION #############


def get_optimizers(model, lr, adam_lr, weight_decay):
    """
    Muon for 2D weights, AdamW for everything else.
    NOTE: Only the 2D weights decay, so eigenvalues and biases are left alone.
    """

    matrices = [param for param in model.parameters() if param.ndim == 2]
    others = [param for param in model.parameters() if param.ndim != 2]

    return [Muon(matrices, lr=lr, weight_decay=weight_decay), AdamW(others, lr=adam_lr, weight_decay=0)]


def compute_fvu(preds, targets):
    """Fraction of variance unexplained, per layer. 0 is perfect, 1 is no better than the mean."""

    # shape: [samples, layers, neurons]
    error = (preds - targets).pow(2).sum(dim=[0, 2])
    total = (targets - targets.mean(dim=0)).pow(2).sum(dim=[0, 2])

    return error / total


def get_covariances(phi):
    """Covariance of each layer, with every layer at unit scale."""

    centered = phi - phi.mean(dim=0)
    centered = centered / centered.pow(2).mean(dim=[0, 2], keepdim=True).sqrt()

    # shape: [layers, latent, latent]
    return torch.einsum("bld,ble->lde", centered, centered) / (len(phi) - 1)


def compute_covariance_loss(phi):
    """Stops latent dimensions from carrying the same signal."""

    covariances = get_covariances(phi)
    off_diagonal = covariances - torch.diag_embed(covariances.diagonal(dim1=1, dim2=2))

    return off_diagonal.pow(2).sum(dim=[1, 2]).mean() / covariances.shape[-1]


def compute_variance_loss(phi, fraction):
    """
    Keeps every latent dimension in use, relative to the scale of its layer.
    NOTE: This only limits how uneven dimensions are, so a layer's total variance stays free.
    """

    # shape: [layers, latent]
    std = (phi.var(dim=0) + 1e-8).sqrt()
    scale = std.pow(2).mean(dim=-1, keepdim=True).sqrt()

    return F.relu(fraction - std / scale).mean()


def compute_isometric_loss(states, phi):
    """
    Matches distances between images, in each layer, relative to that layer's distances.
    NOTE: This makes the latent variance follow the state variance across layers.
    """

    # shape: [layers, samples, samples]
    state_distances = torch.cdist(states.transpose(0, 1), states.transpose(0, 1))
    latent_distances = torch.cdist(phi.transpose(0, 1), phi.transpose(0, 1))

    error = (latent_distances - state_distances).pow(2).sum(dim=[1, 2])
    return (error / state_distances.pow(2).sum(dim=[1, 2])).mean()


def compute_overlap_loss(states, phi):
    """
    Matches inner products between start and end states, relative to their size.
    NOTE: Each layer is centered first, as the offset between layers would otherwise swamp the rest.
    NOTE: With a real start, this ties a mode's first number at the end to how much of its start pattern is still there.
    """

    states, phi = states - states.mean(dim=0), phi - phi.mean(dim=0)

    # shape: [samples, samples]
    state_overlaps = states[:, 0] @ states[:, 1].T
    latent_overlaps = phi[:, 0] @ phi[:, 1].T

    return (latent_overlaps - state_overlaps).pow(2).sum() / state_overlaps.pow(2).sum()


def compute_embedding_loss(clip_head, preds, targets):
    """One minus the cosine between predicted and true CLIP embeddings."""

    return 1 - (clip_head.embed(preds) * clip_head.embed(targets)).sum(dim=-1).mean()


def compute_kl_loss(clip_head, preds, targets):
    """
    KL from CLIP's class probabilities to the surrogate's, as in distillation.
    NOTE: This needs ImageNet's classes, so it is zero elsewhere.
    """

    if clip_head.classes is None:
        return preds.new_zeros(())

    log_targets = clip_head.get_log_probs(targets)
    return F.kl_div(clip_head.get_log_probs(preds), log_targets, log_target=True, reduction="batchmean")


def encode_pairs(kae, states):
    """
    Encodes start and end states, as [samples, 2, latent].
    NOTE: With `shared_modes`, each end keeps the modes picked at its start, so every kept mode is measured at both ends.
    """

    flat = states.flatten(0, 1)
    phi = kae.encode(flat, pairs=True) if getattr(kae, "shared_modes", False) else kae.encode(flat)
    return phi.unflatten(0, states.shape[:2])


def compute_losses(kae, states, std_fraction, clip_head):
    """
    Reconstruction at both ends, prediction of the end from the start, then regularizers.
    NOTE: Errors are relative to the variance of each layer, so neither end dominates.
    """

    # shape: [samples, 2, latent]
    phi = encode_pairs(kae, states)
    recons = kae.decode(phi.flatten(0, 1)).unflatten(0, states.shape[:2])

    pred_phi = kae.koopman_forward(phi[:, :1], 1)
    pred_state = kae.decode(pred_phi[:, 0])[:, None]

    return (
        compute_fvu(recons, states).mean(),
        compute_fvu(pred_phi, phi[:, 1:]).mean(),
        compute_fvu(pred_state, states[:, 1:]).mean(),
        compute_variance_loss(phi, std_fraction),
        compute_covariance_loss(phi),
        compute_isometric_loss(states, phi),
        compute_embedding_loss(clip_head, pred_state[:, 0], states[:, 1]),
        compute_kl_loss(clip_head, pred_state[:, 0], states[:, 1]),
        compute_overlap_loss(states, phi),
    )


def train_one_epoch(kae, states, optimizers, batch_size, lambdas, std_fraction, clip_head, device):
    kae.train()
    totals = torch.zeros(len(lambdas))

    batches = torch.randperm(len(states)).split(batch_size)
    for idx in batches:
        losses = compute_losses(kae, states[idx].to(device).float(), std_fraction, clip_head)
        loss = sum(weight * value for weight, value in zip(lambdas, losses))

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        totals += torch.tensor([value.item() for value in losses])

    return (totals / len(batches)).tolist()


############# EVALUATION #############


def fit_linear(keys, values, reg=100.0):
    """Ridge regression, with a bias."""

    keys = F.pad(keys, (0, 1), value=1).double()
    gram = keys.T @ keys
    eye = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)

    return torch.linalg.solve(gram + reg * eye, keys.T @ values.double()).float()


def apply_linear(operator, x):
    return F.pad(x, (0, 1), value=1) @ operator


def get_accuracy(classifier, states, labels):
    return (apply_linear(classifier, states).argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(kae, states, labels, classifier, clip_head, split):
    """How well the KAE predicts the end state, and whether that prediction keeps the class."""

    kae.eval()

    # shape: [samples, 2, latent]
    phi = encode_pairs(kae, states)
    recons = kae.decode(phi.flatten(0, 1)).unflatten(0, states.shape[:2])

    pred_phi = kae.koopman_forward(phi[:, 0], 1)
    pred_state = kae.decode(pred_phi)

    logs = {
        f"{split}/reconstruction_fvu": compute_fvu(recons, states).mean().item(),
        f"{split}/latent_fvu": compute_fvu(pred_phi[:, None], phi[:, 1:]).item(),
        # NOTE: The pooled FVU above can hide a few images with huge latents, so also track the typical image
        f"{split}/latent_error_per_image": (
            ((pred_phi - phi[:, 1]).pow(2).sum(-1) / phi[:, 1].pow(2).sum(-1).clamp_min(1e-12)).median().item()
        ),
        f"{split}/latent_energy_top1pct": (
            phi[:, 1].pow(2).sum(-1).topk(max(1, len(phi) // 100)).values.sum() / phi[:, 1].pow(2).sum()
        ).item(),
        f"{split}/one_step_fvu": compute_fvu(pred_state[:, None], states[:, 1:]).item(),
        f"{split}/predicted_accuracy": get_accuracy(classifier, pred_state, labels),
        f"{split}/embedding_loss": compute_embedding_loss(clip_head, pred_state, states[:, 1]).item(),
        f"{split}/active_latents": (phi[:, 0] != 0).float().sum(dim=-1).mean().item(),
    }

    # As a surrogate: CLIP's own classifier on the prediction, and how often it gives CLIP's answer
    if clip_head.classes is not None:
        answers = clip_head.classify(pred_state)
        logs[f"{split}/zero_shot_accuracy"] = (answers == labels).float().mean().item()
        logs[f"{split}/clip_agreement"] = (answers == clip_head.classify(states[:, 1])).float().mean().item()
        logs[f"{split}/kl_loss"] = compute_kl_loss(clip_head, pred_state, states[:, 1]).item()

    return logs


@torch.no_grad()
def get_latent_logs(kae, states):
    """How many directions carry variance, and whether latent variance follows state variance."""

    phi = encode_pairs(kae, states)

    # shape: [2, latent]
    # NOTE: In double precision, as single precision fails on the CPU when many dimensions are unused
    eigenvalues = torch.linalg.eigvalsh(get_covariances(phi).double()).clamp(min=0)
    shares = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True)
    ranks = (-(shares * (shares + 1e-20).log()).sum(dim=-1)).exp()

    # shape: [2]
    ratios = phi.var(dim=0).sum(dim=-1) / states.var(dim=0).sum(dim=-1)

    return {
        "latent/rank_start": ranks[0].item(),
        "latent/rank_end": ranks[1].item(),
        "latent/var_ratio_start": ratios[0].item(),
        "latent/var_ratio_end": ratios[1].item(),
    }


def main(
    encoder: str = "ViT-B-32",
    dataset: str = "imagenette",
    start_block: int = 6,
    end_block: int = 12,
    ae_dim: int = 1_024,
    ae_hidden: int = 1_024,
    ae_layers: int = 1,
    top_k: int = 0,
    jordan: bool = False,
    batch_top_k: bool = False,
    shared_modes: bool = False,
    real_start: bool = False,
    standardization: str = "global",
    mlp: bool = False,
    tokens: str = "cls",
    extra_dataset: str | None = None,
    train_fraction: float = 1.0,
    lambda_reconstruction: float = 1.0,
    lambda_latent: float = 1.0,
    lambda_state: float = 1.0,
    lambda_variance: float = 10.0,
    std_fraction: float = 0.5,
    lambda_covariance: float = 1.0,
    lambda_isometric: float = 1.0,
    lambda_embedding: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_overlap: float = 0.0,
    views: int = 10,
    name: str | None = None,
    num_epochs: int = 100,
    batch_size: int = 1_024,
    lr: float = 0.01,
    adam_lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
):
    """Fits one linear operator from a start block to an end block, in a learned latent."""

    config = dict(locals())
    set_seed(seed)
    device = get_device()

    name = name or f"b{start_block}"
    operator = "mlp" if mlp else "jordan" if jordan else "full_k"
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="tokae",
        name=name,
        tags=[
            encoder,
            dataset,
            tokens.replace("_", "+"),
            "start_to_end",
            "augmented" if views > 1 else "clean",
            "sparse" if top_k else "dense",
            operator,
            *(["batch_topk"] if batch_top_k else []),
            *(["shared_modes"] if shared_modes else []),
            *(["real_start"] if real_start else []),
            *([f"{standardization}_standardization"] if standardization != "global" else []),
            *([f"plus_{extra_dataset}"] if extra_dataset else []),
            *([f"train_fraction_{train_fraction:g}"] if train_fraction < 1 else []),
        ],
        config={
            **config,
            "standardization": {
                "global": "global over both blocks",
                "center": "each block centered on its own mean, then one std over both",
                "block": "each block on its own mean and std",
            }[standardization],
            "state": {
                "cls": "CLS token",
                "cls_patch": "CLS token, then the mean patch token",
                "cls_patch_std": "CLS token, then the mean and std of the patch tokens",
                "cls_grid_std": "CLS token, then the mean patch token of each quarter of the grid, then the std of the patch tokens",
            }[tokens],
            "operator": f"{operator}, one step from start to end",
            "head": "DINOv3 ImageNet linear probe" if encoder in DINO_MODELS else "CLIP zero-shot",
        },
    )

    # Get data, keeping the start and end only
    blocks = [start_block - 1, end_block - 1]
    train_states, train_labels = get_states(encoder, dataset, "train", device, blocks, views, tokens)
    val_states, val_labels = get_states(encoder, dataset, "val", device, blocks, tokens=tokens)

    # A random share of the images, with all of their views
    # NOTE: Views are stored one after another, each covering every image in the same order
    if train_fraction < 1:
        num_images = len(train_states) // views
        keep = torch.randperm(num_images)[: int(train_fraction * num_images)].sort().values
        keep = torch.cat([view * num_images + keep for view in range(views)])
        train_states, train_labels = train_states[keep], train_labels[keep]

    # Standardize once. With one std over both blocks, each block keeps its natural variance
    # NOTE: Per-block means remove the offset between blocks, which K, being linear, can only carry in always-on modes
    # NOTE: In place, as ImageNet with several views and patches is too large to copy
    if train_states.dtype == torch.half:
        # NOTE: Half-precision caches are too large to copy, so the statistics are summed in chunks, in double precision
        assert standardization == "global", "half-precision caches only support global standardization"
        total, squares = torch.zeros(train_states.shape[-1], dtype=torch.double), torch.zeros(train_states.shape[-1], dtype=torch.double)
        for chunk in train_states.split(100_000):
            chunk = chunk.double()
            total += chunk.sum(dim=[0, 1])
            squares += chunk.pow(2).sum(dim=[0, 1])
        count = train_states.shape[0] * train_states.shape[1]
        mean = (total / count).float()
        std = ((squares - total.pow(2) / count) / (count - 1)).sqrt().float()
        # NOTE: In chunks, as an in-place op on half with float statistics briefly makes a float copy of the whole tensor
        for chunk in train_states.split(100_000):
            chunk.sub_(mean).div_(std)
    else:
        mean = train_states.mean(dim=[0, 1]) if standardization == "global" else train_states.mean(dim=0)
        train_states.sub_(mean)
        std = train_states.std(dim=0) if standardization == "block" else train_states.std(dim=[0, 1])
        train_states.div_(std)
    val_states = (val_states.float() - mean) / std

    # References: a linear map, and a classifier on the true end state
    # NOTE: These are fit on one sample per view, for memory
    # NOTE: Half-precision caches are wide, so their references use fewer samples
    ref = views if train_states.dtype != torch.half else max(views, len(train_states) // 400_000)
    one_hot = F.one_hot(train_labels[::ref], int(train_labels.max()) + 1).float()
    classifier = fit_linear(train_states[::ref, 1], one_hot)
    linear_map = fit_linear(train_states[::ref, 0], train_states[::ref, 1])
    linear_pred = apply_linear(linear_map, val_states[:, 0])
    wandb.summary.update(
        {
            "baseline/linear_one_step_fvu": compute_fvu(
                linear_pred[:, None], val_states[:, 1:]
            ).item(),
            "baseline/linear_predicted_accuracy": get_accuracy(classifier, linear_pred, val_labels),
            "baseline/true_end_accuracy": get_accuracy(classifier, val_states[:, 1], val_labels),
        }
    )

    # Train
    # Only the training set stays on the CPU
    classifier, val_states, val_labels = classifier.to(device), val_states.to(device), val_labels.to(device)
    # NOTE: The head undoes the end block's standardization, whether stats are shared or per block
    # NOTE: For DINOv3, a linear probe stands in for CLIP's zero-shot head, so "zero_shot" and "clip_agreement" logs read the probe
    head_class = DinoHead if encoder in DINO_MODELS else ClipHead
    clip_head = head_class(encoder, mean.expand(2, -1)[1], std.expand(2, -1)[1], device, zero_shot=dataset == "imagenet")

    if clip_head.classes is not None:
        wandb.summary.update(
            {
                "baseline/linear_zero_shot_accuracy": (
                    (clip_head.classify(linear_pred.to(device)) == val_labels).float().mean().item()
                ),
                "baseline/true_zero_shot_accuracy": (
                    (clip_head.classify(val_states[:, 1]) == val_labels).float().mean().item()
                ),
            }
        )

    # NOTE: Training logs use a sample of at most about 50,000 states
    stride = max(views, len(train_states) // 50_000)
    train_sample, train_sample_labels = train_states[::stride].to(device).float(), train_labels[::stride].to(device)

    # NOTE: Training states stay on the CPU in half precision, so extra images fit
    train_states = train_states.half()

    # Extra unlabeled images only join the training loop, standardized like ImageNet
    if extra_dataset:
        # NOTE: ImageNet-21k holds copies of ImageNet's val images, so its file is built and deduplicated beforehand
        extra_states, _ = get_states(encoder, extra_dataset, "train", device, blocks, tokens=tokens, extract=extra_dataset != "w21")
        extra_states = extra_states.sub_(mean).div_(std).half()
        train_states = torch.cat([train_states, extra_states])
        del extra_states
        wandb.config.update({"num_train_states": len(train_states)})

    kwargs = {
        "in_features": train_states.shape[-1],
        "latent_features": ae_dim,
        "hidden_config": [ae_hidden] * ae_layers,
        "bias": True,
        "batchnorm": False,
        "nonlinearity": "leaky_relu",
    }
    if mlp:
        kae = MlpKoopmanAutoencoder(ae_hidden, **kwargs).to(device)
    elif jordan:
        kae = JordanKoopmanAutoencoder(top_k, batch_top_k, shared_modes, real_start, **kwargs).to(device)
    else:
        kae = SparseKoopmanAutoencoder(top_k=top_k, k_steps=1, **kwargs).to(device)

    optimizers = get_optimizers(kae, lr, adam_lr, weight_decay)
    schedulers = [LinearLR(opt, 1.0, 0.0, total_iters=num_epochs) for opt in optimizers]
    lambdas = (
        lambda_reconstruction,
        lambda_latent,
        lambda_state,
        lambda_variance,
        lambda_covariance,
        lambda_isometric,
        lambda_embedding,
        lambda_kl,
        lambda_overlap,
    )

    # Log which optimizer owns how many parameters
    counts = {
        type(opt).__name__: sum(p.numel() for g in opt.param_groups for p in g["params"])
        for opt in optimizers
    }
    wandb.config.update({"optimizer_params": counts})

    save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
    save_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "std": std,
            "blocks": blocks,
            "standardization": standardization,
            "shared_modes": shared_modes,
            "real_start": real_start,
            "hidden_config": kwargs["hidden_config"],
        },
        save_dir / f"{name}_stats.pt",
    )

    best_score = -float("inf")
    for epoch in range(num_epochs):
        losses = train_one_epoch(
            kae, train_states, optimizers, batch_size, lambdas, std_fraction, clip_head, device
        )
        for scheduler in schedulers:
            scheduler.step()

        logs = {
            "loss/reconstruction": losses[0],
            "loss/latent": losses[1],
            "loss/state": losses[2],
            "loss/variance": losses[3],
            "loss/covariance": losses[4],
            "loss/isometric": losses[5],
            "loss/embedding": losses[6],
            "loss/kl": losses[7],
            "loss/overlap": losses[8],
            **evaluate(kae, train_sample, train_sample_labels, classifier, clip_head, "train"),
            **evaluate(kae, val_states, val_labels, classifier, clip_head, "val"),
        }
        if epoch % 5 == 0:
            logs.update(get_latent_logs(kae, val_states))

        # Keep the best surrogate, or the best prediction when we can't score it with CLIP
        score = logs.get("val/zero_shot_accuracy", -logs["val/one_step_fvu"])
        if score > best_score:
            best_score = score
            kae.save_model(save_dir / f"{name}.safetensors")
            wandb.summary.update({"best/val_one_step_fvu": logs["val/one_step_fvu"], "best/epoch": epoch + 1})
            wandb.summary.update({"best/val_accuracy": logs["val/predicted_accuracy"]})

        wandb.log(logs)

        if (epoch + 1) % 50 == 0:
            logger.info({key: round(logs[key], 4) for key in logs if key.startswith("val/")})

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
