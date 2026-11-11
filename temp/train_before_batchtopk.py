import os
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Imagenette

from koopmann.log import logger
from koopmann.models import Autoencoder, KoopmanAutoencoder
from koopmann.utils import get_device, set_seed

load_dotenv(Path(__file__).parent.parent.parent / ".env")

############# DATA #############


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


@torch.no_grad()
def extract_states(encoder, dataset, split, root, device, views, blocks=None, tokens="cls"):
    """
    Runs CLIP over a dataset, keeping the CLS token after the chosen blocks, or every block.
    NOTE: With more than one view, each pass uses a random crop and flip.
    NOTE: `cls_patch` adds the mean patch token, and `cls_patch_std` also its spread, as later blocks read the patches too.
    """

    model, _, preprocess = open_clip.create_model_and_transforms(
        encoder,
        pretrained="openai",
        force_quick_gelu=True,  # OpenAI's weights were trained with QuickGELU
        cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
    )
    model = model.to(device).eval()

    # NOTE: We start after block 1, as the CLS token is constant before any attention
    states = []
    resblocks = model.visual.transformer.resblocks
    def keep(_, __, output):
        cls, patches = output[:, 0], output[:, 1:]
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
                *preprocess.transforms[2:],  # Everything after the resize and center crop
            ]
        )

    if dataset == "imagenet":
        images = ImageNet(split, preprocess)
    else:
        download = not (root / "imagenette2").exists()
        images = Imagenette(root, split=split, size="full", download=download, transform=preprocess)

    loader = DataLoader(images, batch_size=256, num_workers=16)

    all_states, all_labels = [], []
    for _ in range(views):
        for images, labels in loader:
            states.clear()
            model.encode_image(images.to(device))
            all_states.append(torch.stack(states, dim=1))
            all_labels.append(labels)

    for hook in hooks:
        hook.remove()

    # shape: [samples, layers, neurons]
    return torch.cat(all_states), torch.cat(all_labels)


def get_states(encoder, dataset, split, device, blocks, views=1, tokens="cls"):
    """
    Extracts the states once, then loads the chosen blocks from disk.
    NOTE: These stay on the CPU, and training moves one batch at a time to the GPU.
    NOTE: ImageNet is too large to keep every block, so its files hold the chosen blocks only.
    """

    root = Path(os.getenv("DATASETS_CACHE")) / dataset
    root.mkdir(exist_ok=True)
    suffix = f"_{views}_views" if views > 1 else ""
    saved_blocks = None

    if dataset == "imagenet":
        saved_blocks = blocks
        suffix += f"_blocks_{blocks[0]}_{blocks[1]}"

    path = root / f"clip_{tokens}_{encoder}_{split}{suffix}.pt"

    if not path.exists():
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
    """

    def __init__(self, top_k, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

        num_blocks, remainder = divmod(self.latent_features - 1, 2)
        if remainder:
            raise ValueError("One real eigenvalue plus 2x2 blocks needs an odd latent size.")

        self.real_eigenvalue = nn.Parameter(torch.ones(1))
        self.block_real = nn.Parameter(torch.ones(num_blocks))
        self.block_imag = nn.Parameter(torch.zeros(num_blocks))

    def encode(self, x):
        """Keeps the largest modes. Each mode is kept or dropped whole."""

        phi = super().encode(x)
        if not self.top_k:
            return phi

        real, blocks = phi[:, :1], phi[:, 1:].unflatten(-1, (-1, 2))

        # shape: [samples, modes]
        magnitudes = torch.cat([real.abs(), blocks.norm(dim=-1)], dim=-1)
        threshold = magnitudes.topk(self.top_k // 2, dim=-1).values[:, -1:]
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


def compute_losses(kae, states, std_fraction, clip_head):
    """
    Reconstruction at both ends, prediction of the end from the start, then regularizers.
    NOTE: Errors are relative to the variance of each layer, so neither end dominates.
    """

    # shape: [samples * 2, latent]
    phi = kae.encode(states.flatten(0, 1))
    recons = kae.decode(phi).unflatten(0, states.shape[:2])

    # shape: [samples, 2, latent]
    phi = phi.unflatten(0, states.shape[:2])
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
    )


def train_one_epoch(kae, states, optimizers, batch_size, lambdas, std_fraction, clip_head, device):
    kae.train()
    totals = torch.zeros(len(lambdas))

    batches = torch.randperm(len(states)).split(batch_size)
    for idx in batches:
        losses = compute_losses(kae, states[idx].to(device), std_fraction, clip_head)
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
    phi = kae.encode(states.flatten(0, 1)).unflatten(0, states.shape[:2])
    recons = kae.decode(phi.flatten(0, 1)).unflatten(0, states.shape[:2])

    pred_phi = kae.koopman_forward(phi[:, 0], 1)
    pred_state = kae.decode(pred_phi)

    logs = {
        f"{split}/reconstruction_fvu": compute_fvu(recons, states).mean().item(),
        f"{split}/latent_fvu": compute_fvu(pred_phi[:, None], phi[:, 1:]).item(),
        f"{split}/one_step_fvu": compute_fvu(pred_state[:, None], states[:, 1:]).item(),
        f"{split}/predicted_accuracy": get_accuracy(classifier, pred_state, labels),
        f"{split}/embedding_loss": compute_embedding_loss(clip_head, pred_state, states[:, 1]).item(),
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

    phi = kae.encode(states.flatten(0, 1)).unflatten(0, states.shape[:2])

    # shape: [2, latent]
    eigenvalues = torch.linalg.eigvalsh(get_covariances(phi)).clamp(min=0)
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
    top_k: int = 0,
    jordan: bool = False,
    mlp: bool = False,
    tokens: str = "cls",
    lambda_reconstruction: float = 1.0,
    lambda_latent: float = 1.0,
    lambda_state: float = 1.0,
    lambda_variance: float = 10.0,
    std_fraction: float = 0.5,
    lambda_covariance: float = 1.0,
    lambda_isometric: float = 1.0,
    lambda_embedding: float = 0.0,
    lambda_kl: float = 0.0,
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
        ],
        config={
            **config,
            "standardization": "global over both blocks",
            "state": {
                "cls": "CLS token",
                "cls_patch": "CLS token, then the mean patch token",
                "cls_patch_std": "CLS token, then the mean and std of the patch tokens",
            }[tokens],
            "operator": f"{operator}, one step from start to end",
        },
    )

    # Get data, keeping the start and end only
    blocks = [start_block - 1, end_block - 1]
    train_states, train_labels = get_states(encoder, dataset, "train", device, blocks, views, tokens)
    val_states, val_labels = get_states(encoder, dataset, "val", device, blocks, tokens=tokens)

    # Standardize once, so each block keeps its natural variance
    # NOTE: In place, as ImageNet with several views and patches is too large to copy
    mean = train_states.mean(dim=[0, 1])
    std = train_states.std(dim=[0, 1])
    train_states.sub_(mean).div_(std)
    val_states = (val_states - mean) / std

    # References: a linear map, and a classifier on the true end state
    # NOTE: These are fit on one sample per view, for memory
    one_hot = F.one_hot(train_labels[::views], int(train_labels.max()) + 1).float()
    classifier = fit_linear(train_states[::views, 1], one_hot)
    linear_map = fit_linear(train_states[::views, 0], train_states[::views, 1])
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
    clip_head = ClipHead(encoder, mean, std, device, zero_shot=dataset == "imagenet")

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
    train_sample, train_sample_labels = train_states[::stride].to(device), train_labels[::stride].to(device)

    kwargs = {
        "in_features": train_states.shape[-1],
        "latent_features": ae_dim,
        "hidden_config": [ae_hidden],
        "bias": True,
        "batchnorm": False,
        "nonlinearity": "leaky_relu",
    }
    if mlp:
        kae = MlpKoopmanAutoencoder(ae_hidden, **kwargs).to(device)
    elif jordan:
        kae = JordanKoopmanAutoencoder(top_k, **kwargs).to(device)
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
    )

    # Log which optimizer owns how many parameters
    counts = {
        type(opt).__name__: sum(p.numel() for g in opt.param_groups for p in g["params"])
        for opt in optimizers
    }
    wandb.config.update({"optimizer_params": counts})

    save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
    save_dir.mkdir(exist_ok=True)
    torch.save({"mean": mean, "std": std, "blocks": blocks}, save_dir / f"{name}_stats.pt")

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
