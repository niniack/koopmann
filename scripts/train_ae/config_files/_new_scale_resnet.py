import os
import pdb
import sys
from collections import OrderedDict
from functools import reduce
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
import torch.nn.functional as F
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from safetensors.torch import save_model
from torch import linalg, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from koopmann.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from koopmann.log import logger
from koopmann.models import (
    MLP,
    Autoencoder,
    BaseTorchModel,
    ConvAutoencoder,
    ExponentialKoopmanAutencoder,
)
from koopmann.models.resnet import resnet20, resnet56
from koopmann.models.utils import get_device, parse_safetensors_metadata
from koopmann.scripts.common import load_config
from koopmann.utils import set_seed
from koopmann.visualization import plot_eigenvalues


########################## CONFIG ##########################
class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    num_epochs: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    betas: list[PositiveFloat] | None = None


# ScaleConfig for scale-related parameters
class ScaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_to_scale: str
    scale_location: PositiveInt
    num_scaled_layers: PositiveInt


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ae_dim: PositiveInt
    lambda_reconstruction: NonNegativeFloat
    lambda_prediction: NonNegativeFloat
    lambda_id: NonNegativeFloat
    ae_nonlinearity: Optional[str] = None


# Main Config class
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    scale: ScaleConfig
    autoencoder: AutoencoderConfig
    batch_size: PositiveInt
    print_freq: PositiveInt
    seed: NonNegativeInt = 0
    save_name: Optional[str] = None
    save_dir: Optional[str] = None


########################## LOSSES ##########################
def compute_recons_loss(act_dict, autoencoder, k):
    padded_acts = list(act_dict.values())
    padded_acts = torch.stack(padded_acts, dim=1)  # shape: batch, layers, neurons

    # Latent act for each padded act
    latent_acts = [autoencoder._encode(act) for act in padded_acts]
    latent_acts = torch.stack(latent_acts, dim=1)  # shape: batch, layers, neurons

    #### STATE SPACE
    # Reconstruction with AE
    recons_acts = [autoencoder(act, k=0).reconstruction for act in padded_acts.unbind(dim=1)]
    recons_acts = torch.stack(recons_acts, dim=1)
    recons_error = (padded_acts - recons_acts).pow(2).sum()

    # Total variance for padded acts in state space
    centered_acts = padded_acts - padded_acts.mean(dim=0)
    total_variance_state_space = centered_acts.pow(2).sum()

    return recons_error / total_variance_state_space


########################## TRAIN LOOP ##########################
def train_one_epoch(
    model: nn.Module,
    autoencoder: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> dict:
    # Per epoch losses
    loss_epoch = 0
    ae_recons_loss_epoch = 0

    # Freeze original model
    autoencoder.train()
    model.eval()

    for input, label in train_loader:
        # Unwrap batch
        input, label = input.to(device), label.to(device).squeeze()

        # # Attach model hooks
        # model.hook_model()
        # NOTE: Resnet should already be hooked

        optimizer.zero_grad()

        # Raw forward pass
        with torch.no_grad():
            _ = model.forward(input)

        all_acts = model._forward_activations

        # Compute autoencoder losses
        ae_recons_loss = compute_recons_loss(
            act_dict=all_acts,
            autoencoder=autoencoder,
            k=config.scale.num_scaled_layers,
        )

        # Combine total loss
        loss = config.autoencoder.lambda_reconstruction * ae_recons_loss

        loss.backward()
        optimizer.step()

        # Accumulate losses
        loss_epoch += loss.item()
        ae_recons_loss_epoch += ae_recons_loss.item()

    # Average losses over all batches
    epoch_loss = loss_epoch / len(train_loader)
    avg_recons_loss = ae_recons_loss_epoch / len(train_loader)

    # Return all metrics as a dictionary
    return {
        "epoch_loss": epoch_loss,
        "recons_loss": avg_recons_loss,
    }


########################## MAIN ##########################
def setup(config_path_or_obj: Optional[Path | str | Config] = None):
    # Initialize wandb
    wandb.init(entity="nishantaswani", project="koopmann")

    # Validate
    if config_path_or_obj is None and not dict(wandb.config):
        sys.exit("No configuration found for the run! Provide a file")

    # Update wandb config
    wandb_config_dict = dict(wandb.config)

    config = load_config(
        config_path_or_obj or wandb_config_dict,
        config_model=Config,
    )
    logger.info(config)

    set_seed(config.seed)

    return config


def hook_resnet(resnet_model):
    # Initialize the dictionary to store forward activations
    if not hasattr(resnet_model, "_forward_activations"):
        resnet_model._forward_activations = {}

    def hook_layer(layer, key):
        def _hook(module, input, output):
            if key == 0:
                resnet_model._forward_activations[key] = F.relu(output)
            else:
                resnet_model._forward_activations[key] = output

        # Register the hook on the given layer
        return layer.register_forward_hook(_hook)

    # List of layers to hook
    layers = [
        resnet_model.bn1,
        resnet_model.layer1[0],
        resnet_model.layer1[1],
        resnet_model.layer1[2],
    ]

    # Hook each layer and associate it with a key
    for i, layer in enumerate(layers):
        hook_layer(layer, key=i)


def main(config_path_or_obj: Optional[Path | str | Config] = None):
    # Setup WandB and load config
    config = setup(config_path_or_obj)

    # Device
    device = get_device()

    # Load data
    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    subset_idx = list(range(0, len(dataset) // 10))
    subset = torch.utils.data.Subset(dataset, subset_idx)
    train_loader = create_data_loader(subset, batch_size=config.batch_size, global_seed=config.seed)

    # Load model
    original_model = resnet20()
    checkpoint = torch.load(
        "/home/nsa325/work/koopmann/model_saves/resnets/resnet20.th",
        weights_only=False,
        map_location=device,
    )
    stripped_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    original_model.load_state_dict(stripped_state_dict)
    original_model.to(device).eval()

    # Hook resnet
    hook_resnet(original_model)

    # Grab dimensions
    batch = next(iter(train_loader))
    _ = original_model(torch.randn_like(batch[0]).to(device))
    test_fwd_acts = original_model._forward_activations[config.scale.scale_location - 1]

    # Build autoencoder
    autoencoder = ConvAutoencoder(
        k=config.scale.num_scaled_layers,
        input_channels=test_fwd_acts.shape[1],
        input_height=test_fwd_acts.shape[2],
        input_width=test_fwd_acts.shape[3],
        latent_dimension=config.autoencoder.ae_dim,
        nonlinearity=config.autoencoder.ae_nonlinearity,
    ).to(device)

    autoencoder.summary()

    # Define optimiser and scheduler
    optimizer = optim.AdamW(
        params=list(autoencoder.parameters()),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=(0.9, 0.999) if not config.optim.betas else tuple(config.optim.betas),
    )
    # optimizer = optim.SGD(
    #     params=list(autoencoder.parameters()),
    #     lr=config.optim.learning_rate,
    #     weight_decay=config.optim.weight_decay,
    # )

    # Loop over epochs
    for epoch in range(config.optim.num_epochs):
        # Train step
        losses = train_one_epoch(
            model=original_model,
            autoencoder=autoencoder,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
        )

        # scheduler.step(losses["epoch_loss"])

        # Log metrics
        if (epoch + 1) % config.print_freq == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": losses["epoch_loss"],
                    "recons_loss": losses["recons_loss"],
                }
            )

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Train Loss: {losses['epoch_loss']:.4f}, "
                f"Reconstruction Loss: {losses['recons_loss']:.4f}, "
            )

    # Save model
    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        path = Path(
            config.save_dir,
            f"k_{config.scale.num_scaled_layers}_dim_{config.autoencoder.ae_dim}_loc_{config.scale.scale_location}_autoencoder_{config.save_name}.safetensors",
        )
        autoencoder.save_model(
            path,
            dataset=dataset.name(),
            num_scaled=config.scale.num_scaled_layers,
            scale_location=config.scale.scale_location,
        )

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
