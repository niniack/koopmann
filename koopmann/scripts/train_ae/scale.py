import os
import pdb
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
import torch.nn.functional as F
import wandb
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

from koopmann.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from koopmann.log import logger
from koopmann.models import MLP, Autoencoder, BaseTorchModel, ExponentialKoopmanAutencoder
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


########################## UTILITY ##########################
def pad_act(x, target_size):
    current_size = x.size(1)
    if current_size < target_size:
        pad_size = target_size - current_size
        x = F.pad(x, (0, pad_size), mode="constant", value=0)

    return x


########################## LOSSES ##########################
def compute_recons_loss(act_dict, autoencoder):
    padded_acts = []
    masks = []
    ae_input_size = autoencoder.encoder[0].in_features
    for act in act_dict.values():
        # Pad activations
        padded_acts.append(pad_act(act, ae_input_size))

        # Build a mask to ignore "extra" neurons in downstream activations
        # Only relevant for the second layer
        mask = torch.zeros(ae_input_size, device=act.device)
        curr_size = act.size(-1)
        mask[:curr_size] = 1
        masks.append(mask)

    # Stack lists
    padded_acts = torch.stack(padded_acts, dim=1)  # shape: batch, layers, neurons
    masks = torch.stack(masks, dim=0)  # shape: layers, neurons

    # Total variance, used as a denominator for scaling the reconstruction loss
    masked_centered_acts = padded_acts - padded_acts.mean(dim=0) * masks.unsqueeze(dim=0)
    total_variance = masked_centered_acts.pow(2).sum()

    # Reconstruction with AE
    recons_acts = [autoencoder(x=act, k=0).reconstruction for act in padded_acts.unbind(dim=1)]
    recons_acts = torch.stack(recons_acts, dim=1)

    # Reconstruction error
    masked_diff = (padded_acts - recons_acts) * masks.unsqueeze(dim=0)
    recons_error = (masked_diff).pow(2).sum()

    # Error is scaled with total_variance
    # Note that we didn't bother with dividing either value
    # by the numel because it would cancel out!
    return recons_error / total_variance


def compute_k_prediction_loss(act_dict, autoencoder, k):
    # Extract activations from the dictionary
    act_list = list(act_dict.values())
    target_acts = act_list[-1]  # shape: batch, neurons
    total_variance = (target_acts - target_acts.mean(dim=0)).pow(2).sum()

    # Compute the prediction for the first activation
    # Autoencoder outputs are shaped [layers, batch, neurons]
    all_preds = autoencoder(x=act_list[0], k=k).predictions
    pred_k = all_preds[-1, :, : target_acts.size(-1)]
    recons_error = (pred_k - target_acts).pow(2).sum()

    # Compute and return the loss
    return recons_error / total_variance


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
    ae_prediction_loss_epoch = 0

    for input, label in train_loader:
        # Unwrap batch
        input, label = input.to(device), label.to(device).squeeze()

        # Freeze original model
        autoencoder.train()
        model.eval()

        # Attach model hooks
        model.hook_model()

        optimizer.zero_grad()

        # Raw forward pass
        with torch.no_grad():
            _ = model.forward(input)

        all_acts = model.get_fwd_activations()

        # NOTE: Does not include scaled activations!
        # Get first and last elements without modifying the dict
        first_key, first_value = next(iter(all_acts.items()))
        last_key, last_value = next(iter(reversed(all_acts.items())))
        first_last_acts = {first_key: first_value, last_key: last_value}

        # Compute autoencoder losses
        ae_recons_loss = compute_recons_loss(
            act_dict=all_acts,
            autoencoder=autoencoder,
        )
        ae_pred_loss = compute_k_prediction_loss(
            act_dict=first_last_acts,
            autoencoder=autoencoder,
            k=config.scale.num_scaled_layers,
        )

        # Combine total loss
        loss = (
            config.autoencoder.lambda_reconstruction * ae_recons_loss
            + config.autoencoder.lambda_prediction * ae_pred_loss
        )

        loss.backward()
        optimizer.step()

        # Accumulate losses
        loss_epoch += loss.item()
        ae_recons_loss_epoch += ae_recons_loss.item()
        ae_prediction_loss_epoch += ae_pred_loss.item()

    # Average losses over all batches
    epoch_loss = loss_epoch / len(train_loader)
    avg_recons_loss = ae_recons_loss_epoch / len(train_loader)
    avg_prediction_loss = ae_prediction_loss_epoch / len(train_loader)

    # Return all metrics as a dictionary
    return {
        "epoch_loss": epoch_loss,
        "recons_loss": avg_recons_loss,
        "prediction_loss": avg_prediction_loss,
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


def eval_autoencoder(config: Config, model: nn.Module, autoencoder: nn.Module) -> None:
    model.eval()
    model.hook_model()
    autoencoder.eval()
    # Dataset
    cloned_ds_config = DatasetConfig(
        dataset_name=config.train_data.dataset_name,
        num_samples=config.train_data.num_samples,
        split="test",
        torch_transform=config.train_data.torch_transform,
        seed=config.train_data.seed,
        negative_label=config.train_data.negative_label,
    )
    DatasetClass = get_dataset_class(name=cloned_ds_config.dataset_name)
    dataset = DatasetClass(config=cloned_ds_config)
    test_loader = create_data_loader(dataset, batch_size=config.batch_size, global_seed=config.seed)

    # Compute mean reconstruction error
    k = config.scale.num_scaled_layers
    recons_error = torch.tensor(0.0)
    prediction_error = torch.tensor(0.0)
    for batch in test_loader:
        input, label = batch
        _ = model(input)

        gt_acts = model.get_fwd_activations()
        original_num_layers = len(model.modules)
        first_act = gt_acts[0]
        last_act = gt_acts[original_num_layers - 1]

        # First layer
        recons = autoencoder(
            x=pad_act(first_act, target_size=autoencoder.encoder[0].in_features), k=0
        ).reconstruction
        recons_error += F.mse_loss(first_act, recons)

        # Last layer
        recons = autoencoder(
            x=pad_act(last_act, target_size=autoencoder.encoder[0].in_features), k=0
        ).reconstruction[:, : last_act.size(-1)]
        recons_error += F.mse_loss(last_act, recons)

        all_pred = autoencoder(
            x=pad_act(first_act, target_size=autoencoder.encoder[0].in_features),
            k=k,
        ).predictions
        pred = all_pred[k, :, : last_act.size(-1)]
        prediction_error += F.mse_loss(last_act, pred)

    recons_error = recons_error / len(test_loader)
    prediction_error = prediction_error / len(test_loader)

    # Compute spectral quantities
    with torch.no_grad():
        eigenvalues, _ = linalg.eig(autoencoder.koopman_matrix.linear_layer.weight.detach())

    spectral_radius = torch.max(torch.abs(eigenvalues))
    fig, ax = plot_eigenvalues(eigenvalues)

    # Log
    wandb.log(
        {
            "eval/recons_mse": recons_error,
            "eval/pred_mse": prediction_error,
            "eval/spectral_radius": spectral_radius,
            "scatter_plot": wandb.Image(fig, caption="Eigenvalues"),
        }
    )


def main(config_path_or_obj: Optional[Path | str | Config] = None):
    # Setup WandB and load config
    config = setup(config_path_or_obj)

    # Device
    device = get_device()

    # Load data
    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset, batch_size=config.batch_size, global_seed=config.seed
    )

    # Load model
    original_model, _ = MLP.load_model(file_path=config.scale.model_to_scale)
    original_model.to(device).eval()

    # Build autoencoder
    # autoencoder = Autoencoder(
    #     input_dimension=original_model.modules[config.scale.scale_location].in_features,
    #     latent_dimension=config.autoencoder.ae_dim,
    #     nonlinearity=config.autoencoder.ae_nonlinearity,
    # ).to(device)
    autoencoder = ExponentialKoopmanAutencoder(
        k=config.scale.num_scaled_layers,
        input_dimension=original_model.modules[config.scale.scale_location].in_features,
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
                    "prediction_loss": losses["prediction_loss"],
                }
            )

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Train Loss: {losses['epoch_loss']:.4f}, "
                f"Reconstruction Loss: {losses['recons_loss']:.4f}, "
                f"Prediction Loss: {losses['prediction_loss']:.4f} "
            )

    eval_autoencoder(config, original_model, autoencoder)

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
