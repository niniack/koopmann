import copy
import os
import pdb
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from config_def import Config, KoopmanParam
from torch.utils.data import DataLoader

import wandb
from koopmann.log import logger
from koopmann.models import (
    MLP,
    Autoencoder,
    ExponentialKoopmanAutencoder,
    FrankensteinKoopmanAutoencoder,
    LowRankKoopmanAutoencoder,
    ResMLP,
)
from koopmann.models.layers import Layer
from koopmann.utils import get_device
from scripts.train_ae.losses import (
    compute_eigenvector_shaping_loss,
    compute_isometric_loss,
    compute_k_prediction_loss,
    compute_latent_space_prediction_loss,
    compute_state_space_recons_loss,
)
from scripts.utils import DotDict, get_dataloaders, get_optimizer, setup_config

############## UTILS ###################


def get_model(config, device):
    """Load model based on configuration."""
    if config.scale.model_with_probe:
        is_probed = True
        file_path = config.scale.model_with_probe

        model, _ = MLP.load_model(file_path=file_path)
        model.modules[-2].remove_nonlinearity()
        model.modules[-3].remove_nonlinearity()
    else:
        is_probed = False
        file_path = config.scale.model_to_scale

        if "res" in file_path:
            model, _ = ResMLP.load_model(file_path=file_path)
        else:
            model, _ = MLP.load_model(file_path=file_path)

    model.to(device).eval()
    return model, is_probed


def get_autoencoder(
    config: Config, model: nn.Module, device: torch.device
) -> Tuple[nn.Module, str]:
    """Create autoencoder based on configuration."""
    autoencoder_kwargs = {
        "k_steps": config.scale.num_scaled_layers,
        # "in_features": model.components[config.scale.scale_location].in_channels,
        "in_features": 784,  # TODO: FIX!!!!!!!!!!
        "latent_features": config.autoencoder.ae_dim,
        "hidden_config": config.autoencoder.hidden_config,
        "batchnorm": config.autoencoder.batchnorm,
        "bias": True,
        "rank": config.autoencoder.koopman_rank,
        "nonlinearity": config.autoencoder.ae_nonlinearity,
        "use_eigeninit": False,
    }

    if config.autoencoder.koopman_param == KoopmanParam.exponential:
        autoencoder = ExponentialKoopmanAutencoder(**autoencoder_kwargs).to(device)
        flavor = config.autoencoder.koopman_param.value
    elif config.autoencoder.koopman_param == KoopmanParam.lowrank:
        autoencoder = LowRankKoopmanAutoencoder(**autoencoder_kwargs).to(device)
        flavor = f"{config.autoencoder.koopman_param.value}_{config.autoencoder.koopman_rank}"
    else:
        autoencoder = Autoencoder(**autoencoder_kwargs).to(device)
        flavor = "standard"

    return autoencoder, flavor


def extract_activations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    config: Config,
    is_probed: bool,
    only_first_last: bool,
) -> OrderedDict:
    """Extract activations from model."""
    # Fill activations dict.
    with torch.no_grad():
        _ = model.forward(input_tensor)

    # Grab activations
    all_acts = model.get_forward_activations()

    # If scale_location == 0, prepend the raw input as layer 0
    if config.scale.scale_location == 0:
        temp_all_acts = OrderedDict()
        temp_all_acts[0] = input_tensor.flatten(start_dim=1)
        for key, val in all_acts.items():
            temp_all_acts[key + 1] = val
        all_acts = temp_all_acts

    last_index = -3 if is_probed else -2

    # Get all items once
    items = list(all_acts.items())

    if only_first_last:
        # Only keep first and last items
        first_item = items[0]
        last_item = items[last_index]
        acts_dict = OrderedDict([first_item, last_item])
    else:
        # Keep all items up to and including last_index
        acts_dict = OrderedDict(items[: last_index + 1])

    return acts_dict


############ METRICS ##################
class AutoencoderMetrics:
    def __init__(self):
        """Initialize the metrics computation state."""

        self.metric_to_method_dict = {
            "reconstruction": compute_state_space_recons_loss,
            "state_pred": compute_k_prediction_loss,
            "latent_pred": compute_latent_space_prediction_loss,
            "distance": compute_isometric_loss,
        }

        self.reset()

    def reset(self) -> "AutoencoderMetrics":
        """Reset the internal state to initial values."""

        self.batch_metrics = DotDict()
        for name, method in self.metric_to_method_dict.items():
            self.batch_metrics[f"raw_{name}"] = torch.tensor(0.0, device=get_device())
            self.batch_metrics[f"fvu_{name}"] = torch.tensor(0.0, device=get_device())

        self.batch_metrics["shaping_loss"] = torch.tensor(0.0, device=get_device())
        self.batch_metrics["combined_loss"] = torch.tensor(0.0, device=get_device())

        self.total_metrics = copy.deepcopy(self.batch_metrics)
        self.num_batches = 0

        return self

    def update(
        self,
        model,
        input,
        label,
        autoencoder,
        config,
        is_probed=None,
        only_first_last=True,
    ) -> OrderedDict:
        # Forward pass
        _ = model(input)

        # Get activations
        act_dict = extract_activations(model, input, config, is_probed, only_first_last)

        # Get number of steps
        k_steps = config.scale.num_scaled_layers

        # Compute each loss
        for name, method in self.metric_to_method_dict.items():
            raw, fvu = method(
                act_dict=act_dict,
                autoencoder=autoencoder,
                k=k_steps,
                probed=is_probed,
            )

            self.batch_metrics[f"raw_{name}"] = raw
            self.batch_metrics[f"fvu_{name}"] = fvu

        self.batch_metrics["shaping_loss"] = compute_eigenvector_shaping_loss(
            act_dict, autoencoder, label
        )

        # Update totals
        self.num_batches += 1
        for key, value in self.batch_metrics.items():
            if value is not None:
                self.total_metrics[key] += value.detach()

        return act_dict

    def log_metrics(self, epoch: int, prefix: str) -> None:
        """Log training metrics to wandb."""

        log_dict = {}
        log_dict["epoch"] = epoch
        for key, value in self.total_metrics.items():
            log_dict[f"{prefix}/{key}"] = value / self.num_batches

        wandb.log(log_dict, step=epoch)

    def set_combined_loss(self, loss):
        self.total_metrics["combined_loss"] += loss.detach()
        return self

    def compute(self) -> DotDict:
        if self.num_batches == 0:
            self.avg_metrics = DotDict({k: 0.0 for k in self.total_metrics.keys()})

        self.avg_metrics = DotDict(
            {k: v.item() / self.num_batches for k, v in self.total_metrics.items()}
        )

        return self.avg_metrics


########## COMPUTATION ################
def eval_log_autoencoder(model, autoencoder, test_loader, device, config, is_probed, epoch):
    # Initialize metrics tracker
    metrics = AutoencoderMetrics()

    # Set up models
    model.eval().hook_model()
    autoencoder.train()

    # Forward
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device).squeeze()
            metrics.update(model, input, label, autoencoder, config, is_probed)

    # Log metrics
    metrics.log_metrics(epoch, "eval")


def train_one_epoch(model, autoencoder, train_loader, device, config, is_probed, epoch, optimizer):
    """Train autoencoder for one epoch."""

    # Initialize metrics tracker
    metrics = AutoencoderMetrics()

    # Set up models
    model.to(device).eval().hook_model()
    autoencoder.to(device).train()

    # Training loop
    for inputs, labels in train_loader:
        # Setup
        batch_size = inputs.size(0)
        inputs_ = inputs.clone()
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        optimizer.zero_grad()

        # Compute losses
        act_dict = metrics.update(
            model, inputs, labels, autoencoder, config, is_probed, only_first_last=True
        )

        # Loss weighting
        lambda_reconstruction = config.autoencoder.lambda_reconstruction
        lambda_state_pred = config.autoencoder.lambda_state_pred
        lambda_latent_pred = config.autoencoder.lambda_latent_pred
        lambda_isometric = config.autoencoder.lambda_isometric
        lambda_shaping_loss = 1e-3

        # Weighted loss
        loss = (
            lambda_reconstruction * metrics.batch_metrics.fvu_reconstruction
            + lambda_state_pred * metrics.batch_metrics.fvu_state_pred
            + lambda_latent_pred * metrics.batch_metrics.raw_latent_pred
            + lambda_isometric * metrics.batch_metrics.fvu_distance
            # + lambda_shaping_loss * metrics.batch_metrics.shaping_loss
        )

        # # NOTE: adhoc implementation of nuclear norm
        # nuc_norm = torch.tensor(0.0).to(device)
        # for name, module in autoencoder.named_modules():
        #     if isinstance(module, torch.nn.Linear):
        #         if "decoder" in name or "encoder" in name:
        #             weight = module.weight
        #             nuc_norm += torch.linalg.norm(weight, ord="nuc")
        #     elif "koopman" in name and isinstance(module, Layer):
        #         weight = (module.components.lora_up.weight @ module.components.lora_down.weight).T
        #         nuc_norm += torch.linalg.norm(weight, ord="nuc")
        # loss += 1e-4 * nuc_norm

        # Track combined loss
        metrics.set_combined_loss(loss)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Calculate averages
    avg_metrics = metrics.compute()

    # Log
    if (epoch + 1) % config.print_freq == 0:
        metrics.log_metrics(epoch, "train")

    # Return results
    return avg_metrics


def save_autoencoder(autoencoder, config, flavor):
    """Save trained autoencoder model."""
    if not config.save_dir:
        return None

    os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)

    filename = (
        f"dim_{config.autoencoder.ae_dim}_"
        f"k_{config.scale.num_scaled_layers}_"
        f"loc_{config.scale.scale_location}_"
        f"{flavor}_"
        f"autoencoder_{config.save_name}.safetensors"
    )
    ae_path = Path(config.save_dir, filename)
    autoencoder.save_model(ae_path, suffix=None)

    return ae_path


def main(config_path_or_obj: Optional[Union[Path, str, Config]] = None):
    """Main function to train the autoencoder."""

    # Setup
    config = setup_config(config_path_or_obj, Config)
    device = get_device()

    # Get data
    train_loader, test_loader, test_dataset = get_dataloaders(config=config)

    # Load model and create autoencoder
    model, is_probed = get_model(config=config, device=device)
    autoencoder, flavor = get_autoencoder(config=config, model=model, device=device)
    autoencoder.summary()

    # Setup optimizer
    optimizer = get_optimizer(config, autoencoder)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.optim.num_epochs)

    # Training loop
    for epoch in range(config.optim.num_epochs):
        # Train step
        train_losses = train_one_epoch(
            model=model,
            autoencoder=autoencoder,
            train_loader=train_loader,
            device=device,
            config=config,
            is_probed=is_probed,
            epoch=epoch,
            optimizer=optimizer,
        )

        scheduler.step()

        # Evaluate
        if (epoch + 1) % config.print_freq == 0:
            eval_log_autoencoder(
                model=model,
                autoencoder=autoencoder,
                test_loader=test_loader,
                device=device,
                config=config,
                is_probed=is_probed,
                epoch=epoch,
            )

            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Train Loss: {train_losses['combined_loss']:.4f}, "
            )

    # Save model
    _ = save_autoencoder(autoencoder, config, flavor)

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
