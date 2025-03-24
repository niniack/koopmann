import copy
import os
import pdb
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_def import Config, KoopmanParam
from torch.utils.data import DataLoader

import wandb
from analysis.adv_attacks import adv_attack
from koopmann.log import logger
from koopmann.models import (
    MLP,
    Autoencoder,
    ExponentialKoopmanAutencoder,
    LowRankKoopmanAutoencoder,
    ResMLP,
)
from koopmann.utils import get_device
from scripts.train_ae.losses import (
    calculate_replacement_error,
    compute_eigenvector_shaping_loss,
    compute_isometric_loss,
    compute_k_prediction_loss,
    compute_latent_space_prediction_loss,
    compute_state_space_recons_loss,
)
from scripts.utils import (
    get_dataloaders,
    get_optimizer,
    setup_config,
)


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

    # Grab first and last activations
    last_index = -3 if is_probed else -2
    first_key, first_value = next(iter(all_acts.items()))
    last_key, last_value = list(all_acts.items())[last_index]

    # # # Z-score activations
    # first_value = first_value - first_value.mean(dim=1, keepdim=True)
    # first_value = (first_value / torch.norm(first_value, "fro")) * 100

    # last_value = last_value - last_value.mean(dim=1, keepdim=True)
    # last_value = (last_value / torch.norm(last_value, "fro")) * 100

    return OrderedDict({first_key: first_value, last_key: last_value})


class DotDict(dict):
    """Dictionary subclass that provides attribute access to keys."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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

        # self.batch_metrics["replacement_error"] = torch.tensor(0.0, device=get_device())
        self.batch_metrics["shaping_loss"] = torch.tensor(0.0, device=get_device())
        self.batch_metrics["combined_loss"] = torch.tensor(0.0, device=get_device())

        self.total_metrics = copy.deepcopy(self.batch_metrics)
        self.num_batches = 0

        return self

    def update(
        self,
        model=None,
        input=None,
        label=None,
        autoencoder=None,
        config=None,
        is_probed=None,
    ) -> "AutoencoderMetrics":
        # If arguments are provided, compute losses
        if all(arg is not None for arg in [model, input, label, autoencoder, config]):
            # Forward pass
            _ = model(input)

            # Get activations
            first_last_acts = extract_activations(model, input, config, is_probed)
            k_steps = config.scale.num_scaled_layers

            for name, method in self.metric_to_method_dict.items():
                raw, fvu = method(
                    act_dict=first_last_acts,
                    autoencoder=autoencoder,
                    k=k_steps,
                    probed=is_probed,
                )

                self.batch_metrics[f"raw_{name}"] = raw
                self.batch_metrics[f"fvu_{name}"] = fvu

            # self.tensors.replacement_error = calculate_replacement_error(
            #     autoencoder=autoencoder,
            #     model=model,
            #     first_value=first_value,
            #     last_index=last_index,
            #     label=label,
            #     k_steps=k_steps,
            # )

            self.batch_metrics["shaping_loss"] = compute_eigenvector_shaping_loss(
                first_last_acts, autoencoder, label
            )

        # Update totals
        self.num_batches += 1
        for key, value in self.batch_metrics.items():
            if value is not None:
                self.total_metrics[key] += value.detach()

        return self

    def log_metrics(self, epoch: int, prefix: str) -> None:
        """Log training metrics to wandb."""

        log_dict = {}
        log_dict["epoch"] = epoch
        for key, value in self.total_metrics.items():
            log_dict[f"{prefix}/{key}"] = value

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


def eval_autoencoder(model, autoencoder, test_loader, config, is_probed, epoch):
    model.eval().hook_model()
    autoencoder.eval()

    autoencoder_metrics = AutoencoderMetrics()
    device = get_device()

    with torch.no_grad():
        for input, label in test_loader:
            input = input.to(device)
            label = label.to(device).squeeze()

            autoencoder_metrics.update(model, input, label, autoencoder, config, is_probed)

    # Calculate averages
    avg_metrics = autoencoder_metrics.compute()
    autoencoder_metrics.log_metrics(epoch, "eval")

    return avg_metrics


def train_one_epoch(model, autoencoder, train_loader, device, optimizer, config, is_probed, epoch):
    """Train autoencoder for one epoch."""

    # Initialize metrics tracker
    metrics = AutoencoderMetrics()

    # Set up models
    model.eval().hook_model()
    autoencoder.train()

    # Training loop
    for input, label in train_loader:
        # Setup
        input, label = input.to(device), label.to(device).squeeze()
        optimizer.zero_grad()

        # Compute metrics (stores tensor values in metrics.tensors)
        metrics.update(model, input, label, autoencoder, config, is_probed)

        # Calculate combined loss using tensor values
        loss = (
            config.autoencoder.lambda_reconstruction * metrics.batch_metrics.fvu_reconstruction
            + config.autoencoder.lambda_state_pred * metrics.batch_metrics.fvu_state_pred
            + config.autoencoder.lambda_latent_pred * metrics.batch_metrics.raw_latent_pred
            + config.autoencoder.lambda_isometric * metrics.batch_metrics.fvu_distance
            # + 1e-3 * metrics.batch_metrics.shaping_loss
        )

        # Track combined loss
        metrics.set_combined_loss(loss)

        # Backward pass
        loss.backward()
        optimizer.step()

        autoencoder.scaler.update_statistics()

    # Calculate averages
    avg_metrics = metrics.compute()

    # Log
    if (epoch + 1) % config.print_freq // 2 == 0:
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
    autoencoder.save_model(ae_path)

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
        losses = train_one_epoch(
            model=model,
            autoencoder=autoencoder,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
            is_probed=is_probed,
            epoch=epoch,
        )

        scheduler.step()

        # Evaluate
        if (epoch + 1) % config.print_freq == 0:
            eval_autoencoder(
                model=model,
                autoencoder=autoencoder,
                test_loader=test_loader,
                config=config,
                is_probed=is_probed,
                epoch=epoch,
            )

            # Print progress
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Train Loss: {losses['combined_loss']:.4f}"
            )
    # Save model
    ae_path = save_autoencoder(autoencoder, config, flavor)

    # # Test adversarial robustness
    # adv_results = adv_attack(
    #     dataset=test_dataset,
    #     model=model,
    #     autoencoder=autoencoder,
    #     scale_idx=config.scale.scale_location,
    #     k_steps=config.scale.num_scaled_layers,
    #     device=device,
    # )

    # for attack_name, scenarios_results in adv_results.items():
    #     # Process each scenario (adv_original_acc_original, adv_original_acc_autoencoder, etc.)
    #     for scenario_name, epsilon_results in scenarios_results.items():
    #         # For each epsilon value, log the accuracy
    #         for epsilon, accuracy in epsilon_results.items():
    #             # Create a descriptive metric name
    #             metric_name = f"{attack_name}/{scenario_name}"

    #             # Log both the epsilon and accuracy
    #             wandb.log({f"{attack_name}/epsilon": epsilon, metric_name: accuracy})

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
