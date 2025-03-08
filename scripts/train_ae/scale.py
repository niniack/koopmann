import os
import pdb
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from config_def import Config, KoopmanParam
from torch import optim
from torch.utils.data import DataLoader

from koopmann.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from koopmann.log import logger
from koopmann.models import (
    MLP,
    Autoencoder,
    ExponentialKoopmanAutencoder,
    LowRankKoopmanAutoencoder,
    ResMLP,
)
from koopmann.models.utils import get_device
from scripts import adv_attacks
from scripts.train_ae.eval import eval_autoencoder, run_adv_attacks
from scripts.train_ae.losses import (
    compute_eigenloss,
    compute_k_prediction_loss,
    compute_latent_space_prediction_loss,
    compute_sparsity_loss,
    compute_state_space_recons_loss,
    linear_warmup,
)
from scripts.utils import setup_config


########################## TRAIN LOOP ##########################
def train_one_epoch(
    model: nn.Module,
    autoencoder: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: Config,
    probed: bool,
    epoch: int,
) -> dict:
    # --- Initialize effective weights ---
    effective_state_pred_weight = linear_warmup(
        epoch=epoch,
        end_epoch=config.optim.num_epochs,
        final_value=config.autoencoder.lambda_prediction,
    )
    effective_latent_pred_weight = linear_warmup(
        epoch=epoch,
        end_epoch=config.optim.num_epochs,
        final_value=config.autoencoder.lambda_id,
    )

    # --- Initialize loss accumulators ---
    epoch_combined_loss = 0.0
    raw_recons_loss_sum, scaled_recons_loss_sum = 0.0, 0.0
    raw_state_pred_loss_sum, scaled_state_pred_loss_sum = 0.0, 0.0
    raw_latent_pred_loss_sum, scaled_latent_pred_loss_sum = 0.0, 0.0

    num_batches = len(train_loader)

    # --- Training loop ---
    for batch_idx, (input, label) in enumerate(train_loader):
        # Setup
        input, label = input.to(device), label.to(device).squeeze()
        autoencoder.train()
        model.eval()
        model.hook_model()
        optimizer.zero_grad()

        with torch.no_grad():
            _ = model.forward(input)

        # TODO: This is quick and dirty
        ###############################################################
        # Preprocess input
        input = input * 0.3081 + 0.1307  # Undo MNIST standardization
        input = 2 * input - 1  # Convert [0,1] range to [-1,1]
        ###############################################################

        # --- Extract activations ---
        all_acts = model.get_fwd_activations()

        # If scale_location == 0, prepend the raw input as layer 0
        if config.scale.scale_location == 0:
            temp_all_acts = OrderedDict()
            temp_all_acts[0] = input.flatten(start_dim=1)
            for key, val in all_acts.items():
                temp_all_acts[key + 1] = val
            all_acts = temp_all_acts

        # We only use first and last states
        last_index = -3 if probed else -2
        first_key, first_value = next(iter(all_acts.items()))
        last_key, last_value = list(all_acts.items())[last_index]
        first_last_acts = OrderedDict({first_key: first_value, last_key: last_value})

        # --- Compute losses ---
        (raw_recons_loss, scaled_recons_loss) = compute_state_space_recons_loss(
            act_dict=first_last_acts,
            autoencoder=autoencoder,
            k=config.scale.num_scaled_layers,
            probed=probed,
        )

        (raw_state_pred_loss, scaled_state_pred_loss) = compute_k_prediction_loss(
            act_dict=first_last_acts,
            autoencoder=autoencoder,
            k=config.scale.num_scaled_layers,
            probed=probed,
        )

        (raw_latent_pred_loss, scaled_latent_pred_loss) = compute_latent_space_prediction_loss(
            act_dict=first_last_acts,
            autoencoder=autoencoder,
            k=config.scale.num_scaled_layers,
            probed=probed,
        )

        # NOTE: Uses effective weight based on linear warmup
        loss = (
            config.autoencoder.lambda_reconstruction * scaled_recons_loss
            + effective_state_pred_weight * scaled_state_pred_loss
            + effective_latent_pred_weight * raw_latent_pred_loss
        )

        loss.backward()
        optimizer.step()

        # --- Accumulate batch metrics ---
        epoch_combined_loss += loss.item()
        raw_recons_loss_sum += raw_recons_loss.item()
        scaled_recons_loss_sum += scaled_recons_loss.item()
        raw_state_pred_loss_sum += raw_state_pred_loss.item()
        scaled_state_pred_loss_sum += scaled_state_pred_loss.item()
        raw_latent_pred_loss_sum += raw_latent_pred_loss.item()
        scaled_latent_pred_loss_sum += scaled_latent_pred_loss.item()

    # --- Compute averages ---
    epoch_combined_loss /= num_batches
    avg_recons_loss_raw = raw_recons_loss_sum / num_batches
    avg_recons_loss_scaled = scaled_recons_loss_sum / num_batches
    avg_state_pred_loss_raw = raw_state_pred_loss_sum / num_batches
    avg_state_pred_loss_scaled = scaled_state_pred_loss_sum / num_batches
    avg_latent_pred_loss_raw = raw_latent_pred_loss_sum / num_batches
    avg_latent_pred_loss_scaled = scaled_latent_pred_loss_sum / num_batches

    # --- Return results ---
    return {
        "epoch_loss": epoch_combined_loss,
        "reconstruction_error": avg_recons_loss_raw,
        "reconstruction_fvu": avg_recons_loss_scaled,
        "state_pred_loss": avg_state_pred_loss_raw,
        "state_pred_fvu": avg_state_pred_loss_scaled,
        "latent_pred_loss": avg_latent_pred_loss_raw,
        "latent_pred_fvu": avg_latent_pred_loss_scaled,
        # For debugging
        "effective_state_pred_weight": effective_state_pred_weight,
        "effective_latent_pred_weight": effective_latent_pred_weight,
    }


########################## MAIN ##########################
def build_and_load_model(config, device):
    if config.scale.model_with_probe:
        model, _ = MLP.load_model(file_path=config.scale.model_with_probe)
        model.modules[-2].remove_nonlinearity()
        model.modules[-3].remove_nonlinearity()
        # model.modules[-3].update_nonlinearity("leaky_relu")
        is_probed = True
    else:
        file_path = config.scale.model_to_scale
        if "residual" in file_path:
            model, _ = ResMLP.load_model(file_path=file_path)
        else:
            model, _ = MLP.load_model(file_path=file_path)
        is_probed = False

    model.to(device).eval()
    return model, is_probed


def build_autoencoder(config, model, device):
    autoencoder_kwargs = {
        "k": config.scale.num_scaled_layers,
        "input_dimension": model.modules[config.scale.scale_location].in_features,
        "latent_dimension": config.autoencoder.ae_dim,
        "nonlinearity": config.autoencoder.ae_nonlinearity,
        "hidden_configuration": config.autoencoder.hidden_config,
        "batchnorm": config.autoencoder.batchnorm,
        "rank": config.autoencoder.koopman_rank,
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


def load_data(config):
    # Load train data
    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset, batch_size=config.batch_size, global_seed=config.seed
    )

    # Load test data
    cloned_ds_config = DatasetConfig(
        dataset_name=config.train_data.dataset_name,
        num_samples=config.train_data.num_samples,
        split="test",
        torch_transform=config.train_data.torch_transform,
        seed=config.train_data.seed,
        negative_label=config.train_data.negative_label,
    )
    dataset = DatasetClass(config=cloned_ds_config)
    test_loader = create_data_loader(
        dataset,
        batch_size=config.batch_size,
        global_seed=config.seed,
    )

    return dataset, train_loader, test_loader


def main(config_path_or_obj: Optional[Path | str | Config] = None):
    # Setup WandB and load config
    config = setup_config(config_path_or_obj, Config)

    # Device
    device = get_device()

    # Load data
    dataset, train_loader, test_loader = load_data(config=config)

    # Build and load models
    model, is_probed = build_and_load_model(config=config, device=device)

    # Build autoencoder
    autoencoder, flavor = build_autoencoder(config=config, model=model, device=device)
    autoencoder.summary()

    # Define optimiser
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
            model=model,
            autoencoder=autoencoder,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
            probed=is_probed,
            epoch=epoch,
        )

        # Log metrics
        if (epoch + 1) % config.print_freq == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": losses["epoch_loss"],
                    "reconstruction_error": losses["reconstruction_error"],
                    "reconstruction_fvu": losses["reconstruction_fvu"],
                    "state_pred_loss": losses["state_pred_loss"],
                    "state_pred_fvu": losses["state_pred_fvu"],
                    "latent_pred_loss": losses["latent_pred_loss"],
                    "latent_pred_fvu": losses["latent_pred_fvu"],
                },
                step=epoch,
            )
            eval_autoencoder(
                model=model,
                autoencoder=autoencoder,
                test_loader=test_loader,
                config=config,
                probed=is_probed,
                epoch=epoch,
            )

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Train Loss: {losses['epoch_loss']:.4f}, "
            )

    # Save model
    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        filename = (
            f"dim_{config.autoencoder.ae_dim}_"
            f"k_{config.scale.num_scaled_layers}_"
            f"loc_{config.scale.scale_location}_"
            f"{flavor}_"
            f"autoencoder_{config.save_name}.safetensors"
        )
        ae_path = Path(config.save_dir, filename)
        autoencoder.save_model(
            ae_path,
            dataset=dataset.name(),
            num_scaled=config.scale.num_scaled_layers,
            scale_location=config.scale.scale_location,
        )

    # # Adversarial attacks
    # attack_results = run_adv_attacks(str(dataset.root), str(model_path), str(ae_path))
    # for attack_name, methods in attack_results.items():
    #     for method_name in methods:
    #         for eps, acc in methods[method_name].items():
    #             wandb.log(
    #                 {
    #                     "attack": attack_name,
    #                     "method": method_name,
    #                     "epsilon": eps,
    #                     f"{attack_name}/{method_name}/accuracy": acc,
    #                 }
    #             )

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
