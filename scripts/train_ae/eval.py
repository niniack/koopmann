from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from config_def import Config
from torch.utils.data import DataLoader

import wandb
from koopmann.models.utils import get_device
from scripts import adv_attacks
from scripts.train_ae.losses import (
    compute_k_prediction_loss,
    compute_latent_space_prediction_loss,
    compute_state_space_recons_loss,
)


########################## EVAL CODE ##########################
def eval_autoencoder(
    model: nn.Module,
    autoencoder: nn.Module,
    test_loader: DataLoader,
    config: Config,
    probed: bool,
    epoch: int,
) -> None:
    """
    Evaluates the autoencoder on the test set. Called at intervals.

    Args:
        model: The frozen model from which we extract layer activations.
        autoencoder: The Koopman-based autoencoder we are evaluating.
        test_loader: Data loader for test set.
        config: Configuration object with dataset info, scale info, etc.
        probed: Whether we are in 'probed' mode or not (affects layer indexing).
        epoch: Current epoch (for logging steps in wandb).
    """

    # Put both models into eval mode
    model.eval()
    model.hook_model()
    autoencoder.eval()

    device = get_device()
    num_batches = len(test_loader)

    # We will accumulate raw/scaled errors for reconstruction & prediction
    total_raw_recons = 0.0
    total_scaled_recons = 0.0

    total_raw_state_pred = 0.0
    total_scaled_state_pred = 0.0

    total_raw_latent_pred = 0.0
    total_scaled_latent_pred = 0.0

    total_replacement_error = 0.0

    def replace_activations_error(autoencoder, k, model, first_value, last_index, label):
        pred_act = autoencoder(first_value, k=k).predictions[-1]  # shape: [batch, neurons]
        output = model.modules[last_index + 1 :](pred_act)
        return F.cross_entropy(output, label.long())

    # Evaluate
    with torch.no_grad():
        for batch in test_loader:
            # Prepare data
            input, label = batch
            input = input.to(device)
            label = label.to(device).squeeze()

            # Forward pass through frozen model to get activations
            _ = model(input)
            all_acts = model.get_fwd_activations()

            # TODO: This is quick and dirty
            ###############################################################
            # Undo MNIST standardization: X_original = X_standardized * std + mean
            input = input * 0.3081 + 0.1307

            # Convert [0,1] range to [-1,1]
            input = 2 * input - 1
            ###############################################################

            if config.scale.scale_location == 0:
                # Add the original inputs as part of `all_acts` and shift keys
                temp_all_acts = OrderedDict()
                temp_all_acts[0] = input.flatten(start_dim=1)
                for key, val in all_acts.items():
                    temp_all_acts[key + 1] = val
                all_acts = temp_all_acts

            # Slice out first & last layers
            last_index = -3 if probed else -2
            first_key, first_value = next(iter(all_acts.items()))
            last_key, last_value = list(all_acts.items())[last_index]
            first_last_acts = OrderedDict({first_key: first_value, last_key: last_value})

            # Compute reconstruction (raw & scaled)
            raw_recons_error, scaled_recons_error = compute_state_space_recons_loss(
                act_dict=first_last_acts,
                autoencoder=autoencoder,
                k=config.scale.num_scaled_layers,
                probed=probed,
            )

            # Compute k-step state prediction (raw & scaled)
            raw_state_pred_error, scaled_state_pred_error = compute_k_prediction_loss(
                act_dict=first_last_acts,
                autoencoder=autoencoder,
                k=config.scale.num_scaled_layers,
                probed=probed,
            )

            # Compute k-step latent prediction (raw & scaled)
            raw_latent_pred_error, scaled_latent_pred_error = compute_latent_space_prediction_loss(
                act_dict=first_last_acts,
                autoencoder=autoencoder,
                k=config.scale.num_scaled_layers,
                probed=probed,
            )

            # Replacement error
            replacement_error = replace_activations_error(
                autoencoder=autoencoder,
                k=config.scale.num_scaled_layers,
                model=model,
                first_value=first_value,
                last_index=last_index,
                label=label,
            )

            # Accumulate
            total_raw_recons += raw_recons_error.item()
            total_scaled_recons += scaled_recons_error.item()
            total_raw_state_pred += raw_state_pred_error.item()
            total_scaled_state_pred += scaled_state_pred_error.item()
            total_raw_latent_pred += raw_latent_pred_error.item()
            total_scaled_latent_pred += scaled_latent_pred_error.item()
            total_replacement_error += replacement_error.item()

    # Averages
    avg_raw_recons = total_raw_recons / num_batches
    avg_scaled_recons = total_scaled_recons / num_batches
    avg_raw_state_pred = total_raw_state_pred / num_batches
    avg_state_scaled_pred = total_scaled_state_pred / num_batches
    avg_raw_latent_pred = total_raw_latent_pred / num_batches
    avg_state_latent_pred = total_scaled_latent_pred / num_batches
    avg_replacement_error = total_replacement_error / num_batches

    # Log to wandb
    wandb.log(
        {
            "eval/recons_raw": avg_raw_recons,
            "eval/recons_scaled": avg_scaled_recons,
            "eval/state_pred_raw": avg_raw_state_pred,
            "eval/state_pred_scaled": avg_state_scaled_pred,
            "eval/latent_pred_raw": avg_raw_latent_pred,
            "eval/latent_pred_scaled": avg_state_latent_pred,
            "eval/replacement_error": avg_replacement_error,
        },
        step=epoch,
    )


def run_adv_attacks(data_path: str, model_path: str, ae_path: str):
    return adv_attacks.main.callback(data_path, model_path, ae_path)
