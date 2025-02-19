import math
import os
import pdb
import sys
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import reduce
from pathlib import Path
from typing import List, Literal, Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from config_def import Config
from torch import linalg, optim
from torch.nn.utils.parametrizations import _Orthogonal, orthogonal
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

import wandb
from koopmann.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from koopmann.log import logger
from koopmann.models import MLP, Autoencoder, ExponentialKoopmanAutencoder
from koopmann.models.utils import get_device
from koopmann.utils import set_seed
from koopmann.visualization import plot_eigenvalues
from scripts.common import load_config, setup_config


########################## UTILITY ##########################
def pad_act(x, target_size):
    current_size = x.size(1)
    if current_size < target_size:
        pad_size = target_size - current_size
        x = F.pad(x, (0, pad_size), mode="constant", value=0)

    return x


def _prepare_padded_acts_and_masks(act_dict, autoencoder):
    """
    Returns:
      padded_acts: list of padded activation tensors (one per layer)
      masks: list of 1/0 masks for ignoring "extra" neurons in padded activations
    """

    # Autoencoder input dimension
    ae_input_size = autoencoder.encoder[0].in_features
    padded_acts = []
    masks = []

    # Iterate through all activations
    for act in act_dict.values():
        # Pad activations
        padded_act = pad_act(act, ae_input_size)
        padded_acts.append(padded_act)

        # Construct a mask that has '1' up to original size, then 0
        mask = torch.zeros(ae_input_size, device=act.device)
        mask[: act.size(-1)] = 1
        masks.append(mask)

    return padded_acts, masks


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
    device = get_device()
    k = config.scale.num_scaled_layers
    recons_error = 0.0
    prediction_error = 0.0

    # Iterate through loader
    for batch in test_loader:
        input, label = batch
        input, label = input.to(device), label.to(device).squeeze()
        _ = model(input)

        gt_acts = model.get_fwd_activations()
        for act in gt_acts:
            recons = autoencoder(
                x=pad_act(act, target_size=autoencoder.encoder[0].in_features), k=0
            ).reconstruction[:, : act.size(-1)]
            recons_error += F.mse_loss(act.to(device), recons.to(device)).item()

        # Use the *raw input* to predict forward
        all_pred = autoencoder(
            x=pad_act(input, target_size=autoencoder.encoder[0].in_features),
            k=k,
        ).predictions

        # Compare final predicted activations to the final ground truth activations
        last_act = gt_acts[-1]
        pred = all_pred[-1, :, : last_act.size(-1)]
        prediction_error += F.mse_loss(last_act.to(device), pred.to(device)).item()

    recons_error = recons_error / len(test_loader)
    prediction_error = prediction_error / len(test_loader)

    # Compute spectral quantities
    with torch.no_grad():
        eigenvalues, _ = linalg.eig(autoencoder.koopman_matrix.linear_layer.weight.detach())

    spectral_radius = torch.max(torch.abs(eigenvalues))
    fig, axes = plot_eigenvalues(
        eigenvalues_dict={(k, autoencoder.latent_dimension): eigenvalues},
        tile_size=4,
        num_rows=1,
        num_cols=1,
    )
    # Log
    wandb.log(
        {
            "eval/recons_mse": recons_error,
            "eval/pred_mse": prediction_error,
            "eval/spectral_radius": spectral_radius,
            "scatter_plot": wandb.Image(fig, caption="Eigenvalues"),
        }
    )


########################## Reconstruction Loss ##########################
def compute_state_space_recons_loss(act_dict, autoencoder):
    """
    Computes the reconstruction loss in the *state space* across all layers.
    Returns reconstruction MSE scaled by total variance (across all layers).
    """
    # Prepare padded activations and masks
    # shape: [batch, layer, norms]
    padded_acts, masks = _prepare_padded_acts_and_masks(act_dict, autoencoder)

    # Stack along layers dimension
    padded_acts = torch.stack(padded_acts, dim=1)  # shape: [batch, layers, neurons]
    masks = torch.stack(masks, dim=0)  # shape: [layers, neurons]

    # Reconstruct each layer’s padded activation (k=0 => no Koopman stepping)
    recons_acts = [
        autoencoder(padded_act, k=0).reconstruction for padded_act in padded_acts.unbind(dim=1)
    ]

    # Same shape as padded_acts, shape: [batch, layers, neurons]
    recons_acts = torch.stack(recons_acts, dim=1)

    # MSE in state space, ignoring masked-out neurons
    masked_diff = (padded_acts - recons_acts) * masks.unsqueeze(dim=0)
    recons_error = masked_diff.pow(2).sum()

    # Total variance in state space
    masked_centered_acts = (padded_acts - padded_acts.mean(dim=0)) * masks.unsqueeze(dim=0)
    total_variance_state_space = masked_centered_acts.pow(2).sum()

    # Return ratio = (MSE) / (variance)
    return recons_error / total_variance_state_space


# def compute_state_space_recons_loss(act_dict, autoencoder, optimizer):
#     """
#     Computes reconstruction loss in state space via learnable projections.
#     Each layer's activations are projected to a fixed D-dim space, reconstructed,
#     and then inverted using an orthonormal projection (so that the inverse is simply Q^T).
#     The loss is the MSE in the original space scaled by total variance.
#     """
#     # Cache projections on the autoencoder object.
#     if not hasattr(autoencoder, "random_projections"):
#         autoencoder.random_projections = nn.ParameterDict()

#     loss = 0.0
#     total_var = 0.0
#     D = autoencoder.encoder[0].in_features  # target projection dimension

#     # Iterate over all activations in the dictionary
#     for key, x in act_dict.items():
#         batch_size, d = x.shape
#         key = str(key)

#         # Get (or create) a learnable projection Q: shape [D, d]
#         # NOTE: is the separate init for key == 0 really necessary?
#         if key not in autoencoder.random_projections:
#             if key == "0":
#                 Q = torch.eye(D, device=x.device)
#             else:
#                 Q = torch.randn(D, d, device=x.device) / math.sqrt(D)
#             autoencoder.random_projections[key] = nn.Parameter(Q)
#             optimizer.add_param_group({"params": autoencoder.random_projections[key]})

#         # Grab Q
#         Q = autoencoder.random_projections[key]

#         # Enforce orthonormal columns via QR decomposition.
#         # NOTE: Q_orth will be [D, d] with Q_orth^T Q_orth = I.
#         Q_orth, _ = torch.linalg.qr(Q)

#         # Project d-dimensional object to D-dimensional space: [batch, D]
#         x_proj = x @ Q_orth.T

#         # Reconstruct in D-dim space using the autoencoder (k=0 means no Koopman stepping!)
#         recon_proj = autoencoder(x_proj, k=0).reconstruction

#         # Invert the projection using the fact that the pseudo-inverse of an orthonormal Q is Q^T.
#         x_recons = recon_proj @ Q_orth

#         # Accumulate reconstruction loss (squared error) and variance in original space.
#         diff = x - x_recons
#         loss += diff.pow(2).sum()

#         x_mean = x.mean(dim=0, keepdim=True)
#         total_var += (x - x_mean).pow(2).sum()

#     return loss / total_var


########################## K-Step Prediction Loss (State Space) ##########################
def compute_k_prediction_loss(act_dict, autoencoder, k):
    """
    Computes the k-step prediction loss in *state space* for the final activation.
    Compares the predicted state at k steps with the actual final-layer activation.
    """
    # Extract activations from the dictionary
    act_list = [act_dict[k] for k in sorted(act_dict, key=int)]

    # We'll predict the final-layer activation from the first-layer activation
    target_acts = act_list[-1]  # shape: [batch, neurons]
    total_variance = (target_acts - target_acts.mean(dim=0)).pow(2).sum()

    Q0 = autoencoder.random_projections[str(0)]
    QN = autoencoder.random_projections[str(5)]

    # Enforce orthonormal columns via QR decomposition.
    # Note: Q_orth will be [D, d] with Q_orth^T Q_orth = I.
    # Q0_orth, _ = torch.linalg.qr(Q0)
    # QN_orth, _ = torch.linalg.qr(QN)

    Q0_orth = Q0
    QN_orth = QN

    # Project to D-dimensional space: [batch, D]
    x_proj = act_list[0] @ Q0_orth.T

    # Get autoencoder predictions: shape [layers, batch, neurons]
    all_preds = autoencoder(x=x_proj, k=k).predictions

    # The final prediction is all_preds[-1]
    pred_k = all_preds[-1] @ QN_orth
    state_space_pred_error = (pred_k - target_acts).pow(2).sum()

    return state_space_pred_error / total_variance


########################## K-Step Prediction Loss (Latent Space) ##########################
def compute_latent_space_prediction_loss(act_dict, autoencoder, k):
    """
    Computes the k-step prediction loss purely in the *latent space*.
    We encode each layer’s activation into latent space, then compare
    the predicted next-layer embedding (via the Koopman matrix)
    with the actual final-layer embedding.
    """
    # Extract activations from the dictionary
    act_list = [act_dict[k] for k in sorted(act_dict, key=int)]

    Q0 = autoencoder.random_projections[str(0)]
    QN = autoencoder.random_projections[str(5)]

    # Enforce orthonormal columns via QR decomposition.
    # Note: Q_orth will be [D, d] with Q_orth^T Q_orth = I.
    # Q0_orth, _ = torch.linalg.qr(Q0)
    # QN_orth, _ = torch.linalg.qr(QN)

    Q0_orth = Q0
    QN_orth = QN

    # Project to D-dimensional space: [batch, D]
    latent_first_act = autoencoder._encode(act_list[0] @ Q0_orth.T)
    latent_last_act = autoencoder._encode(act_list[-1] @ QN_orth.T)

    # Koopman k-step: multiply the *first* layer’s latent by K^k
    K_weight = autoencoder.koopman_matrix.linear_layer.weight
    K_effective_weight = torch.linalg.matrix_power(K_weight, k)
    predicted_last_act = latent_first_act @ K_effective_weight.T

    # Compare predicted embedding with the actual *last-layer* embedding
    latent_error = (latent_last_act - predicted_last_act).pow(2).sum()

    # Total variance in final layer’s latent
    latent_last_centered_acts = latent_last_act - latent_last_act.mean(dim=0)
    total_variance_latent_space = latent_last_centered_acts.pow(2).sum()

    return latent_error / total_variance_latent_space


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
    ae_state_prediction_loss_epoch = 0
    ae_latent_pred_loss_epoch = 0

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

        # Get activations
        all_acts = model.get_fwd_activations()

        if config.scale.scale_location == 0:
            # Add the original inputs as part of the `all_acts` dict.
            # But that requires shifting each key down one
            temp_all_acts = OrderedDict()
            temp_all_acts[0] = input.flatten(start_dim=1)
            for key, val in all_acts.items():
                temp_all_acts[key + 1] = val

            all_acts = temp_all_acts

        # Get first and last elements without modifying the dict
        first_key, first_value = next(iter(all_acts.items()))
        last_key, last_value = next(iter(reversed(all_acts.items())))
        first_last_acts = OrderedDict({first_key: first_value, last_key: last_value})

        # Compute autoencoder losses
        ae_recons_loss = compute_state_space_recons_loss(
            act_dict=first_last_acts,
            autoencoder=autoencoder,
            # optimizer=optimizer,
        )

        ae_state_pred_loss = torch.tensor(0.0)
        ae_latent_pred_loss = torch.tensor(0.0)

        # ae_state_pred_loss = compute_k_prediction_loss(
        #     act_dict=first_last_acts,
        #     autoencoder=autoencoder,
        #     k=config.scale.num_scaled_layers,
        # )
        # ae_latent_pred_loss = compute_latent_space_prediction_loss(
        #     act_dict=first_last_acts,
        #     autoencoder=autoencoder,
        #     k=config.scale.num_scaled_layers,
        # )

        # Combine total loss
        loss = (
            config.autoencoder.lambda_reconstruction * ae_recons_loss
            + config.autoencoder.lambda_prediction * ae_state_pred_loss
            + config.autoencoder.lambda_id * ae_latent_pred_loss
        )

        loss.backward()
        optimizer.step()

        # Accumulate losses
        loss_epoch += loss.item()
        ae_recons_loss_epoch += ae_recons_loss.item()
        ae_state_prediction_loss_epoch += ae_state_pred_loss.item()
        ae_latent_pred_loss_epoch += ae_latent_pred_loss.item()

    # Average losses over all batches
    epoch_loss = loss_epoch / len(train_loader)
    avg_recons_loss = ae_recons_loss_epoch / len(train_loader)
    avg_state_pred_loss = ae_state_prediction_loss_epoch / len(train_loader)
    avg_latent_pred_loss = ae_latent_pred_loss_epoch / len(train_loader)

    # Return all metrics as a dictionary
    return {
        "epoch_loss": epoch_loss,
        "recons_loss": avg_recons_loss,
        "state_pred_loss": avg_state_pred_loss,
        "observable_pred_loss": avg_latent_pred_loss,
    }


########################## MAIN ##########################
def main(config_path_or_obj: Optional[Path | str | Config] = None):
    # Setup WandB and load config
    config = setup_config(config_path_or_obj, Config)

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

    # Load model
    original_model, _ = MLP.load_model(file_path=config.scale.model_to_scale)
    original_model.to(device).eval()

    # Build autoencoder
    autoencoder_kwargs = {
        "k": config.scale.num_scaled_layers,
        "input_dimension": original_model.modules[config.scale.scale_location].in_features,
        "latent_dimension": config.autoencoder.ae_dim,
        "nonlinearity": config.autoencoder.ae_nonlinearity,
        "hidden_configuration": config.autoencoder.hidden_config,
        "batchnorm": config.autoencoder.batchnorm,
    }
    if config.autoencoder.exp_param:
        autoencoder = ExponentialKoopmanAutencoder(**autoencoder_kwargs).to(device)
    else:
        autoencoder = Autoencoder(**autoencoder_kwargs).to(device)

    autoencoder.summary()

    # Define optimiser and scheduler
    optimizer = optim.AdamW(
        params=list(autoencoder.parameters()),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=(0.9, 0.999) if not config.optim.betas else tuple(config.optim.betas),
    )

    scheduler = StepLR(
        optimizer,
        step_size=80,  # decay every 50 epochs
        gamma=0.7,  # multiply LR by this factor
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
        scheduler.step()

        # Log metrics
        if (epoch + 1) % config.print_freq == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": losses["epoch_loss"],
                    "recons_loss": losses["recons_loss"],
                    "state_pred_loss": losses["state_pred_loss"],
                    "observable_pred_loss": losses["observable_pred_loss"],
                },
                step=epoch,
            )

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Train Loss: {losses['epoch_loss']:.4f}, "
                f"Reconstruction Loss: {losses['recons_loss']:.4f}, "
                f"State Prediction Loss: {losses['state_pred_loss']:.4f} "
                f"Observable Prediction Loss: {losses['observable_pred_loss']:.4f} "
            )

    # eval_autoencoder(config, original_model, autoencoder)

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

################# V1 (boohoo) ##############################
# def compute_state_space_recons_loss(act_dict, autoencoder):
#     """
#     Computes the reconstruction loss in the *state space* across all layers.
#     Returns reconstruction MSE scaled by total variance (across all layers).
#     """
#     # Prepare padded activations and masks
#     # shape: [batch, layer, norms]
#     padded_acts, masks = _prepare_padded_acts_and_masks(act_dict, autoencoder)

#     # Stack along layers dimension
#     padded_acts = torch.stack(padded_acts, dim=1)  # shape: [batch, layers, neurons]
#     masks = torch.stack(masks, dim=0)  # shape: [layers, neurons]
#     pdb.set_trace()

#     # Reconstruct each layer’s padded activation (k=0 => no Koopman stepping)
#     recons_acts = [
#         autoencoder(padded_act, k=0).reconstruction for padded_act in padded_acts.unbind(dim=1)
#     ]

#     # Same shape as padded_acts, shape: [batch, layers, neurons]
#     recons_acts = torch.stack(recons_acts, dim=1)
#     pdb.set_trace()

#     # MSE in state space, ignoring masked-out neurons
#     masked_diff = (padded_acts - recons_acts) * masks.unsqueeze(dim=0)
#     recons_error = masked_diff.pow(2).sum()
#     pdb.set_trace()

#     # Total variance in state space
#     masked_centered_acts = (padded_acts - padded_acts.mean(dim=0)) * masks.unsqueeze(dim=0)
#     total_variance_state_space = masked_centered_acts.pow(2).sum()
#     pdb.set_trace()

#     # Return ratio = (MSE) / (variance)
#     return recons_error / total_variance_state_space

################# V2 (works best!) ##############################
# def compute_state_space_recons_loss(act_dict, autoencoder, optimizer):
#     """
#     Computes reconstruction loss in state space via learnable projections.
#     Each layer's activations are projected to a fixed D-dim space, reconstructed,
#     and then inverted using an orthonormal projection (so that the inverse is simply Q^T).
#     The loss is the MSE in the original space scaled by total variance.
#     """
#     # Cache projections on the autoencoder object.
#     if not hasattr(autoencoder, "random_projections"):
#         autoencoder.random_projections = nn.ParameterDict()

#     loss = 0.0
#     total_var = 0.0
#     D = autoencoder.encoder[0].in_features  # target projection dimension

#     # Iterate over all activations in the dictionary
#     for key, x in act_dict.items():
#         batch_size, d = x.shape
#         key = str(key)

#         # Get (or create) a learnable projection Q: shape [D, d]
#         # NOTE: is the separate init for key == 0 really necessary?
#         if key not in autoencoder.random_projections:
#             if key == "0":
#                 Q = torch.eye(D, device=x.device)
#             else:
#                 Q = torch.randn(D, d, device=x.device) / math.sqrt(D)
#             autoencoder.random_projections[key] = nn.Parameter(Q)
#             optimizer.add_param_group({"params": autoencoder.random_projections[key]})

#         # Grab Q
#         Q = autoencoder.random_projections[key]

#         # Enforce orthonormal columns via QR decomposition.
#         # NOTE: Q_orth will be [D, d] with Q_orth^T Q_orth = I.
#         Q_orth, _ = torch.linalg.qr(Q)

#         # Project d-dimensional object to D-dimensional space: [batch, D]
#         x_proj = x @ Q_orth.T

#         # Reconstruct in D-dim space using the autoencoder (k=0 means no Koopman stepping!)
#         recon_proj = autoencoder(x_proj, k=0).reconstruction

#         # Invert the projection using the fact that the pseudo-inverse of an orthonormal Q is Q^T.
#         x_recons = recon_proj @ Q_orth

#         # Accumulate reconstruction loss (squared error) and variance in original space.
#         diff = x - x_recons
#         loss += diff.pow(2).sum()

#         x_mean = x.mean(dim=0, keepdim=True)
#         total_var += (x - x_mean).pow(2).sum()

#     return loss / total_var

################# V3 ##############################
# def compute_state_space_recons_loss(act_dict, autoencoder, optimizer):
#     """
#     Computes reconstruction loss in state space via learnable projections.
#     Each layer's activations are projected to a fixed D-dim space, reconstructed,
#     and then inverted using an orthogonal projection achieved via nn.utils.parametrizations.orthogonal.

#     For each key:
#       - The MSE in the original space is computed and normalized by the total variance.
#       - This normalized loss is then weighted by the number of features in that layer,
#         so that layers with more features (e.g. 784) naturally contribute more than
#         layers with fewer features (e.g. 10).
#     """
#     # Cache projection modules on the autoencoder object.
#     if not hasattr(autoencoder, "random_projections"):
#         autoencoder.random_projections = nn.ModuleDict()

#     weighted_loss_sum = 0.0  # Accumulates each key's weighted (normalized) loss.
#     total_features = 0  # Total number of features across all keys.
#     ae_input_dim = autoencoder.encoder[0].in_features  # Target projection dimension.
#     epsilon = 1e-8  # Safeguard to prevent division by zero.

#     # Iterate over all activations in the dictionary.
#     for key, x in act_dict.items():
#         batch_size, act_dim = x.shape
#         total_features += act_dim  # Count features for weighting later.
#         key_str = str(key)

#         # Create a projection module (a linear layer with orthogonal weight)
#         # if not already present.
#         if key_str not in autoencoder.random_projections:
#             # Create a linear layer mapping from d -> D.
#             # Note: The weight of this layer will be of shape [D, d].
#             linear_layer = nn.Linear(act_dim, ae_input_dim, bias=False, device=x.device)

#             # Initialize the weight exactly as you did before.
#             with torch.no_grad():
#                 if key == "0":
#                     init_weight = torch.eye(ae_input_dim, device=x.device)
#                 else:
#                     init_weight = torch.randn(ae_input_dim, act_dim, device=x.device) / math.sqrt(
#                         ae_input_dim
#                     )
#                 linear_layer.weight.copy_(init_weight)

#             # Register the orthogonal parameterization on the weight parameter.
#             # This replaces the original weight with a parametrized version that is always orthogonal.
#             parameterize.register_parametrization(
#                 linear_layer, "weight", _Orthogonal(linear_layer.weight, "householder")
#             )
#             autoencoder.random_projections[key_str] = linear_layer
#             optimizer.add_param_group(
#                 {"params": autoencoder.random_projections[key_str].parameters()}
#             )

#         # Get the projection module.
#         proj = autoencoder.random_projections[key_str]

#         # Extract the weight Q, which is [D, d] and orthogonal (i.e., Q^T Q ~ I).
#         Q = proj.weight

#         # Project the d-dimensional activation to the D-dimensional state space.
#         x_proj = x @ Q.T

#         # Reconstruct in D-dim space using the autoencoder (k=0 means no Koopman stepping!)
#         recon_proj = autoencoder(x_proj, k=0).reconstruction

#         # Invert the projection using the fact that Q has orthonormal columns.
#         x_recons = recon_proj @ Q

#         # Compute the squared error (MSE) for this key.
#         diff = x - x_recons
#         layer_loss = diff.pow(2).sum()

#         # Compute the total variance in the original space.
#         x_mean = x.mean(dim=0, keepdim=True)
#         layer_var = (x - x_mean).pow(2).sum()

#         # Normalize loss by variance (with epsilon safeguard).
#         normalized_loss = layer_loss / (layer_var + epsilon)

#         # Weight this key's loss by its number of features.
#         # weighted_loss_sum += act_dim * normalized_loss
#         weighted_loss_sum += normalized_loss

#     # Combine the weighted losses, normalizing by the total number of features.
#     # final_loss = weighted_loss_sum / total_features
#     # return final_loss
#     return weighted_loss_sum


# def compute_k_prediction_loss(act_dict, autoencoder, k):
#     """
#     Computes the k-step prediction loss in *state space* for the final activation.
#     Compares the predicted state at k steps with the actual final-layer activation.
#     """
#     # Extract activations from the dictionary
#     act_list = list(act_dict.values())

#     # We'll predict the final-layer activation from the first-layer activation
#     target_acts = act_list[-1]  # shape: [batch, neurons]
#     total_variance = (target_acts - target_acts.mean(dim=0)).pow(2).sum()

#     # Get autoencoder predictions: shape [layers, batch, neurons]
#     all_preds = autoencoder(x=act_list[0], k=k).predictions

#     # The final prediction is all_preds[-1], but we may slice to the correct size
#     pred_k = all_preds[-1, :, : target_acts.size(-1)]
#     state_space_pred_error = (pred_k - target_acts).pow(2).sum()

#     return state_space_pred_error / total_variance

# def compute_latent_space_prediction_loss(act_dict, autoencoder, k):
#     """
#     Computes the k-step prediction loss purely in the *latent space*.
#     We encode each layer’s activation into latent space, then compare
#     the predicted next-layer embedding (via the Koopman matrix)
#     with the actual final-layer embedding.
#     """
#     # Prepare padded activations (we don't need masks for latent space)
#     padded_acts, _ = _prepare_padded_acts_and_masks(act_dict, autoencoder)

#     # Encode each layer’s padded activation into latent space
#     latent_acts = [autoencoder._encode(padded_act) for padded_act in padded_acts]
#     latent_acts = torch.stack(latent_acts, dim=1)  # shape: [batch, layers, latent_dim]

#     # Koopman k-step: multiply the *first* layer’s latent by K^k
#     K = autoencoder.koopman_matrix.linear_layer.weight.T.detach()
#     embedded_act = latent_acts[:, 0, :] @ linalg.matrix_power(K, k)

#     # Compare predicted embedding with the actual *last-layer* embedding
#     latent_error = (embedded_act - latent_acts[:, -1, :]).pow(2).sum()

#     # Total variance in final layer’s latent
#     latent_last_centered_acts = latent_acts[:, -1, :] - latent_acts[:, -1, :].mean(dim=0)
#     total_variance_latent_space = latent_last_centered_acts.pow(2).sum()

#     return latent_error / total_variance_latent_space


############################### V4 ########################################

# class QRParametrization(nn.Module):
#     def forward(self, X):
#         # X is our unconstrained parameter of shape [D, d].
#         # We assume D >= d so that we can get an orthonormal matrix Q with Q^T Q = I.
#         Q, R = torch.linalg.qr(X, mode="reduced")
#         # Optionally, adjust the signs so that the diagonal of R is positive.
#         diag = torch.diagonal(R, dim1=-2, dim2=-1)
#         Q = Q * diag.sign().unsqueeze(0)
#         return Q


# def compute_state_space_recons_loss(act_dict, autoencoder, optimizer):
#     """
#     Computes reconstruction loss in state space via learnable projections.
#     Each layer's activations are projected to a fixed D-dim space, reconstructed,
#     and then inverted using an orthogonal projection achieved via a QR parameterization.

#     For each key:
#       - The MSE in the original space is computed and normalized by the total variance.
#       - This normalized loss is then weighted by the number of features in that layer.
#     """
#     # Cache projection modules on the autoencoder object.
#     if not hasattr(autoencoder, "random_projections"):
#         autoencoder.random_projections = nn.ModuleDict()

#     weighted_loss_sum = 0.0  # Accumulate each key's weighted (normalized) loss.
#     ae_input_dim = autoencoder.encoder[0].in_features  # Target projection dimension.
#     epsilon = 1e-8  # Safeguard to prevent division by zero.

#     # Iterate over all activations in the dictionary.
#     for key, x in act_dict.items():
#         batch_size, act_dim = x.shape
#         key_str = str(key)

#         # Create a projection module (a linear layer with orthogonal weight)
#         # if not already present.
#         if key_str not in autoencoder.random_projections:
#             # Create a linear layer mapping from act_dim -> ae_input_dim.
#             # Weight shape will be [ae_input_dim, act_dim].
#             linear_layer = nn.Linear(act_dim, ae_input_dim, bias=False, device=x.device)

#             with torch.no_grad():
#                 # For key "0", if dimensions match, initialize to identity.
#                 # if key_str == "0":
#                 #     init_weight = torch.eye(ae_input_dim, device=x.device)
#                 # else:
#                 init_weight = torch.empty(ae_input_dim, act_dim, device=x.device)
#                 nn.init.orthogonal_(init_weight)
#                 linear_layer.weight.copy_(init_weight)

#             # Register our QR-based orthogonal parameterization.
#             parametrize.register_parametrization(linear_layer, "weight", QRParametrization())
#             autoencoder.random_projections[key_str] = linear_layer
#             optimizer.add_param_group(
#                 {"params": autoencoder.random_projections[key_str].parameters()}
#             )

#         # Get the projection module.
#         proj = autoencoder.random_projections[key_str]

#         # Extract the orthogonal weight Q (shape [ae_input_dim, act_dim]).
#         Q = proj.weight

#         # Project the activation x (shape [batch, act_dim]) to the D-dimensional state space.
#         x_proj = x @ Q.T

#         # Reconstruct in D-dim space using the autoencoder (k=0 means no Koopman stepping!)
#         recon_proj = autoencoder(x_proj, k=0).reconstruction

#         # Invert the projection using the fact that Q has orthonormal columns.
#         x_recons = recon_proj @ Q

#         # Compute the squared error (MSE) for this key.
#         diff = x - x_recons
#         layer_loss = diff.pow(2).sum()

#         # Compute the total variance in the original space.
#         x_mean = x.mean(dim=0, keepdim=True)
#         layer_var = (x - x_mean).pow(2).sum()

#         # Normalize loss by variance (with epsilon safeguard).
#         normalized_loss = layer_loss / (layer_var + epsilon)

#         # Weight this key's loss by its number of features.
#         weighted_loss_sum += normalized_loss

#     return weighted_loss_sum
