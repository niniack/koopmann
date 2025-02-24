import pdb

import torch
from torch import linalg

from scripts.common import prepare_padded_acts_and_masks


########################## K-Step Loss (Latent Space) ##########################
def compute_latent_space_prediction_loss(act_dict, autoencoder, k, probed) -> tuple:
    """
    Computes the k-step prediction loss purely in the *latent space*.
    We encode each layer’s activation into latent space, then compare
    the predicted next-layer embedding (via the Koopman matrix)
    with the actual final-layer embedding.
    """
    return _latent_prediction_loss(act_dict, autoencoder, k)


def _latent_prediction_loss(act_dict, autoencoder, k) -> tuple:
    # Prepare padded activations (we don't need masks for latent space)
    padded_acts, _ = prepare_padded_acts_and_masks(act_dict, autoencoder)

    # Encode each layer’s padded activation into latent space
    latent_acts = [autoencoder._encode(padded_act) for padded_act in padded_acts]
    latent_acts = torch.stack(latent_acts, dim=1)  # shape: [batch, layers, latent]

    # NOTE: Is this right? I think we should be detaching these to prevent a complicated
    # computational graph.
    latent_acts = latent_acts.detach()

    # Koopman step: multiply the *first* layer’s latent by K^k
    # NOTE: this K used to be transposed.
    K = autoencoder.koopman_matrix.linear_layer.weight.T
    embedded_act = latent_acts[:, 0, :] @ linalg.matrix_power(K, k)

    # Compare predicted embedding with the *last-layer* embedding
    # Square the difference, average across batch
    latent_error = (embedded_act - latent_acts[:, -1, :]).pow(2).mean(dim=0)  # shape: [latent]

    # Total variance in final layer’s latent
    latent_last_centered_acts = latent_acts[:, -1, :] - latent_acts[:, -1, :].mean(dim=0)
    total_variance_latent_space = latent_last_centered_acts.pow(2).mean(dim=0)  # shape: [latent]

    # Average FVU across latent dimension
    avg_fvu = (latent_error / total_variance_latent_space).mean()

    # Average MSE across latent dimension
    avg_mse = latent_error.mean()

    return avg_mse, avg_fvu


########################## K-Step Prediction Loss (State Space) ##########################
def compute_k_prediction_loss(act_dict, autoencoder, k, probed) -> tuple:
    """
    Computes the k-step prediction loss in *state space* for the final activation.
    Compares the predicted state at k steps with the actual final-layer activation.
    """
    return _k_prediction_loss(act_dict, autoencoder, k)


def _k_prediction_loss(act_dict, autoencoder, k) -> tuple:
    # Extract activations from the dictionary
    act_list = list(act_dict.values())

    starting_acts = act_list[0]  # shape: [batch, neurons]
    target_acts = act_list[-1]  # shape: [batch, neurons]

    # Get autoencoder predictions: shape [layers, batch, neurons]
    all_preds = autoencoder(x=starting_acts, k=k).predictions

    # The final prediction is stored at all_preds[-1].
    pred_k = all_preds[-1, :, : target_acts.size(-1)]
    state_space_pred_error = (pred_k - target_acts).pow(2).mean(dim=0)  # shape: [neurons]

    # Total variance
    total_variance = (target_acts - target_acts.mean(dim=0)).pow(2).mean(dim=0)  # shape: [neurons]

    # Average FVU across neuron dimension
    avg_fvu = (state_space_pred_error / total_variance).mean()

    # Average MSE across neuron dimension
    avg_mse = state_space_pred_error.mean()

    return avg_mse, avg_fvu


#################################### Reconstruction Loss ####################################
def compute_state_space_recons_loss(act_dict, autoencoder, k, probed) -> tuple:
    if probed:
        return _probing_recons_loss(act_dict, autoencoder)
    else:
        return _padding_recons_loss(act_dict, autoencoder)


def _padding_recons_loss(act_dict, autoencoder) -> tuple:
    """
    Computes the reconstruction loss in the *state space* across all layers.
    Returns reconstruction MSE scaled by total variance (aross all layers).
    """
    # Prepare padded activations and masks
    # shape: [batch, layer, norms]
    padded_acts, masks = prepare_padded_acts_and_masks(act_dict, autoencoder)

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
    return recons_error, recons_error / total_variance_state_space


def _probing_recons_loss(act_dict, autoencoder) -> tuple:
    # Activations
    acts = torch.stack(list(act_dict.values()), dim=1)  # [batch, layers, neurons]
    batch_size, num_layers, num_neurons = acts.shape

    # Reconstruct each layer
    recons = torch.stack(
        [autoencoder(acts[:, i, :], k=0).reconstruction for i in range(num_layers)], dim=1
    )
    recons_error = (recons - acts).pow(2).mean(dim=0)  # [layers, neurons]

    # Variance per layer
    total_variance = (acts - acts.mean(dim=0)).pow(2).mean(dim=0)  # [layers, neurons]
    total_variance += 1e-8

    # Average FVU across latent dimension
    avg_fvu = (recons_error / total_variance).mean(dim=1).sum()

    # Average MSE across latent dimension
    avg_mse = recons_error.mean(dim=1).sum()

    return avg_mse, avg_fvu


def compute_sparsity_loss(act_dict, autoencoder, k, probed) -> tuple:
    # Prepare padded activations (we don't need masks for latent space)
    padded_acts, _ = prepare_padded_acts_and_masks(act_dict, autoencoder)

    # Encode each layer's padded activation into latent space
    latent_acts = [autoencoder._encode(padded_act) for padded_act in padded_acts]
    latent_acts = torch.stack(latent_acts, dim=1)  # shape: [batch, layers, latent]

    # Average across batch to get batch-independent statistic
    # Then sum across latent dimension to get total feature usage per layer
    l1_per_layer = torch.abs(latent_acts).mean(dim=0).sum(dim=1)  # shape: [layers]

    # Sum across layers for total sparsity
    total_l1 = l1_per_layer.sum()

    return total_l1, total_l1
