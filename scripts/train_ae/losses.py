import pdb

import torch
import torch.nn.functional as F
from torch import linalg


def linear_warmup(
    epoch: int,
    final_value: float,
    end_epoch: int,
    start_epoch: int = 0,
) -> float:
    """
    Linearly scale from 0.0 to final_value across [start_epoch, end_epoch].
    - If epoch < start_epoch: return 0.0
    - If epoch >= end_epoch: return final_value
    - Else: scale linearly
    """
    if epoch < start_epoch:
        return 0.0
    elif epoch >= end_epoch:
        return final_value
    else:
        # Fraction of the way from start_epoch to end_epoch
        alpha = (epoch - start_epoch) / float(end_epoch - start_epoch)
        return alpha * final_value


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


########################## Eigenvalue Loss##########################
def compute_eigenloss(act_dict, autoencoder, k, probed) -> tuple:
    """
    Highly optimized eigenvalue loss function that leverages the low-rank
    parametrization of the Koopman matrix to compute eigenvalues efficiently.

    This implementation correctly accesses the parametrization to extract
    the matrix factors.
    """
    # Percentage of eigenvalues to penalize (top by magnitude)
    TOP_EIGENVALUE_PERCENT = 0.7

    # Get the Koopman matrix's linear layer
    koopman_layer = autoencoder.koopman_matrix.linear_layer

    # Get the dimension and rank
    n = koopman_layer.weight.shape[0]  # Matrix dimension
    r = autoencoder.rank  # Low rank value (assuming it's stored in the autoencoder)

    # Method 1: Access through parametrizations module
    try:
        # Starting from PyTorch 1.12+, this is the correct way to access the original tensor
        original = koopman_layer.parametrizations.weight.original
        left = original[:, :r]  # Shape: n × r
        right = original[:, r:].t()  # Shape: r × n (after transpose)
    except (AttributeError, IndexError):
        # Alternative approach: If the above fails, we can use a direct computation approach
        # Since we know the matrix is low-rank, we can use the regular weight
        # and perform SVD to recover equivalent factors

        # Get the full Koopman matrix
        koopman_weights = koopman_layer.weight

        # Perform SVD to get the factors
        U, S, Vh = torch.linalg.svd(koopman_weights, full_matrices=False)

        # Take only the top r singular values/vectors
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        # Construct equivalent left and right factors
        left = U_r @ torch.diag(torch.sqrt(S_r))
        right = torch.diag(torch.sqrt(S_r)) @ Vh_r

    # Compute the small matrix whose eigenvalues match the non-zero eigenvalues of K
    small_matrix = right @ left  # Shape: r × r

    # Compute eigenvalues of the small matrix
    eigenvalues = torch.linalg.eigvals(small_matrix)  # Shape: [r]

    # Sort by magnitude
    eigenvalues_abs = torch.abs(eigenvalues)

    # Select top percentage to penalize
    num_top = max(1, int(TOP_EIGENVALUE_PERCENT * r))
    _, top_indices = torch.topk(eigenvalues_abs, num_top)
    top_eigenvalues = eigenvalues[top_indices]

    # Calculate distance from target (1+0i)
    target = torch.ones(1, device=eigenvalues.device, dtype=torch.complex64)
    distances = torch.abs(top_eigenvalues - target)

    # Compute the loss
    static_eigen_loss = torch.mean(distances**2)

    return (static_eigen_loss, static_eigen_loss)


########################## Sparsity Loss##########################
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


def pad_act(x, target_size):
    current_size = x.size(1)
    if current_size < target_size:
        pad_size = target_size - current_size
        x = F.pad(x, (0, pad_size), mode="constant", value=0)

    return x


def prepare_padded_acts_and_masks(act_dict, autoencoder):
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
