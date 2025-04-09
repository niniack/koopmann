import pdb

import torch
import torch.nn.functional as F
from torch import linalg


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
    padded_acts, masks = prepare_padded_acts_and_masks(act_dict, autoencoder)

    # Stack along layers dimension
    # shape: [batch, layers, neurons]
    padded_acts = torch.stack(padded_acts, dim=1)

    # shape: [1, layers, neurons]
    masks = torch.stack(masks, dim=0).unsqueeze(dim=0)

    # Reconstruct each layer’s padded activation
    # k=0 means no Koopman stepping
    recons_acts = [
        autoencoder(padded_act, k=0).reconstruction for padded_act in padded_acts.unbind(dim=1)
    ]

    # Stack reconstructed acts
    # shape: [batch, layers, neurons]
    recons_acts = torch.stack(recons_acts, dim=1)

    # Difference in state space, ignoring masked-out neurons
    # shape: [batch, layers, neurons]
    masked_diff = (padded_acts - recons_acts) * masks

    # MSE across batch and neuron dimensions
    # shape: [layers]
    recons_error = masked_diff.pow(2).mean(dim=[0, 2])

    # Total variance in state space
    # shape: [batch, layers, neurons]
    masked_centered_acts = (padded_acts - padded_acts.mean(dim=[0], keepdim=True)) * masks

    # shape: [layers]
    total_variance_state_space = masked_centered_acts.pow(2).mean(dim=[0, 2])

    return recons_error.mean(), (recons_error / total_variance_state_space).mean()


def _probing_recons_loss(act_dict, autoencoder) -> tuple:
    raise NotImplementedError()

    # # Activations
    # acts = torch.stack(list(act_dict.values()), dim=1)  # [batch, layers, neurons]
    # batch_size, num_layers, num_neurons = acts.shape

    # # Reconstruct each layer
    # recons = torch.stack(
    #     [autoencoder(acts[:, i, :], k=0).reconstruction for i in range(num_layers)], dim=1
    # )
    # recons_error = (recons - acts).pow(2).mean(dim=0)  # [layers, neurons]

    # # Variance per layer
    # total_variance = (acts - acts.mean(dim=0)).pow(2).mean(dim=0)  # [layers, neurons]
    # total_variance += 1e-8

    # # Average FVU across latent dimension
    # avg_fvu = (recons_error / total_variance).mean(dim=1).sum()

    # # Average MSE across latent dimension
    # avg_mse = recons_error.mean(dim=1).sum()

    # return avg_mse, avg_fvu


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
    latent_acts = [autoencoder.encode(padded_act) for padded_act in padded_acts]

    # shape: [batch, layers, latent]
    latent_acts = torch.stack(latent_acts, dim=1)

    # NOTE: Is this right? I think we should be detaching these to prevent a complicated
    # computational graph.
    # latent_acts = latent_acts.detach()

    # Koopman step: multiply the *first* layer’s latent by K^k
    K_matrix = autoencoder.koopman_weights.T

    # shape: [batch, neurons]
    embedded_act = latent_acts[:, 0, :] @ linalg.matrix_power(K_matrix, autoencoder.k_steps)

    # Square the difference, average across all dimensions
    # shape: [1]
    latent_error = (embedded_act - latent_acts[:, -1, :]).pow(2).mean(dim=[0, 1])

    # Total variance in final layer’s latent
    latent_last_centered_acts = latent_acts[:, -1, :] - latent_acts[:, -1, :].mean(dim=0)

    # shape: [1]
    total_variance_latent_space = latent_last_centered_acts.pow(2).mean(dim=[0, 1])

    return latent_error, latent_error / total_variance_latent_space


########################## K-Step Prediction Loss (State Space) ##########################
def compute_k_prediction_loss(act_dict, autoencoder, k, probed) -> tuple:
    """
    Computes the k-step prediction loss in *state space* for the final activation.
    Compares the predicted state at k steps with the actual final-layer activation.
    """
    return _k_prediction_loss(act_dict, autoencoder, k)


def _k_prediction_loss(act_dict, autoencoder, k) -> tuple:
    padded_acts, _ = prepare_padded_acts_and_masks(act_dict, autoencoder)

    # shape: [batch, neurons]
    starting_acts = padded_acts[0]
    target_acts = padded_acts[-1]

    # Get autoencoder predictions
    # shape [layers, batch, neurons]
    all_preds = autoencoder(x=starting_acts, k=k).predictions

    # The final prediction is stored at all_preds[-1].
    pred_k = all_preds[-1, :, : target_acts.size(-1)]

    # Square the difference and average across all dimensions
    # shape: [1]
    state_space_pred_error = (pred_k - target_acts).pow(2).mean(dim=[0, 1])

    # Total variance
    # shape: [1]
    total_variance = (target_acts - target_acts.mean(dim=0)).pow(2).mean(dim=[0, 1])

    return state_space_pred_error, state_space_pred_error / total_variance


########################## Isometric Loss ##########################
def compute_isometric_loss(act_dict, autoencoder, k, probed) -> tuple:
    padded_acts, _ = prepare_padded_acts_and_masks(act_dict, autoencoder)
    starting_acts = padded_acts[0]  # shape: [batch, neurons]
    target_acts = padded_acts[-1]  # shape: [batch, neurons]

    # Encode both starting and target states
    encoded_starting = autoencoder.components.encoder(starting_acts)
    encoded_target = autoencoder.components.encoder(target_acts)

    # Pairwise distances for starting
    start_dists = torch.cdist(starting_acts, starting_acts)
    enc_start_dists = torch.cdist(encoded_starting, encoded_starting)

    # Pairwise distances for target states
    target_dists = torch.cdist(target_acts, target_acts)
    enc_target_dists = torch.cdist(encoded_target, encoded_target)

    # Isometric errors
    start_iso_error = (enc_start_dists - start_dists).pow(2).mean()
    target_iso_error = (enc_target_dists - target_dists).pow(2).mean()
    iso_error = start_iso_error + target_iso_error

    # Variance for normalization
    start_variance = start_dists.var()
    target_variance = target_dists.var()

    # FVU (Fraction of Variance Unexplained) for each component
    start_fvu = start_iso_error / (start_variance + 1e-8)
    target_fvu = target_iso_error / (target_variance + 1e-8)

    # Average FVU
    iso_fvu = (start_fvu + target_fvu) / 2.0

    return iso_error, iso_fvu.mean()


########################## Shaping Loss ##########################
def compute_eigenvector_shaping_loss(act_dict, autoencoder, labels) -> torch.Tensor:
    act_list = list(act_dict.values())
    initial_acts = act_list[0]
    final_acts = act_list[-1]

    # Get class representatives
    unique_labels = torch.unique(labels)
    first_indices = torch.tensor([torch.where(labels == lbl)[0][0].item() for lbl in unique_labels])

    # Encode and normalize
    target_right_directions = autoencoder.encode(initial_acts[first_indices]).detach()
    target_right_directions = target_right_directions / torch.norm(
        target_right_directions, dim=1, keepdim=True
    )

    target_left_directions = autoencoder.encode(final_acts[first_indices]).detach()
    target_left_directions = target_left_directions / torch.norm(
        target_left_directions, dim=1, keepdim=True
    )

    # Koopman matrix
    K_matrix = autoencoder.koopman_weights.T

    # Calculate cosine similarity between t^T K and t^T
    loss = 0.0
    for right, left in zip(target_right_directions, target_left_directions):
        # Apply Koopman matrix: t^T @ K
        left_transformed = left @ K_matrix

        # K @ t
        right_transformed = K_matrix @ right

        # Normalize transformed vector for cosine similarity
        left_transformed_norm = left_transformed / (torch.norm(left_transformed) + 1e-8)
        right_transformed_norm = right_transformed / (torch.norm(right_transformed) + 1e-8)

        # Loss: 1 - cos_sim (perfect alignment gives 0 loss)
        left_residual = 1 - torch.abs(torch.dot(left_transformed_norm, left))
        right_residual = 1 - torch.abs(torch.dot(right_transformed_norm, right))

        loss += left_residual
    return loss


########################## Replacement ##########################
def calculate_replacement_error(autoencoder, model, first_value, last_index, label, k_steps):
    """Calculate replacement error."""
    pred_act = autoencoder(first_value, k=k_steps).predictions[-1]
    output = model.components[last_index + 1 :](pred_act)
    return F.cross_entropy(output, label.long())


########################## Padding ##########################
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
    ae_input_size = autoencoder.components.encoder[0].in_channels
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
