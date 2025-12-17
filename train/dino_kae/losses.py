import torch.nn.functional as F


def compute_state_recons_loss(autoencoder, x_i, x_j):
    x_i_recon = autoencoder.decode(autoencoder.encode(x_i))
    x_j_recon = autoencoder.decode(autoencoder.encode(x_j))

    loss_i = F.mse_loss(x_i_recon, x_i)
    loss_j = F.mse_loss(x_j_recon, x_j)

    return loss_i + loss_j


def compute_latent_prediction_loss(autoencoder, x_i, x_j):
    z_j_pred = autoencoder.koopman_forward(autoencoder.encode(x_i))
    z_j = autoencoder.encode(x_j)
    loss = F.mse_loss(z_j_pred, z_j)

    return loss


def compute_state_prediction_loss(autoencoder, x_i, x_j):
    x_j_pred, _ = autoencoder.forward(x_i)
    x_j_pred = x_j_pred.squeeze()
    loss = F.mse_loss(x_j_pred, x_j)

    return loss
