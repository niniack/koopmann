import torch.nn.functional as F


def compute_state_recons_loss(autoencoder, x_i, x_j):
    x_i_recon = autoencoder.decode(autoencoder.encode(x_i))
    x_j_recon = autoencoder.decode(autoencoder.encode(x_j))

    loss_i = F.mse_loss(x_i_recon, x_i)
    loss_j = F.mse_loss(x_j_recon, x_j)

    return 0.5 * (loss_i + loss_j)


def compute_latent_prediction_loss(autoencoder, x_i, x_j):
    z_j_pred = autoencoder.koopman_forward(autoencoder.encode(x_i))
    z_j = autoencoder.encode(x_j)

    return F.mse_loss(z_j_pred, z_j)


def compute_state_prediction_loss(autoencoder, x_i, x_j, k_steps=None):
    phi_i = autoencoder.encode(x_i)
    x_j_pred_latent = autoencoder.koopman_forward(phi_i, k_steps=k_steps)
    x_j_pred = autoencoder.decode(x_j_pred_latent)

    return F.mse_loss(x_j_pred, x_j)
