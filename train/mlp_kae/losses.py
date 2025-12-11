import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F

import wandb


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


############ METRICS ##################
# class AutoencoderMetrics:
#     def __init__(self, device):
#         """Initialize the metrics computation state."""

#         self.metric_to_method_dict = {
#             "reconstruction": compute_state_recons_loss,
#             "state_pred": compute_state_prediction_loss,
#             "latent_pred": compute_latent_prediction_loss,
#         }
#         self.device = device
#         self.reset()

#     def reset(self) -> "AutoencoderMetrics":
#         """Reset the internal state to initial values."""

#         self.batch_metrics = DotDict()
#         for name, _ in self.metric_to_method_dict.items():
#             self.batch_metrics[f"raw_{name}"] = torch.tensor(0.0, device=self.device)
#             self.batch_metrics[f"fvu_{name}"] = torch.tensor(0.0, device=self.device)

#         # Extras
#         self.batch_metrics["combined_loss"] = torch.tensor(0.0, device=self.device)

#         self.total_metrics = copy.deepcopy(self.batch_metrics)
#         self.num_batches = 0

#         return self

#     def update(self, autoencoder, act_dict, k_steps) -> OrderedDict:
#         """Compute losses."""
#         # Compute each core loss
#         for name, method in self.metric_to_method_dict.items():
#             raw, fvu = method(
#                 act_dict=act_dict,
#                 autoencoder=autoencoder,
#                 k=k_steps,
#             )

#             self.batch_metrics[f"raw_{name}"] = raw
#             self.batch_metrics[f"fvu_{name}"] = fvu

#         # # Non-core loss
#         # self.batch_metrics["shaping_loss"] = compute_eigenvector_shaping_loss(
#         #     act_dict, autoencoder, labels
#         # )

#         # Update totals
#         self.num_batches += 1
#         for key, value in self.batch_metrics.items():
#             self.total_metrics[key] += value.detach()

#     def log_metrics(self, epoch: int, prefix: str) -> None:
#         """Log training metrics to wandb."""

#         log_dict = {}
#         log_dict["epoch"] = epoch
#         for key, value in self.total_metrics.items():
#             log_dict[f"{prefix}/{key}"] = value / self.num_batches

#         wandb.log(log_dict, step=epoch)

#     def set_weighted_loss(self, loss):
#         self.total_metrics["combined_loss"] += loss.detach()
#         return self

#     def compute(self) -> DotDict:
#         if self.num_batches == 0:
#             self.avg_metrics = DotDict({k: 0.0 for k in self.total_metrics.keys()})

#         self.avg_metrics = DotDict(
#             {k: v.item() / self.num_batches for k, v in self.total_metrics.items()}
#         )

#         return self.avg_metrics
