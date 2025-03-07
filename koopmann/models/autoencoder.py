__all__ = [
    "Autoencoder",
    "KoopmanAutoencoder",
    # "ExponentialKoopmanAutencoder",
    # "LowRankKoopmanAutoencoder",
]

import warnings
from collections import namedtuple
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer

VanillaAutoencoderResult = namedtuple("VanillaAutoencoderResult", "latent reconstruction")
KoopmanAutoencoderResult = namedtuple("KoopmanAutoencoderResult", "predictions reconstruction")


class Autoencoder(BaseTorchModel):
    """
    Autoencoder model.
    """

    def __init__(
        self,
        in_features: int = 2,
        latent_features: int = 4,
        hidden_config: Optional[list[int]] = None,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: str = "leakyrelu",
    ):
        super().__init__()

        self.in_features = in_features
        self.latent_features = latent_features
        self.hidden_config = hidden_config
        self.bias = bias
        self.batchnorm = batchnorm
        self.nonlinearity = nonlinearity

        # Warning
        if latent_features <= in_features:
            warnings.warn(
                f"The latent dimension {latent_features} should probably be "
                f"larger than the input dimension {in_features}!"
            )

        # Set up autoencoder architecture
        if not hidden_config:
            channel_dims = [
                (in_features, latent_features),
            ]
        else:
            dims_list = [in_features, latent_features]
            dims_list = dims_list[:1] + hidden_config + dims_list[1:]
            channel_dims = [(dims_list[i - 1], dims_list[i]) for i in range(1, len(dims_list))]

        # Build components
        encoder = self._build_encoder(channel_dims)
        self.components.add_module("encoder", encoder)

        decoder = self._build_decoder(channel_dims)
        self.components.add_module("decoder", decoder)

    def _build_encoder(self, channel_dims) -> nn.Sequential:
        """Returns the encoder in a sequential container."""

        encoder = nn.Sequential()
        for i in range(0, len(channel_dims), 1):
            encoder_layer = LinearLayer(
                in_channels=channel_dims[i][0],
                out_channels=channel_dims[i][1],
                bias=self.bias,
                batchnorm=self.batchnorm,
                nonlinearity=self.nonlinearity,
            )

            encoder_layer.apply(LinearLayer.init_weights)
            encoder.add_module(f"encoder_{i}", encoder_layer)

        return encoder

    def _build_decoder(self, channel_dims) -> nn.Sequential:
        """Returns the decoder in a sequential container."""

        decoder = nn.Sequential()
        for i in range(len(channel_dims) - 1, -1, -1):
            decoder_layer = LinearLayer(
                in_channels=channel_dims[i][1],
                out_channels=channel_dims[i][0],
                bias=self.bias,
                batchnorm=self.batchnorm,
                nonlinearity=self.nonlinearity if (i != 0) else None,
            )

            decoder_layer.apply(LinearLayer.init_weights)
            decoder.add_module(f"decoder_{i}", decoder_layer)

        return decoder

    def encode(self, x):
        ## Pre-encoder bias
        # x_bar = x - self.decoder[-1].linear_layer.bias
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x: float):
        phi_x = self.encode(x)
        x_recons = self.decode(phi_x)
        return VanillaAutoencoderResult(phi_x, x_recons)

    def _get_basic_metadata(self) -> dict[str, Any]:
        """Get model-specific metadata for serialization."""
        return {
            "in_features": self.in_features,
            "latent_features": self.latent_features,
            "hidden_config": self.hidden_config,
            "bias": self.bias,
            "batchnorm": self.batchnorm,
            "nonlinearity": self.nonlinearity,
        }


class KoopmanAutoencoder(Autoencoder):
    def __init__(
        self,
        k_steps: int,
        in_features: int = 2,
        latent_features: int = 4,
        hidden_config: Optional[list[int]] = None,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: str = "leakyrelu",
        use_eigeninit: Optional[bool] = False,
    ):
        super().__init__(
            in_features=in_features,
            latent_features=latent_features,
            hidden_config=hidden_config,
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
        )

        self.k_steps = k_steps
        koopman_matrix = LinearLayer(
            in_channels=latent_features,
            out_channels=latent_features,
            bias=False,
            batchnorm=False,
            nonlinearity=None,
        )
        if use_eigeninit:
            eigeninit(koopman_matrix.components.linear.weight, theta=0.3)

        # Rebuild container
        # NOTE: Pytorch doesn't have a great way to insert into nn.Sequential
        temp_container = nn.Sequential()
        temp_container.add_module("encoder", self.components.encoder)
        temp_container.add_module("koopman_matrix", koopman_matrix)
        temp_container.add_module("decoder", self.components.decoder)
        self.components = temp_container

    def forward(self, x: float, intermediate=False):
        phi_x = self.encode(x)
        x_recons = self.decode(phi_x)

        # Stores all intermediate predictions
        if intermediate:
            # Advance latent variable k times
            prediction = [phi_x]
            for i in range(1, self.k_steps + 1):
                prev_pred = prediction[i - 1]
                prediction.append(self.components.koopman_matrix(prev_pred))

            # Batched decoding
            # Shape: [steps, batch, latent_dim]
            stacked_predictions = torch.stack(prediction, dim=0)
            steps, batch_size, latent_dim = stacked_predictions.size()
            # Shape: [steps * batch, feature_dim]
            reshaped_predictions = stacked_predictions.view(-1, latent_dim)
            decoded = self.decoder(reshaped_predictions)
            # Shape: [steps, batch, latent_dim]
            x_k = decoded.view(steps, batch_size, -1)

        # Faster way, but no intermediate stores
        else:
            K_weight = self.koopman_matrix.linear_layer.weight
            K_effective_weight = torch.linalg.matrix_power(K_weight, self.k_steps)
            # NOTE: this K is transposed because of
            # how torch handles matrix multiplication!
            x_k = phi_x @ K_effective_weight.T
            x_k = self.decode(x_k)

            # For compatibility
            x_k = x_k.unsqueeze(0)

        return KoopmanAutoencoderResult(x_k, x_recons)

    @property
    def koopman_weights(self):
        return self.components.koopman_matrix.components.linear.weight

    def _get_basic_metadata(self) -> dict[str, Any]:
        """Get model-specific metadata for serialization."""
        return {
            "k_steps": self.k_steps,
            "in_features": self.in_features,
            "latent_features": self.latent_features,
            "hidden_config": self.hidden_config,
            "bias": self.bias,
            "batchnorm": self.batchnorm,
            "nonlinearity": self.nonlinearity,
        }


def eigeninit(weight: torch.Tensor, theta: float = 0.7) -> None:
    """
    Initialization for Koopman matrix weights.

    The magnitudes of the eigenvalues are set to be between 0 and 1,
    with theta determining the probability of 1.
    Directly modifies the input tensor `weight` in-place.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(weight)

    # Represent eigenvalues in polar coordinates
    polar_mags = torch.abs(eigenvalues)
    polar_phase = torch.angle(eigenvalues)

    # Sample with slab-spike distribution
    num_unique = len(torch.unique(polar_mags, sorted=False))
    bernoulli_trials = torch.distributions.Bernoulli(theta).sample([num_unique])
    uniform_trials = torch.distributions.Uniform(0, 1).sample([num_unique]) * (1 - bernoulli_trials)
    result_trials = bernoulli_trials + uniform_trials

    # Sample new magnitudes, while preserving conjugate pairs!
    new_polar_mags = torch.empty_like(polar_mags)
    new_polar_mags[0] = result_trials[0]
    j = 1
    for i in range(1, new_polar_mags.size(0)):
        if torch.isclose(polar_mags[i], polar_mags[i - 1]):
            new_polar_mags[i] = new_polar_mags[i - 1]
        else:
            new_polar_mags[i] = result_trials[j]
            j += 1

    # Rebuild eigenvalues with new magnitudes
    new_eigenvalues = torch.polar(new_polar_mags, polar_phase)

    # Construct new weight matrix in-place
    with torch.no_grad():  # Precaution
        weight.copy_(
            torch.real(eigenvectors @ torch.diag(new_eigenvalues) @ torch.linalg.inv(eigenvectors))
        )


# class MatrixExponential(nn.Module):
#     def __init__(self, k, dim):
#         super().__init__()
#         self.k = k  # Number of steps
#         self.dim = dim

#     def forward(self, X):
#         return torch.matrix_exp(X / self.k)  # Scale M by 1/k

#     # # Custom initialization function using matrix logarithm
#     # def right_inverse(self, X):
#     #     # Define a target matrix (e.g., an orthogonal matrix or one with specific eigenvalues)
#     #     dummy_layer = nn.Linear(in_features=self.dim, out_features=self.dim)
#     #     eigeninit(dummy_layer.weight, theta=1.0)
#     #     target_matrix = dummy_layer.weight

#     #     # Compute the matrix logarithm of the target matrix
#     #     log_matrix = self.k * logm(target_matrix)

#     #     return torch.Tensor(log_matrix)


# class ExponentialKoopmanAutencoder(Autoencoder):
#     """
#     Autoencoder model.
#     """

#     def __init__(
#         self,
#         rank: int,
#         k: int,
#         input_dimension: int = 2,
#         latent_dimension: int = 4,
#         hidden_configuration: Optional[List[Int]] = None,
#         nonlinearity: str = "leakyrelu",
#         batchnorm: bool = False,
#     ):
#         super().__init__(
#             k,
#             input_dimension,
#             latent_dimension,
#             hidden_configuration,
#             nonlinearity,
#             batchnorm,
#             rank,
#         )

#         parametrize.register_parametrization(
#             self.koopman_matrix.linear_layer, "weight", MatrixExponential(k=k, dim=latent_dimension)
#         )


# class LowRankFactorization(nn.Module):
#     """
#     Parameterizes a matrix to be of rank r by factoring it into two matrices.
#     Similar to how the Symmetric class ensures symmetry, this ensures low rank.
#     """

#     def __init__(self, n: int, r: int):
#         super().__init__()
#         self.n = n  # dimension of square matrix
#         self.r = r  # target rank

#     def forward(self, X):
#         # Use the first n*r elements of X for left factor and the rest for right factor
#         left = X[:, : self.r]  # Shape: n × r
#         right = X[:, self.r :].t()  # Shape: r × n  (after transpose)
#         return left @ right

#     def right_inverse(self, X):
#         # When someone assigns a matrix to the weight, decompose it via SVD
#         U, S, Vh = torch.linalg.svd(X)
#         left = U[:, : self.r] @ torch.diag(torch.sqrt(S[: self.r]))
#         right = torch.diag(torch.sqrt(S[: self.r])) @ Vh[: self.r, :]
#         # Return in the format our forward method expects
#         return torch.cat([left, right.t()], dim=1)


# class LowRankKoopmanAutoencoder(Autoencoder):
#     """
#     Autoencoder model.
#     """

#     def __init__(
#         self,
#         rank: int,
#         k: int,
#         input_dimension: int = 2,
#         latent_dimension: int = 4,
#         hidden_configuration: Optional[List[Int]] = None,
#         nonlinearity: str = "leakyrelu",
#         batchnorm: bool = False,
#     ):
#         super().__init__(
#             k,
#             input_dimension,
#             latent_dimension,
#             hidden_configuration,
#             nonlinearity,
#             batchnorm,
#             rank,
#         )

#         parametrize.register_parametrization(
#             self.koopman_matrix.linear_layer,
#             "weight",
#             LowRankFactorization(n=latent_dimension, r=rank),
#         )
