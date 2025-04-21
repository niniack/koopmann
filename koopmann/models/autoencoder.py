__all__ = [
    "Autoencoder",
    "KoopmanAutoencoder",
    "ExponentialKoopmanAutencoder",
    "LowRankKoopmanAutoencoder",
    "FrankensteinKoopmanAutoencoder",
]

import warnings
from collections import namedtuple
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer, LoRALinearLayer
from koopmann.models.utils import eigeninit

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
        nonlinearity: str = "leaky_relu",
        **kwargs,
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
            # NOTE: spectral normalization
            # spectral_norm(encoder_layer.components.linear)
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
            # NOTE: spectral normalization
            # spectral_norm(decoder_layer.components.linear)
            decoder.add_module(f"decoder_{i}", decoder_layer)

        return decoder

    def encode(self, x):
        x = self.components.encoder(x)
        return x

    def decode(self, x):
        x = self.components.decoder(x)
        return x

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


######################################################
class KoopmanAutoencoder(Autoencoder):
    """
    Koopman autoencoder model.
    """

    def __init__(
        self,
        k_steps: int,
        in_features: int = 2,
        latent_features: int = 4,
        hidden_config: Optional[list[int]] = None,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: str = "leaky_relu",
        use_eigeninit: Optional[bool] = False,
        **kwargs,
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

    def forward(self, x: float, intermediate=False, k=None):
        phi_x = self.encode(x)
        x_recons = self.decode(phi_x)

        if k is None:
            k = self.k_steps

        # Stores all intermediate predictions
        if intermediate:
            # Advance latent variable k times
            prediction = [phi_x]
            for i in range(1, k + 1):
                prev_pred = prediction[i - 1]
                prediction.append(self.components.koopman_matrix(prev_pred))

            # Batched decoding
            # Shape: [steps, batch, latent_dim]
            stacked_predictions = torch.stack(prediction, dim=0)
            steps, batch_size, latent_dim = stacked_predictions.size()
            # Shape: [steps * batch, feature_dim]
            reshaped_predictions = stacked_predictions.view(-1, latent_dim)
            decoded = self.components.decoder(reshaped_predictions)
            # Shape: [steps, batch, latent_dim]
            x_k = decoded.view(steps, batch_size, -1)

        # Faster way, but no intermediate stores
        else:
            if k == 1:
                x_k = self.components.koopman_matrix(phi_x)
            else:
                K_effective_weight = torch.linalg.matrix_power(self.koopman_weights, k)
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
        metadata = super()._get_basic_metadata()
        metadata.update(
            {
                "k_steps": self.k_steps,
            }
        )

        return metadata


class FrankensteinKoopmanAutoencoder(nn.Module):
    def __init__(self, koopman_autoencoder, original_model, scale_idx=0):
        super().__init__()

        # Settings
        self.scale_idx = scale_idx
        self.rank = koopman_autoencoder.rank
        self.k_steps = koopman_autoencoder.k_steps
        self.latent_features = koopman_autoencoder.latent_features

        # Components
        self.start_components = nn.Sequential(
            original_model.components[: self.scale_idx],
        )
        self.koopman_autoencoder = koopman_autoencoder
        self.end_components = nn.Sequential(original_model.components[-1:])

    def forward(self, x) -> torch.Tensor:
        x = self.start_components(x)
        x = self.koopman_autoencoder(x, k=self.k_steps).predictions[-1]
        x = self.end_components(x)
        return x


######################################################
class LowRankKoopmanAutoencoder(KoopmanAutoencoder):
    """
    Koopman autoencoder model with low rank parameterization.
    """

    def __init__(
        self,
        rank: int,
        k_steps: int,
        in_features: int = 2,
        latent_features: int = 4,
        hidden_config: Optional[list[int]] = None,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: str = "leaky_relu",
        use_eigeninit: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            k_steps,
            in_features,
            latent_features,
            hidden_config,
            bias,
            batchnorm,
            nonlinearity,
            use_eigeninit,
        )
        self.rank = rank

        koopman_matrix = LoRALinearLayer(
            in_channels=latent_features,
            out_channels=latent_features,
            rank=rank,
            bias=False,
            batchnorm=False,
            nonlinearity=None,
        )

        self.components.koopman_matrix = koopman_matrix

    @property
    def koopman_weights(self):
        return (
            self.components.koopman_matrix.components.lora_up.weight
            @ self.components.koopman_matrix.components.lora_down.weight
        )

    # NOTE: this makes it symmetric
    # def forward(self, x, k):
    #     # Update lora_up weight to be transpose of lora_down before computation
    #     with torch.no_grad():
    #         self.components.koopman_matrix.components.lora_up.weight.copy_(
    #             self.components.koopman_matrix.components.lora_down.weight.t()
    #         )

    #     # Proceed with the normal forward pass
    #     return super().forward(x)

    def _get_basic_metadata(self) -> dict[str, Any]:
        """Get model-specific metadata for serialization."""
        metadata = super()._get_basic_metadata()
        metadata.update(
            {
                "rank": self.rank,
            }
        )

        return metadata


######################################################
class ExponentialKoopmanAutencoder(KoopmanAutoencoder):
    """
    Koopman autoencoder model with exp parameterization.
    """

    def __init__(
        self,
        k_steps: int,
        in_features: int = 2,
        latent_features: int = 4,
        hidden_config: Optional[list[int]] = None,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: str = "leaky_relu",
        use_eigeninit: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            k_steps,
            in_features,
            latent_features,
            hidden_config,
            bias,
            batchnorm,
            nonlinearity,
            use_eigeninit,
        )

        parametrize.register_parametrization(
            self.components.koopman_matrix.components.linear,
            "weight",
            MatrixExponential(
                k_steps=k_steps,
                latent_features=latent_features,
            ),
        )


class MatrixExponential(nn.Module):
    def __init__(self, k_steps, latent_features):
        super().__init__()
        self.k_steps = k_steps  # Number of steps
        self.latent_features = latent_features

    def forward(self, X):
        return torch.matrix_exp(X / self.k_steps)  # Scale M by 1/k


class LowRankFactorization(nn.Module):
    def __init__(self, n: int, r: int):
        super().__init__()

        self.n = n  # dimension of square matrix
        self.r = r  # target rank

    def forward(self, X):
        # Use the first n*r elements of X for left factor and the rest for right factor
        left = X[:, : self.r]  # Shape: n × r
        right = X[:, self.r :].t()  # Shape: r × n  (after transpose)

        return left @ right

    def right_inverse(self, X):
        # When someone assigns a matrix to the weight, decompose it via SVD
        U, S, Vh = torch.linalg.svd(X)

        left = U[:, : self.r] @ torch.diag(torch.sqrt(S[: self.r]))
        right = torch.diag(torch.sqrt(S[: self.r])) @ Vh[: self.r, :]

        # Return in the format our forward method expects
        return torch.cat([left, right.t()], dim=1)
