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
from deprecated import deprecated

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer, LoRALinearLayer
from koopmann.models.utils import eigeninit

VanillaAutoencoderResult = namedtuple("VanillaAutoencoderResult", "latent reconstruction")
KoopmanAutoencoderResult = namedtuple("KoopmanAutoencoderResult", "predictions reconstruction")


### STANDARD AUTOENCODER
class Autoencoder(BaseTorchModel):
    """Base autoencoder model."""

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
                nonlinearity=self.nonlinearity if (i != len(channel_dims) - 1) else None,
                # nonlinearity=self.nonlinearity,
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
        """Encodes state."""
        x = x.unsqueeze(1)
        x = self.components.encoder(x)
        return x

    def decode(self, x):
        """Decodes observable."""
        x = self.components.decoder(x)
        return x

    def forward(self, x: float):
        """Forward method for vanilla autoencoder."""
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


### KOOPMAN AUTOENCODER
class KoopmanAutoencoder(Autoencoder):
    """Standard Koopman autoencoder model."""

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
        koopman_matrix.components.linear.weight.data.copy_(torch.eye(latent_features))

        if use_eigeninit:
            eigeninit(koopman_matrix.components.linear.weight, theta=0.3)

        # Rebuild container
        # NOTE: Pytorch doesn't have a great way to insert into nn.Sequential
        temp_container = nn.Sequential()
        temp_container.add_module("encoder", self.components.encoder)
        temp_container.add_module("koopman_matrix", koopman_matrix)
        temp_container.add_module("decoder", self.components.decoder)
        self.components = temp_container

    @property
    def koopman_weights(self):
        return self.components.koopman_matrix.components.linear.weight

    def koopman_forward(self, observable, k_steps):
        # NOTE: this K is transposed because of
        # how torch handles matrix multiplication!
        return observable @ torch.linalg.matrix_power(self.koopman_weights.T, k_steps)

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
                new_pred = self.koopman_forward(prev_pred, k_steps=1)
                prediction.append(new_pred)

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
            x_k = self.koopman_forward(phi_x, k_steps=k)
            x_k = self.decode(x_k)

            # For compatibility
            x_k = x_k.unsqueeze(0)

        return KoopmanAutoencoderResult(x_k, x_recons)

    def _get_basic_metadata(self) -> dict[str, Any]:
        """Get model-specific metadata for serialization."""
        metadata = super()._get_basic_metadata()
        metadata.update({"k_steps": self.k_steps})

        return metadata


### EXPONENTIAL PARAM KOOPMAN AUTOENCODER
class ExponentialKoopmanAutencoder(KoopmanAutoencoder):
    """Koopman autoencoder model with exp parameterization."""

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

        # NOTE: Weird convention
        # K is (weirdly) parameterized as V^-1 @ D @ V
        # Following convention, we apply x @ K.T
        # K.T = V.T @ D @ V^-1.T
        # The benefit of this is that entering the eignespace does not require computing an inverse.
        self._V = nn.Linear(latent_features, latent_features, bias=False)
        nn.init.xavier_normal_(self._V.weight)
        self._D = nn.Parameter(torch.ones(latent_features))

        class Lambda(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x):
                return self.func(x)

        self.components.koopman_matrix = nn.Sequential(Lambda(lambda x: x @ self.koopman_weights.T))

    @property
    def V(self):
        return self._V.weight

    @property
    def V_inv(self):
        eye = torch.eye(self.V.shape[0], device=self.V.device)
        return torch.linalg.solve(self.V, eye)

    @property
    def D_exp(self):
        return torch.matrix_exp(torch.diag(self._D) / self.k_steps)

    @property
    def koopman_weights(self):
        return self.V_inv @ self.D_exp @ self.V

    def koopman_forward(self, observable, k_steps):
        return observable @ self.V.T @ torch.linalg.matrix_power(self.D_exp, k_steps) @ self.V_inv.T

    def koopman_eigenspace(self, observable):
        return observable @ self.V.T


@deprecated(
    reason="Previously used with torch parameterizations on a linear weight matrix. It's now directly done using an eigendecomposition."
)
class MatrixExponential(nn.Module):
    def __init__(self, k_steps, latent_features):
        super().__init__()
        self.k_steps = k_steps  # Number of steps
        self.latent_features = latent_features

    def forward(self, X):
        return torch.matrix_exp(X / self.k_steps)  # Scale M by 1/k


### LOW RANK PARAM KOOPMAN AUTOENCODER
class LowRankKoopmanAutoencoder(KoopmanAutoencoder):
    """Koopman autoencoder model with low rank parameterization."""

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

        self.components.koopman_matrix = LoRALinearLayer(
            in_channels=latent_features,
            out_channels=latent_features,
            rank=rank,
            bias=False,
            batchnorm=False,
            nonlinearity=None,
        )

    @property
    def koopman_weights(self):
        return (
            self.components.koopman_matrix.components.lora_up.weight
            @ self.components.koopman_matrix.components.lora_down.weight
        )

    def _get_basic_metadata(self) -> dict[str, Any]:
        """Get model-specific metadata for serialization."""
        metadata = super()._get_basic_metadata()
        metadata.update(
            {
                "rank": self.rank,
            }
        )

        return metadata


# NOTE: Irrelevant for now
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
