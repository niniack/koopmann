__all__ = [
    "Autoencoder",
]
import warnings
from collections import namedtuple
from typing import Any, Optional

import torch.nn as nn

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer

VanillaAutoencoderResult = namedtuple(
    "VanillaAutoencoderResult", "latent reconstruction"
)


### Standard Autoencoder
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
            channel_dims = [
                (dims_list[i - 1], dims_list[i]) for i in range(1, len(dims_list))
            ]

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
                batchnorm=self.batchnorm if (i != len(channel_dims) - 1) else None,
                nonlinearity=self.nonlinearity
                if (i != len(channel_dims) - 1)
                else None,
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
                batchnorm=self.batchnorm if (i != 0) else None,
                nonlinearity=self.nonlinearity if (i != 0) else None,
            )

            decoder_layer.apply(LinearLayer.init_weights)
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
