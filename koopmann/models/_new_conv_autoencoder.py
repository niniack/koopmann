__all__ = [
    "ConvAutoencoder",
]

import warnings
from ast import literal_eval
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from jaxtyping import Float, Int
from safetensors.torch import load_model, save_model
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import Conv2DLayer, LinearLayer
from koopmann.models.utils import (
    StringtoClassNonlinearity,
    get_device,
    parse_safetensors_metadata,
)

AutoencoderResult = namedtuple("AutoencoderResult", "predictions reconstruction")


class ConvAutoencoder(BaseTorchModel):
    """
    Convolutional autoencoder model.
    """

    def __init__(
        self,
        k: int,
        input_channels: int = 2,
        input_height: int = 32,
        input_width: int = 32,
        latent_dimension: int = 4,
        hidden_configuration: Optional[List[Int]] = None,
        nonlinearity: str = "leakyrelu",
    ):
        super().__init__()

        # if latent_dimension <= input_dimension:
        #     warnings.warn(
        #         f"The latent dimension {latent_dimension} should probably be larger than the input dimension {input_dimension}!"
        #     )

        self.steps = k
        self.input_channels = input_channels
        self.latent_dimension = latent_dimension
        self.input_height = input_height
        self.input_width = input_width

        # Convert string nonlinearity to class
        self.nonlinearity = nonlinearity
        nonlinearity = StringtoClassNonlinearity[nonlinearity].value

        # Set up autoencoder architecture
        if not hidden_configuration:
            channel_dims = [
                (input_channels, input_channels * 4),
                # (input_channels * 4, input_channels * 16),
                # (input_channels * 4, input_channels * 8),
                # (input_channels * 8, input_channels * 16),
            ]
            self.hidden_configuration = channel_dims
        else:
            self.hidden_configuration = hidden_configuration
            raise NotImplementedError("Custom hidden configuration does not work yet!")

        ################## ENCODER #################
        self._encoder = nn.Sequential()
        current_height, current_width = input_height, input_width

        # Define kernel and stride parameters (commonly reused in this block)
        kernel_size = 3
        stride = 2
        padding = 1

        # Convolutional layers
        for i, (in_channels, out_channels) in enumerate(channel_dims):
            self._encoder.add_module(
                f"conv2d_{i}",
                Conv2DLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    nonlinearity=nonlinearity,
                    bias=True,
                    batchnorm=True,
                    hook=False,
                ),
            )
            self._encoder.apply(Conv2DLayer.init_weights)

            # Update spatial dimensions after each convolution
            current_height = (current_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            current_width = (current_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        # Calculate the flattened size of the output
        self.final_spatial_dims = (current_height, current_width)
        self.flattened_size = channel_dims[-1][1] * current_height * current_width

        # Validate dimensions
        if current_height <= 0 or current_width <= 0:
            raise ValueError(
                f"Invalid spatial dimensions after convolution: height={current_height}, width={current_width}. Check input dimensions and channel_dims."
            )

        # Linear layer for latent space
        self._encoder.add_module(
            f"linear_{i+1}",
            LinearLayer(
                in_features=self.flattened_size,
                out_features=latent_dimension,
                nonlinearity=nonlinearity,
                bias=True,
                batchnorm=False,
                hook=False,
            ),
        )

        ################## DECODER #################
        self._decoder = nn.Sequential()

        # Linear layer from latent space
        self._decoder.add_module(
            "latent_linear_decoder",
            LinearLayer(
                in_features=latent_dimension,
                out_features=self.flattened_size,
                nonlinearity=nonlinearity,
                bias=True,
                batchnorm=False,
                hook=False,
            ),
        )

        # Convolutional transpose layers
        for i in range(len(channel_dims) - 1, -1, -1):
            self._decoder.add_module(
                f"conv_transpose2d_{i}",
                Conv2DLayer(
                    in_channels=channel_dims[i][1],
                    out_channels=channel_dims[i][0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    nonlinearity=nonlinearity if i != 0 else None,
                    bias=True,
                    batchnorm=True,
                    hook=False,
                    transpose=True,
                ),
            )
            self._decoder.apply(Conv2DLayer.init_weights)

    @property
    def encoder(self) -> nn.Sequential:
        """Returns the encoder in a sequential container."""
        return self._encoder

    @property
    def decoder(self) -> nn.Sequential:
        """Returns the decoder in a sequential container."""
        return self._decoder

    @property
    def koopman_matrix(self) -> nn.Sequential:
        """Returns the Koopman matrix in a sequential container."""
        return self._koopman_matrix

    def hook_model(self) -> None:
        raise NotImplementedError()

    @property
    def modules(self) -> nn.Sequential:
        """Returns the autoencoder modules in a sequential container."""
        return nn.Sequential(self._koopman_matrix, *(list(self._encoder) + list(self._decoder)))

    def _encode(self, x):
        # Access the final decoder convolutional layer
        final_decoder_conv2d = self.decoder[-1].layers.conv2d

        # Subtract the bias from the input if it exists
        x_bar = (
            (x - final_decoder_conv2d.bias.view(1, -1, 1, 1))
            if final_decoder_conv2d.bias is not None
            else x
        )

        # Convolutional encoding
        x = self._encoder[:-1](x_bar)  # Pass through all layers except the linear layer

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass through the linear layer
        x = self._encoder[-1](x)

        return x

    def _decode(self, x):
        # Pass through the linear layer to reconstruct flattened dimensions
        x = self._decoder[0](x)  # First linear layer

        # Unflatten using the dynamically calculated final spatial dimensions
        batch_size = x.size(0)
        final_channels = self.hidden_configuration[-1][1]
        final_height, final_width = self.final_spatial_dims
        x = x.view(batch_size, final_channels, final_height, final_width)

        # Pass through the rest of the decoder
        x = self._decoder[1:](x)
        return x

    def forward(self, x: Float[Tensor, "batch features"], k: Int = 0) -> AutoencoderResult:
        """Forward."""

        # Encode
        phi_x = self._encode(x)

        # Reconstruct
        x_recons = self._decode(phi_x)

        return AutoencoderResult(torch.tensor(-1), x_recons)

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        """Get forward activations."""
        pass

    @classmethod
    def load_model(cls, file_path: str | Path):
        """Load model."""

        # Assert path
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(
            k=literal_eval(metadata["k"]),
            input_channels=literal_eval(metadata["input_channels"]),
            input_width=literal_eval(metadata["input_width"]),
            input_height=literal_eval(metadata["input_height"]),
            latent_dimension=literal_eval(metadata["latent_dimension"]),
            nonlinearity=metadata["nonlinearity"],
        )

        # Load weights
        load_model(model, file_path, device=get_device())

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model."""

        metadata = {
            "k": str(self.steps),
            "input_channels": str(self.input_channels),
            "input_width": str(self.input_width),
            "input_height": str(self.input_height),
            "latent_dimension": str(self.latent_dimension),
            "hidden_configuration": str(self.hidden_configuration),
            "nonlinearity": str(self.nonlinearity),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        save_model(self, Path(file_path), metadata=metadata)
