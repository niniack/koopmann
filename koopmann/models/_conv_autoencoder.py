__all__ = [
    "MatrixExponential",
    "ConvAutoencoder",
    "ConvExponentialKoopmanAutencoder",
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
                (input_channels, input_channels * 2),
                (input_channels * 2, input_channels * 4),
                (input_channels * 4, latent_dimension),
            ]
            self.hidden_configuration = channel_dims
        else:
            self.hidden_configuration = hidden_configuration
            raise NotImplementedError("Custom hidden configuration does not work yet!")

        ################## ENCODER #################
        self._encoder = nn.Sequential()
        current_height, current_width = input_height, input_width

        # Convolutional layers
        for i in range(len(channel_dims)):
            self._encoder.add_module(
                f"conv2d_{i}",
                Conv2DLayer(
                    in_channels=channel_dims[i][0],
                    out_channels=channel_dims[i][1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    nonlinearity=nonlinearity,
                    bias=True,
                    batchnorm=True,
                    hook=False,
                ),
            )
            # Update spatial dimensions after each convolution
            current_height = (current_height + 2 * 1 - 3) // 2 + 1
            current_width = (current_width + 2 * 1 - 3) // 2 + 1

        self.flattened_size = channel_dims[-1][1] * current_height * current_width

        # Linear layer for latent space
        self._encoder.add_module(
            "latent_linear",
            LinearLayer(
                in_features=self.flattened_size,
                out_features=latent_dimension,
                nonlinearity=nonlinearity,
                bias=True,
                batchnorm=False,
                hook=False,
            ),
        )

        ######################
        # Koopman matrix
        ######################
        self._koopman_matrix = LinearLayer(
            in_features=latent_dimension,
            out_features=latent_dimension,
            nonlinearity=None,
            bias=False,
            batchnorm=False,
            hook=False,
        )

        # eigeninit(self._koopman_matrix.linear_layer.weight, theta=0.3)

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

        # Subtract the bias from the input
        x_bar = (
            (x - final_decoder_conv2d.bias.view(1, -1, 1, 1))
            if final_decoder_conv2d.bias is not None
            else x
        )

        # Apply all encoder layers except the final one
        x = self._encoder[:-1](x_bar)

        # Flatten the output while keeping the batch dimension
        x = x.flatten(start_dim=1, end_dim=-1)

        # Pass through the final linear layer
        return self._encoder[-1](x)

    def _decode(self, x):
        # Pass through latent linear decoder
        x = self._decoder[0](x)

        # Reshape to match the input dimensions of ConvTranspose2d
        channels = self.hidden_configuration[-1][-1]
        spatial_dim = int((x.size(-1) // channels) ** (1 / 2))
        x = x.view(x.size(0), channels, spatial_dim, spatial_dim)

        return self._decoder[1:](x)

    def forward(self, x: Float[Tensor, "batch features"], k: Int = 0) -> AutoencoderResult:
        """Forward."""

        # Encode
        phi_x = self._encode(x)

        # Reconstruct
        x_recons = self._decode(phi_x)

        ##########################################
        # Faster way, but no intermediate stores
        ##########################################
        K_weight = self.koopman_matrix.linear_layer.weight
        K_effective_weight = torch.linalg.matrix_power(K_weight, k)
        x_k = phi_x @ K_effective_weight.T
        x_k = self._decode(x_k)

        # For compatibility
        x_k = x_k.unsqueeze(0)

        return AutoencoderResult(x_k, x_recons)

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        """Get forward activations."""
        pass
        # activations = OrderedDict()
        # for i, layer in enumerate(self.features):
        #     if layer.is_hooked:
        #         activations[i] = (
        #             layer.forward_activations if not detach else layer.forward_activations.detach()
        #         )

        # return activations

    @classmethod
    def load_model(cls, file_path: str | Path):
        """Load model."""

        # Assert path
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(
            k=literal_eval(metadata["steps"]),
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
            "k": str(self.input_width),
            "input_channels": str(self.input_width),
            "input_width": str(self.input_width),
            "input_height": str(self.input_width),
            "latent_dimension": str(self.latent_dimension),
            "hidden_configuration": str(self.hidden_configuration),
            "nonlinearity": str(self.nonlinearity),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        save_model(self, Path(file_path), metadata=metadata)


class MatrixExponential(nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        self.k = k  # Number of steps
        self.dim = dim

    def forward(self, X):
        return torch.matrix_exp(X / self.k)  # Scale M by 1/k

    # # Custom initialization function using matrix logarithm
    # def right_inverse(self, X):
    #     # Define a target matrix (e.g., an orthogonal matrix or one with specific eigenvalues)
    #     dummy_layer = nn.Linear(in_features=self.dim, out_features=self.dim)
    #     eigeninit(dummy_layer.weight, theta=1.0)
    #     target_matrix = dummy_layer.weight

    #     # Compute the matrix logarithm of the target matrix
    #     log_matrix = self.k * logm(target_matrix)

    #     return torch.Tensor(log_matrix)


class ConvExponentialKoopmanAutencoder(ConvAutoencoder):
    """
    Autoencoder model.
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
        super().__init__(
            k,
            input_channels,
            input_height,
            input_width,
            latent_dimension,
            hidden_configuration,
            nonlinearity,
        )

        parametrize.register_parametrization(
            self.koopman_matrix.linear_layer, "weight", MatrixExponential(k=k, dim=latent_dimension)
        )
