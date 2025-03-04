__all__ = ["ResMLP"]

from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_model, save_model
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer
from koopmann.models.utils import StringtoClassNonlinearity, get_device, parse_safetensors_metadata

from .residual_block import ResidualBlock


class ResMLP(BaseTorchModel):
    """
    Residual MLP
    """

    def __init__(
        self,
        input_dimension: int = 2,
        output_dimension: int = 2,
        hidden_dimension: int = 512,
        num_blocks: int = 16,  # Paper uses 16 blocks for Type 1
        nonlinearity: str = "relu",
        bias: bool = True,
        batchnorm: bool = True,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.num_blocks = num_blocks
        self.nonlinearity = nonlinearity
        nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value
        self.bias = bias
        self.batchnorm = batchnorm

        # Input projection layer (from input dimension to hidden dimension)
        self.input_layer = LinearLayer(
            in_features=input_dimension,
            out_features=hidden_dimension,
            nonlinearity=nonlinearity_module,
            bias=bias,
            batchnorm=batchnorm,
            hook=False,
        )
        self.input_layer.apply(LinearLayer.init_weights)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dimension=hidden_dimension,
                    nonlinearity=nonlinearity,
                    bias=bias,
                    batchnorm=batchnorm,
                    hook=False,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output projection layer (from hidden dimension to output dimension)
        self.output_layer = LinearLayer(
            in_features=hidden_dimension,
            out_features=output_dimension,
            nonlinearity=None,  # No activation for the output layer
            bias=bias,
            batchnorm=False,  # Typically no batch norm in the final layer
            hook=False,
        )
        self.output_layer.apply(LinearLayer.init_weights)

    def hook_model(self) -> None:
        """Add hooks to all layers to capture activations."""
        # Remove all previous hooks
        self.input_layer.remove_hook()
        for block in self.residual_blocks:
            block.remove_hook()
        self.output_layer.remove_hook()

        # Add back hooks
        self.input_layer.setup_hook()
        for block in self.residual_blocks:
            block.setup_hook()
        self.output_layer.setup_hook()

    @property
    def modules(self) -> list:
        """Returns a list of all modules in order.
        Preserves the exact same format as the original implementation.
        """
        # Collect all modules in the forward pass order
        all_modules = [self.input_layer]
        all_modules.extend(self.residual_blocks)
        all_modules.append(self.output_layer)
        return all_modules

    def forward(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the ResMLP."""
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x, _ = block(x)  # We only need the activated output

        return self.output_layer(x)

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        """Returns an OrderedDict of forward activations from all hooked layers.
        Preserves the exact numeric indices as in the original implementation.
        """
        activations = OrderedDict()

        # Get input layer activations
        if self.input_layer.is_hooked:
            activations[0] = (
                self.input_layer.forward_activations
                if not detach
                else self.input_layer.forward_activations.detach()
            )

        # Get residual block activations
        for i, block in enumerate(self.residual_blocks):
            if block.is_hooked:
                activations[i + 1] = (
                    block.forward_activations if not detach else block.forward_activations.detach()
                )

        # Get output layer activations
        if self.output_layer.is_hooked:
            activations[len(self.residual_blocks) + 1] = (
                self.output_layer.forward_activations
                if not detach
                else self.output_layer.forward_activations.detach()
            )

        return activations

    @classmethod
    def load_model(cls, file_path: str | Path):
        """Load model from a file."""
        # Assert path exists
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(
            input_dimension=literal_eval(metadata["input_dimension"]),
            output_dimension=literal_eval(metadata["output_dimension"]),
            hidden_dimension=literal_eval(metadata["hidden_dimension"]),
            num_blocks=literal_eval(metadata["num_blocks"]),
            nonlinearity=metadata["nonlinearity"],
            bias=literal_eval(metadata["bias"]),
            batchnorm=literal_eval(metadata["batchnorm"]),
        )
        model.train()

        # Load weights
        load_model(model, file_path, device=get_device())

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model to a file."""
        metadata = {
            "input_dimension": str(self.input_dimension),
            "output_dimension": str(self.output_dimension),
            "hidden_dimension": str(self.hidden_dimension),
            "num_blocks": str(self.num_blocks),
            "nonlinearity": str(self.nonlinearity),
            "bias": str(self.bias),
            "batchnorm": str(self.batchnorm),
        }

        # Add any additional metadata
        for key, value in kwargs.items():
            metadata[key] = str(value)

        self.eval()
        save_model(self, Path(file_path), metadata=metadata)
