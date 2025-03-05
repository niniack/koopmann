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
    Residual multi-layer perceptron.
    """

    def __init__(
        self,
        input_dimension: int = 2,
        output_dimension: int = 2,
        config: list = [8],  # Number of neurons per hidden layer.
        nonlinearity: str = "relu",
        bias: bool = True,
        batchnorm: bool = True,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.config = config
        self.nonlinearity = nonlinearity
        nonlinearity = StringtoClassNonlinearity[nonlinearity].value
        self.bias = bias
        self.batchnorm = batchnorm
        self.full_config = [input_dimension, *config, output_dimension]

        # Input projection layer (from input dimension to hidden dimension)
        input_layer = LinearLayer(
            in_features=input_dimension,
            out_features=self.full_config[1],
            nonlinearity=nonlinearity,
            bias=bias,
            batchnorm=batchnorm,
            hook=False,
        )
        input_layer.apply(LinearLayer.init_weights)

        # Residual blocks
        residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dimension=self.full_config[i],
                    nonlinearity=nonlinearity,
                    bias=bias,
                    batchnorm=batchnorm,
                    hook=False,
                )
                for i in range(1, len(self.full_config) - 1)
            ]
        )

        # Output projection layer (from hidden dimension to output dimension)
        output_layer = LinearLayer(
            in_features=self.full_config[-2],
            out_features=output_dimension,
            nonlinearity=None,  # No activation for the output layer
            bias=bias,
            batchnorm=False,  # Typically no batch norm in the final layer
            hook=False,
        )
        output_layer.apply(LinearLayer.init_weights)

        self._features = nn.Sequential(input_layer, *residual_blocks, output_layer)

    def hook_model(self) -> None:
        # Remove all previous hooks
        for layer in self.modules:
            layer.remove_hook()

        # Add back hooks
        for layer in self.modules:
            layer.setup_hook()

    @property
    def modules(self) -> nn.Sequential:
        return self._features

    def forward(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the ResMLP."""
        for layer in self.modules:
            if isinstance(layer, ResidualBlock):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        activations = OrderedDict()

        for i, layer in enumerate(self.modules):
            if i == len(self.modules) - 1:
                continue
            elif isinstance(layer, ResidualBlock) and layer.is_hooked:
                acts, patts = layer.forward_activations
                activations[i + 1] = acts if not detach else acts.detach()
            elif layer.is_hooked:
                activations[0] = (
                    layer.forward_activations if not detach else layer.forward_activations.detach()
                )

        return activations

    def get_fwd_acts_patts(self, detach=False) -> OrderedDict:
        activations = OrderedDict()
        patterns = OrderedDict()

        for i, layer in enumerate(self.modules):
            if i == len(self.modules) - 1:
                continue
            elif i == 0 and layer.is_hooked:
                activations[0] = (
                    layer.forward_activations if not detach else layer.forward_activations.detach()
                )
                patterns[0] = activations[0]
            elif isinstance(layer, ResidualBlock) and layer.is_hooked:
                acts, patts = layer.forward_activations
                activations[i + 1] = acts if not detach else acts.detach()
                patterns[i + 1] = patts if not detach else patts.detach()
            elif layer.is_hooked:
                activations[i] = (
                    layer.forward_activations if not detach else layer.forward_activations.detach()
                )
                patterns[i] = activations[i]

        return activations, patterns

    def insert_layer(
        self,
        index: int,
        out_features: int = None,
        nonlinearity: Optional[Literal["none"]] = None,
    ):
        # Get nonlinerity
        if nonlinearity and nonlinearity == "none":
            nonlinearity = None
        elif nonlinearity:
            nonlinearity = StringtoClassNonlinearity[nonlinearity].value
        else:
            nonlinearity = StringtoClassNonlinearity[self.nonlinearity].value

        # Convert container to list
        layers = list(self.modules)

        # Configure new layer
        in_features = layers[index - 1].out_features
        if not out_features:
            out_features = layers[index].in_features

        # Convert container to list and insert
        new_layer = LinearLayer(
            in_features=in_features,
            out_features=out_features,
            nonlinearity=nonlinearity if not index == (len(layers) + 1) - 1 else None,
            batchnorm=True,
            bias=self.bias,
            hook=False,
        )
        layers.insert(index, new_layer)

        # Convert back to container
        self._features = nn.Sequential(*layers)

        # Update config
        self.full_config.insert(index + 1, out_features)
        self.config = self.full_config[1:-1]

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
            config=literal_eval(metadata["config"]),
            nonlinearity=metadata["nonlinearity"],
            bias=literal_eval(metadata["bias"]),
            batchnorm=literal_eval(metadata["batchnorm"]),
        )
        model.train()

        # Load weights
        load_model(model, file_path, device=get_device())

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model."""
        metadata = {
            "input_dimension": str(self.input_dimension),
            "output_dimension": str(self.output_dimension),
            "config": str(self.config),
            "nonlinearity": str(self.nonlinearity),
            "bias": str(self.bias),
            "batchnorm": str(self.batchnorm),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        self.eval()
        save_model(self, Path(file_path), metadata=metadata)
