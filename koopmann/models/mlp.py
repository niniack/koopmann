# mlp.py
__all__ = ["MLP"]

from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .base import BaseTorchModel
from .layers import LinearLayer
from .utils import StringtoClassNonlinearity, get_device, parse_safetensors_metadata


class MLP(BaseTorchModel):
    """
    Multi-layer perceptron with improved architecture.
    """

    def __init__(
        self,
        input_dimension: int = 2,
        output_dimension: int = 2,
        config: List[int] = [8],  # Number of neurons per hidden layer
        nonlinearity: str = "relu",
        bias: bool = True,
        batchnorm: bool = True,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.config = config
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batchnorm = batchnorm
        self.full_config = [input_dimension, *config, output_dimension]

        # Convert string nonlinearity to class
        nonlinearity_module = (
            StringtoClassNonlinearity[nonlinearity].value if nonlinearity else None
        )

        # Create layers
        self._features = nn.Sequential()

        for i in range(len(self.full_config) - 1):
            # For the last layer, don't apply nonlinearity
            layer_nonlinearity = None if i == len(self.full_config) - 2 else nonlinearity_module

            layer = LinearLayer(
                in_features=self.full_config[i],
                out_features=self.full_config[i + 1],
                nonlinearity=layer_nonlinearity,
                batchnorm=batchnorm,
                bias=bias,
            )
            layer.apply(LinearLayer.init_weights)

            self._features.add_module(f"layer_{i}", layer)

    @property
    def modules(self) -> nn.Sequential:
        """Returns the sequential container of layers."""
        return self._features

    def forward(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the MLP."""
        return self.modules(x)

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        """Get forward activations from all hooked layers."""
        activations = OrderedDict()
        for i, (name, layer) in enumerate(self._features.named_modules()):
            if hasattr(layer, "is_hooked") and layer.is_hooked:
                activations[i] = (
                    layer.forward_activations if not detach else layer.forward_activations.detach()
                )

        return activations

    def insert_layer(
        self,
        index: int,
        out_features: int = None,
        nonlinearity: Optional[Literal["none"]] = None,
    ):
        """Insert a new layer at the specified index."""
        # Get nonlinearity
        if nonlinearity and nonlinearity == "none":
            nonlinearity_module = None
        elif nonlinearity:
            nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value
        else:
            nonlinearity_module = StringtoClassNonlinearity[self.nonlinearity].value

        # Convert container to list
        layers = list(self.modules)

        # Configure new layer
        in_features = layers[index - 1].out_features
        if not out_features:
            out_features = layers[index].in_features

        # Create the new layer
        new_layer = LinearLayer(
            in_features=in_features,
            out_features=out_features,
            nonlinearity=nonlinearity_module if index != len(layers) else None,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        new_layer.apply(LinearLayer.init_weights)

        # Insert the layer
        layers.insert(index, new_layer)

        # Rebuild sequential container
        self._features = nn.Sequential(*layers)

        # Update configuration
        self.full_config.insert(index + 1, out_features)
        self.config = self.full_config[1:-1]

    def remove_layer(self, index: int):
        """Remove a layer at the specified index."""
        # Update configuration
        self.full_config = self.full_config[: index + 1] + self.full_config[index + 2 :]
        self.config = self.full_config[1:-1]

        # Remove from layers
        layers = list(self._features)
        layers.pop(index)

        # Rebuild sequential container
        self._features = nn.Sequential(*layers)

    @classmethod
    def load_model(cls, file_path: Union[str, Path], **kwargs):
        """Load model from a file."""
        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Prepare parameters
        model_params = {
            "input_dimension": literal_eval(metadata["input_dimension"]),
            "output_dimension": literal_eval(metadata["output_dimension"]),
            "config": literal_eval(metadata["config"]),
            "nonlinearity": metadata["nonlinearity"],
            "bias": literal_eval(metadata["bias"]),
            "batchnorm": literal_eval(metadata["batchnorm"]),
        }

        # Use generic load method from base class
        return cls.generic_load_model(file_path, model_params)

    def save_model(self, file_path: Union[str, Path], **kwargs):
        """Save model to a file."""
        metadata = {
            "input_dimension": str(self.input_dimension),
            "output_dimension": str(self.output_dimension),
            "config": str(self.config),
            "nonlinearity": str(self.nonlinearity),
            "bias": str(self.bias),
            "batchnorm": str(self.batchnorm),
        }

        # Add any additional metadata
        for key, value in kwargs.items():
            metadata[key] = str(value)

        # Use generic save method from base class
        self.generic_save_model(file_path, metadata)
