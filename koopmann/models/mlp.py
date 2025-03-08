# mlp.py
__all__ = ["MLP"]

import warnings
from typing import Any, Dict, Optional

import torch.nn as nn
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer


class MLP(BaseTorchModel):
    """
    Multi-layer perceptron.
    NOTE: `hidden_config` defines the output dimension of the hidden linear layers
    """

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 2,
        hidden_config: list[int] = [8],
        bias: bool = True,
        batchnorm: bool = True,
        nonlinearity: str = "relu",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_config = hidden_config
        self.bias = bias
        self.batchnorm = batchnorm
        self.nonlinearity = nonlinearity
        self.full_config = [in_features, *hidden_config, out_features]

        for i in range(len(self.full_config) - 1):
            # For the last layer, we don't use a nonlinearity
            layer_nonlinearity = None if i == len(self.full_config) - 2 else nonlinearity

            layer = LinearLayer(
                in_channels=self.full_config[i],
                out_channels=self.full_config[i + 1],
                bias=bias,
                batchnorm=batchnorm,
                nonlinearity=layer_nonlinearity,
            )
            layer.apply(LinearLayer.init_weights)

            self.components.add_module(f"linear_{i}", layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.components(x)

    def insert_layer(
        self,
        index: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        nonlinearity: Optional[str] = None,
    ):
        """Insert a linear layer at the specified index."""
        # Ensure we're not inserting at the boundaries
        if index <= 0 or index >= len(self.components):
            raise ValueError(
                f"Cannot insert at position {index}. Must be between 1 and {len(self.components)-1}."
            )

        # Convert container to list
        layers = list(self.components)

        # Configure dimensions based on surrounding layers
        if in_channels is None:
            in_channels = layers[index - 1].out_channels
        if out_channels is None:
            out_channels = layers[index].in_channels

        # Dimension validation
        if (
            in_channels != layers[index - 1].out_channels
            or out_channels != layers[index].in_channels
        ):
            warnings.warn("Dimension mismatch may cause errors during forward pass")

        # Create the new layer
        new_layer = LinearLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            nonlinearity=nonlinearity,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        new_layer.apply(LinearLayer.init_weights)

        # Insert the layer
        layers.insert(index, new_layer)

        # Rebuild sequential container
        self.components = nn.Sequential(*layers)

        # Update configuration - adjust index for hidden_config (subtract 1 since hidden_config doesn't include input layer)
        self.hidden_config.insert(index - 1, out_channels)
        self.full_config = [self.in_features, *self.hidden_config, self.out_features]

    def remove_layer(self, index: int):
        """Remove a linear layer at the specified index."""
        # Ensure we have enough layers and aren't removing boundaries
        if len(self.components) <= 2:
            raise ValueError("Cannot remove layer: minimum model size is 2 layers")

        if index <= 0 or index >= len(self.components) - 1:
            raise ValueError(
                f"Cannot remove layer at position {index}. Must be between 1 and {len(self.components)-2}."
            )

        # Remove from layers
        layers = list(self.components)
        _ = layers.pop(index)

        # Ensure connectivity after removal
        if layers[index - 1].out_channels != layers[index].in_channels:
            warnings.warn("Removing this layer may cause dimension mismatch during forward pass")

        # Rebuild sequential container
        self.components = nn.Sequential(*layers)

        # Update configuration - adjust index for hidden_config
        self.hidden_config.pop(index - 1)
        self.full_config = [self.in_features, *self.hidden_config, self.out_features]

    def _get_basic_metadata(self) -> Dict[str, Any]:
        """Get model-specific metadata for serialization."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_config": self.hidden_config,
            "bias": self.bias,
            "batchnorm": self.batchnorm,
            "nonlinearity": self.nonlinearity,
        }
