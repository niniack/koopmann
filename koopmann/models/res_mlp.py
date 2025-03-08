# resmlp.py
__all__ = ["ResMLP"]

import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer
from koopmann.models.residual_blocks import LinearResidualBlock


class ResMLP(BaseTorchModel):
    """
    Residual multi-layer perceptron with improved architecture.
    NOTE: `hidden_config` defines the (uniform) number of channels for the residual blocks.
    """

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 2,
        hidden_config: list[int] = [8],
        bias: bool = True,
        batchnorm: bool = True,
        nonlinearity: str = "relu",
        stochastic_depth_prob: float = 0.0,
        stochastic_depth_mode: str = "batch",
    ):
        super().__init__()

        if len(hidden_config) == 0:
            raise ValueError("Config must contain at least one block!")

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_config = hidden_config
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batchnorm = batchnorm
        self.stochastic_depth_prob = stochastic_depth_prob
        self.stochastic_depth_mode = stochastic_depth_mode
        self.full_config = [in_features, *hidden_config, out_features]

        # Input projection layer
        input_layer = LinearLayer(
            in_channels=in_features,
            out_channels=self.full_config[1],
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
        )
        input_layer.apply(LinearLayer.init_weights)
        self.components.add_module("input_layer", input_layer)

        # Residual blocks with linearly increasing drop probability
        num_blocks = len(self.full_config) - 2
        for i in range(1, len(self.full_config) - 1):
            # Linear increase in drop probability with depth
            # First block has near 0, last block has stochastic_depth_prob
            if num_blocks > 1 and stochastic_depth_prob > 0:
                block_index = i - 1
                block_drop_prob = stochastic_depth_prob * block_index / (num_blocks - 1)
            else:
                block_drop_prob = 0.0

            block = LinearResidualBlock(
                channels=self.full_config[i],
                bias=bias,
                batchnorm=batchnorm,
                nonlinearity=nonlinearity,
                drop_prob=block_drop_prob,
                stoch_mode=stochastic_depth_mode,
            )
            self.components.add_module(f"residual_block_{i}", block)

        if len(hidden_config) > 0:
            # Output projection layer
            output_layer = LinearLayer(
                in_channels=self.full_config[-2],
                out_channels=out_features,
                nonlinearity=None,
                bias=bias,
                batchnorm=False,
            )
            output_layer.apply(LinearLayer.init_weights)
            self.components.add_module("output_layer", output_layer)

    def forward(self, x: float) -> Tensor:
        """Forward pass through the ResMLP."""

        x = self.components[0](x)
        for block in self.components[1:-1]:
            x, _ = block(x)
        x = self.components[-1](x)

        return x

    def get_fwd_acts_patts(self, detach=False) -> Tuple[OrderedDict, OrderedDict]:
        """Get both activations and patterns from layers."""
        raise NotImplementedError("Not yet.")

    def insert_residual_block(
        self,
        index: int,
        channels: Optional[int] = None,
        nonlinearity: Optional[str] = None,
    ):
        """Insert a new residual block at the specified index in the hidden layers."""
        # Get current modules
        modules_list = list(self.components.children())

        # Validate index - only allow inserting between first and last layer
        if index <= 0 or index >= len(modules_list):
            raise ValueError(
                f"Cannot insert at position {index}. Must be between 1 and {len(modules_list)-1}."
            )

        # Determine nonlinearity
        if nonlinearity is None:
            nonlinearity = self.nonlinearity

        # Determine channels based on surrounding layers if not specified
        if channels is None:
            # Try to infer from surrounding layers
            prev_module = modules_list[index - 1]
            next_module = modules_list[index]

            # Get output channels of previous layer
            prev_channels = getattr(prev_module, "out_channels", None)
            if prev_channels is None and hasattr(prev_module, "components"):
                # Try to get from last component
                last_comp = list(prev_module.components.values())[-1]
                prev_channels = getattr(last_comp, "out_channels", None)

            # Get input channels of next layer
            next_channels = getattr(next_module, "in_channels", None)
            if next_channels is None and hasattr(next_module, "components"):
                # Try to get from first component
                first_comp = list(next_module.components.values())[0]
                next_channels = getattr(first_comp, "in_channels", None)

            # Use surrounding channels if they match
            if prev_channels is not None and prev_channels == next_channels:
                channels = prev_channels
            else:
                # Default to config value
                channels = self.full_config[index]
                warnings.warn(f"Could not determine channels from context. Using {channels}.")

        # Create a new residual block
        new_block = LinearResidualBlock(
            channels=channels,
            nonlinearity=nonlinearity,
            bias=self.bias,
            batchnorm=self.batchnorm,
            drop_prob=0.0,  # Initialize with no stochastic depth
            stoch_mode=self.stochastic_depth_mode,
        )

        # Insert the block
        modules_list.insert(index, new_block)

        # Update the modules
        self.components = nn.Sequential(*modules_list)

        # Update configuration
        self.full_config.insert(index, channels)
        self.hidden_config = self.full_config[1:-1]

        # Update stochastic depth probabilities
        self._update_stochastic_depth_probs()

    def remove_residual_block(self, index: int):
        """Remove a layer at the specified index."""
        # Get current modules
        modules_list = list(self.components.children())

        # Validate index - only allow removing hidden layers
        if index <= 0 or index >= len(modules_list) - 1:
            raise ValueError(
                f"Cannot remove layer at position {index}. Must be between 1 and {len(modules_list)-2}."
            )

        # Remove the layer
        _ = modules_list.pop(index)

        # Update the modules
        self.components = nn.Sequential(*modules_list)

        # Update configuration
        self.full_config.pop(index)
        self.hidden_config = self.full_config[1:-1]

        # Update stochastic depth probabilities
        self._update_stochastic_depth_probs()

    def _update_stochastic_depth_probs(self):
        """Update stochastic depth probabilities based on layer depth."""
        residual_blocks = [
            module for module in self.components if isinstance(module, LinearResidualBlock)
        ]

        num_blocks = len(residual_blocks)
        if (
            num_blocks > 1
            and hasattr(self, "stochastic_depth_prob")
            and self.stochastic_depth_prob > 0
        ):
            for i, block in enumerate(residual_blocks):
                if hasattr(block, "drop_prob"):
                    block.drop_prob = self.stochastic_depth_prob * i / (num_blocks - 1)
                else:
                    warnings.warn(
                        f"Block {i} doesn't have drop_prob attribute. Skipping stochastic depth update."
                    )

    def _get_basic_metadata(self) -> Dict[str, Any]:
        """Get model-specific metadata for serialization."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_config": self.hidden_config,
            "bias": self.bias,
            "batchnorm": self.batchnorm,
            "nonlinearity": self.nonlinearity,
            "stochastic_depth_prob": self.stochastic_depth_prob,
            "stochastic_depth_mode": self.stochastic_depth_mode,
        }
