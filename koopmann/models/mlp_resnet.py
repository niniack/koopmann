# resmlp.py
__all__ = ["ResMLP"]

from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .base import BaseTorchModel
from .layers import LinearLayer
from .residual_blocks import LinearResidualBlock
from .utils import StringtoClassNonlinearity, get_device, parse_safetensors_metadata


class ResMLP(BaseTorchModel):
    """
    Residual multi-layer perceptron with improved architecture.
    """

    def __init__(
        self,
        input_dimension: int = 2,
        output_dimension: int = 2,
        config: List[int] = [8],  # Number of neurons per hidden layer
        nonlinearity: str = "relu",
        bias: bool = True,
        batchnorm: bool = True,
        stochastic_depth_prob: float = 0.0,  # Max probability for stochastic depth
        stochastic_depth_mode: str = "batch",  # Mode for stochastic depth: "batch" or "row"
    ):
        super().__init__()

        if len(config) == 0:
            raise ValueError("Config must contain blocks!")

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.config = config
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batchnorm = batchnorm
        self.stochastic_depth_prob = stochastic_depth_prob
        self.stochastic_depth_mode = stochastic_depth_mode
        self.full_config = [input_dimension, *config, output_dimension]

        # Convert string nonlinearity to class
        nonlinearity_module = (
            StringtoClassNonlinearity[nonlinearity].value if nonlinearity else None
        )

        # Create sequential container
        self._features = nn.Sequential()

        # Input projection layer (from input dimension to hidden dimension)
        input_layer = LinearLayer(
            in_features=input_dimension,
            out_features=self.full_config[1],
            nonlinearity=nonlinearity_module,
            bias=bias,
            batchnorm=batchnorm,
        )
        input_layer.apply(LinearLayer.init_weights)
        self._features.add_module("input_layer", input_layer)

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
                dimension=self.full_config[i],
                nonlinearity=nonlinearity_module,
                bias=bias,
                batchnorm=batchnorm,
                drop_prob=block_drop_prob,
                stoch_mode=stochastic_depth_mode,
            )
            self._features.add_module(f"residual_block_{i}", block)

        if len(config) > 0:
            # Output projection layer (from hidden dimension to output dimension)
            output_layer = LinearLayer(
                in_features=self.full_config[-2],
                out_features=output_dimension,
                nonlinearity=None,  # No activation for the output layer
                bias=bias,
                batchnorm=False,  # Typically no batch norm in the final layer
            )
            output_layer.apply(LinearLayer.init_weights)
            self._features.add_module("output_layer", output_layer)

    @property
    def modules(self) -> nn.Sequential:
        """Returns the sequential container of layers."""
        return self._features

    def forward(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the ResMLP."""
        for module in self.modules.children():
            if isinstance(module, LinearResidualBlock):
                x, _ = module(x)
            else:
                x = module(x)
        return x

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        """Get forward activations from all hooked layers."""
        activations = OrderedDict()
        for i, (name, module) in enumerate(self._features.named_modules()):
            if hasattr(module, "is_hooked") and module.is_hooked:
                if isinstance(module, LinearResidualBlock):
                    acts, _ = module.forward_activations
                    activations[i] = acts if not detach else acts.detach()
                else:
                    activations[i] = (
                        module.forward_activations
                        if not detach
                        else module.forward_activations.detach()
                    )

        return activations

    def get_fwd_acts_patts(self, detach=False) -> Tuple[OrderedDict, OrderedDict]:
        """Get both activations and patterns from layers."""
        activations = OrderedDict()
        patterns = OrderedDict()

        for i, (name, module) in enumerate(self._features.named_modules()):
            if i == len(self._features) - 1:
                continue  # Skip the output layer for patterns
            elif hasattr(module, "is_hooked") and module.is_hooked:
                if isinstance(module, LinearResidualBlock):
                    acts, patts = module.forward_activations
                    activations[i] = acts if not detach else acts.detach()
                    patterns[i] = patts if not detach else patts.detach()
                else:
                    activations[i] = (
                        module.forward_activations
                        if not detach
                        else module.forward_activations.detach()
                    )
                    patterns[i] = activations[i]  # For non-residual blocks

        return activations, patterns

    def insert_layer(
        self,
        index: int,
        out_features: int = None,
        nonlinearity: Optional[Literal["none"]] = None,
    ):
        """Insert a new layer at the specified index."""
        # Convert string nonlinearity to module
        if nonlinearity and nonlinearity == "none":
            nonlinearity_module = None
        elif nonlinearity:
            nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value
        else:
            nonlinearity_module = StringtoClassNonlinearity[self.nonlinearity].value

        # Get current modules
        modules_list = list(self.modules.children())

        # Configure new layer parameters
        in_features = None
        is_last_layer = index == len(modules_list)

        # Determine in_features based on surrounding layers
        if index > 0:
            prev_module = modules_list[index - 1]
            if isinstance(prev_module, LinearLayer):
                in_features = prev_module.out_features
            elif isinstance(prev_module, LinearResidualBlock):
                in_features = prev_module.out_features

        # If not provided, determine out_features based on next layer
        if out_features is None and not is_last_layer:
            next_module = modules_list[index]
            if isinstance(next_module, LinearLayer):
                out_features = next_module.in_features
            elif isinstance(next_module, LinearResidualBlock):
                out_features = next_module.in_features

        # Default to current dimension if still not determined
        if in_features is None:
            in_features = self.full_config[index]
        if out_features is None:
            out_features = in_features  # Use same as input if not specified

        # Create new layer - either a LinearLayer or LinearResidualBlock based on context
        if index == 0 or index == len(modules_list):  # Input or output layer
            new_layer = LinearLayer(
                in_features=in_features,
                out_features=out_features,
                nonlinearity=None if is_last_layer else nonlinearity_module,
                bias=self.bias,
                batchnorm=False if is_last_layer else self.batchnorm,
            )
            new_layer.apply(LinearLayer.init_weights)
        else:
            # Residual block for internal layers
            new_layer = LinearResidualBlock(
                dimension=out_features,
                nonlinearity=nonlinearity_module,
                bias=self.bias,
                batchnorm=self.batchnorm,
                drop_prob=0.0,  # Initialize with no stochastic depth
                stoch_mode=self.stochastic_depth_mode,
            )

        # Insert the new layer
        modules_list.insert(index, new_layer)

        # Update the modules
        self._features = nn.Sequential(*modules_list)

        # Update configuration
        self.full_config.insert(index + 1, out_features)
        self.config = self.full_config[1:-1]

        # Update stochastic depth probabilities
        if self.stochastic_depth_prob > 0:
            self._update_stochastic_depth_probs()

    def remove_layer(self, index: int):
        """Remove a layer at the specified index."""
        # Get current modules
        modules_list = list(self.modules.children())

        # Remove the layer
        removed_module = modules_list.pop(index)

        # Update the modules
        self._features = nn.Sequential(*modules_list)

        # Update configuration
        full_config_index = index + 1 if index > 0 else index
        self.full_config = (
            self.full_config[:full_config_index] + self.full_config[full_config_index + 1 :]
        )
        self.config = self.full_config[1:-1]

        # Update stochastic depth probabilities
        if self.stochastic_depth_prob > 0:
            self._update_stochastic_depth_probs()

    def _update_stochastic_depth_probs(self):
        """Update stochastic depth probabilities based on layer depth."""
        residual_blocks = [
            module for module in self.modules.children() if isinstance(module, LinearResidualBlock)
        ]

        num_blocks = len(residual_blocks)
        if num_blocks > 1 and self.stochastic_depth_prob > 0:
            for i, block in enumerate(residual_blocks):
                block.drop_prob = self.stochastic_depth_prob * i / (num_blocks - 1)

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
            "stochastic_depth_prob": literal_eval(metadata.get("stochastic_depth_prob", "0.0")),
            "stochastic_depth_mode": metadata.get("stochastic_depth_mode", "batch"),
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
            "stochastic_depth_prob": str(self.stochastic_depth_prob),
            "stochastic_depth_mode": str(self.stochastic_depth_mode),
        }

        # Add any additional metadata
        for key, value in kwargs.items():
            metadata[key] = str(value)

        # Use generic save method from base class
        self.generic_save_model(file_path, metadata)
