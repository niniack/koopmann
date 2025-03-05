# convresnet.py
__all__ = ["ConvResNet"]

from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_model, save_model
from torch import Tensor

from .base import BaseTorchModel
from .layers import Conv2DLayer, LayerType
from .residual_blocks import Conv2DResidualBlock
from .utils import StringtoClassNonlinearity, get_device, parse_safetensors_metadata


class ConvResNet(BaseTorchModel):
    """
    Convolutional Residual Network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,  # Typically number of classes for classification
        input_size: Tuple[int, int] = (32, 32),  # Input image size (height, width)
        channels_config: List[int] = [64, 128, 256],  # Number of channels per stage
        blocks_per_stage: List[int] = [2, 2, 2],  # Number of residual blocks per stage
        nonlinearity: str = "relu",
        bias: bool = True,
        batchnorm: bool = True,
        stochastic_depth_prob: float = 0.0,  # Max probability for stochastic depth
        stochastic_depth_mode: str = "batch",  # Mode for stochastic depth: "batch" or "row"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.channels_config = channels_config
        self.blocks_per_stage = blocks_per_stage
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batchnorm = batchnorm
        self.stochastic_depth_prob = stochastic_depth_prob
        self.stochastic_depth_mode = stochastic_depth_mode

        # Convert string nonlinearity to class
        nonlinearity_class = StringtoClassNonlinearity[nonlinearity].value

        # Initial convolution layer
        self._features = nn.Sequential()
        self._features.add_module(
            "initial_conv",
            Conv2DLayer(
                in_channels=in_channels,
                out_channels=channels_config[0],
                kernel_size=7,  # Typical initial kernel size
                stride=2,  # Typical initial stride
                padding=3,  # Same padding for 7x7 kernel
                nonlinearity=nonlinearity_class,
                bias=bias,
                batchnorm=batchnorm,
            ),
        )

        # Max pooling after initial convolution
        self._features.add_module("max_pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Create stages with residual blocks
        current_channels = channels_config[0]

        # Calculate total number of blocks for stochastic depth scaling
        total_blocks = sum(blocks_per_stage)
        block_idx = 0

        # Build each stage
        for stage_idx, (num_blocks, stage_channels) in enumerate(
            zip(blocks_per_stage, channels_config)
        ):
            # First block in the stage may downsample (except for the first stage)
            first_stride = 1 if stage_idx == 0 else 2

            # Add first block of the stage (potential downsampling)
            block_name = f"stage_{stage_idx+1}_block_1"

            # Calculate drop probability for stochastic depth
            drop_prob = 0.0
            if stochastic_depth_prob > 0:
                drop_prob = stochastic_depth_prob * block_idx / total_blocks
                block_idx += 1

            self._features.add_module(
                block_name,
                Conv2DResidualBlock(
                    in_channels=current_channels,
                    out_channels=stage_channels,
                    kernel_size=3,
                    stride=first_stride,
                    nonlinearity=nonlinearity_class,
                    bias=bias,
                    batchnorm=batchnorm,
                    drop_prob=drop_prob,
                    stoch_mode=stochastic_depth_mode,
                ),
            )

            current_channels = stage_channels

            # Add remaining blocks in the stage
            for block_num in range(2, num_blocks + 1):
                block_name = f"stage_{stage_idx+1}_block_{block_num}"

                # Calculate drop probability for stochastic depth
                drop_prob = 0.0
                if stochastic_depth_prob > 0:
                    drop_prob = stochastic_depth_prob * block_idx / total_blocks
                    block_idx += 1

                self._features.add_module(
                    block_name,
                    Conv2DResidualBlock(
                        in_channels=current_channels,
                        out_channels=stage_channels,
                        kernel_size=3,
                        stride=1,  # No stride for subsequent blocks in stage
                        nonlinearity=nonlinearity_class,
                        bias=bias,
                        batchnorm=batchnorm,
                        drop_prob=drop_prob,
                        stoch_mode=stochastic_depth_mode,
                    ),
                )

        # Global average pooling
        self._features.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))

        # Fully connected layer for final output
        self._features.add_module("fc", nn.Linear(channels_config[-1], out_channels))

    def hook_model(self) -> None:
        """Set up activation hooks for all layers."""
        # Remove all previous hooks
        for name, module in self._features.named_modules():
            if hasattr(module, "remove_hook"):
                module.remove_hook()

        # Add back hooks
        for name, module in self._features.named_modules():
            if hasattr(module, "setup_hook"):
                module.setup_hook()

    @property
    def modules(self) -> nn.Sequential:
        """Returns all modules in the model."""
        return self._features

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Tensor:
        """Forward pass through the ConvResNet."""
        for name, layer in self._features.named_children():
            if isinstance(layer, Conv2DResidualBlock):
                x, _ = layer(x)
            elif name == "fc":
                # Flatten before the final FC layer
                x = torch.flatten(x, 1)
                x = layer(x)
            else:
                x = layer(x)
        return x

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        """Get forward activations from all hooked layers."""
        activations = OrderedDict()
        for i, (name, layer) in enumerate(self._features.named_modules()):
            if hasattr(layer, "is_hooked") and layer.is_hooked:
                if isinstance(layer, Conv2DResidualBlock):
                    acts, patts = layer.forward_activations
                    activations[i] = acts if not detach else acts.detach()
                else:
                    activations[i] = (
                        layer.forward_activations
                        if not detach
                        else layer.forward_activations.detach()
                    )

        return activations

    def get_fwd_acts_patts(self, detach=False) -> Tuple[OrderedDict, OrderedDict]:
        """Get both activations and patterns from residual blocks."""
        activations = OrderedDict()
        patterns = OrderedDict()

        for i, (name, layer) in enumerate(self._features.named_modules()):
            if hasattr(layer, "is_hooked") and layer.is_hooked:
                if isinstance(layer, Conv2DResidualBlock):
                    acts, patts = layer.forward_activations
                    activations[i] = acts if not detach else acts.detach()
                    patterns[i] = patts if not detach else patts.detach()
                else:
                    activations[i] = (
                        layer.forward_activations
                        if not detach
                        else layer.forward_activations.detach()
                    )
                    patterns[i] = activations[i]

        return activations, patterns

    @classmethod
    def load_model(cls, file_path: str | Path):
        """Load model from a file."""
        # Assert path exists
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(
            in_channels=literal_eval(metadata["in_channels"]),
            out_channels=literal_eval(metadata["out_channels"]),
            input_size=literal_eval(metadata["input_size"]),
            channels_config=literal_eval(metadata["channels_config"]),
            blocks_per_stage=literal_eval(metadata["blocks_per_stage"]),
            nonlinearity=metadata["nonlinearity"],
            bias=literal_eval(metadata["bias"]),
            batchnorm=literal_eval(metadata["batchnorm"]),
            stochastic_depth_prob=literal_eval(metadata.get("stochastic_depth_prob", "0.0")),
            stochastic_depth_mode=metadata.get("stochastic_depth_mode", "batch"),
        )
        model.train()

        # Load weights
        load_model(model, file_path, device=get_device())

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model."""
        metadata = {
            "in_channels": str(self.in_channels),
            "out_channels": str(self.out_channels),
            "input_size": str(self.input_size),
            "channels_config": str(self.channels_config),
            "blocks_per_stage": str(self.blocks_per_stage),
            "nonlinearity": str(self.nonlinearity),
            "bias": str(self.bias),
            "batchnorm": str(self.batchnorm),
            "stochastic_depth_prob": str(self.stochastic_depth_prob),
            "stochastic_depth_mode": str(self.stochastic_depth_mode),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        self.eval()
        save_model(self, Path(file_path), metadata=metadata)
