# convresnet.py
__all__ = ["ConvResNet"]

from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torch import Tensor

from koopmann.mixins.serializable import Serializable
from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import Conv2DLayer
from koopmann.models.residual_blocks import Conv2DResidualBlock
from koopmann.models.utils import StringtoClassNonlinearity
from koopmann.utils import get_device

# convresnet.py
__all__ = ["ConvResNet"]


class ConvResNet(BaseTorchModel):
    """
    Convolutional Residual Network.
    Designed to mimic the architecture from the reference implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,  # Typically number of classes for classification
        input_size: Tuple[int, int] = (32, 32),  # Input image size (height, width)
        channels_config: List[int] = [64],  # Number of channels per stage
        blocks_per_stage: List[int] = [8],  # Number of residual blocks per stage
        nonlinearity: str = "relu",
        bias: bool = True,
        batchnorm: bool = True,
        stochastic_depth_prob: float = 0.0,  # Max probability for stochastic depth
        stochastic_depth_mode: str = "batch",  # Mode for stochastic depth: "batch" or "row"
        initial_downsample_factor: int = 1,  # Similar to 'pol' in reference
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
        self.initial_downsample_factor = initial_downsample_factor

        # Convert string nonlinearity to class
        nonlinearity_class = StringtoClassNonlinearity[nonlinearity].value

        self._features = nn.Sequential()

        # Calculate parameters based on input size and downsample factor
        kernel_size = 2 * self.initial_downsample_factor + 1
        stride = self.initial_downsample_factor

        # Determine padding based on input size (MNIST vs CIFAR)
        if self.input_size[0] == 28:  # MNIST size
            padding = kernel_size // 2 + 1
        else:  # CIFAR or other sizes
            padding = kernel_size // 2

        # Calculate spatial dimensions after initial conv
        nimsize = input_size[0] // initial_downsample_factor

        # Calculate pooling kernel size as in original
        pos = min(4, nimsize)

        # Initial convolution layer with dynamic parameters
        self._features.add_module(
            "initial_conv",
            Conv2DLayer(
                in_channels=in_channels,
                out_channels=channels_config[0],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                nonlinearity=nonlinearity_class,
                bias=bias,
                batchnorm=batchnorm,
            ),
        )

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

        # Average pooling based on architecture style
        if len(channels_config) > 1:  # Multi-stage architecture (like conv=2)
            # In the conv=2 case they use a fixed pooling size of 2
            self._features.add_module("avg_pool", nn.AvgPool2d(2))
            # Account for spatial reduction and doubling channels
            feature_dim = channels_config[-1] * (nimsize // 2) ** 2
        else:  # Single stage architecture (like conv=1)
            # In the conv=1 case they use the calculated pool size
            self._features.add_module("avg_pool", nn.AvgPool2d(pos))
            # Account for spatial reduction
            feature_dim = channels_config[-1] * (nimsize // pos) ** 2

        # Fully connected layer for final output
        self._features.add_module("fc", nn.Linear(feature_dim, out_channels))

    def hook_model(self) -> None:
        """Set up activation hooks for all layers."""
        # Remove all previous hooks
        for mod in self._features:
            if hasattr(mod, "remove_hook"):
                mod.remove_hook()

        # Add back hooks
        for mod in self._features:
            if hasattr(mod, "setup_hook"):
                mod.setup_hook()

    @property
    def modules(self) -> nn.Sequential:
        """Returns all modules in the model."""
        return self._features

    def forward(self, x: Tensor) -> Tensor:
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
        for i, layer in enumerate(self._features):
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

        for i, layer in enumerate(self._features):
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
        metadata = Serializable.parse_safetensors_metadata(file_path=file_path)

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
            initial_downsample_factor=literal_eval(metadata.get("initial_downsample_factor", "1")),
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
            "initial_downsample_factor": str(self.initial_downsample_factor),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        self.eval()
        save_model(self, Path(file_path), metadata=metadata)
