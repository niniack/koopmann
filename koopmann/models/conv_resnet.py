# convresnet.py
__all__ = ["ConvResNet", "resnet18", "resnet18_mnist"]

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import Conv2DLayer, LinearLayer
from koopmann.models.residual_blocks import Conv2DResidualBlock


def resnet18():
    return ConvResNet(
        in_channels=3,
        out_features=10,
        input_size=(32, 32),
        hidden_config=[64, 128, 256, 512],
        blocks_per_stage=[2, 2, 2, 2],
        bias=False,
        batchnorm=True,
        nonlinearity="relu",
        initial_downsample_factor=2,
    )


def resnet18_mnist():
    return ConvResNet(
        in_channels=1,
        out_features=10,
        input_size=(28, 28),
        hidden_config=[64, 128, 256, 512],
        blocks_per_stage=[2, 2, 2, 2],
        bias=False,
        batchnorm=True,
        nonlinearity="relu",
        initial_downsample_factor=2,
    )


class ConvResNet(BaseTorchModel):
    """
    Convolutional Residual Network.
    Designed to mimic the architecture from the reference implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_features: int = 10,  # Typically number of classes for classification
        input_size: Tuple[int, int] = (32, 32),  # Input image size (height, width)
        hidden_config: List[int] = [64],  # Number of channels per stage
        blocks_per_stage: List[int] = [8],  # Number of residual blocks per stage
        bias: bool = True,
        batchnorm: bool = True,
        nonlinearity: str = "relu",
        stochastic_depth_prob: float = 0.0,  # Max probability for stochastic depth
        stochastic_depth_mode: str = "batch",  # Mode for stochastic depth: "batch" or "row"
        initial_downsample_factor: int = 1,  # Similar to 'pol' in reference
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.input_size = input_size
        self.hidden_config = hidden_config
        self.blocks_per_stage = blocks_per_stage
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batchnorm = batchnorm
        self.stochastic_depth_prob = stochastic_depth_prob
        self.stochastic_depth_mode = stochastic_depth_mode
        self.initial_downsample_factor = initial_downsample_factor

        # Calculate spatial dimensions after initial conv
        nimsize = input_size[0] // initial_downsample_factor

        # Calculate pooling kernel size as in original
        pos = min(4, nimsize)

        # Initial convolution layer with dynamic parameters
        self.components.add_module(
            "initial_conv",
            Conv2DLayer(
                in_channels=in_channels,
                out_channels=hidden_config[0],
                kernel_size=3,  # 7 for imagenet
                stride=1,  # 2 for imagenet
                padding=1,  # 3 for imagenet
                bias=bias,
                batchnorm=batchnorm,
                nonlinearity=nonlinearity,
            ),
        )

        # Add maxpool after the initial layer
        # self.components.add_module("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Calculate total number of blocks for stochastic depth scaling
        total_blocks = sum(blocks_per_stage)
        block_idx = 0

        # Track the current channel count
        current_channels = hidden_config[0]

        # Build each stage
        for stage_idx, (num_blocks, stage_channels) in enumerate(
            zip(blocks_per_stage, hidden_config)
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

            # First block of stage handles transition from previous stage
            self.components.add_module(
                block_name,
                Conv2DResidualBlock(
                    in_channels=current_channels,
                    out_channels=stage_channels,
                    kernel_size=3,
                    stride=first_stride,
                    bias=bias,
                    batchnorm=batchnorm,
                    nonlinearity=nonlinearity,
                    drop_prob=drop_prob,
                    stoch_mode=stochastic_depth_mode,
                ),
            )

            # Update current channels for next blocks
            current_channels = stage_channels

            # Add remaining blocks in the stage
            for block_num in range(2, num_blocks + 1):
                block_name = f"stage_{stage_idx+1}_block_{block_num}"

                # Calculate drop probability for stochastic depth
                drop_prob = 0.0
                if stochastic_depth_prob > 0:
                    drop_prob = stochastic_depth_prob * block_idx / total_blocks
                    block_idx += 1

                self.components.add_module(
                    block_name,
                    Conv2DResidualBlock(
                        in_channels=current_channels,
                        out_channels=current_channels,  # Same in/out channels for non-transition blocks
                        kernel_size=3,
                        stride=1,
                        bias=bias,
                        batchnorm=batchnorm,
                        nonlinearity=nonlinearity,
                        drop_prob=drop_prob,
                        stoch_mode=stochastic_depth_mode,
                    ),
                )

        self.components.add_module("avg_pool", nn.AdaptiveAvgPool2d((1, 1)))

        # Fully connected layer for final output
        self.components.add_module(
            "fc",
            LinearLayer(
                in_channels=hidden_config[-1],
                out_channels=out_features,
                bias=False,
                batchnorm=False,
                nonlinearity=None,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        Returns logits (pre-softmax activations).
        """

        # All but FC layer
        for block in self.components[:-1]:
            if isinstance(block, Conv2DResidualBlock):
                x, _ = block(x)
            else:
                x = block(x)

        # Flatten before FC layer
        x = torch.flatten(x, start_dim=1)

        # Final FC layer
        x = self.components[-1](x)

        return x

    def get_fwd_acts_patts(self, detach=False) -> Tuple[OrderedDict, OrderedDict]:
        """Get both activations and patterns from layers."""
        raise NotImplementedError("Not yet.")

    def _get_basic_metadata(self) -> Dict[str, Any]:
        """Get model-specific metadata for serialization."""
        return {
            "in_channels": self.in_channels,
            "out_features": self.out_features,
            "input_size": self.input_size,
            "hidden_config": self.hidden_config,
            "blocks_per_stage": self.blocks_per_stage,
            "bias": self.bias,
            "batchnorm": self.batchnorm,
            "nonlinearity": self.nonlinearity,
            "stochastic_depth_prob": self.stochastic_depth_prob,
            "stochastic_depth_mode": self.stochastic_depth_mode,
            "initial_downsample_factor": self.initial_downsample_factor,
        }
