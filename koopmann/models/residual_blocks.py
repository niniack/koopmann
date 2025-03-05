# residual_blocks.py
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import stochastic_depth

from .layers import Conv2DLayer, LinearLayer
from .utils import StringtoClassNonlinearity


class BaseResidualBlock(nn.Module, ABC):
    """
    Base class for all residual blocks.
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        stoch_mode: str = "batch",
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.stoch_mode = stoch_mode
        self._forward_activations = None
        self._handle = None

    @property
    def forward_activations(self) -> tuple:
        """Returns tuple of forward activations from hook."""
        return self._forward_activations

    @property
    def is_hooked(self) -> bool:
        """Returns boolean indicating whether block has hook."""
        return self.hook

    def setup_hook(self):
        """Sets up a hook to capture activations."""

        def _hook(module, input, output):
            self._forward_activations = output

        self.hook = True
        self._handle = self.register_forward_hook(_hook)

    def remove_hook(self):
        """Tears down the hook."""
        self.hook = False
        if self._handle:
            self._handle.remove()
            self._handle = None

    def identity_block(self, x: Tensor) -> Tensor:
        """
        Identity mapping for the residual connection.
        Default is to return the input unchanged.
        """
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the residual block.
        Returns both the activated output and the activation pattern.
        """
        identity = self.identity_block(x)

        # Apply residual branch
        out = self.residual_branch(x)

        # Apply stochastic depth
        out = stochastic_depth(out, p=self.drop_prob, mode=self.stoch_mode, training=self.training)

        # Add the identity connection
        out_with_skip = out + identity

        # Apply activation
        activated_out = F.relu(out_with_skip)

        # Calculate activation pattern (which elements are positive)
        activation_pattern = out * (out_with_skip > 0)

        return activated_out, activation_pattern

    def residual_branch(self, x: Tensor) -> Tensor:
        """Apply the residual branch operations."""
        for layer in self.layers.values():
            x = layer(x)
        return x


class LinearResidualBlock(BaseResidualBlock):
    """
    Residual block for ResMLP, consisting of two fully-connected layers.
    """

    def __init__(
        self,
        dimension: int,
        nonlinearity: Union[str, nn.Module] = "relu",
        bias: bool = False,
        batchnorm: bool = True,
        drop_prob: float = 0.0,
        stoch_mode: str = "batch",
    ):
        super().__init__(drop_prob=drop_prob, stoch_mode=stoch_mode)

        self.dimension = dimension
        self.in_features = dimension
        self.out_features = dimension

        # Convert nonlinearity to module if it's a string
        if isinstance(nonlinearity, str):
            nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value
        else:
            nonlinearity_module = nonlinearity

        self.layers = nn.ModuleDict()

        # First fully-connected layer
        fc1 = LinearLayer(
            in_features=dimension,
            out_features=dimension,
            nonlinearity=nonlinearity_module,
            bias=bias,
            batchnorm=batchnorm,
        )
        fc1.apply(LinearLayer.init_weights)
        self.layers["fc1"] = fc1

        # Second fully-connected layer
        fc2 = LinearLayer(
            in_features=dimension,
            out_features=dimension,
            nonlinearity=None,  # No nonlinearity after the second layer as per ResNet design
            bias=bias,
            batchnorm=batchnorm,
        )
        fc2.apply(LinearLayer.init_weights)
        self.layers["fc2"] = fc2


class Conv2DResidualBlock(BaseResidualBlock):
    """
    Residual block for ConvResNet, consisting of two convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        nonlinearity: Union[str, nn.Module] = "relu",
        bias: bool = False,
        batchnorm: bool = True,
        drop_prob: float = 0.0,
        stoch_mode: str = "batch",
    ):
        super().__init__(drop_prob=drop_prob, stoch_mode=stoch_mode)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Determine if we need a projection shortcut (1x1 conv) for changing dimensions
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv2DLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                nonlinearity=None,
                bias=False,
                batchnorm=batchnorm,
            )
            self.downsample.apply(Conv2DLayer.init_weights)

        # Convert nonlinearity to module if it's a string
        if isinstance(nonlinearity, str):
            nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value
        else:
            nonlinearity_module = nonlinearity

        self.layers = nn.ModuleDict()

        # First convolutional layer
        padding = kernel_size // 2  # Same padding
        conv1 = Conv2DLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            nonlinearity=nonlinearity_module,
            bias=bias,
            batchnorm=batchnorm,
        )
        conv1.apply(Conv2DLayer.init_weights)
        self.layers["conv1"] = conv1

        # Second convolutional layer
        conv2 = Conv2DLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,  # Always stride 1 for the second conv
            padding=padding,
            nonlinearity=None,  # No nonlinearity after the second layer as per ResNet design
            bias=bias,
            batchnorm=batchnorm,
        )
        conv2.apply(Conv2DLayer.init_weights)
        self.layers["conv2"] = conv2

    def identity_block(self, x: Tensor) -> Tensor:
        """Apply identity mapping with optional projection for dimension mismatch."""
        if self.downsample is not None:
            return self.downsample(x)
        return x
