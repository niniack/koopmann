# residual_blocks.py
import warnings
from abc import ABC
from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import stochastic_depth

from koopmann.mixins.hookable import Hookable

from .layers import Conv2DLayer, LinearLayer
from .utils import StringtoClassNonlinearity


class BaseResidualBlock(nn.Module, ABC, Hookable):
    """
    Base class for residual blocks.
    """

    def __init__(
        self,
        bias: bool,
        batchnorm: bool,
        nonlinearity: str,
        in_channels: int,
        out_channels: int,
        drop_prob: float,
        stoch_mode: str,
    ):
        nn.Module.__init__(self)  # Initialize nn.Module
        Hookable.__init__(self)  # Initialize Hookable

        self.bias = bias
        self.batchnorm = batchnorm
        self.nonlinearity = nonlinearity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob
        self.stoch_mode = stoch_mode
        self.components = nn.ModuleDict()

        # Handle nonlinearity
        if nonlinearity is None:
            pass
        elif not isinstance(nonlinearity, str):
            raise ValueError("Nonlinearity should be a string!")
        elif nonlinearity != "relu":
            warnings.warn("The current implementation only correctly supports `relu`")
        else:
            self.nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value

    def residual_branch(self, x: Tensor) -> Tensor:
        """Apply the residual branch."""
        for layer in self.components.values():
            x = layer(x)
        return x

    def identity_block(self, x: Tensor) -> Tensor:
        """Apply identity mapping"""
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        NOTE: Returns both the activated output and the 'activation pattern'.
        """

        identity = self.identity_block(x)
        branch_output = self.residual_branch(x)

        # Apply stochastic depth
        branch_output = stochastic_depth(
            branch_output, p=self.drop_prob, mode=self.stoch_mode, training=self.training
        )

        out = identity + branch_output
        activated_out = F.relu(out)

        # Calculate activation pattern (which elements are positive)
        # TODO: This currently only works for ReLU!
        activation_pattern = branch_output * (out > 0)

        return activated_out, activation_pattern


class LinearResidualBlock(BaseResidualBlock):
    """
    Residual block for ResMLP, with two layers
    NOTE: This implementation assumes that both layers are linear and have the same dimension.
    NOTE: The `channels` param refers to dimension of the linear layers.
    """

    def __init__(
        self,
        channels: int,
        bias: bool = False,
        batchnorm: bool = True,
        nonlinearity: str = "relu",
        drop_prob: float = 0.0,
        stoch_mode: str = "batch",
    ):
        super().__init__(
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
            in_channels=channels,
            out_channels=channels,
            drop_prob=drop_prob,
            stoch_mode=stoch_mode,
        )

        # First fully-connected layer
        fc1 = LinearLayer(
            in_channels=channels,
            out_channels=channels,
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
        )
        fc1.apply(LinearLayer.init_weights)
        self.components["fc1"] = fc1

        # Second fully-connected layer
        fc2 = LinearLayer(
            in_channels=channels,
            out_channels=channels,
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=None,
        )
        fc2.apply(LinearLayer.init_weights)
        self.components["fc2"] = fc2


class Conv2DResidualBlock(BaseResidualBlock):
    """
    Residual block for ConvResNet, consisting of two convolutional layers.
    NOTE: This implementation assumes that both layers are have the same channels.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        batchnorm: bool = True,
        nonlinearity: str = "relu",
        drop_prob: float = 0.0,
        stoch_mode: str = "batch",
    ):
        super().__init__(
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
            in_channels=channels,
            out_channels=channels,
            drop_prob=drop_prob,
            stoch_mode=stoch_mode,
        )
        self.kernel_size = kernel_size
        self.stride = stride

        # Determine if we need a projection shortcut (1x1 conv) for changing dimensions
        self.downsample = None
        if stride != 1:
            self.downsample = Conv2DLayer(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                nonlinearity=None,
                bias=False,
                batchnorm=batchnorm,
            )
            self.downsample.apply(Conv2DLayer.init_weights)

        # First convolutional layer
        padding = kernel_size // 2  # Same padding
        conv1 = Conv2DLayer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            nonlinearity=nonlinearity,
            bias=bias,
            batchnorm=batchnorm,
        )
        conv1.apply(Conv2DLayer.init_weights)
        self.components["conv1"] = conv1

        # Second convolutional layer
        conv2 = Conv2DLayer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,  # Always stride 1 for the second conv
            padding=padding,
            nonlinearity=None,
            bias=bias,
            batchnorm=batchnorm,
        )
        conv2.apply(Conv2DLayer.init_weights)
        self.components["conv2"] = conv2

    def identity_block(self, x: Tensor) -> Tensor:
        """Apply identity mapping with optional projection for dimension mismatch."""
        if self.downsample is not None:
            return self.downsample(x)
        return x
