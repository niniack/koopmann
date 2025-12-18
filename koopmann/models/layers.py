# layers.py
__all__ = ["Layer", "LinearLayer"]

from abc import ABC
from typing import Optional

import torch.nn as nn

from koopmann.mixins.hookable import Hookable
from koopmann.models.utils import StringtoClassNonlinearity


class Layer(nn.Module, ABC, Hookable):
    """Abstract base class for all layer types."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nonlinearity: Optional[str],
    ):
        nn.Module.__init__(self)  # Initialize nn.Module
        Hookable.__init__(self)  # Initialize Hookable

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.components = nn.ModuleDict()

        # Handle nonlinearity
        if nonlinearity is None:
            pass
        elif not isinstance(nonlinearity, str):
            raise ValueError("Nonlinearity should be a string!")
        else:
            self.nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value

    def get_component(self, name):
        if name in self.components.keys():
            return self.components[name]
        else:
            return None

    @classmethod
    def init_weights(cls, module: nn.Module):
        """Initialize weights"""

        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class LinearLayer(Layer):
    """
    Linear layer with built-in batchnorm and nonlinearity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        batchnorm: bool | None = False,
        nonlinearity: Optional[str] = "relu",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            nonlinearity=nonlinearity,
        )

        # Linear component
        self.components["linear"] = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
        )

        # Batchnorm (optional)
        if batchnorm:
            self.components["batchnorm"] = nn.BatchNorm1d(out_channels)

        # Nonlinearity (optional)
        if nonlinearity is not None:
            self.components["nonlinearity"] = self.nonlinearity_module()

    def forward(self, x):
        # Flatten
        if len(x.shape) > 2:
            x = x.flatten(start_dim=1)

        for component in self.components.values():
            x = component(x)

        return x


class Conv2dLayer(Layer):
    """
    2D convolutional layer with optional BatchNorm2d and nonlinearity.

    Expects input of shape (N, C_in, H, W) and returns (N, C_out, H_out, W_out).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        batchnorm: bool | None = False,
        nonlinearity: Optional[str] = "relu",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            nonlinearity=nonlinearity,
        )

        # Conv2d component
        self.components["conv"] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Batchnorm (optional)
        if batchnorm:
            self.components["batchnorm"] = nn.BatchNorm2d(out_channels)

        # Nonlinearity (optional)
        if nonlinearity is not None:
            self.components["nonlinearity"] = self.nonlinearity_module()

    def forward(self, x):
        # x: (N, C_in, H, W)
        for component in self.components.values():
            x = component(x)
        # x: (N, C_out, H_out, W_out)
        return x
