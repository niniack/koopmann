# layers.py
__all__ = ["Layer", "LinearLayer", "Conv2DLayer"]

import math
from abc import ABC
from typing import Optional, Tuple, Union

import torch.nn as nn

from koopmann.mixins.hookable import Hookable
from koopmann.models.utils import StringtoClassNonlinearity


class Layer(nn.Module, ABC, Hookable):
    """Abstract base class for all layer types."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        batchnorm: bool,
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
        batchnorm: bool = False,
        nonlinearity: Optional[str] = "relu",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            batchnorm=batchnorm,
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


class LoRALinearLayer(Layer):
    """
    Linear layer with low-rank decomposition, built-in batchnorm and nonlinearity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: Optional[str] = "relu",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
        )

        self.rank = rank

        # First projection: in_channels -> rank
        self.components["lora_down"] = nn.Linear(
            in_features=in_channels,
            out_features=rank,
            bias=False,
        )

        # Second projection: rank -> out_channels
        self.components["lora_up"] = nn.Linear(
            in_features=rank,
            out_features=out_channels,
            bias=bias,
        )

        # Batchnorm (optional)
        if batchnorm:
            self.components["batchnorm"] = nn.BatchNorm1d(out_channels)

        # Nonlinearity (optional)
        if nonlinearity is not None:
            self.components["nonlinearity"] = self.nonlinearity_module()

        # Apply weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize down projection with small random values
        nn.init.kaiming_uniform_(self.components["lora_down"].weight, a=math.sqrt(5))
        # Initialize up projection with zeros for stable training start
        nn.init.kaiming_uniform_(self.components["lora_up"].weight, a=math.sqrt(5))
        # Initialize bias if present
        if self.components["lora_up"].bias is not None:
            bound = 1 / math.sqrt(self.rank)
            nn.init.uniform_(self.components["lora_up"].bias, -bound, bound)

    def forward(self, x):
        # Flatten
        if len(x.shape) > 2:
            x = x.flatten(start_dim=1)

        for component in self.components.values():
            x = component(x)

        return x


class Conv2DLayer(Layer):
    """
    2D Convolutional layer with built-in batchnorm and nonlinearity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: Optional[str] = "relu",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Conv2d
        self.components["conv"] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Batchnorm (optional)
        if batchnorm:
            self.components["batchnorm"] = nn.BatchNorm2d(out_channels)

        # Nonlinearity (optional)
        if nonlinearity is not None:
            self.components["nonlinearity"] = self.nonlinearity_module()

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("Expects 4D input!")

        for component in self.components.values():
            x = component(x)

        return x


class Conv1DLayer(Layer):
    """
    1D Convolutional layer with built-in batchnorm and nonlinearity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: Optional[str] = "relu",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Conv1d
        self.components["conv"] = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Batchnorm (optional)
        if batchnorm:
            self.components["batchnorm"] = nn.BatchNorm1d(out_channels)

        # Nonlinearity (optional)
        if nonlinearity is not None:
            self.components["nonlinearity"] = self.nonlinearity_module()

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError("Expects 3D input!")

        for component in self.components.values():
            x = component(x)

        return x
