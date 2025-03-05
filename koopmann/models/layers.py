# layers.py
__all__ = ["LayerType", "Layer", "LinearLayer", "Conv2DLayer"]

from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .utils import StringtoClassNonlinearity, pad_act


class LayerType(Enum):
    """Defines the type of layer."""

    LINEAR = "linear"
    CONV2D = "conv2d"


class Layer(nn.Module, ABC):
    """Abstract base class for all layer types."""

    def __init__(self):
        super().__init__()
        self._forward_activations = None
        self._handle: Optional[RemovableHandle] = None
        self._is_hooked = False
        self.layers = nn.ModuleDict()

    @property
    @abstractmethod
    def layer_type(self) -> LayerType:
        """Returns the type of layer."""
        pass

    @property
    @abstractmethod
    def in_features(self) -> int:
        """Returns the input dimension/channels."""
        pass

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Returns the output dimension/channels."""
        pass

    @property
    def forward_activations(self) -> Tensor:
        """Returns tensor of forward activations from hook."""
        return self._forward_activations

    @property
    def is_hooked(self) -> bool:
        """Returns boolean indicating whether layer has hook."""
        return self._is_hooked

    def setup_hook(self):
        """Sets up a hook to capture activations."""
        # Remove any existing hook first
        self.remove_hook()

        def _hook(module, input, output):
            self._forward_activations = output

        self._is_hooked = True

        # Get the last layer in the execution order
        if hasattr(self, "_layer_order") and len(self._layer_order) > 0:
            last_layer_name = self._layer_order[-1]
            last_layer = self.layers[last_layer_name]
            self._handle = last_layer.register_forward_hook(_hook)
        else:
            # Fallback for any derived classes not using ModuleDict
            # Find the last layer by position
            modules = list(self.modules())
            if len(modules) > 1:  # Skip self
                last_module = modules[-1]
                self._handle = last_module.register_forward_hook(_hook)

    def remove_hook(self):
        """Tears down the hook."""
        self._is_hooked = False
        if self._handle:
            self._handle.remove()
            self._handle = None
        self._forward_activations = None

    def get_layer(self, name):
        if name in self.layers.keys():
            return self.layers[name]
        else:
            return None

    @classmethod
    def init_weights(cls, module: nn.Module):
        """Initialize weights for all supported module types."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class LinearLayer(Layer):
    """
    Linear layer with the option for a batch norm and an activation with a hook.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nonlinearity: Union[str, nn.Module, None],
        bias: bool = True,
        batchnorm: bool = False,
    ):
        # Call super().__init__() first
        super().__init__()

        # Create a ModuleDict instead of Sequential for named access
        self.layers = nn.ModuleDict()

        # Linear
        self.layers["linear"] = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

        # Batchnorm (optional)
        if batchnorm:
            self.layers["batchnorm"] = nn.BatchNorm1d(out_features)

        # Nonlinearity (optional)
        if nonlinearity is not None:
            if isinstance(nonlinearity, str):
                nonlinearity = StringtoClassNonlinearity[nonlinearity].value
            if nonlinearity:
                self.layers["nonlinearity"] = nonlinearity()

        # Store layer parameters
        self._in_features = in_features
        self._out_features = out_features
        self._nonlinearity = nonlinearity
        self._bias = bias
        self._batchnorm = batchnorm

        # Store execution order
        self._layer_order = ["linear"]
        if batchnorm:
            self._layer_order.append("batchnorm")
        if nonlinearity is not None and nonlinearity:
            self._layer_order.append("nonlinearity")

    @property
    def layer_type(self) -> LayerType:
        return LayerType.LINEAR

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.flatten(start_dim=1)

        # Process through layers in order
        for name in self._layer_order:
            x = self.layers[name](x)

        return x


class Conv2DLayer(Layer):
    """
    2D Convolutional layer with the option for a batch norm and an activation with a hook.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        nonlinearity: Union[str, nn.Module, None] = "relu",
        bias: bool = True,
        batchnorm: bool = False,
    ):
        # Call super().__init__() first
        super().__init__()

        # Create a ModuleDict instead of Sequential for named access
        self.layers = nn.ModuleDict()

        # Conv2d
        self.layers["conv"] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Batchnorm (optional)
        if batchnorm:
            self.layers["batchnorm"] = nn.BatchNorm2d(out_channels)

        # Nonlinearity (optional)
        if nonlinearity is not None:
            if isinstance(nonlinearity, str):
                nonlinearity = StringtoClassNonlinearity[nonlinearity].value
            if nonlinearity:
                self.layers["nonlinearity"] = nonlinearity()

        # Store layer parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._nonlinearity = nonlinearity
        self._bias = bias
        self._batchnorm = batchnorm

        # Store execution order
        self._layer_order = ["conv"]
        if batchnorm:
            self._layer_order.append("batchnorm")
        if nonlinearity is not None and nonlinearity:
            self._layer_order.append("nonlinearity")

    @property
    def layer_type(self) -> LayerType:
        return LayerType.CONV2D

    @property
    def in_features(self) -> int:
        return self._in_channels

    @property
    def out_features(self) -> int:
        return self._out_channels

    def forward(self, x):
        # Ensure input has the right dimensions (N, C, H, W)
        if len(x.shape) == 2:
            # If it's (N, F), reshape to (N, C, H, W) by assuming square image
            batch_size = x.shape[0]
            features = x.shape[1]

            # Determine dimensions - assumes square image
            side_length = int(torch.sqrt(torch.tensor(features / self._in_channels)))

            if side_length**2 * self._in_channels != features:
                raise ValueError(
                    f"Cannot reshape {features} features into square image with {self._in_channels} channels"
                )

            x = x.view(batch_size, self._in_channels, side_length, side_length)

        # Process through layers in order
        for name in self._layer_order:
            x = self.layers[name](x)

        return x
