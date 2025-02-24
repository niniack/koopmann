__all__ = ["LinearLayer", "Conv2DLayer"]

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .utils import StringtoClassNonlinearity


class Layer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def kwargs(self) -> dict:
        """Return keyword arguments dictionary."""
        pass

    @property
    @abstractmethod
    def forward_activations(self) -> Tensor:
        """Returns tensor of forward activations from hook."""
        pass

    @property
    @abstractmethod
    def is_hooked(self) -> bool:
        """Returns boolean indicating whether layer has hook."""
        pass

    @abstractmethod
    def setup_hook(self) -> bool:
        """Sets up a hook."""
        pass

    @abstractmethod
    def remove_hook(self) -> bool:
        """Tears down the hook."""
        pass


#######################################################################################
####################################### LINEAR ########################################
#######################################################################################


class LinearLayer(Layer):
    """
    Linear layer with the option for a batch norm and an activation with a hook.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nonlinearity: str | nn.Module,
        bias: bool = True,
        batchnorm: bool = False,
        hook: bool = False,
    ):
        super().__init__()

        self.layers = nn.Sequential()

        # Linear
        self.layers.add_module(
            "linear", nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        )

        # Batchnorm
        if batchnorm:
            self.layers.add_module("batchnorm", nn.BatchNorm1d(out_features))

        # Nonlinearity
        if isinstance(nonlinearity, str):
            nonlinearity = StringtoClassNonlinearity[nonlinearity].value

        if nonlinearity:
            self.layers.add_module("nonlinearity", nonlinearity())

        self._kwargs = OrderedDict(
            {
                "in_features": in_features,
                "out_features": out_features,
                "nonlinearity": nonlinearity,
                "bias": bias,
                "hook": hook,
            }
        )

        self._forward_activations: Tensor = None

        # Hook
        self._handle: RemovableHandle = None
        if hook:
            self.setup_hook()

    @property
    def kwargs(self) -> dict:
        """Returns keyword arguments dictionary."""
        return self._kwargs

    @property
    def in_features(self) -> dict:
        """Returns input dimension."""
        return self._kwargs["in_features"]

    @property
    def out_features(self) -> dict:
        """Returns output dimension."""
        return self._kwargs["out_features"]

    @property
    def linear_layer(self) -> dict:
        """Returns input dimension."""
        return self.layers[0]

    @property
    def forward_activations(self) -> Tensor:
        """Returns tensor of forward activations from hook."""
        return self._forward_activations

    @property
    def is_hooked(self) -> bool:
        """Returns boolean indicating whether layer has hook."""
        return self._kwargs["hook"]

    def setup_hook(self):
        """Sets up a hook."""

        def _hook(module, input, output):
            self._forward_activations = output

        self._kwargs["hook"] = True

        layer = self.layers[-1]
        self._handle = layer.register_forward_hook(_hook)

    def remove_hook(self):
        """Tears down the hook."""
        self._kwargs["hook"] = False
        if self._handle:
            self._handle.remove()

    def remove_nonlinearity(self):
        self._kwargs["nonlinearity"] = None
        self.layers = nn.Sequential(
            OrderedDict(
                (name, layer)
                for name, layer in self.layers.named_children()
                if name != "nonlinearity"
            )
        )

    def update_nonlinearity(self, nonlinearity, **kwargs):
        self._kwargs["nonlinearity"] = nonlinearity
        new_nonlinearity_layer = StringtoClassNonlinearity[nonlinearity].value
        new_ordered_dict = OrderedDict()
        for name, layer in self.layers.named_children():
            if name == "nonlinearity":
                continue
            new_ordered_dict[name] = layer
        new_ordered_dict["nonlinearity"] = new_nonlinearity_layer(**kwargs)
        self.layers = nn.Sequential(new_ordered_dict)

    def forward(self, x):
        self._forward_activations = None
        x = x.flatten(start_dim=1)
        return self.layers(x)

    @staticmethod
    def init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)


#######################################################################################
####################################### CONV2D ########################################
#######################################################################################
class Conv2DLayer(Layer):
    """
    Conv2D layer with the option for batch norm, an activation, a hook, and transpose.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        nonlinearity: str | nn.Module = None,
        bias: bool = True,
        batchnorm: bool = False,
        hook: bool = False,
        transpose: bool = False,
    ):
        super().__init__()

        self.layers = nn.Sequential()

        # Conv2D or ConvTranspose2D
        if transpose:
            self.layers.add_module(
                "conv2d",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=0,
                    bias=bias,
                ),
            )
        else:
            self.layers.add_module(
                "conv2d",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
            )

        # Batchnorm
        if batchnorm:
            self.layers.add_module("batchnorm", nn.BatchNorm2d(out_channels))

        # Nonlinearity
        if isinstance(nonlinearity, str):
            nonlinearity = StringtoClassNonlinearity[nonlinearity].value

        if nonlinearity:
            self.layers.add_module("nonlinearity", nonlinearity())

        self._kwargs = OrderedDict(
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "nonlinearity": nonlinearity,
                "bias": bias,
                "hook": hook,
                "transpose": transpose,
            }
        )

        self._forward_activations: Tensor = None

        # Hook
        self._handle: RemovableHandle = None
        if hook:
            self.setup_hook()

    @property
    def kwargs(self) -> dict:
        """Returns keyword arguments dictionary."""
        return self._kwargs

    @property
    def in_channels(self) -> int:
        """Returns input channels."""
        return self._kwargs["in_channels"]

    @property
    def out_channels(self) -> int:
        """Returns output channels."""
        return self._kwargs["out_channels"]

    @property
    def conv_layer(self) -> nn.Module:
        """Returns the Conv2D or ConvTranspose2D layer."""
        return self.layers[0]

    @property
    def forward_activations(self) -> Tensor:
        """Returns tensor of forward activations from hook."""
        return self._forward_activations

    @property
    def is_hooked(self) -> bool:
        """Returns boolean indicating whether layer has hook."""
        return self._kwargs["hook"]

    def setup_hook(self):
        """Sets up a hook."""

        def _hook(module, input, output):
            self._forward_activations = output

        self._kwargs["hook"] = True

        layer = self.layers[-1]
        self._handle = layer.register_forward_hook(_hook)

    def remove_hook(self):
        """Tears down the hook."""
        self._kwargs["hook"] = False
        if self._handle:
            self._handle.remove()

    def forward(self, x):
        self._forward_activations = None
        return self.layers(x)

    @staticmethod
    def init_weights(module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)
