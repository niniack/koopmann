__all__ = ["LinearLayer"]

from abc import ABC, abstractmethod
from collections import OrderedDict

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
