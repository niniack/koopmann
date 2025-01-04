__all__ = ["BaseTorchModel"]

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torchinfo


class BaseTorchModel(nn.Module, ABC):
    """Base PyTorch abstract class."""

    @property
    @abstractmethod
    def modules(self) -> nn.Sequential:
        """Returns a container of hidden layers"""
        pass

    @abstractmethod
    def forward(self) -> torch.tensor:
        """Forward pass through the model"""
        pass

    @abstractmethod
    def get_fwd_activations(self) -> OrderedDict:
        """Returns forward activations from the layers of a model"""
        pass

    @abstractmethod
    def load_model(self):
        """Load model."""
        pass

    @abstractmethod
    def save_model(self):
        """Save model."""
        pass

    def summary(self):
        """"""
        return torchinfo.summary(self, row_settings=["var_names", "ascii_only"])
