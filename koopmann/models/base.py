# base.py
__all__ = ["BaseTorchModel"]

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch.nn as nn
import torchinfo
from torch import Tensor

from koopmann.mixins.serializable import Serializable


class BaseTorchModel(nn.Module, Serializable, ABC):
    """Base PyTorch abstract class with common functionality."""

    def __init__(self):
        super().__init__()
        self.components = nn.Sequential()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model"""
        pass

    def hook_model(self) -> "BaseTorchModel":
        """Set up hooks for all components that support it."""
        # Remove all previous hooks
        for component in self.components:
            if hasattr(component, "remove_hook"):
                component.remove_hook()

        # Add back hooks
        for component in self.components:
            if hasattr(component, "setup_hook"):
                component.setup_hook()

        return self

    def get_forward_activations(self, detach: bool = True) -> OrderedDict:
        """Returns forward activations from all hooked components in the model."""
        act_dict = OrderedDict()

        for i, component in enumerate(self.components):
            if hasattr(component, "is_hooked") and component.is_hooked:
                acts = component.forward_activations

                if isinstance(acts, tuple):
                    # For residual blocks
                    acts = acts[0]

                if detach and acts is not None:
                    acts = acts.detach()

                act_dict[i] = acts

        return act_dict

    def summary(self, *input_size: int) -> str:
        """Generate a summary of the model architecture."""
        if input_size:
            return torchinfo.summary(
                self, input_size=input_size, row_settings=["var_names", "ascii_only"]
            )
        else:
            return torchinfo.summary(self, row_settings=["var_names", "ascii_only"])
