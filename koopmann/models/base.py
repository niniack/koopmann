# base.py
__all__ = ["BaseTorchModel"]

from abc import ABC, abstractmethod
from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchinfo
from safetensors.torch import load_model, save_model
from torch import Tensor

from .utils import get_device, parse_safetensors_metadata


class BaseTorchModel(nn.Module, ABC):
    """Base PyTorch abstract class with common functionality."""

    @property
    @abstractmethod
    def modules(self) -> nn.Sequential:
        """Returns a container of model layers"""
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model"""
        pass

    @abstractmethod
    def get_fwd_activations(self, detach: bool = True) -> OrderedDict:
        """Returns forward activations from the layers of a model"""
        pass

    def hook_model(self) -> None:
        """Set up hooks for all modules that support it."""
        # Remove all previous hooks
        for module in self.modules():
            if hasattr(module, "remove_hook"):
                module.remove_hook()

        # Add back hooks
        for module in self.modules():
            if hasattr(module, "setup_hook"):
                module.setup_hook()

    @classmethod
    @abstractmethod
    def load_model(cls, file_path: Union[str, Path], **kwargs) -> Tuple[Any, Dict]:
        """
        Load model from a file.

        Returns:
            Tuple[Any, Dict]: The loaded model and its metadata.
        """
        pass

    @abstractmethod
    def save_model(self, file_path: Union[str, Path], **kwargs) -> None:
        """Save model to a file."""
        pass

    def summary(self, *input_size: int) -> str:
        """Generate a summary of the model architecture."""
        if input_size:
            return torchinfo.summary(
                self, input_size=input_size, row_settings=["var_names", "ascii_only"]
            )
        else:
            return torchinfo.summary(self, row_settings=["var_names", "ascii_only"])

    def generic_save_model(self, file_path: Union[str, Path], metadata: Dict[str, str]) -> None:
        """Generic implementation of save_model for child classes."""
        self.eval()  # Set model to evaluation mode before saving
        save_model(self, Path(file_path), metadata=metadata)

    @classmethod
    def generic_load_model(
        cls, file_path: Union[str, Path], model_params: Dict[str, Any], strict: bool = True
    ) -> Tuple[Any, Dict]:
        """Generic implementation of load_model for child classes."""
        # Assert path exists
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(**model_params)

        # Load weights
        load_model(model, file_path, device=get_device(), strict=strict)

        return model, metadata
