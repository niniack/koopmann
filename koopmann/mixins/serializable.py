__all__ = ["Serializable"]

import inspect
import json
import os
from abc import ABC, abstractmethod
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import safetensors.torch as st
import torch.nn as nn

from koopmann.utils import get_device


class Serializable(ABC):
    """Mixin that adds serialization capabilities."""

    @abstractmethod
    def _get_basic_metadata(self) -> Dict[str, Any]:
        """
        This method MUST be implemented by all classes inheriting from Serializable.
        It should return a dictionary containing all parameters needed to reconstruct
        the model (e.g., dimensions, configurations, hyperparameters).

        """
        pass

    @staticmethod
    def parse_safetensors_metadata(file_path: Union[str, Path]) -> dict:
        """Parse the model's metadata from the safetensors file."""

        # Convert Path to string if needed
        file_path = str(file_path)

        header_size = 8
        meta_data = {}
        if os.stat(file_path).st_size > header_size:
            with open(file_path, "rb") as f:
                b8 = f.read(header_size)
                if len(b8) == header_size:
                    header_len = int.from_bytes(b8, "little", signed=False)
                    headers = f.read(header_len)
                    if len(headers) == header_len:
                        meta_data = sorted(
                            json.loads(headers.decode("utf-8"))
                            .get("__metadata__", meta_data)
                            .items()
                        )
        meta_data_dict = {}
        for k, v in meta_data:
            meta_data_dict[k] = v
        return meta_data_dict

    def save_model(self, file_path: Union[str, Path], **metadata) -> None:
        """Save model to file with metadata."""
        path = Path(file_path)

        # Determine if it's a directory or if it's missing an extension
        if path.is_dir() or not path.suffix:
            model_name = self.__class__.__name__.lower()
            filename = f"{model_name}.safetensors"

            # If path is a file without extension, use it as a prefix
            if not path.is_dir() and not path.suffix:
                filename = f"{path.name}.safetensors"
                path = path.parent

            # Create the full path
            final_path = path / filename
        else:
            # User provided a complete filename
            final_path = path
            # Ensure it has the correct extension
            if final_path.suffix != ".safetensors":
                final_path = final_path.with_suffix(".safetensors")

        # Ensure directory exists
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect basic metadata
        basic_metadata = self._get_basic_metadata()

        # Add standard fields
        standard_metadata = {
            "model_class": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
        }

        # Merge all metadata (user-provided overrides everything)
        combined_metadata = {**standard_metadata, **basic_metadata, **metadata}

        # Convert all metadata to strings for safetensors
        string_metadata = {k: str(v) for k, v in combined_metadata.items()}

        # Save using safetensors
        st.save_model(self, final_path, metadata=string_metadata)

        return final_path  # Return the actual path used, so the user knows where it was saved

    @classmethod
    def load_model(cls, file_path: Union[str, Path], **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model from file."""
        # Parse metadata
        metadata = cls.parse_safetensors_metadata(file_path)

        # Convert metadata values from strings
        parsed_metadata = cls._parse_metadata(metadata)

        # Filter metadata to only include constructor parameters

        init_params = inspect.signature(cls.__init__).parameters.keys()
        init_params = [p for p in init_params if p != "self"]

        # Filter metadata to only include init parameters
        init_kwargs = {k: v for k, v in parsed_metadata.items() if k in init_params}

        # Update with explicit kwargs (which override metadata)
        init_kwargs.update(kwargs)

        # Create model instance
        model = cls(**init_kwargs)

        # Load weights
        st.load_model(model, file_path, device=get_device())

        return model, parsed_metadata

    @classmethod
    def _parse_metadata(cls, metadata: Dict[str, str]) -> Dict[str, Any]:
        """Parse metadata values from strings to appropriate types."""
        parsed = {}
        for key, value in metadata.items():
            try:
                parsed[key] = literal_eval(value)
            except (ValueError, SyntaxError):
                parsed[key] = value

        return parsed
