__all__ = ["Serializable"]

import inspect
import json
import os
from abc import ABC, abstractmethod
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
    def parse_safetensors_metadata(file_path: Union[str, Path]) -> Dict[str, str]:
        """Parse the model's metadata from the safetensors file."""

        # Convert Path to string if needed
        file_path = str(file_path)

        # safetensors files store an 8-byte little-endian header length followed
        # by a JSON blob. We only need the '__metadata__' key if present.
        header_size = 8

        # If file is too small to contain a header, return empty metadata.
        try:
            if os.stat(file_path).st_size <= header_size:
                return {}
        except OSError:
            return {}

        # Read metadata
        try:
            with open(file_path, "rb") as f:
                header_bytes = f.read(header_size)
                if len(header_bytes) != header_size:
                    return {}

                # Interpret the first 8 bytes as a little-endian integer
                # which gives the length of the following JSON header.
                header_len = int.from_bytes(header_bytes, "little", signed=False)
                headers = f.read(header_len)
                if len(headers) != header_len:
                    return {}

                try:
                    headers_json = json.loads(headers.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return {}

                # Extract the '__metadata__' dict if present. Values stored
                # in safetensors metadata are strings; we keep them as-is here
                # and convert types later in `_parse_metadata`.
                metadata = headers_json.get("__metadata__", {})
                if isinstance(metadata, dict):
                    return dict(metadata)
                # If metadata is present but not a dict, return empty.
                return {}
        except OSError:
            return {}

    def save_model(
        self, file_path: Union[str, Path], suffix: Optional[str] = None, **metadata
    ) -> Path:
        """Save model to file with metadata."""
        path = Path(file_path)

        # Determine if it's a directory or if it's missing an extension
        if path.is_dir() or not path.suffix:
            model_name = self.__class__.__name__.lower()

            # Apply suffix if provided
            # `suffix` is appended to the generated model name (if provided).
            # An empty string or None means no suffix is used.
            if suffix and suffix != "":
                model_name = f"{model_name}_{suffix}"

            filename = f"{model_name}.safetensors"

            # If path is a file without extension, use it as a prefix
            if not path.is_dir() and not path.suffix:
                filename = f"{path.name}.safetensors"
                path = path.parent

            # Create the full path
            final_path = path / filename

        # User provided a complete filename
        else:
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
        # Safetensors requires string metadata; cast here and leave parsing
        # of types to `_parse_metadata` when re-loading.
        string_metadata = {k: str(v) for k, v in combined_metadata.items()}

        # Save using safetensors
        st.save_model(self, final_path, metadata=string_metadata)

        return final_path

    @classmethod
    def load_model(cls, file_path: Union[str, Path], **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load a model from a safetensors file."""

        # Try parsing metadata
        try:
            metadata = cls.parse_safetensors_metadata(file_path)
            parsed_metadata = cls._parse_metadata(metadata)
        except Exception as e:
            raise ValueError(f"Failed to load metadata from {file_path}: {e}")

        # Get constructor signature and filter metadata to only include
        # explicit constructor parameters (exclude `self`, *args, **kwargs).
        # This avoids passing unexpected keys from metadata into `__init__`.
        sig = inspect.signature(cls.__init__)
        init_param_names = {
            name
            for name, param in sig.parameters.items()
            if name != "self"
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        init_kwargs = {k: v for k, v in parsed_metadata.items() if k in init_param_names}
        init_kwargs.update(kwargs)

        # Load model with kwargs
        try:
            model = cls(**init_kwargs)
            st.load_model(model, file_path, device=get_device())
        except Exception as e:
            raise ValueError(f"Failed to load model from {file_path}: {e}")

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
