__all__ = ["StringtoClassNonlinearity", "get_device"]

import json
import os
from enum import Enum

import torch
import torch.nn.functional as F
from torch import device, nn


class StringtoClassNonlinearity(Enum):
    """Convert string represenation of nonlinearity to `torch.nn` class"""

    relu = nn.ReLU
    leakyrelu = nn.LeakyReLU
    sigmoid = nn.Sigmoid
    gelu = nn.GELU
    tanh = nn.Tanh


def pad_act(x, target_size):
    current_size = x.size(1)
    if current_size < target_size:
        pad_size = target_size - current_size
        x = F.pad(x, (0, pad_size), mode="constant", value=0)

    return x


def get_device() -> device:
    """Return fastest device."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def parse_safetensors_metadata(file_path: str) -> dict:
    """Parse the model's metadata from the safetensors file."""
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
                        json.loads(headers.decode("utf-8")).get("__metadata__", meta_data).items()
                    )
    meta_data_dict = {}
    for k, v in meta_data:
        meta_data_dict[k] = v
    return meta_data_dict
