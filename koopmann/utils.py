import random

import numpy as np
import torch
from torch import device, nn
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassAccuracy


def get_device() -> device:
    """Return fastest device."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def set_seed(seed: int) -> None:
    """Set a random seed for reproducibility across PyTorch, NumPy, and Python's `random` module.

    Args:
        seed (int): The random seed value.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def compute_model_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device,
) -> torch.Tensor:
    """Computes the classification accuracy of a model on a given dataset."""
    model.eval()
    metric = MulticlassAccuracy()

    for batch in dataloader:
        input, target = batch
        input, target = input.to(device), target.to(device)
        output = model(input)  # Avoid explicit `forward()` call; use `model()`
        metric.update(output, target.squeeze())  # Ensure target shape compatibility

    return metric.compute()
