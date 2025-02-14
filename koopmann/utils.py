import random

import numpy as np
import torch
from jaxtyping import Int
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassAccuracy


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
    model: nn.Module,  # Neural network model
    dataset: Dataset,  # Dataset to evaluate
    batch_size: Int = 256,  # Batch size for data loading
) -> torch.Tensor:
    """Computes the classification accuracy of a model on a given dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataset (Dataset): The dataset to compute accuracy on.
        batch_size (Int, optional): The batch size for DataLoader. Defaults to 256.

    Returns:
        torch.Tensor: The computed accuracy as a tensor.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size)
    metric = MulticlassAccuracy()

    for batch in dataloader:
        input, target = batch
        output = model(input)  # Avoid explicit `forward()` call; use `model()`
        metric.update(output, target.squeeze())  # Ensure target shape compatibility

    return metric.compute()
