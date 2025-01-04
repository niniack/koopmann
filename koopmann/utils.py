import random

import numpy as np
import torch
from jaxtyping import Int
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassAccuracy


def set_seed(seed: int) -> None:
    """Set a random seed for random, PyTorch and NumPy."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def compute_model_accuracy(
    model: nn.Module,  # Model
    dataset: Dataset,  # Dataset
    batch_size: Int = 256,
):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    metric = MulticlassAccuracy()
    for i, batch in enumerate(dataloader):
        input, target = batch
        output = model.forward(input)
        metric.update(output, target.squeeze())

    return metric.compute()
