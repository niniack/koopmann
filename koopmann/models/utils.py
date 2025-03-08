__all__ = ["StringtoClassNonlinearity"]

import json
import os
from enum import Enum

import torch
import torch.nn.functional as F
from torch import device, nn


class StringtoClassNonlinearity(Enum):
    """Convert string represenation of nonlinearity to `torch.nn` class"""

    relu = nn.ReLU
    leaky_relu = nn.LeakyReLU
    sigmoid = nn.Sigmoid
    gelu = nn.GELU
    tanh = nn.Tanh


def eigeninit(weight: torch.Tensor, theta: float = 0.7) -> None:
    """
    Initialization for Koopman matrix weights.

    The magnitudes of the eigenvalues are set to be between 0 and 1,
    with theta determining the probability of 1.
    Directly modifies the input tensor `weight` in-place.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(weight)

    # Represent eigenvalues in polar coordinates
    polar_mags = torch.abs(eigenvalues)
    polar_phase = torch.angle(eigenvalues)

    # Sample with slab-spike distribution
    num_unique = len(torch.unique(polar_mags, sorted=False))
    bernoulli_trials = torch.distributions.Bernoulli(theta).sample([num_unique])
    uniform_trials = torch.distributions.Uniform(0, 1).sample([num_unique]) * (1 - bernoulli_trials)
    result_trials = bernoulli_trials + uniform_trials

    # Sample new magnitudes, while preserving conjugate pairs!
    new_polar_mags = torch.empty_like(polar_mags)
    new_polar_mags[0] = result_trials[0]
    j = 1
    for i in range(1, new_polar_mags.size(0)):
        if torch.isclose(polar_mags[i], polar_mags[i - 1]):
            new_polar_mags[i] = new_polar_mags[i - 1]
        else:
            new_polar_mags[i] = result_trials[j]
            j += 1

    # Rebuild eigenvalues with new magnitudes
    new_eigenvalues = torch.polar(new_polar_mags, polar_phase)

    # Construct new weight matrix in-place
    with torch.no_grad():  # Precaution
        weight.copy_(
            torch.real(eigenvectors @ torch.diag(new_eigenvalues) @ torch.linalg.inv(eigenvectors))
        )
