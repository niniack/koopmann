# processor.py

from typing import Tuple

import torch
from torch import Tensor


class Processor:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

        self.mu_a: Tensor | None = None
        self.std_a: Tensor | None = None
        self.mu_b: Tensor | None = None
        self.std_b: Tensor | None = None
        self.R: Tensor | None = None  # maps a_norm -> b_norm

    @torch.no_grad()
    def fit(self, a: Tensor, b: Tensor) -> None:
        assert a.ndim == 2 and b.ndim == 2
        assert a.shape == b.shape

        # per-feature stats
        self.mu_a = a.mean(dim=0)
        self.std_a = a.std(dim=0, correction=1).clamp_min(self.eps)

        self.mu_b = b.mean(dim=0)
        self.std_b = b.std(dim=0, correction=1).clamp_min(self.eps)

        # normalize using those stats
        a_norm = (a - self.mu_a) / self.std_a
        b_norm = (b - self.mu_b) / self.std_b

        # Procrustes: a_norm @ R ≈ b_norm
        C = a_norm.T @ b_norm  # (D, D)
        U, _, Vh = torch.linalg.svd(C, full_matrices=False)
        self.R = U @ Vh  # (D, D)

    def normalize_a(self, x: Tensor) -> Tensor:
        assert self.mu_a is not None, "call fit() first"
        mu = self.mu_a.to(x.device)
        std = self.std_a.to(x.device)
        return (x - mu) / std

    def normalize_b(self, x: Tensor) -> Tensor:
        assert self.mu_b is not None, "call fit() first"
        mu = self.mu_b.to(x.device)
        std = self.std_b.to(x.device)
        return (x - mu) / std

    def denormalize_a(self, x_norm: Tensor) -> Tensor:
        assert self.mu_a is not None, "call fit() first"
        mu = self.mu_a.to(x_norm.device)
        std = self.std_a.to(x_norm.device)
        return x_norm * std + mu

    def denormalize_b(self, x_norm: Tensor) -> Tensor:
        assert self.mu_b is not None, "call fit() first"
        mu = self.mu_b.to(x_norm.device)
        std = self.std_b.to(x_norm.device)
        return x_norm * std + mu

    def align(self, a_norm: Tensor, b_norm: Tensor) -> Tuple[Tensor, Tensor]:
        assert self.R is not None, "call fit() first"
        R = self.R.to(a_norm.device)
        b_aligned = b_norm @ R.T  # align b into a's frame
        return a_norm, b_aligned

    def undo_align(self, a_norm: Tensor, b_aligned: Tensor) -> Tuple[Tensor, Tensor]:
        assert self.R is not None, "call fit() first"
        R = self.R.to(a_norm.device)
        b_norm = b_aligned @ R  # approx original b_norm
        return a_norm, b_norm

    def transform(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        a_norm = self.normalize_a(a)
        b_norm = self.normalize_b(b)
        return self.align(a_norm, b_norm)

    def fit_transform(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        self.fit(a, b)
        return self.transform(a, b)
