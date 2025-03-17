__all__ = [
    "AnalyticEncoder",
]

import warnings
from collections import namedtuple
from typing import Any, Optional

import torch
import torch.nn as nn
from pydantic import ValidationError, validate_call
from pydantic.types import NonNegativeInt, PositiveInt
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import Layer, LinearLayer
from koopmann.models.utils import eigeninit


class AnalyticEncoder(BaseTorchModel):
    """
    Analytic encoder model.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_config: list[int],
        bias: bool,
        batchnorm: bool,
        nonlinearity: str,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_config = hidden_config
        self.bias = bias
        self.batchnorm = batchnorm
        self.nonlinearity = nonlinearity
        self.full_config = [in_features, *hidden_config, out_features]

        warnings.warn("This constructor currently *only* works for vanilla MLPs.")

        # Weights list
        self.layers_list = nn.ModuleList()
        for i in range(len(self.full_config) - 1):
            # For the last layer, we don't use a nonlinearity
            layer_nonlinearity = None if i == len(self.full_config) - 2 else nonlinearity

            layer = LinearLayer(
                in_channels=self.full_config[i],
                out_channels=self.full_config[i + 1],
                bias=bias,
                batchnorm=batchnorm,
                nonlinearity=layer_nonlinearity,
            )
            self.layers_list.append(layer)
        self.num_layers = len(self.layers_list)

    @classmethod
    def from_model(cls, model: BaseTorchModel) -> BaseTorchModel:
        model.eval()
        encoder = cls(
            in_features=model.in_features,
            out_features=model.out_features,
            hidden_config=model.hidden_config,
            bias=model.bias,
            batchnorm=model.batchnorm,
            nonlinearity=model.nonlinearity,
        )

        encoder._collect_layers(model)
        return encoder

    def _collect_layers(self, model: BaseTorchModel):
        model_layers_list = nn.ModuleList()
        for component in model.components:
            if isinstance(component, Layer):
                model_layers_list.append(component)
            elif hasattr(component, "components"):
                child_list = self._collect_layers(component)
                model_layers_list.extend(child_list)

        assert len(model_layers_list) == len(self.layers_list)
        self.layers_list = model_layers_list

    @validate_call
    def _compute_collapsed_weight(
        self, start: Optional[NonNegativeInt] = 0, end: Optional[PositiveInt] = None
    ):
        """
        Follows slicing convention, `start` is inclusive and `end` is non-inclusive.
        """
        # Input validation
        if end is None:
            end = self.num_layers
        elif end > self.num_layers:
            raise ValueError(f"`end`: {end} cannot exceed {self.num_layers}")

        if start > end:
            raise ValueError(f"`start`: {start} exceeds `end`: {end}")
        elif start == end:
            return torch.eye(self.layers_list[start - 1].in_channels)

        # Initialize identity with shape of `in_channels`
        identity = torch.eye(self.layers_list[start].in_channels)

        # Collect all weights
        # [W_0, W_1, ... , W_{t-1}]
        weights = [layer.components.linear.weight for layer in self.layers_list[start:end]]

        # Reverse and append identity
        # [W_{t-1}, ... , W_1, W_0, I]
        weights = weights[::-1]
        weights.append(identity)

        # Chain multiply all layers
        # W_{t-1} @ ... @ W_1 @ W_0 @ I
        Q_weight = torch.linalg.multi_dot(weights)

        return Q_weight

    def compute_full_q(self):
        return self._compute_collapsed_weight()

    @validate_call
    def compute_c_vectors(self, t_step: PositiveInt) -> list[Tensor]:
        # Input validation
        if t_step > self.num_layers:
            raise ValueError(f"`t_step`: {t_step} cannot exceed {self.num_layers}")

        # Get t-1 layer for its bias shape
        final_bias_eye = torch.eye(self.layers_list[t_step - 1].out_channels)

        # Compute collapsed weights
        # We don't include the full collapse, which starts from 0
        # [Q_{t-2}, ... , Q_0, I]
        q_weights = [self._compute_collapsed_weight(start=i, end=t_step) for i in range(1, t_step)]
        q_weights.append(final_bias_eye)

        # Collect biases
        # [b_0, b_1, .. , b_{t-1}]
        biases = []
        for layer in self.layers_list[:t_step]:
            bias = layer.components.linear.bias
            if bias is not None:
                biases.append(bias)
            else:
                biases.append(torch.zeros(layer.out_channels))

        # Compute product
        # [Q_{t-2} @ b_0, ... , I @ b_{t-1}]
        q_bias_product = [torch.matmul(q, b) for q, b in zip(q_weights, biases)]

        # Canonical suffix sum
        ############
        c_len = len(q_bias_product)

        # Preallocate space
        c_vectors = [0] * c_len

        # Last element is the final bias I @ b_{t-1}
        c_vectors[c_len - 1] = q_bias_product[c_len - 1]

        # Iterate backwards and add previous + Q@b
        for i in range(len(q_bias_product) - 2, -1, -1):
            c_vectors[i] = c_vectors[i + 1] + q_bias_product[i]
        ############

        return c_vectors

    def build_operator(self):
        # Get the full Q matrix and c vectors
        full_q = self.compute_full_q()
        c_vectors = self.compute_c_vectors(t_step=self.num_layers)

        # Get the dimensions
        q_rows, q_cols = full_q.shape
        dtype = full_q.dtype
        device = full_q.device

        # Find number of rows and columsn
        num_blocks = len(c_vectors) + 1  # c vectors plus the final zero block
        total_rows = num_blocks * q_rows
        total_cols = q_cols + 1

        # For sparse constructor
        indices = []
        values = []

        # Add entries from the Q matrix (top-left block)
        q_nonzero = full_q.nonzero()
        for idx in range(q_nonzero.shape[0]):
            i, j = q_nonzero[idx, 0].item(), q_nonzero[idx, 1].item()
            indices.append([i, j])
            values.append(full_q[i, j].item())

        # Add entries from the c vectors (along the right column)
        for block_idx, c_vector in enumerate(c_vectors):
            row_offset = block_idx * q_rows
            c_nonzero = c_vector.nonzero().squeeze(1)  # Get indices of non-zero elements

            for i in c_nonzero:
                i_item = i.item()
                indices.append([row_offset + i_item, q_cols])
                values.append(c_vector[i_item].item())

        # Convert to tensors for the sparse constructor
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device).t()
        values_tensor = torch.tensor(values, dtype=dtype, device=device)

        # Create the sparse tensor
        sparse_operator = torch.sparse_coo_tensor(
            indices_tensor, values_tensor, (total_rows, total_cols), dtype=dtype, device=device
        )

        return sparse_operator

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        ones = torch.ones((x.size(0), 1))

        x_obs = torch.cat((x, ones), dim=-1)
        sparse_operator = self.build_operator()

        res = torch.matmul(x_obs, sparse_operator.T)
        return res

    def _get_basic_metadata(self) -> dict[str, Any]:
        """Get model-specific metadata for serialization."""
        return {}
