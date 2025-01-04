__all__ = ["MLP"]


from ast import literal_eval
from collections import OrderedDict
from pathlib import Path

import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_model, save_model
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer
from koopmann.models.utils import StringtoClassNonlinearity, get_device, parse_safetensors_metadata


class MLP(BaseTorchModel):
    """
    Multi-layer perceptron.
    """

    def __init__(
        self,
        input_dimension: int = 2,
        output_dimension: int = 2,
        config: list = [8],  # Number of neurons per hidden layer.
        nonlinearity: str = "relu",
        bias: bool = True,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.config = config
        self.nonlinearity = nonlinearity
        nonlinearity = StringtoClassNonlinearity[nonlinearity].value
        self.bias = bias
        full_config = [input_dimension, *config, output_dimension]

        layers = [None] * (len(full_config) - 1)
        for i in range(len(full_config) - 1):
            layers[i] = LinearLayer(
                in_features=full_config[i],
                out_features=full_config[i + 1],
                nonlinearity=nonlinearity if not i == len(full_config) - 2 else None,
                batchnorm=True,
                bias=bias,
                hook=False,
            )
            layers[i].apply(LinearLayer.init_weights)
        self._features = nn.Sequential(*layers)

    def hook_model(self) -> None:
        # Remove all previous hooks
        for layer in self.modules:
            layer.remove_hook()

        # Add back hooks
        for layer in self.modules:
            layer.setup_hook()

    @property
    def modules(self) -> nn.Sequential:
        return self._features

    def forward(self, x: Float[Tensor, "batch features"]) -> Tensor:
        return self.modules(x)

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        activations = OrderedDict()
        for i, layer in enumerate(self.modules):
            if layer.is_hooked:
                activations[i] = (
                    layer.forward_activations if not detach else layer.forward_activations.detach()
                )

        return activations

    @classmethod
    def load_model(cls, file_path: str | Path):
        """Load model."""

        # Assert path
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(
            input_dimension=literal_eval(metadata["input_dimension"]),
            output_dimension=literal_eval(metadata["output_dimension"]),
            config=literal_eval(metadata["config"]),
            nonlinearity=metadata["nonlinearity"],
            bias=literal_eval(metadata["bias"]),
        )
        model.train()

        # Load weights
        load_model(model, file_path, device=get_device())

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model."""

        metadata = {
            "input_dimension": str(self.input_dimension),
            "output_dimension": str(self.output_dimension),
            "config": str(self.config),
            "nonlinearity": str(self.nonlinearity),
            "bias": str(self.bias),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        self.eval()
        save_model(self, Path(file_path), metadata=metadata)
