from pathlib import Path

import torch
from torch import testing

from koopmann.models.mlp import MLP


def test_init_mlp():
    mlp = MLP(input_dimension=2, output_dimension=2, config=[8], nonlinearity="relu", bias=True)

    assert mlp.modules[0].linear_layer.weight.shape == torch.Size([8, 2])
    assert mlp.modules[0].linear_layer.bias.shape == torch.Size([8])

    assert mlp.modules[-1].linear_layer.weight.shape == torch.Size([2, 8])
    assert mlp.modules[-1].linear_layer.bias.shape == torch.Size([2])


def test_save_load_mlp(tmp_path):
    mlp = MLP(
        input_dimension=2,
        output_dimension=2,
        config=[8],
        nonlinearity="relu",
        bias=True,
    )

    path = Path.joinpath(tmp_path, "autoencoder.safetensors")
    mlp.save_model(path)
    mlp_loaded = MLP.load_model(file_path=path)

    testing.assert_close(
        mlp.modules[0].linear_layer.weight, mlp_loaded.modules[0].linear_layer.weight
    )
