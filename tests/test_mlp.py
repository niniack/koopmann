from pathlib import Path

import torch
from torch import testing

from koopmann.models.mlp import MLP
from koopmann.models.resmlp import ResMLP


def test_init_mlp():
    mlp = MLP(
        input_dimension=2,
        output_dimension=2,
        config=[8],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
    )

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
        batchnorm=True,
    )

    path = Path.joinpath(tmp_path, "autoencoder.safetensors")
    mlp.save_model(path)
    mlp_loaded, _ = MLP.load_model(file_path=path)

    testing.assert_close(
        mlp.modules[0].linear_layer.weight, mlp_loaded.modules[0].linear_layer.weight
    )


def test_insert_mlp_layer(tmp_path):
    mlp = MLP(
        input_dimension=2,
        output_dimension=2,
        config=[8, 6],
        nonlinearity="relu",
        bias=True,
    )

    expanded_dim = 25
    dummy_mlp = MLP(
        input_dimension=2,
        output_dimension=2,
        config=[8, 6, expanded_dim, 6],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
    )

    insert_index = len(mlp.modules) - 1
    mlp.insert_layer(index=insert_index, out_features=expanded_dim)
    mlp.insert_layer(index=insert_index + 1)

    assert mlp.full_config == dummy_mlp.full_config


########################################################


def test_init_resmlp():
    resmlp = ResMLP(
        input_dimension=2,
        output_dimension=2,
        config=[8],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
    )

    assert resmlp.modules[0].linear_layer.weight.shape == torch.Size([8, 2])
    assert resmlp.modules[0].linear_layer.bias.shape == torch.Size([8])

    assert resmlp.modules[-1].linear_layer.weight.shape == torch.Size([2, 8])
    assert resmlp.modules[-1].linear_layer.bias.shape == torch.Size([2])


def test_save_load_resmlp(tmp_path):
    resmlp = ResMLP(
        input_dimension=2,
        output_dimension=2,
        config=[8],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
    )

    path = Path.joinpath(tmp_path, "autoencoder.safetensors")
    resmlp.save_model(path)
    resmlp_loaded, _ = ResMLP.load_model(file_path=path)

    testing.assert_close(
        resmlp.modules[0].linear_layer.weight, resmlp_loaded.modules[0].linear_layer.weight
    )


def test_insert_resmlp_layer(tmp_path):
    resmlp = ResMLP(
        input_dimension=2,
        output_dimension=2,
        config=[8],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
    )

    expanded_dim = 25

    insert_index = len(resmlp.modules) - 1
    resmlp.insert_layer(index=insert_index, out_features=expanded_dim)
    resmlp.insert_layer(index=insert_index + 1)

    print(resmlp)
