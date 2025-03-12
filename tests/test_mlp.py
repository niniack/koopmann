from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.mlp import MLP


@pytest.mark.parametrize("in_features", [2, 10])
@pytest.mark.parametrize("out_features", [2, 5])
@pytest.mark.parametrize("hidden_config", [[8], [8, 16], []])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("nonlinearity", ["relu", "leaky_relu"])
def test_init_mlp(
    in_features,
    out_features,
    hidden_config,
    bias,
    batchnorm,
    nonlinearity,
):
    # Create MLP with various configurations
    mlp = MLP(
        in_features=in_features,
        out_features=out_features,
        hidden_config=hidden_config,
        bias=bias,
        batchnorm=batchnorm,
        nonlinearity=nonlinearity,
    )

    # Verify full configuration
    expected_full_config = [in_features, *hidden_config, out_features]
    assert mlp.full_config == expected_full_config

    # Check number of layers
    assert len(mlp.components) == len(expected_full_config) - 1

    # Verify layer configurations
    for i, layer in enumerate(mlp.components):
        # Check layer dimensions
        assert layer.components.linear.weight.shape == torch.Size(
            [expected_full_config[i + 1], expected_full_config[i]]
        )

        # Check bias
        layer_bias = layer.components.linear.bias
        if bias:
            assert layer_bias is not None
        else:
            assert layer_bias is None

        # Check nonlinearity
        if i < len(mlp.components) - 1:
            assert isinstance(layer.components.nonlinearity, nn.Module)
        else:
            with pytest.raises(AttributeError):
                _ = layer.components.nonlinearity

        # Check batchnorm
        if batchnorm:
            assert isinstance(layer.components.batchnorm, nn.BatchNorm1d)
        else:
            with pytest.raises(AttributeError):
                _ = layer.components.batchnorm

    # Verify forward pass
    batch_size = 5
    input_tensor = testing.make_tensor((batch_size, in_features), device="cpu", dtype=torch.float32)
    output = mlp(input_tensor)
    assert output.shape == torch.Size([batch_size, out_features])


def test_save_load_mlp(tmp_path):
    # Create original MLP
    mlp = MLP(
        in_features=2,
        out_features=10,
        hidden_config=[8, 16],
        nonlinearity="leaky_relu",
        bias=True,
        batchnorm=False,
    )

    # Save and load the model
    path = Path.joinpath(tmp_path, "mlp_model.safetensors")
    mlp.save_model(path)
    mlp_loaded, _ = MLP.load_model(file_path=path)

    # Verify configurations are the same
    assert mlp.in_features == mlp_loaded.in_features
    assert mlp.out_features == mlp_loaded.out_features
    assert mlp.hidden_config == mlp_loaded.hidden_config
    assert mlp.nonlinearity == mlp_loaded.nonlinearity
    assert mlp.bias == mlp_loaded.bias
    assert mlp.batchnorm == mlp_loaded.batchnorm

    # Compare layer weights and biases
    for orig_layer, loaded_layer in zip(mlp.components, mlp_loaded.components):
        orig_linear = orig_layer.components.linear
        loaded_linear = loaded_layer.components.linear
        testing.assert_close(orig_linear.weight, loaded_linear.weight)
        if orig_layer.components.linear.bias is not None:
            testing.assert_close(orig_linear.bias, loaded_linear.bias)


@pytest.mark.parametrize("initial_config", [[8], [8, 16, 32, 64], []])
def test_insert_remove_mlp_layer(initial_config):
    # Create initial MLP
    mlp = MLP(
        in_features=2,
        out_features=2,
        hidden_config=initial_config,
        nonlinearity="relu",
        bias=True,
    )

    # Store initial configuration
    initial_full_config = mlp.full_config.copy()

    # Test layer insertion at different indices
    for insert_index in range(1, len(mlp.components) - 1):
        # Insert layer with explicit dimensions
        expanded_dim = 25
        mlp.insert_layer(index=insert_index, out_channels=expanded_dim)

        # Verify configuration update
        expected_config = initial_full_config.copy()
        expected_config.insert(insert_index, expanded_dim)
        assert mlp.full_config == expected_config

        # Verify layer count increased
        assert len(mlp.components) == len(initial_full_config)

        # Test layer removal
        mlp.remove_layer(index=insert_index)

        # Verify configuration reverted
        assert mlp.full_config == initial_full_config


@pytest.mark.parametrize("detach", [True, False])
def test_forward_activations(detach):
    # Create ResMLP with multiple layers
    mlp = MLP(
        in_features=2,
        out_features=2,
        hidden_config=[8, 8],
        nonlinearity="relu",
        bias=True,
    )

    mlp.eval().hook_model().to("cpu")

    # Perform forward pass
    input = testing.make_tensor((5, 2), device="cpu", dtype=torch.float32)
    _ = mlp(input)

    # Test get_fwd_activations
    activations = mlp.get_forward_activations(detach)
    assert len(activations) == len(mlp.components)
    assert isinstance(activations, OrderedDict)

    for act in activations.values():
        assert act.requires_grad != detach
