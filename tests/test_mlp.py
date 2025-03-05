from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import LinearLayer
from koopmann.models.mlp import MLP
from koopmann.models.mlp_resnet import ResMLP
from koopmann.models.residual_blocks import LinearResidualBlock


@pytest.mark.parametrize("input_dimension", [2, 10])
@pytest.mark.parametrize("output_dimension", [2, 5])
@pytest.mark.parametrize("config", [[8], [8, 16], []])
@pytest.mark.parametrize("nonlinearity", ["relu", "leakyrelu"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_init_mlp(
    input_dimension,
    output_dimension,
    config,
    nonlinearity,
    bias,
    batchnorm,
):
    # Create MLP with various configurations
    mlp = MLP(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        config=config,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
    )

    # Verify full configuration
    expected_full_config = [input_dimension, *config, output_dimension]
    assert mlp.full_config == expected_full_config

    # Check number of layers
    assert len(mlp.modules) == len(expected_full_config) - 1

    # Verify layer configurations
    for i, layer in enumerate(mlp.modules):
        # Check layer dimensions
        linear_layer = layer.get_layer("linear")
        assert linear_layer.weight.shape == torch.Size(
            [expected_full_config[i + 1], expected_full_config[i]]
        )

        # Check bias configuration
        if bias:
            assert linear_layer.bias is not None
            assert linear_layer.bias.shape == torch.Size([expected_full_config[i + 1]])
        else:
            assert linear_layer.bias is None

        # Check nonlinearity
        if i < len(mlp.modules) - 1:
            # Hidden layers should have nonlinearity
            assert layer.get_layer("nonlinearity") is not None
        else:
            # Last layer should not have nonlinearity
            assert layer.get_layer("nonlinearity") is None

        # Check batch normalization
        if batchnorm:
            assert layer.get_layer("batchnorm") is not None
        else:
            assert layer.get_layer("batchnorm") is None

    # Verify forward pass
    input_tensor = torch.randn(5, input_dimension)
    output = mlp(input_tensor)
    assert output.shape == torch.Size([5, output_dimension])


def test_save_load_mlp(tmp_path):
    # Create original MLP
    mlp = MLP(
        input_dimension=2,
        output_dimension=2,
        config=[8, 16],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
    )

    # Save and load the model
    path = Path.joinpath(tmp_path, "mlp_model.safetensors")
    mlp.save_model(path)
    mlp_loaded, _ = MLP.load_model(file_path=path)

    # Verify configurations are the same
    assert mlp.input_dimension == mlp_loaded.input_dimension
    assert mlp.output_dimension == mlp_loaded.output_dimension
    assert mlp.config == mlp_loaded.config
    assert mlp.nonlinearity == mlp_loaded.nonlinearity
    assert mlp.bias == mlp_loaded.bias
    assert mlp.batchnorm == mlp_loaded.batchnorm

    # Compare layer weights and biases
    for orig_layer, loaded_layer in zip(mlp.modules, mlp_loaded.modules):
        testing.assert_close(
            orig_layer.get_layer("linear").weight, loaded_layer.get_layer("linear").weight
        )
        if orig_layer.get_layer("linear").bias is not None:
            testing.assert_close(
                orig_layer.get_layer("linear").bias, loaded_layer.get_layer("linear").bias
            )


@pytest.mark.parametrize(
    "initial_config",
    [
        [8],  # Single hidden layer
        [8, 16],  # Multiple hidden layers
        [],  # No hidden layers
    ],
)
def test_insert_remove_mlp_layer(initial_config):
    # Create initial MLP
    mlp = MLP(
        input_dimension=2,
        output_dimension=2,
        config=initial_config,
        nonlinearity="relu",
        bias=True,
    )

    # Store initial configuration
    initial_full_config = mlp.full_config.copy()

    # Test layer insertion at different indices
    for insert_index in range(len(mlp.modules)):
        # Insert layer with explicit dimensions
        expanded_dim = 25
        mlp.insert_layer(index=insert_index, out_features=expanded_dim)

        # Verify configuration update
        expected_config = initial_full_config.copy()
        expected_config.insert(insert_index + 1, expanded_dim)
        assert mlp.full_config == expected_config

        # Verify layer count increased
        assert len(mlp.modules) == len(initial_full_config)

        # Test layer removal
        mlp.remove_layer(index=insert_index)

        # Verify configuration reverted
        assert mlp.full_config == initial_full_config

    # Additional test for default insertion
    mlp.insert_layer(index=len(mlp.modules) - 1)
    assert len(mlp.modules) == len(initial_full_config)


########################################################


@pytest.mark.parametrize("input_dimension", [2, 10])
@pytest.mark.parametrize("output_dimension", [2, 5])
@pytest.mark.parametrize("config", [[8], [8, 8]])
@pytest.mark.parametrize("nonlinearity", ["relu", "leakyrelu"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("stochastic_depth_prob", [0.0, 0.3])
@pytest.mark.parametrize("stochastic_depth_mode", ["batch", "row"])
def test_init_resmlp(
    input_dimension,
    output_dimension,
    config,
    nonlinearity,
    bias,
    batchnorm,
    stochastic_depth_prob,
    stochastic_depth_mode,
):
    # Create ResMLP with various configurations
    resmlp = ResMLP(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        config=config,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
        stochastic_depth_prob=stochastic_depth_prob,
        stochastic_depth_mode=stochastic_depth_mode,
    )

    # Verify full configuration
    expected_full_config = [input_dimension, *config, output_dimension]
    assert resmlp.full_config == expected_full_config

    # # Check number of layers
    # assert len(resmlp.modules) == len(expected_full_config)

    # Verify first (input) layer
    input_layer = resmlp.modules[0]
    assert isinstance(input_layer, LinearLayer)
    assert input_layer.get_layer("linear").weight.shape == torch.Size(
        [expected_full_config[1], input_dimension]
    )

    # Verify output layer
    output_layer = resmlp.modules[-1]
    assert isinstance(output_layer, LinearLayer)
    assert output_layer.get_layer("linear").weight.shape == torch.Size(
        [output_dimension, expected_full_config[-2]]
    )

    # Verify layer configurations
    for i, layer in list(enumerate(resmlp.modules))[1:-1]:
        # Check residual block configuration
        assert isinstance(layer, LinearResidualBlock)
        assert layer.in_features == expected_full_config[i]

        # Check stochastic depth
        if len(config) > 1 and stochastic_depth_prob > 0:
            expected_drop_prob = stochastic_depth_prob * (i - 1) / (len(config) - 1)
            assert layer.drop_prob == pytest.approx(expected_drop_prob)
        else:
            assert layer.drop_prob == 0.0

    # Verify forward pass
    input_tensor = torch.randn(5, input_dimension)
    output = resmlp(input_tensor)
    assert output.shape == torch.Size([5, output_dimension])


def test_save_load_resmlp(tmp_path):
    # Create original ResMLP
    resmlp = ResMLP(
        input_dimension=2,
        output_dimension=2,
        config=[8, 16],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
        stochastic_depth_prob=0.3,
        stochastic_depth_mode="batch",
    )

    # Save and load the model
    path = Path.joinpath(tmp_path, "resmlp_model.safetensors")
    resmlp.save_model(path)
    resmlp_loaded, _ = ResMLP.load_model(file_path=path)

    # Verify configurations are the same
    assert resmlp.input_dimension == resmlp_loaded.input_dimension
    assert resmlp.output_dimension == resmlp_loaded.output_dimension
    assert resmlp.config == resmlp_loaded.config
    assert resmlp.nonlinearity == resmlp_loaded.nonlinearity
    assert resmlp.bias == resmlp_loaded.bias
    assert resmlp.batchnorm == resmlp_loaded.batchnorm
    assert resmlp.stochastic_depth_prob == resmlp_loaded.stochastic_depth_prob
    assert resmlp.stochastic_depth_mode == resmlp_loaded.stochastic_depth_mode

    # Compare layer weights and biases
    for orig_layer, loaded_layer in zip(resmlp.modules, resmlp_loaded.modules):
        if hasattr(orig_layer, "get_layer"):
            orig_linear = orig_layer.get_layer("linear")
            loaded_linear = loaded_layer.get_layer("linear")

            testing.assert_close(orig_linear.weight, loaded_linear.weight)
            if orig_linear.bias is not None:
                testing.assert_close(orig_linear.bias, loaded_linear.bias)


@pytest.mark.parametrize(
    "initial_config",
    [
        [8],  # Single hidden layer
        [8, 16],  # Multiple hidden layers
    ],
)
def test_insert_remove_resmlp_layer(initial_config):
    # Create initial ResMLP
    resmlp = ResMLP(
        input_dimension=2,
        output_dimension=2,
        config=initial_config,
        nonlinearity="relu",
        bias=True,
        stochastic_depth_prob=0.3,
    )

    # Store initial configuration
    initial_full_config = resmlp.full_config.copy()

    # Test layer insertion at different indices
    # Exclude first (input layer) and last (output layer) indices
    for insert_index in range(1, len(resmlp.modules) - 1):
        # Insert layer with explicit dimensions
        expanded_dim = 25
        resmlp.insert_layer(index=insert_index, out_features=expanded_dim)

        # Verify configuration update
        expected_config = initial_full_config.copy()
        expected_config.insert(insert_index + 1, expanded_dim)
        assert resmlp.full_config == expected_config

        # Verify layer type based on insertion location
        inserted_layer = list(resmlp.modules)[insert_index]
        if insert_index < len(initial_full_config) - 1:
            # Internal layer should be a residual block
            assert isinstance(inserted_layer, LinearResidualBlock)

        # Test layer removal
        resmlp.remove_layer(index=insert_index)

        # Verify configuration reverted
        assert resmlp.full_config == initial_full_config

    # Additional test for default insertion
    initial_modules_count = len(resmlp.modules)
    resmlp.insert_layer(index=len(resmlp.modules) - 1)
    assert len(resmlp.modules) == initial_modules_count + 1


def test_forward_activations_and_patterns():
    # Create ResMLP with multiple layers
    resmlp = ResMLP(
        input_dimension=2,
        output_dimension=2,
        config=[8, 8],
        nonlinearity="relu",
        bias=True,
    )

    # Setup hooks
    for module in resmlp.modules:
        if hasattr(module, "setup_hook"):
            module.setup_hook()

    # Perform forward pass
    input_tensor = torch.randn(5, 2)
    _ = resmlp(input_tensor)

    # Test get_fwd_activations
    activations = resmlp.get_fwd_activations()
    assert len(activations) > 0

    # Test get_fwd_acts_patts
    activations, patterns = resmlp.get_fwd_acts_patts()
    assert len(activations) > 0
    assert len(patterns) > 0
    assert len(activations) == len(patterns)
