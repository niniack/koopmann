from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import testing

from koopmann.models.layers import LinearLayer
from koopmann.models.res_mlp import ResMLP
from koopmann.models.residual_blocks import LinearResidualBlock


@pytest.mark.parametrize("in_features", [2, 10])
@pytest.mark.parametrize("out_features", [2, 5])
@pytest.mark.parametrize("hidden_config", [[8], [8, 8]])
@pytest.mark.parametrize("nonlinearity", ["relu", "leaky_relu"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("stochastic_depth_prob", [0.0, 0.3])
@pytest.mark.parametrize("stochastic_depth_mode", ["batch", "row"])
def test_init_resmlp(
    in_features,
    out_features,
    hidden_config,
    nonlinearity,
    bias,
    batchnorm,
    stochastic_depth_prob,
    stochastic_depth_mode,
):
    # Create ResMLP with various configurations
    resmlp = ResMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_config=hidden_config,
        bias=bias,
        batchnorm=batchnorm,
        nonlinearity=nonlinearity,
        stochastic_depth_prob=stochastic_depth_prob,
        stochastic_depth_mode=stochastic_depth_mode,
    )

    # Verify full configuration
    expected_full_config = [in_features, *hidden_config, out_features]
    assert resmlp.full_config == expected_full_config

    # Check number of layers
    assert len(resmlp.components) == len(expected_full_config)

    # Verify first (input) layer
    input_layer = resmlp.components[0]
    assert isinstance(input_layer, LinearLayer)
    assert input_layer.components.linear.weight.shape == torch.Size(
        [expected_full_config[1], in_features]
    )

    # Verify output layer
    output_layer = resmlp.components[-1]
    assert isinstance(output_layer, LinearLayer)
    assert output_layer.components.linear.weight.shape == torch.Size(
        [out_features, expected_full_config[-2]]
    )

    # Verify layer configurations
    for i, layer in list(enumerate(resmlp.components))[1:-1]:
        # Check residual block configuration
        assert isinstance(layer, LinearResidualBlock)
        assert layer.in_channels == expected_full_config[i]

        # Check stochastic depth
        if len(hidden_config) > 1 and stochastic_depth_prob > 0:
            expected_drop_prob = stochastic_depth_prob * (i - 1) / (len(hidden_config) - 1)
            assert layer.drop_prob == pytest.approx(expected_drop_prob)
        else:
            assert layer.drop_prob == 0.0

    # Verify forward pass
    batch_size = 5
    input_tensor = testing.make_tensor((batch_size, in_features), device="cpu", dtype=torch.float32)
    output = resmlp(input_tensor)
    assert output.shape == torch.Size([batch_size, out_features])


def test_save_load_resmlp(tmp_path):
    # Create original ResMLP
    resmlp = ResMLP(
        in_features=2,
        out_features=2,
        hidden_config=[8, 16],
        bias=True,
        batchnorm=True,
        nonlinearity="relu",
        stochastic_depth_prob=0.3,
        stochastic_depth_mode="batch",
    )

    # Save and load the model
    path = Path.joinpath(tmp_path, "resmlp_model.safetensors")
    resmlp.save_model(path)
    resmlp_loaded, _ = ResMLP.load_model(file_path=path)

    # Verify configurations are the same
    assert resmlp.in_features == resmlp_loaded.in_features
    assert resmlp.out_features == resmlp_loaded.out_features
    assert resmlp.hidden_config == resmlp_loaded.hidden_config
    assert resmlp.nonlinearity == resmlp_loaded.nonlinearity
    assert resmlp.bias == resmlp_loaded.bias
    assert resmlp.batchnorm == resmlp_loaded.batchnorm
    assert resmlp.stochastic_depth_prob == resmlp_loaded.stochastic_depth_prob
    assert resmlp.stochastic_depth_mode == resmlp_loaded.stochastic_depth_mode


@pytest.mark.parametrize("initial_config", [[8], [8, 16, 32, 64]])
def test_insert_remove_resmlp_layer(initial_config):
    # Create initial ResMLP
    resmlp = ResMLP(
        in_features=2,
        out_features=2,
        hidden_config=initial_config,
        nonlinearity="relu",
        bias=True,
        stochastic_depth_prob=0.3,
    )

    # Store initial configuration
    initial_full_config = resmlp.full_config.copy()

    # Test layer insertion at different indices
    # Exclude first (input layer) and last (output layer) indices
    for insert_index in range(1, len(resmlp.components) - 1):
        # Insert layer with explicit dimensions
        expanded_dim = 25
        resmlp.insert_residual_block(index=insert_index, channels=expanded_dim)

        # Verify configuration update
        expected_config = initial_full_config.copy()
        expected_config.insert(insert_index, expanded_dim)
        assert resmlp.full_config == expected_config

        # Verify layer type based on insertion location
        inserted_layer = list(resmlp.components)[insert_index]
        if insert_index < len(initial_full_config) - 1:
            # Internal layer should be a residual block
            assert isinstance(inserted_layer, LinearResidualBlock)

        # Test layer removal
        resmlp.remove_residual_block(index=insert_index)

        # Verify configuration reverted
        assert resmlp.full_config == initial_full_config


@pytest.mark.parametrize("detach", [True, False])
def test_forward_activations(detach):
    # Create ResMLP with multiple layers
    resmlp = ResMLP(
        in_features=2,
        out_features=2,
        hidden_config=[8, 8],
        nonlinearity="relu",
        bias=True,
    )

    resmlp.eval().hook_model().to("cpu")

    # Perform forward pass
    input = testing.make_tensor((5, 2), device="cpu", dtype=torch.float32)
    _ = resmlp(input)

    # Test get_fwd_activations
    activations = resmlp.get_forward_activations(detach)
    assert len(activations) == len(resmlp.components)
    assert isinstance(activations, OrderedDict)

    for act in activations.values():
        assert act.requires_grad != detach
