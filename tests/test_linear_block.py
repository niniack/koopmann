import pytest
import torch
import torch.nn as nn

from koopmann.models.residual_blocks import LinearResidualBlock


@pytest.mark.parametrize("dimension", [64, 128])
@pytest.mark.parametrize("nonlinearity", ["relu", nn.ReLU])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("drop_prob", [0.0, 0.3])
@pytest.mark.parametrize("stoch_mode", ["batch", "row"])
def test_linearresblock_inits(
    dimension,
    nonlinearity,
    bias,
    batchnorm,
    drop_prob,
    stoch_mode,
):
    # Test block initialization
    mod = LinearResidualBlock(
        dimension=dimension,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
        drop_prob=drop_prob,
        stoch_mode=stoch_mode,
    )

    # Check basic attributes
    assert mod.dimension == dimension
    assert mod.in_features == dimension
    assert mod.out_features == dimension

    # Check layers existence
    assert "fc1" in mod.layers
    assert "fc2" in mod.layers

    # Create input tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, dimension)

    # Test forward pass
    mod.train()  # Set to training mode
    output, activation_pattern = mod(input_tensor)

    # Check output shape
    assert output.shape == input_tensor.shape

    # Check activation pattern
    assert activation_pattern.shape == output.shape

    # Verify stochastic depth probability
    assert mod.drop_prob == drop_prob
    assert mod.stoch_mode == stoch_mode

    # Verify layer configurations
    fc1 = mod.layers["fc1"]
    fc2 = mod.layers["fc2"]

    # Check first fully-connected layer
    assert fc1.in_features == dimension
    assert fc1.out_features == dimension
    assert fc1._bias == bias
    assert fc1._batchnorm == batchnorm
    assert isinstance(fc1.get_layer("nonlinearity"), nn.ReLU)

    # Check second fully-connected layer
    assert fc2.in_features == dimension
    assert fc2.out_features == dimension
    assert fc2._bias == bias
    assert fc2._batchnorm == batchnorm
    assert fc2.get_layer("nonlinearity") is None  # No nonlinearity for second layer

    # Test hook setup and removal
    mod.setup_hook()
    assert mod.is_hooked
    mod(input_tensor)  # Trigger forward pass to capture activations
    assert mod.forward_activations is not None

    mod.remove_hook()
    assert not mod.is_hooked
