import pytest
import torch

from koopmann.models.residual_blocks import LinearResidualBlock


@pytest.mark.parametrize("dimension", [64, 128])
@pytest.mark.parametrize("nonlinearity", ["relu", "leaky_relu"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("drop_prob", [0.0, 0.3])
@pytest.mark.parametrize("stoch_mode", ["batch", "row"])
def test_linearresblock_inits(
    dimension,
    bias,
    batchnorm,
    nonlinearity,
    drop_prob,
    stoch_mode,
):
    # Test block initialization
    block = LinearResidualBlock(
        channels=dimension,
        bias=bias,
        batchnorm=batchnorm,
        nonlinearity=nonlinearity,
        drop_prob=drop_prob,
        stoch_mode=stoch_mode,
    )

    # Check basic attributes
    assert block.in_channels == dimension
    assert block.out_channels == dimension

    # Check layers existence
    assert "fc1" in block.components
    assert "fc2" in block.components

    # Test forward pass
    input_tensor = torch.randn(2, dimension)
    block.train()
    output, activation_pattern = block(input_tensor)

    # Check output shape
    assert output.shape == input_tensor.shape

    # Check activation pattern
    assert activation_pattern.shape == output.shape

    # Verify stochastic depth probability
    assert block.drop_prob == drop_prob
    assert block.stoch_mode == stoch_mode

    # Verify layer configurations
    fc1 = block.components["fc1"]
    fc2 = block.components["fc2"]

    # # Check first fully-connected layer
    # assert fc1.in_channels == dimension
    # assert fc1.out_channels == dimension
    # assert fc1._bias == bias
    # assert fc1._batchnorm == batchnorm
    # assert isinstance(fc1.get_layer("nonlinearity"), nn.ReLU)

    # # Check second fully-connected layer
    # assert fc2.in_channels == dimension
    # assert fc2.out_channels == dimension
    # assert fc2._bias == bias
    # assert fc2._batchnorm == batchnorm
    # assert fc2.get_layer("nonlinearity") is None

    # Test hook setup and removal
    block.setup_hook()
    assert block.is_hooked
    block(input_tensor)  # Trigger forward pass to capture activations
    assert block.forward_activations is not None

    block.remove_hook()
    assert not block.is_hooked
