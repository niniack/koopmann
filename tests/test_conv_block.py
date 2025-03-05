import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.residual_blocks import Conv2DResidualBlock, LinearResidualBlock


@pytest.mark.parametrize("in_channels, out_channels", [(16, 16), (32, 32)])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("nonlinearity", ["relu", nn.ReLU])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("drop_prob", [0.0, 0.3])
@pytest.mark.parametrize("stoch_mode", ["batch", "row"])
def test_convresblock_inits(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    nonlinearity,
    bias,
    batchnorm,
    drop_prob,
    stoch_mode,
):
    # Test block initialization
    mod = Conv2DResidualBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
        drop_prob=drop_prob,
        stoch_mode=stoch_mode,
    )

    # Check basic attributes
    assert mod.in_channels == in_channels
    assert mod.out_channels == out_channels
    assert mod.stride == stride

    # Check layers existence
    assert "conv1" in mod.layers
    assert "conv2" in mod.layers

    # Verify downsampling layer when needed
    if stride != 1 or in_channels != out_channels:
        assert mod.downsample is not None
    else:
        assert mod.downsample is None

    # Create input tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, in_channels, 32, 32)

    # Test forward pass
    mod.train()  # Set to training mode
    output, activation_pattern = mod(input_tensor)

    # Check output shape
    expected_height = (32 + 2 * (mod.layers["conv1"]._padding) - kernel_size) // stride + 1
    assert output.shape == (batch_size, out_channels, expected_height, expected_height)

    # Check activation pattern
    assert activation_pattern.shape == output.shape

    # Verify stochastic depth probability
    assert mod.drop_prob == drop_prob
    assert mod.stoch_mode == stoch_mode

    # Verify layer configurations
    conv1 = mod.layers["conv1"]
    conv2 = mod.layers["conv2"]

    # Check first convolutional layer
    assert conv1.get_layer("conv").in_channels == in_channels
    assert conv1.get_layer("conv").out_channels == out_channels
    assert conv1._bias == bias
    assert conv1._batchnorm == batchnorm
    assert isinstance(conv1.get_layer("nonlinearity"), nn.ReLU)

    # Check second convolutional layer
    assert conv2.get_layer("conv").in_channels == in_channels
    assert conv2.get_layer("conv").out_channels == out_channels
    assert conv2._bias == bias
    assert conv2._batchnorm == batchnorm
    assert conv2.get_layer("nonlinearity") is None  # No nonlinearity for second layer

    # Test hook setup and removal
    mod.setup_hook()
    assert mod.is_hooked
    mod(input_tensor)  # Trigger forward pass to capture activations
    assert mod.forward_activations is not None

    mod.remove_hook()
    assert not mod.is_hooked
