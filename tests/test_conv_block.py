import pytest
import torch

from koopmann.models.residual_blocks import Conv2DResidualBlock


@pytest.mark.parametrize("channels", [16, 32])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("nonlinearity", ["relu", "leakyrelu", None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("drop_prob", [0.0, 0.3])
@pytest.mark.parametrize("stoch_mode", ["batch", "row"])
def test_convresblock_inits(
    channels,
    kernel_size,
    stride,
    nonlinearity,
    bias,
    batchnorm,
    drop_prob,
    stoch_mode,
):
    # Test block initialization
    block = Conv2DResidualBlock(
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        batchnorm=batchnorm,
        nonlinearity=nonlinearity,
        drop_prob=drop_prob,
        stoch_mode=stoch_mode,
    )

    # Check basic attributes
    assert block.in_channels == channels
    assert block.out_channels == channels
    assert block.stride == stride

    # Check layers existence
    assert "conv1" in block.components
    assert "conv2" in block.components

    # Verify downsampling layer when needed
    if stride != 1:
        assert block.downsample is not None
    else:
        assert block.downsample is None

    # Test forward pass
    N, C, H, W = 2, channels, 32, 32
    input_tensor = torch.randn(N, C, H, W)
    block.train()
    output, activation_pattern = block(input_tensor)

    # Check output shape
    expected_height = (32 + 2 * (block.components.conv1.padding) - kernel_size) // stride + 1
    assert output.shape == (N, C, expected_height, expected_height)

    # Check activation pattern
    assert activation_pattern.shape == output.shape

    # Verify stochastic depth probability
    assert block.drop_prob == drop_prob
    assert block.stoch_mode == stoch_mode

    # Verify layer configurations
    conv1 = block.components.conv1
    conv2 = block.components.conv2

    # # Check first convolutional layer
    # assert conv1.get_layer("conv").in_channels == channels
    # assert conv1.get_layer("conv").out_channels == channels
    # assert conv1._bias == bias
    # assert conv1._batchnorm == batchnorm
    # assert isinstance(conv1.get_layer("nonlinearity"), nn.ReLU)

    # # Check second convolutional layer
    # assert conv2.get_layer("conv").in_channels == channels
    # assert conv2.get_layer("conv").out_channels == channels
    # assert conv2._bias == bias
    # assert conv2._batchnorm == batchnorm
    # assert conv2.get_layer("nonlinearity") is None

    # Test hook setup and removal
    block.setup_hook()
    assert block.is_hooked
    block(input_tensor)  # Trigger forward pass to capture activations
    assert block.forward_activations is not None

    block.remove_hook()
    assert not block.is_hooked
