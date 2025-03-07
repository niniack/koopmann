import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import Conv2DLayer


@pytest.mark.parametrize("in_channels, out_channels", [(3, 16), (16, 32)])
@pytest.mark.parametrize("kernel_size", [3, (3, 3)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("nonlinearity", ["relu", "leakyrelu", None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_conv2d_inits(
    in_channels, out_channels, kernel_size, stride, padding, nonlinearity, bias, batchnorm
):
    layer = Conv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
    )

    # Check dimension
    assert layer.components.conv.weight.shape == torch.Size(
        [
            out_channels,
            in_channels,
            kernel_size if isinstance(kernel_size, int) else kernel_size[0],
            kernel_size if isinstance(kernel_size, int) else kernel_size[1],
        ]
    )

    # Check nonlinearity
    if nonlinearity:
        assert isinstance(layer.components.nonlinearity, nn.Module)
    else:
        with pytest.raises(AttributeError):
            _ = layer.components.nonlinearity

    # Check bias
    if bias:
        assert layer.components.conv.bias.shape == torch.Size([out_channels])
    else:
        assert layer.components.conv.bias is None

    # Check batchnorm
    if batchnorm:
        assert isinstance(layer.components.batchnorm, nn.BatchNorm2d)
    else:
        with pytest.raises(AttributeError):
            _ = layer.components.batchnorm


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_conv2d_setup_and_remove_hook(bias, batchnorm):
    batch_size = 10
    in_channels, out_channels = 3, 16
    height, width = 32, 32
    layer = Conv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        nonlinearity="relu",
        bias=bias,
        batchnorm=batchnorm,
    )

    # Assert no hook
    assert not layer.is_hooked

    # Hook
    layer.setup_hook()
    assert layer.is_hooked

    # Check output and activations
    input = testing.make_tensor(
        (batch_size, in_channels, height, width), device="cpu", dtype=torch.float32
    )
    output = layer(input)
    testing.assert_close(layer.forward_activations, output)

    # Remove hook
    layer.remove_hook()
    assert not layer.is_hooked

    # Ensure no activations
    _ = layer(input)
    assert layer.forward_activations is None
