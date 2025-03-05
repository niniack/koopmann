import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import Conv2DLayer


@pytest.mark.parametrize("in_channels, out_channels", [(3, 16), (16, 32)])
@pytest.mark.parametrize("kernel_size", [3, (3, 3)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("nonlinearity", ["relu", nn.ReLU])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_conv2d_inits(
    in_channels, out_channels, kernel_size, stride, padding, nonlinearity, bias, batchnorm
):
    mod = Conv2DLayer(
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
    assert mod.get_layer("conv").weight.shape == torch.Size(
        [
            out_channels,
            in_channels,
            kernel_size if isinstance(kernel_size, int) else kernel_size[0],
            kernel_size if isinstance(kernel_size, int) else kernel_size[1],
        ]
    )

    # Check nonlinearity
    if nonlinearity:
        isinstance(mod.get_layer("nonlinearity"), nn.ReLU)
    else:
        assert mod.get_layer("nonlinearity") is None

    # Check bias
    if bias:
        assert mod.get_layer("conv").bias.shape == torch.Size([out_channels])
    else:
        assert mod.get_layer("conv").bias is None

    # Check batchnorm
    if batchnorm:
        assert isinstance(mod.get_layer("batchnorm"), nn.BatchNorm2d)
    else:
        assert mod.get_layer("batchnorm") is None


def test_conv2d_setup_and_remove_hook():
    batch_size = 10
    in_channels, out_channels = 3, 16
    height, width = 32, 32
    mod = Conv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
    )

    # Assert no hook
    assert not mod.is_hooked

    # Hook
    mod.setup_hook()
    assert mod.is_hooked

    # Check output and activations
    input = testing.make_tensor(
        (batch_size, in_channels, height, width), device="cpu", dtype=torch.float32
    )
    output = mod(input)
    testing.assert_close(mod.forward_activations, output)

    # Remove hook
    mod.remove_hook()
    assert not mod.is_hooked

    # Ensure no activations
    _ = mod(input)
    assert mod.forward_activations is None
