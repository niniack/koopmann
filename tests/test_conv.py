import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import Conv2dLayer


@pytest.mark.parametrize("in_channels, out_channels", [(3, 8), (16, 32)])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("nonlinearity", ["relu", "leaky_relu", None])
def test_conv2d_inits(
    in_channels,
    out_channels,
    kernel_size,
    nonlinearity,
    bias,
    batchnorm,
):
    # Use stride=1, padding chosen so H,W are preserved
    padding = 0 if kernel_size == 1 else 1

    layer = Conv2dLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        bias=bias,
        batchnorm=batchnorm,
        nonlinearity=nonlinearity,
    )

    conv = layer.components.conv
    assert isinstance(conv, nn.Conv2d)

    # Check weight shape
    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size

    assert conv.weight.shape == torch.Size([out_channels, in_channels, kH, kW])

    # Check bias
    if bias:
        assert conv.bias.shape == torch.Size([out_channels])
    else:
        assert conv.bias is None

    # Check batchnorm
    if batchnorm:
        assert isinstance(layer.components.batchnorm, nn.BatchNorm2d)
    else:
        with pytest.raises(AttributeError):
            _ = layer.components.batchnorm

    # Check nonlinearity
    if nonlinearity is not None:
        assert isinstance(layer.components.nonlinearity, nn.Module)
    else:
        with pytest.raises(AttributeError):
            _ = layer.components.nonlinearity

    # Check forward shape on a dummy input
    batch_size, H, W = 4, 16, 16
    x = testing.make_tensor(
        (batch_size, in_channels, H, W),
        device="cpu",
        dtype=torch.float32,
    )
    y = layer(x)

    # For stride=1 and our padding choice, H,W should be preserved
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2] == H
    assert y.shape[3] == W


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_conv2d_setup_and_remove_hook(bias, batchnorm):
    batch_size = 2
    in_channels, out_channels = 3, 5
    H, W = 8, 8

    layer = Conv2dLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        nonlinearity="relu",
        bias=bias,
        batchnorm=batchnorm,
    )

    # Initially no hook
    assert not layer.is_hooked

    # Install hook
    layer.setup_hook()
    assert layer.is_hooked

    # Forward pass
    x = testing.make_tensor(
        (batch_size, in_channels, H, W),
        device="cpu",
        dtype=torch.float32,
    )
    y = layer(x)

    # forward_activations should match output
    testing.assert_close(layer.forward_activations, y)

    # Remove hook
    layer.remove_hook()
    assert not layer.is_hooked

    # After removing hook, forward_activations should stay None
    _ = layer(x)
    assert layer.forward_activations is None
