import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import Conv2DLayer, LinearLayer


@pytest.mark.parametrize("in_features, out_features", [(5, 22), (34, 67)])
@pytest.mark.parametrize("nonlinearity", ["relu", nn.ReLU])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("hook", [True, False])
def test_linear_inits(in_features, out_features, nonlinearity, bias, batchnorm, hook):
    layer = LinearLayer(
        in_features=in_features,
        out_features=out_features,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
        hook=hook,
    )

    # Check dimension
    assert layer.linear_layer.weight.shape == torch.Size([out_features, in_features])

    # # Check nonlinearity
    # if nonlinearity:
    #     isinstance(layer.activation_layer, nn.ReLU)
    # else:
    #     assert layer.activation_layer is None

    # Check bias
    if bias:
        assert layer.linear_layer.bias.shape == torch.Size([out_features])
    else:
        assert layer.linear_layer.bias is None

    # # Check batchnorm
    # assert (batchnorm and layer.batchnorm_layer is not None) or (
    #     not batchnorm and layer.batchnorm_layer is None
    # )

    # Check hook
    assert layer.is_hooked == hook
    assert (hook and layer._handle is not None) or (not hook and layer._handle is None)


def test_linear_setup_and_remove_hook():
    batch_size = 10
    in_features, out_features = 5, 5
    layer = LinearLayer(
        in_features=in_features,
        out_features=out_features,
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
        hook=False,
    )

    # Assert no hook
    assert not layer.is_hooked

    # Hook
    layer.setup_hook()
    assert layer.is_hooked

    # Check output and activations
    input = testing.make_tensor((batch_size, in_features), device="cpu", dtype=torch.float32)
    output = layer(input)
    testing.assert_close(layer.forward_activations, output)

    # Remove hook
    layer.remove_hook()
    assert not layer.is_hooked

    # Ensure no activations
    _ = layer(input)
    assert layer.forward_activations is None


@pytest.mark.parametrize("in_channels, out_channels", [(3, 16), (16, 32)])
@pytest.mark.parametrize("kernel_size", [3, (3, 3)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("nonlinearity", ["relu", nn.ReLU])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("hook", [True, False])
def test_conv2d_inits(
    in_channels, out_channels, kernel_size, stride, padding, nonlinearity, bias, batchnorm, hook
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
        hook=hook,
    )

    # Check dimension
    assert layer.conv_layer.weight.shape == torch.Size(
        [
            out_channels,
            in_channels,
            kernel_size if isinstance(kernel_size, int) else kernel_size[0],
            kernel_size if isinstance(kernel_size, int) else kernel_size[1],
        ]
    )

    # Check bias
    if bias:
        assert layer.conv_layer.bias.shape == torch.Size([out_channels])
    else:
        assert layer.conv_layer.bias is None

    # Check hook
    assert layer.is_hooked == hook
    assert (hook and layer._handle is not None) or (not hook and layer._handle is None)


def test_conv2d_setup_and_remove_hook():
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
        bias=True,
        batchnorm=True,
        hook=False,
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
