import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import LinearLayer


@pytest.mark.parametrize("in_features, out_features", [(5, 22), (34, 67)])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("nonlinearity", ["relu", "leaky_relu", None])
def test_linear_inits(in_features, out_features, nonlinearity, bias, batchnorm):
    layer = LinearLayer(
        in_channels=in_features,
        out_channels=out_features,
        bias=bias,
        batchnorm=batchnorm,
        nonlinearity=nonlinearity,
    )

    # Check dimension
    assert layer.components.linear.weight.shape == torch.Size([out_features, in_features])

    # Check nonlinearity
    if nonlinearity:
        assert isinstance(layer.components.nonlinearity, nn.Module)
    else:
        with pytest.raises(AttributeError):
            _ = layer.components.nonlinearity

    # Check bias
    if bias:
        assert layer.components.linear.bias.shape == torch.Size([out_features])
    else:
        assert layer.components.linear.bias is None

    # Check batchnorm
    if batchnorm:
        assert isinstance(layer.components.batchnorm, nn.BatchNorm1d)
    else:
        with pytest.raises(AttributeError):
            _ = layer.components.batchnorm


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_linear_setup_and_remove_hook(bias, batchnorm):
    batch_size = 10
    in_features, out_features = 5, 5
    layer = LinearLayer(
        in_channels=in_features,
        out_channels=out_features,
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
    input = testing.make_tensor((batch_size, in_features), device="cpu", dtype=torch.float32)
    output = layer(input)
    testing.assert_close(layer.forward_activations, output)

    # Remove hook
    layer.remove_hook()
    assert not layer.is_hooked

    # Ensure no activations
    _ = layer(input)
    assert layer.forward_activations is None
