import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import LinearLayer


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


def test_setup_and_remove_hook():
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
