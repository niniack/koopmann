import pytest
import torch
import torch.nn as nn
from torch import testing

from koopmann.models.layers import LinearLayer


@pytest.mark.parametrize("in_features, out_features", [(5, 22), (34, 67)])
@pytest.mark.parametrize("nonlinearity", ["relu", nn.ReLU])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_linear_inits(in_features, out_features, nonlinearity, bias, batchnorm):
    mod = LinearLayer(
        in_features=in_features,
        out_features=out_features,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
    )

    # Check dimension
    assert mod.get_layer("linear").weight.shape == torch.Size([out_features, in_features])

    # Check nonlinearity
    if nonlinearity:
        isinstance(mod.get_layer("nonlinearity"), nn.ReLU)
    else:
        assert mod.get_layer("nonlinearity") is None

    # Check bias
    if bias:
        assert mod.get_layer("linear").bias.shape == torch.Size([out_features])
    else:
        assert mod.get_layer("linear").bias is None

    # Check batchnorm
    if batchnorm:
        assert isinstance(mod.get_layer("batchnorm"), nn.BatchNorm1d)
    else:
        assert mod.get_layer("batchnorm") is None


def test_linear_setup_and_remove_hook():
    batch_size = 10
    in_features, out_features = 5, 5
    mod = LinearLayer(
        in_features=in_features,
        out_features=out_features,
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
    input = testing.make_tensor((batch_size, in_features), device="cpu", dtype=torch.float32)
    output = mod(input)
    testing.assert_close(mod.forward_activations, output)

    # Remove hook
    mod.remove_hook()
    assert not mod.is_hooked

    # Ensure no activations
    _ = mod(input)
    assert mod.forward_activations is None
