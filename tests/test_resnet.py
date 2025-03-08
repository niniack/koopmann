from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.testing as testing

from koopmann.models.conv_resnet import ConvResNet


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [10, 100])
@pytest.mark.parametrize("input_size", [(32, 32), (64, 64)])
@pytest.mark.parametrize("channels_config", [[64, 128, 256], [32, 64, 128]])
@pytest.mark.parametrize("blocks_per_stage", [[2, 2, 2], [1, 1, 1]])
@pytest.mark.parametrize("nonlinearity", ["relu", "leaky_relu"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("stochastic_depth_prob", [0.0, 0.3])
@pytest.mark.parametrize("stochastic_depth_mode", ["batch", "row"])
def test_init_convresnet(
    in_channels,
    out_channels,
    input_size,
    channels_config,
    blocks_per_stage,
    nonlinearity,
    bias,
    batchnorm,
    stochastic_depth_prob,
    stochastic_depth_mode,
):
    # Create ConvResNet with various configurations
    model = ConvResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        input_size=input_size,
        channels_config=channels_config,
        blocks_per_stage=blocks_per_stage,
        nonlinearity=nonlinearity,
        bias=bias,
        batchnorm=batchnorm,
        stochastic_depth_prob=stochastic_depth_prob,
        stochastic_depth_mode=stochastic_depth_mode,
    )

    # Verify basic attributes
    assert model.in_channels == in_channels
    assert model.out_channels == out_channels
    assert model.input_size == input_size
    assert model.channels_config == channels_config
    assert model.blocks_per_stage == blocks_per_stage

    # Check model structure
    modules = list(model.modules.children())

    # Check initial convolution
    assert isinstance(modules[0], nn.Module)
    assert modules[0].__class__.__name__ == "Conv2DLayer"

    # Check max pooling
    assert isinstance(modules[1], nn.MaxPool2d)

    # Check residual blocks
    total_blocks = sum(blocks_per_stage)
    residual_blocks = [m for m in modules if m.__class__.__name__ == "Conv2DResidualBlock"]
    assert len(residual_blocks) == total_blocks

    # Check global average pooling
    assert isinstance(modules[-2], nn.AdaptiveAvgPool2d)

    # Check final fully connected layer
    assert isinstance(modules[-1], nn.Linear)
    assert modules[-1].out_features == out_channels

    # Verify stochastic depth configuration
    if stochastic_depth_prob > 0:
        block_drop_probs = [block.drop_prob for block in residual_blocks]
        assert len(set(block_drop_probs)) > 1  # Ensure varying drop probabilities

    # Test forward pass
    input_tensor = torch.randn(5, in_channels, *input_size)
    output = model(input_tensor)
    assert output.shape == torch.Size([5, out_channels])


def test_save_load_convresnet(tmp_path):
    # Create original ConvResNet
    model = ConvResNet(
        in_channels=3,
        out_channels=10,
        input_size=(32, 32),
        channels_config=[64, 128, 256],
        blocks_per_stage=[2, 2, 2],
        nonlinearity="relu",
        bias=True,
        batchnorm=True,
        stochastic_depth_prob=0.3,
        stochastic_depth_mode="batch",
    )

    # Save and load the model
    path = Path.joinpath(tmp_path, "convresnet_model.safetensors")
    model.save_model(path)
    loaded_model, metadata = ConvResNet.load_model(file_path=path)

    # Verify configurations are the same
    assert model.in_channels == loaded_model.in_channels
    assert model.out_channels == loaded_model.out_channels
    assert model.input_size == loaded_model.input_size
    assert model.channels_config == loaded_model.channels_config
    assert model.blocks_per_stage == loaded_model.blocks_per_stage
    assert model.nonlinearity == loaded_model.nonlinearity
    assert model.bias == loaded_model.bias
    assert model.batchnorm == loaded_model.batchnorm
    assert model.stochastic_depth_prob == loaded_model.stochastic_depth_prob
    assert model.stochastic_depth_mode == loaded_model.stochastic_depth_mode

    # Compare a few layer weights (initial conv and final FC)
    initial_layers = list(model.modules.children())
    loaded_layers = list(loaded_model.modules.children())

    # Compare initial convolution layer weights
    testing.assert_close(
        initial_layers[0].get_layer("conv").weight, loaded_layers[0].get_layer("conv").weight
    )

    # Compare final FC layer weights
    testing.assert_close(initial_layers[-1].weight, loaded_layers[-1].weight)


def test_hook_and_activation_methods():
    # Create ConvResNet
    model = ConvResNet(
        in_channels=3,
        out_channels=10,
        input_size=(32, 32),
        channels_config=[64, 128, 256],
        blocks_per_stage=[2, 2, 2],
    )

    # Hook the model
    model.hook_model()

    # Verify hooks are set
    for name, module in model.modules.named_modules():
        if hasattr(module, "is_hooked"):
            assert module.is_hooked

    # Perform forward pass
    input_tensor = torch.randn(5, 3, 32, 32)
    _ = model(input_tensor)

    # Test get_fwd_activations
    activations = model.get_fwd_activations()
    assert len(activations) > 0

    # Test get_fwd_acts_patts
    activations, patterns = model.get_fwd_acts_patts()
    assert len(activations) > 0
    assert len(patterns) > 0
    assert len(activations) == len(patterns)


def test_edge_cases():
    # Test with minimal configuration
    model_minimal = ConvResNet(
        in_channels=1,
        out_channels=2,
        input_size=(16, 16),
        channels_config=[16],
        blocks_per_stage=[1],
    )

    # Forward pass with small input
    input_tensor = torch.randn(2, 1, 16, 16)
    output = model_minimal(input_tensor)
    assert output.shape == torch.Size([2, 2])

    # Test with extreme stochastic depth
    model_max_dropout = ConvResNet(
        in_channels=3, out_channels=10, input_size=(32, 32), stochastic_depth_prob=1.0
    )

    # Ensure model can be instantiated and forward pass works
    input_tensor = torch.randn(2, 3, 32, 32)
    output = model_max_dropout(input_tensor)
    assert output.shape == torch.Size([2, 10])
