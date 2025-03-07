from pathlib import Path

import pytest
import torch
from torch import nn, testing

from koopmann.models.autoencoder import KoopmanAutoencoder


@pytest.mark.parametrize("k_steps", [3, 8])
@pytest.mark.parametrize("in_features", [2, 10])
@pytest.mark.parametrize("latent_features", [15, 20])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("nonlinearity", ["relu", "leakyrelu"])
def test_init_autoencoder(k_steps, in_features, latent_features, bias, batchnorm, nonlinearity):
    autoencoder = KoopmanAutoencoder(
        k_steps=k_steps,
        in_features=in_features,
        latent_features=latent_features,
        bias=bias,
        batchnorm=batchnorm,
        nonlinearity=nonlinearity,
    )

    for i in range(len(autoencoder.components.encoder)):
        encoder_layer = autoencoder.components.encoder[i]
        decoder_layer = autoencoder.components.decoder[-i]

        # Check batchnorm
        if batchnorm:
            assert isinstance(encoder_layer.components.batchnorm, nn.BatchNorm1d)
            assert isinstance(encoder_layer.components.batchnorm, nn.BatchNorm1d)
        else:
            with pytest.raises(AttributeError):
                _ = encoder_layer.components.batchnorm
                _ = decoder_layer.components.batchnorm

        # Check bias
        if bias:
            assert encoder_layer.components.linear.bias is not None
            assert decoder_layer.components.linear.bias is not None
        else:
            assert encoder_layer.components.linear.bias is None
            assert decoder_layer.components.linear.bias is None

    # Koopman matrix shape
    assert autoencoder.koopman_weights.shape == torch.Size([latent_features, latent_features])


def test_save_load_autoencoder(tmp_path):
    autoencoder = KoopmanAutoencoder(
        k_steps=2,
        in_features=2,
        latent_features=5,
        bias=True,
        batchnorm=True,
        nonlinearity="leakyrelu",
    )

    path = Path.joinpath(tmp_path, "autoencoder.safetensors")
    autoencoder.save_model(path)
    ae_loaded, _ = KoopmanAutoencoder.load_model(file_path=path)

    testing.assert_close(ae_loaded.koopman_weights, autoencoder.koopman_weights)
