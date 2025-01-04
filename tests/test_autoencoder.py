from pathlib import Path

import torch
from torch import testing

from koopmann.models.autoencoder import Autoencoder


def test_init_autoencoder():
    latent_dimension = 20
    autoencoder = Autoencoder(
        input_dimension=5,
        latent_dimension=latent_dimension,
        nonlinearity="leakyrelu",
    )

    for i in range(len(autoencoder.encoder)):
        # # Assert no batchnorm
        # assert autoencoder.encoder[i].batchnorm_layer is None
        # assert autoencoder.decoder[i].batchnorm_layer is None

        # # Assert bias exists
        # assert autoencoder.encoder[i].linear_layer.bias is not None
        # assert autoencoder.decoder[i].linear_layer.bias is not None

        # Assert no hooks
        assert autoencoder.encoder[i].is_hooked is False
        assert autoencoder.decoder[i].is_hooked is False

    # Assert Koopman matrix shape
    assert autoencoder.koopman_matrix.linear_layer.weight.shape == torch.Size(
        [latent_dimension, latent_dimension]
    )


def test_save_load_autoencoder(tmp_path):
    autoencoder = Autoencoder(
        input_dimension=5,
        latent_dimension=20,
        nonlinearity="leakyrelu",
    )

    path = Path.joinpath(tmp_path, "autoencoder.safetensors")
    autoencoder.save_model(path)
    ae_loaded = Autoencoder.load_model(file_path=path)

    testing.assert_close(
        ae_loaded.koopman_matrix.linear_layer.weight, autoencoder.koopman_matrix.linear_layer.weight
    )
