__all__ = ["Autoencoder", "ExponentialKoopmanAutencoder"]

import warnings
from ast import literal_eval
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from jaxtyping import Float, Int
from safetensors.torch import load_model, save_model
from torch import Tensor

from koopmann.models.base import BaseTorchModel
from koopmann.models.layers import LinearLayer
from koopmann.models.utils import StringtoClassNonlinearity, get_device, parse_safetensors_metadata

AutoencoderResult = namedtuple("AutoencoderResult", "predictions reconstruction")


class Autoencoder(BaseTorchModel):
    """
    Autoencoder model.
    """

    def __init__(
        self,
        input_dimension: int = 2,
        latent_dimension: int = 4,
        hidden_configuration: Optional[List[Int]] = None,
        nonlinearity: str = "leakyrelu",
    ):
        super().__init__()
        if latent_dimension <= input_dimension:
            warnings.warn(
                f"The latent dimension {latent_dimension} should probably be larger than the input dimension {input_dimension}!"
            )
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension

        # Convert string nonlinearity to class
        self.nonlinearity = nonlinearity
        nonlinearity = StringtoClassNonlinearity[nonlinearity].value

        # Set up autoencoder architecture
        if not hidden_configuration:
            self.hidden_configuration = [input_dimension * 2]
            channel_dims = [
                (input_dimension, input_dimension * 2),
                (input_dimension * 2, latent_dimension),
            ]
        else:
            self.hidden_configuration = hidden_configuration
            raise NotImplementedError("Custom hidden configuration does not work yet!")

        ################## ENCODER #################
        self._encoder = nn.Sequential()
        for i in range(0, len(channel_dims), 1):
            self._encoder.append(
                LinearLayer(
                    in_features=channel_dims[i][0],
                    out_features=channel_dims[i][1],
                    nonlinearity=nonlinearity,
                    bias=True,
                    batchnorm=False,
                    hook=False,
                )
            )
            self._encoder.apply(LinearLayer.init_weights)

        ######################
        # Koopman matrix
        ######################
        self._koopman_matrix = LinearLayer(
            in_features=latent_dimension,
            out_features=latent_dimension,
            nonlinearity=None,
            bias=False,
            batchnorm=False,
            hook=False,
        )
        eigeninit(self._koopman_matrix.linear_layer.weight, theta=0.7)

        ################## DECODER #################
        self._decoder = nn.Sequential()
        for i in range(len(channel_dims) - 1, -1, -1):
            self._decoder.append(
                LinearLayer(
                    in_features=channel_dims[i][1],
                    out_features=channel_dims[i][0],
                    nonlinearity=nonlinearity if (i != 0) else None,
                    bias=True,
                    batchnorm=False,
                    hook=False,
                )
            )
            self._decoder.apply(LinearLayer.init_weights)

    @property
    def encoder(self) -> nn.Sequential:
        """Returns the encoder in a sequential container."""
        return self._encoder

    @property
    def decoder(self) -> nn.Sequential:
        """Returns the decoder in a sequential container."""
        return self._decoder

    @property
    def koopman_matrix(self) -> nn.Sequential:
        """Returns the Koopman matrix in a sequential container."""
        return self._koopman_matrix

    def hook_model(self) -> None:
        raise NotImplementedError()

    @property
    def modules(self) -> nn.Sequential:
        """Returns the autoencoder modules in a sequential container."""
        return nn.Sequential(self._koopman_matrix, *(list(self._encoder) + list(self._decoder)))

    def forward(self, x: Float[Tensor, "batch features"], k: Int = 0) -> AutoencoderResult:
        """Forward."""
        # Pre-encoder bias
        x_bar = x - self.decoder[-1].linear_layer.bias

        # Encode
        phi_x = self.encoder(x_bar)

        # Reconstruct
        x_recons = self.decoder(phi_x)

        # Advance latent variable k times
        prediction = [phi_x]
        for i in range(1, k + 1):
            prev_pred = prediction[i - 1]
            prediction.append(self.koopman_matrix(prev_pred))

        # Batched decoding
        # Shape: [steps, batch, latent_dim]
        stacked_predictions = torch.stack(prediction, dim=0)
        steps, batch_size, latent_dim = stacked_predictions.size()
        # Shape: [steps * batch, feature_dim]
        reshaped_predictions = stacked_predictions.view(-1, latent_dim)
        decoded = self.decoder(reshaped_predictions)
        # Shape: [steps, batch, latent_dim]
        x_k = decoded.view(steps, batch_size, -1)

        return AutoencoderResult(x_k, x_recons)

    def get_fwd_activations(self, detach=True) -> OrderedDict:
        """Get forward activations."""
        activations = OrderedDict()
        for i, layer in enumerate(self.features):
            if layer.is_hooked:
                activations[i] = (
                    layer.forward_activations if not detach else layer.forward_activations.detach()
                )

        return activations

    @classmethod
    def load_model(cls, file_path: str | Path):
        """Load model."""

        # Assert path
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(
            input_dimension=literal_eval(metadata["input_dimension"]),
            latent_dimension=literal_eval(metadata["latent_dimension"]),
            nonlinearity=metadata["nonlinearity"],
        )

        # Load weights
        load_model(model, file_path, device=get_device())

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model."""

        metadata = {
            "input_dimension": str(self.input_dimension),
            "latent_dimension": str(self.latent_dimension),
            "hidden_configuration": str(self.hidden_configuration),
            "nonlinearity": str(self.nonlinearity),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        save_model(self, Path(file_path), metadata=metadata)


class MatrixExponential(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k  # Number of steps

    def forward(self, X):
        return torch.matrix_exp(X / self.k)  # Scale M by 1/k


class ExponentialKoopmanAutencoder(Autoencoder):
    """
    Autoencoder model.
    """

    def __init__(
        self,
        k: int,
        input_dimension: int = 2,
        latent_dimension: int = 4,
        hidden_configuration: Optional[List[Int]] = None,
        nonlinearity: str = "leakyrelu",
    ):
        super().__init__(
            input_dimension,
            latent_dimension,
            hidden_configuration,
            nonlinearity,
        )
        self.steps = k

        parametrize.register_parametrization(
            self.koopman_matrix.linear_layer, "weight", MatrixExponential(k=k)
        )

    @classmethod
    def load_model(cls, file_path: str | Path):
        """Load model."""

        # Assert path
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = parse_safetensors_metadata(file_path=file_path)

        # Load base model
        model = cls(
            input_dimension=literal_eval(metadata["input_dimension"]),
            latent_dimension=literal_eval(metadata["latent_dimension"]),
            nonlinearity=metadata["nonlinearity"],
            k=literal_eval(metadata["steps"]),
        )

        # Load weights
        load_model(model, file_path, device=get_device())

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model."""

        metadata = {
            "input_dimension": str(self.input_dimension),
            "latent_dimension": str(self.latent_dimension),
            "hidden_configuration": str(self.hidden_configuration),
            "nonlinearity": str(self.nonlinearity),
            "steps": str(self.steps),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        save_model(self, Path(file_path), metadata=metadata)


def eigeninit(weight: torch.Tensor, theta: float = 0.7) -> None:
    """
    Initialization for Koopman matrix weights.

    The magnitudes of the eigenvalues are set to be between 0 and 1,
    with theta determining the probability of 1.
    Directly modifies the input tensor `weight` in-place.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(weight)

    # Represent eigenvalues in polar coordinates
    polar_mags = torch.abs(eigenvalues)
    polar_phase = torch.angle(eigenvalues)

    # Sample with slab-spike distribution
    num_unique = len(torch.unique(polar_mags, sorted=False))
    bernoulli_trials = torch.distributions.Bernoulli(theta).sample([num_unique])
    uniform_trials = torch.distributions.Uniform(0, 1).sample([num_unique]) * (1 - bernoulli_trials)
    result_trials = bernoulli_trials + uniform_trials

    # Sample new magnitudes, while preserving conjugate pairs!
    new_polar_mags = torch.empty_like(polar_mags)
    new_polar_mags[0] = result_trials[0]
    j = 1
    for i in range(1, new_polar_mags.size(0)):
        if torch.isclose(polar_mags[i], polar_mags[i - 1]):
            new_polar_mags[i] = new_polar_mags[i - 1]
        else:
            new_polar_mags[i] = result_trials[j]
            j += 1

    # Rebuild eigenvalues with new magnitudes
    new_eigenvalues = torch.polar(new_polar_mags, polar_phase)

    # Construct new weight matrix in-place
    with torch.no_grad():  # Precaution
        weight.copy_(
            torch.real(eigenvectors @ torch.diag(new_eigenvalues) @ torch.linalg.inv(eigenvectors))
        )
