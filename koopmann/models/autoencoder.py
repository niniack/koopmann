__all__ = [
    "Autoencoder",
    "ExponentialKoopmanAutencoder",
    "LowRankKoopmanAutoencoder",
]

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
from koopmann.models.utils import (
    StringtoClassNonlinearity,
    eigeninit,
    get_device,
    parse_safetensors_metadata,
)

AutoencoderResult = namedtuple("AutoencoderResult", "predictions reconstruction")


class Autoencoder(BaseTorchModel):
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
        batchnorm: bool = False,
        rank: int = None,
    ):
        super().__init__()

        if latent_dimension <= input_dimension:
            warnings.warn(
                f"The latent dimension {latent_dimension} should probably be "
                f"larger than the input dimension {input_dimension}!"
            )

        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.steps = k
        self.batchnorm = batchnorm
        self.rank = rank

        # Store random projections in a ModuleDict as non-trainable parameters
        self.random_projections = nn.ParameterDict()

        # Convert string nonlinearity to class
        self.nonlinearity = nonlinearity
        nonlinearity = StringtoClassNonlinearity[nonlinearity].value

        # Set up autoencoder architecture
        if not hidden_configuration:
            self.hidden_configuration = None
            channel_dims = [
                (input_dimension, latent_dimension),
            ]
        else:
            self.hidden_configuration = hidden_configuration
            dims_list = [input_dimension, latent_dimension]
            dims_list = dims_list[:1] + hidden_configuration + dims_list[1:]

            channel_dims = [(dims_list[i - 1], dims_list[i]) for i in range(1, len(dims_list))]

        ################## ENCODER #################
        self._encoder = nn.Sequential()
        for i in range(0, len(channel_dims), 1):
            self._encoder.append(
                LinearLayer(
                    in_features=channel_dims[i][0],
                    out_features=channel_dims[i][1],
                    nonlinearity=nonlinearity,
                    bias=True,
                    batchnorm=batchnorm,
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
        eigeninit(self._koopman_matrix.linear_layer.weight, theta=0.3)

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

    def _encode(self, x):
        ## Pre-encoder bias
        # x_bar = x - self.decoder[-1].linear_layer.bias
        return self.encoder(x)

    def _decode(self, x):
        return self.decoder(x)

    def forward(
        self,
        x: Float[Tensor, "batch features"],
        k: Int = 0,
        intermediate=False,
    ) -> AutoencoderResult:
        """Forward."""

        # Encode
        phi_x = self._encode(x)

        # Reconstruct
        x_recons = self._decode(phi_x)

        ##########################################
        # If you like slow code, you'll enjoy this!
        # It stores all intermediate predictions
        # but, we never actually use them, so off with its head.
        ##########################################
        if intermediate:
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

        ##########################################
        # Faster way, but no intermediate stores
        ##########################################
        else:
            K_weight = self.koopman_matrix.linear_layer.weight
            K_effective_weight = torch.linalg.matrix_power(K_weight, k)
            # NOTE: this K is transposed because of how torch handles matrix multiplication!
            x_k = phi_x @ K_effective_weight.T
            x_k = self._decode(x_k)

            # For compatibility
            x_k = x_k.unsqueeze(0)

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
    def load_model(
        cls,
        file_path: str | Path,
        strict: bool = True,
        remove_param: bool = True,
        **kwargs,
    ):
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
            batchnorm=literal_eval(metadata["batchnorm"]),
            hidden_configuration=literal_eval(metadata["hidden_configuration"]),
            rank=literal_eval(metadata["rank"]),
            **kwargs,
        )

        # Load weights
        load_model(model, file_path, device=get_device(), strict=strict)

        # Remove parameterizations
        if remove_param and torch.nn.utils.parametrize.is_parametrized(
            model.koopman_matrix.linear_layer
        ):
            torch.nn.utils.parametrize.remove_parametrizations(
                model.koopman_matrix.linear_layer, "weight"
            )

        return model, metadata

    def save_model(self, file_path: str | Path, **kwargs):
        """Save model."""

        metadata = {
            "input_dimension": str(self.input_dimension),
            "latent_dimension": str(self.latent_dimension),
            "hidden_configuration": str(self.hidden_configuration),
            "nonlinearity": str(self.nonlinearity),
            "steps": str(self.steps),
            "batchnorm": str(self.batchnorm),
            "rank": str(self.rank),
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        save_model(self, Path(file_path), metadata=metadata)


class MatrixExponential(nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        self.k = k  # Number of steps
        self.dim = dim

    def forward(self, X):
        return torch.matrix_exp(X / self.k)  # Scale M by 1/k

    # # Custom initialization function using matrix logarithm
    # def right_inverse(self, X):
    #     # Define a target matrix (e.g., an orthogonal matrix or one with specific eigenvalues)
    #     dummy_layer = nn.Linear(in_features=self.dim, out_features=self.dim)
    #     eigeninit(dummy_layer.weight, theta=1.0)
    #     target_matrix = dummy_layer.weight

    #     # Compute the matrix logarithm of the target matrix
    #     log_matrix = self.k * logm(target_matrix)

    #     return torch.Tensor(log_matrix)


class ExponentialKoopmanAutencoder(Autoencoder):
    """
    Autoencoder model.
    """

    def __init__(
        self,
        rank: int,
        k: int,
        input_dimension: int = 2,
        latent_dimension: int = 4,
        hidden_configuration: Optional[List[Int]] = None,
        nonlinearity: str = "leakyrelu",
        batchnorm: bool = False,
    ):
        super().__init__(
            k,
            input_dimension,
            latent_dimension,
            hidden_configuration,
            nonlinearity,
            batchnorm,
            rank,
        )

        parametrize.register_parametrization(
            self.koopman_matrix.linear_layer, "weight", MatrixExponential(k=k, dim=latent_dimension)
        )


class LowRankFactorization(nn.Module):
    """
    Parameterizes a matrix to be of rank r by factoring it into two matrices.
    Similar to how the Symmetric class ensures symmetry, this ensures low rank.
    """

    def __init__(self, n: int, r: int):
        super().__init__()
        self.n = n  # dimension of square matrix
        self.r = r  # target rank

    def forward(self, X):
        # Use the first n*r elements of X for left factor and the rest for right factor
        left = X[:, : self.r]  # Shape: n × r
        right = X[:, self.r :].t()  # Shape: r × n  (after transpose)
        return left @ right

    def right_inverse(self, X):
        # When someone assigns a matrix to the weight, decompose it via SVD
        U, S, Vh = torch.linalg.svd(X)
        left = U[:, : self.r] @ torch.diag(torch.sqrt(S[: self.r]))
        right = torch.diag(torch.sqrt(S[: self.r])) @ Vh[: self.r, :]
        # Return in the format our forward method expects
        return torch.cat([left, right.t()], dim=1)


class LowRankKoopmanAutoencoder(Autoencoder):
    """
    Autoencoder model.
    """

    def __init__(
        self,
        rank: int,
        k: int,
        input_dimension: int = 2,
        latent_dimension: int = 4,
        hidden_configuration: Optional[List[Int]] = None,
        nonlinearity: str = "leakyrelu",
        batchnorm: bool = False,
    ):
        super().__init__(
            k,
            input_dimension,
            latent_dimension,
            hidden_configuration,
            nonlinearity,
            batchnorm,
            rank,
        )

        parametrize.register_parametrization(
            self.koopman_matrix.linear_layer,
            "weight",
            LowRankFactorization(n=latent_dimension, r=rank),
        )
