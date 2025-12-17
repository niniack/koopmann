__all__ = [
    "KoopmanAutoencoder",
    "ParamExponentialKoopmanAutencoder",
]
from collections import namedtuple
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from koopmann.models.autoencoder import Autoencoder
from koopmann.models.layers import LinearLayer
from koopmann.models.utils import eigeninit

KoopmanAutoencoderResult = namedtuple(
    "KoopmanAutoencoderResult", "predictions reconstruction"
)


### Vanilla Koopman Autoencoder
class KoopmanAutoencoder(Autoencoder):
    """Standard Koopman autoencoder model."""

    def __init__(
        self,
        k_steps: int,
        in_features: int = 2,
        latent_features: int = 4,
        hidden_config: Optional[list[int]] = None,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: str = "leaky_relu",
        use_eigeninit: Optional[bool] = False,
    ):
        super().__init__(
            in_features=in_features,
            latent_features=latent_features,
            hidden_config=hidden_config,
            bias=bias,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
        )
        self.k_steps = k_steps

        koopman_matrix = LinearLayer(
            in_channels=latent_features,
            out_channels=latent_features,
            bias=False,
            batchnorm=False,
            nonlinearity=None,
        )
        koopman_matrix.components.linear.weight.data.copy_(torch.eye(latent_features))

        if use_eigeninit:
            eigeninit(koopman_matrix.components.linear.weight, theta=0.3)

        # Rebuild container
        # NOTE: Pytorch doesn't have a great way to insert into nn.Sequential
        temp_container = nn.Sequential()
        temp_container.add_module("encoder", self.components.encoder)
        temp_container.add_module("koopman_matrix", koopman_matrix)
        temp_container.add_module("decoder", self.components.decoder)
        self.components = temp_container

    @property
    def koopman_weights(self):
        return self.components.koopman_matrix.components.linear.weight

    def koopman_forward(self, observable, k_steps=None):
        if k_steps is None:
            k_steps = self.k_steps

        # NOTE: this K is transposed because of
        # how torch handles matrix multiplication!
        return observable @ torch.linalg.matrix_power(self.koopman_weights.T, k_steps)

    def forward(
        self, x: torch.Tensor, intermediate: bool = False, k: int | None = None
    ) -> KoopmanAutoencoderResult:
        """Forward method for koopman autoencoder."""

        phi_x = self.encode(x)
        x_recons = self.decode(phi_x)

        if k is None:
            k = self.k_steps

        # Stores all intermediate predictions
        if intermediate:
            # Advance latent variable k times
            prediction = [phi_x]
            for i in range(1, k + 1):
                prev_pred = prediction[i - 1]
                new_pred = self.koopman_forward(prev_pred, k_steps=1)
                prediction.append(new_pred)

            # Batched decoding
            # Shape: [steps, batch, latent_dim]
            stacked_predictions = torch.stack(prediction, dim=0)
            steps, batch_size, latent_dim = stacked_predictions.size()

            # Shape: [steps * batch, feature_dim]
            reshaped_predictions = stacked_predictions.view(-1, latent_dim)
            decoded = self.components.decoder(reshaped_predictions)

            # Shape: [steps, batch, latent_dim]
            x_k = decoded.view(steps, batch_size, -1)

        # Faster way, but no intermediate stores
        else:
            x_k = self.koopman_forward(phi_x, k_steps=k)
            x_k = self.decode(x_k)

            # For compatibility
            x_k = x_k.unsqueeze(0)

        return KoopmanAutoencoderResult(x_k, x_recons)

    def _get_basic_metadata(self) -> dict[str, Any]:
        """Get model-specific metadata for serialization."""
        metadata = super()._get_basic_metadata()
        metadata.update({"k_steps": self.k_steps})

        return metadata


### Exponential Koopman Autoencoder
class ParamExponentialKoopmanAutencoder(KoopmanAutoencoder):
    """Koopman autoencoder model with exp parameterization."""

    def __init__(
        self,
        k_steps: int,
        in_features: int = 2,
        latent_features: int = 4,
        hidden_config: Optional[list[int]] = None,
        bias: bool = True,
        batchnorm: bool = False,
        nonlinearity: str = "leaky_relu",
        use_eigeninit: Optional[bool] = False,
    ):
        super().__init__(
            k_steps,
            in_features,
            latent_features,
            hidden_config,
            bias,
            batchnorm,
            nonlinearity,
            use_eigeninit,
        )

        parametrize.register_parametrization(
            self.components.koopman_matrix.components.linear,
            "weight",
            MatrixExponential(
                k_steps=k_steps,
                latent_features=latent_features,
            ),
        )


class MatrixExponential(nn.Module):
    def __init__(self, k_steps, latent_features):
        super().__init__()
        self.k_steps = k_steps  # Number of steps
        self.latent_features = latent_features

    def forward(self, X):
        return torch.matrix_exp(X / self.k_steps)  # Scale M by 1/k
