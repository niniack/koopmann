from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from torchvision.ops import stochastic_depth  # Import the built-in function

from koopmann.models.layers import Layer, LinearLayer
from koopmann.models.utils import StringtoClassNonlinearity


class ResidualBlock(nn.Module):
    """
    Residual block for ResMLP, consisting of two fully-connected layers.
    Following the paper, each residual block contains two fully-connected layers.

    """

    def __init__(
        self,
        dimension: int = 512,
        nonlinearity: str | nn.Module = "relu",
        bias: bool = False,
        batchnorm: bool = True,
        hook: bool = False,
        drop_prob: float = 0.0,  # Probability for stochastic depth
        stoch_mode: str = "batch",  # Mode for stochastic depth: "batch" or "row"
    ):
        super().__init__()

        self.dimension = dimension
        self.hook = hook
        self.in_features = dimension
        self.out_features = dimension
        self._forward_activations = None
        self._handle = None
        self.drop_prob = drop_prob
        self.stoch_mode = stoch_mode

        # Create the residual branch
        if isinstance(nonlinearity, str):
            nonlinearity_module = StringtoClassNonlinearity[nonlinearity].value
        else:
            nonlinearity_module = nonlinearity

        # First fully-connected layer - Preserving direct fc1 reference
        self.fc1 = LinearLayer(
            in_features=dimension,
            out_features=dimension,
            nonlinearity=nonlinearity_module,
            bias=bias,
            batchnorm=batchnorm,
            hook=False,
        )
        self.fc1.apply(LinearLayer.init_weights)

        # Second fully-connected layer - Preserving direct fc2 reference
        self.fc2 = LinearLayer(
            in_features=dimension,
            out_features=dimension,
            nonlinearity=None,  # No nonlinearity after the second layer as per ResNet design
            bias=bias,
            batchnorm=batchnorm,
            hook=False,
        )
        self.fc2.apply(LinearLayer.init_weights)

        # If hook is requested, set it up
        if hook:
            self.setup_hook()

    @property
    def forward_activations(self) -> tuple:
        """Returns tensor of forward activations from hook."""
        return self._forward_activations

    @property
    def is_hooked(self) -> bool:
        """Returns boolean indicating whether layer has hook."""
        return self.hook

    def setup_hook(self):
        """Sets up a hook to capture activations."""

        def _hook(module, input, output):
            # The output is a tuple: (activated_out, activation_pattern)
            # Preserving exactly the same hook behavior
            self._forward_activations = output

        self.hook = True
        self._handle = self.register_forward_hook(_hook)

    def remove_hook(self):
        """Tears down the hook."""
        self.hook = False
        if self._handle:
            self._handle.remove()
            self._handle = None

    def forward(self, x):
        """
        Forward pass through the residual block.
        Returns both the activated output and the activation pattern.
        """
        identity = x

        # Apply first layer with activation - using direct references as in original
        out = self.fc1(x)  # contains batchnorm + ReLU

        # Apply second layer without activation - using direct references as in original
        out = self.fc2(out)  # does not contain ReLU

        # Apply stochastic depth using torchvision's implementation
        out = stochastic_depth(out, p=self.drop_prob, mode=self.stoch_mode, training=self.training)

        # Add the identity connection
        out_with_skip = out + identity

        # Apply ReLU activation
        activated_out = F.relu(out_with_skip)

        # Calculate activation pattern (which elements are positive)
        activation_pattern = out * (out_with_skip > 0)

        return activated_out, activation_pattern
