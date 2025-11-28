__all__ = ["Hookable"]

from typing import Any, Optional


class Hookable:
    """Mixin to manage hooks for modules (layers, blocks)"""

    def __init__(self) -> None:
        self._forward_activations: Optional[Any] = None
        self._handle: Optional[Any] = None

    @property
    def forward_activations(self) -> Optional[Any]:
        """Get forward activations."""
        return self._forward_activations

    @property
    def is_hooked(self) -> bool:
        """Returns whether the module is hooked."""
        return bool(self._handle)

    def setup_hook(self, target_module=None):
        """Sets up a hook to capture activations."""

        # Remove any existing hook
        self.remove_hook()

        # Hook definition
        def _hook(module, input, output):
            self._forward_activations = output

        if target_module is None:
            target_module = self

        # Validate target has register_forward_hook to provide a clearer error for misuse
        if not hasattr(target_module, "register_forward_hook"):
            raise TypeError(
                "target_module must be a torch.nn.Module or support register_forward_hook"
            )

        # Housekeeping
        self._handle = target_module.register_forward_hook(_hook)

    def remove_hook(self):
        """Removes the hook."""

        # Housekeeping
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._forward_activations = None
        self._is_hooked = False
