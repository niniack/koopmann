from collections import OrderedDict


class Hookable:
    """Mixin to manage hooks for modules (layers, blocks)"""

    def __init__(self):
        self._forward_activations = None
        self._handle = None
        self._is_hooked = False

    @property
    def forward_activations(self):
        """Get forward activations."""
        return self._forward_activations

    @property
    def is_hooked(self):
        """Returns whether the module is hooked."""
        return self._is_hooked

    def setup_hook(self, target_module=None):
        """Sets up a hook to capture activations."""
        # Remove any existing hook
        self.remove_hook()

        # Hook definition
        def _hook(module, input, output):
            self._forward_activations = output

        if target_module is None:
            target_module = self

        # Housekeeping
        self._is_hooked = True
        self._handle = target_module.register_forward_hook(_hook)

    def remove_hook(self):
        """Removes the hook."""

        # Housekeeping
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._forward_activations = None
        self._is_hooked = False
