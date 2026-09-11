"""Public entry point for ghost_backend.geometry.io."""
import sys
from ghost_backend.geometry import io as _implementation

sys.modules[__name__] = _implementation
