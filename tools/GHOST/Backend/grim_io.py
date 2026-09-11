"""Public entry point for ghost_backend.io.grim."""
import sys
from ghost_backend.io import grim as _implementation

sys.modules[__name__] = _implementation
