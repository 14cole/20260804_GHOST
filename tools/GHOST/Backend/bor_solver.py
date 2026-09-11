"""Public entry point for ghost_backend.bor.solver."""
import sys
from ghost_backend.bor import solver as _implementation

sys.modules[__name__] = _implementation
